############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_config_export_redaction.py: Role-scoped config backup
#     export (2.9.9)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Role-scoping for /admin/backup/export (2.9.9).

The export dumps every configuration table with no redaction, and was gated
only on ``has_admin_read`` — which includes read-only auditors. That handed
an auditor local-account password hashes, API key hashes, and the secret
app_config values, several of which (``catalog.enrich_api_key``,
``catalog.brave_api_key``) are directly usable credentials rather than
hashes.

Full admins still get a restorable dump. Anyone below that gets the same
configuration with credential material nulled, and a redacted file is
refused by the restore path — importing one would create local accounts
with no password hash and API keys with no hash at all.

crud.py is spec-loaded via its source rather than imported, to avoid the
db package import chain (see MEMORY.md "Import Chain Gotcha").
"""

import ast
import json
import re
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
CRUD_SRC = (_APP_DIR / "db" / "crud.py").read_text()
ROUTES_SRC = (_APP_DIR / "dashboard" / "routes.py").read_text()
BACKUP_HTML = (_APP_DIR / "dashboard" / "templates" / "admin" / "backup.html").read_text()


def _extract(name):
    """Return the source of one top-level (async) def or assignment block."""
    pattern = re.compile(
        rf"^(?:async )?def {name}\(.*?(?=^@|^(?:async )?def |^class |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(CRUD_SRC)
    assert m, f"{name} not found in crud.py"
    return m.group(0)


def _load_redactor():
    """Execute just the redaction helpers, with no imports."""
    ns = {"re": re}
    tree = ast.parse(CRUD_SRC)
    wanted = {"_SECRET_TABLE_COLUMNS", "_SECRET_CONFIG_KEY_RE", "_REDACTED"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & wanted:
                exec(compile(ast.Module([node], []), "<crud>", "exec"), ns)
        elif isinstance(node, ast.FunctionDef) and node.name == "_redact_row":
            exec(compile(ast.Module([node], []), "<crud>", "exec"), ns)
    assert "_redact_row" in ns, "_redact_row not found"
    return ns


@pytest.fixture(scope="module")
def redactor():
    return _load_redactor()


class TestRedactRow:
    """Behavioral: the secret value must not survive in the emitted payload."""

    def test_password_hash_is_nulled(self, redactor):
        row = redactor["_redact_row"](
            "users", {"username": "alice", "password_hash": "$argon2id$v=19$SECRET"}
        )
        assert row["password_hash"] is None
        assert row["username"] == "alice"

    def test_api_key_hashes_are_nulled(self, redactor):
        row = redactor["_redact_row"](
            "api_keys",
            {"name": "k", "key_hash": "$argon2id$HASH", "key_sha256": "abc123", "key_prefix": "mr2_abcd"},
        )
        assert row["key_hash"] is None
        assert row["key_sha256"] is None
        assert row["key_prefix"] == "mr2_abcd", "the display prefix is not a secret"

    @pytest.mark.parametrize(
        "key",
        [
            "email.smtp_password",
            "catalog.enrich_api_key",
            "catalog.brave_api_key",
            "voice.tts_api_key",
            "voice.stt_api_key",
            "dlp.internal_api_key_raw",
            "saml.sp_private_key",
        ],
    )
    def test_secret_config_values_are_nulled(self, redactor, key):
        row = redactor["_redact_row"]("app_config", {"key": key, "value": "SENTINEL"})
        assert row["value"] is None, f"{key} leaked"

    @pytest.mark.parametrize(
        "key",
        [
            "ocr.max_tokens",
            "stats.token_offset",
            "vid.token_cost_per_second",
            "retention.batch_size",
            "branding.org_name",
            "dlp.enabled",
        ],
    )
    def test_ordinary_settings_survive(self, redactor, key):
        """Matching bare 'token' would blank three real settings."""
        row = redactor["_redact_row"]("app_config", {"key": key, "value": "42"})
        assert row["value"] == "42", f"{key} was redacted but is not a secret"

    def test_non_secret_tables_untouched(self, redactor):
        row = redactor["_redact_row"]("groups", {"name": "admin", "token_budget": 100})
        assert row == {"name": "admin", "token_budget": 100}

    def test_no_sentinel_survives_serialization(self, redactor):
        """Assert on the serialized payload, not the dict — a nested
        representation must not sneak the value through."""
        payload = [
            redactor["_redact_row"]("users", {"password_hash": "SENTINEL-PW"}),
            redactor["_redact_row"]("api_keys", {"key_hash": "SENTINEL-KEY"}),
            redactor["_redact_row"]("app_config", {"key": "email.smtp_password", "value": "SENTINEL-SMTP"}),
        ]
        blob = json.dumps(payload)
        for sentinel in ("SENTINEL-PW", "SENTINEL-KEY", "SENTINEL-SMTP"):
            assert sentinel not in blob


class TestExportContract:
    def test_export_takes_a_redact_flag_defaulting_off(self):
        src = _extract("export_config_tables")
        assert "redact_secrets: bool = False" in src, "full export must stay the default"
        assert "if redact_secrets:" in src

    def test_export_stamps_the_metadata_flag(self):
        src = _extract("export_config_tables")
        assert '"redacted": redact_secrets' in src

    def test_import_refuses_a_redacted_backup(self):
        src = _extract("import_config_tables")
        assert 'data["metadata"].get("redacted")' in src
        assert "RedactedBackupError" in src
        assert "class RedactedBackupError" in CRUD_SRC

    def test_secret_matcher_excludes_bare_token(self, redactor):
        """ocr.max_tokens / stats.token_offset / vid.token_cost_per_second are
        real prod settings that a 'token' match would blank.

        Asserted against the compiled pattern, not a source slice — slicing on
        the first ')' goes vacuous the moment the pattern grows a group.
        """
        pattern = redactor["_SECRET_CONFIG_KEY_RE"]
        assert not pattern.search("ocr.max_tokens")
        assert not pattern.search("stats.token_offset")
        assert not pattern.search("vid.token_cost_per_second")
        assert pattern.search("email.smtp_password")


class TestRedactionCoversEveryCredentialColumn:
    """A drift guard over the exported schema.

    The first version of this change redacted `users` and `api_keys` only and
    shipped `nodes.sidecar_key` — a PLAINTEXT shared secret that authenticates
    every GPU node's sidecar and gates /ollama/pull and /ollama/delete. Every
    other surface returns just `sidecar_key_set: bool`; the export was the one
    place emitting the raw value. This test fails when a new credential-shaped
    column appears in an exported table and is not classified.
    """

    # Columns in exported tables that look credential-shaped and are knowingly
    # NOT redacted, with the reason. Anything not here and not redacted fails.
    KNOWN_NON_SECRETS = {
        # first 8 chars of a high-entropy key; the identifier the admin UI shows
        ("api_keys", "key_prefix"),
        # opaque IdP subject identifier, not a credential; needed so a restored
        # account still links to its SSO identity
        ("users", "azure_oid"),
        ("users", "sso_subject"),
        ("users", "sso_provider"),
        # integer policy settings, not credentials
        ("groups", "api_key_expiry_days"),
        ("groups", "max_api_keys"),
    }

    def _exported_tables(self):
        block = CRUD_SRC[CRUD_SRC.index("_CONFIG_TABLES = ["):]
        block = block[: block.index("]")]
        return set(re.findall(r"\(\s*(\w+)\s*,", block))

    def test_no_unclassified_credential_columns(self, redactor):
        import ast as _ast

        models_src = (_APP_DIR / "db" / "models.py").read_text()
        tree = _ast.parse(models_src)

        # class name -> (__tablename__, [column names])
        classes = {}
        for node in tree.body:
            if not isinstance(node, _ast.ClassDef):
                continue
            tablename, cols = None, []
            for stmt in node.body:
                if isinstance(stmt, _ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, _ast.Name) and t.id == "__tablename__":
                            if isinstance(stmt.value, _ast.Constant):
                                tablename = stmt.value.value
                elif isinstance(stmt, _ast.AnnAssign) and isinstance(stmt.target, _ast.Name):
                    # Real columns only — a relationship() is not exported.
                    fn = stmt.value.func if isinstance(stmt.value, _ast.Call) else None
                    fn_name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if fn_name == "mapped_column":
                        cols.append(stmt.target.id)
            if tablename:
                classes[node.name] = (tablename, cols)

        secret_re = re.compile(
            r"password|secret|api_key|_key$|^key_|private|credential|token_hash|salt",
            re.IGNORECASE,
        )
        redacted = redactor["_SECRET_TABLE_COLUMNS"]

        offenders = []
        for cls_name in self._exported_tables():
            if cls_name not in classes:
                continue
            table, cols = classes[cls_name]
            for col in cols:
                if not secret_re.search(col):
                    continue
                if col in redacted.get(table, set()):
                    continue
                if (table, col) in self.KNOWN_NON_SECRETS:
                    continue
                offenders.append(f"{table}.{col}")

        assert not offenders, (
            "credential-shaped columns are exported unredacted: "
            f"{sorted(offenders)} — add them to _SECRET_TABLE_COLUMNS in crud.py, "
            "or to KNOWN_NON_SECRETS here with a reason."
        )

    def test_sidecar_key_specifically_is_redacted(self, redactor):
        row = redactor["_redact_row"](
            "nodes", {"name": "lynx", "sidecar_url": "http://lynx:9100", "sidecar_key": "SENTINEL"}
        )
        assert row["sidecar_key"] is None
        assert row["sidecar_url"] == "http://lynx:9100"


class TestExportRouteScoping:
    def test_route_redacts_for_non_admins(self):
        i = ROUTES_SRC.index("async def admin_backup_export")
        fn = ROUTES_SRC[i: ROUTES_SRC.index("@dashboard_router.post", i)]
        assert "redact = not user.group.is_admin" in fn
        assert "redact_secrets=redact" in fn

    def test_redacted_download_is_named_distinctly(self):
        i = ROUTES_SRC.index("async def admin_backup_export")
        fn = ROUTES_SRC[i: ROUTES_SRC.index("@dashboard_router.post", i)]
        assert '"-redacted" if redact else ""' in fn

    def test_export_is_audit_logged(self):
        i = ROUTES_SRC.index("async def admin_backup_export")
        fn = ROUTES_SRC[i: ROUTES_SRC.index("@dashboard_router.post", i)]
        assert "log_admin_action" in fn

    def test_auditors_are_told_their_copy_is_redacted(self):
        assert "is <strong>redacted</strong>" in BACKUP_HTML
        assert "{% if export_is_redacted %}" in BACKUP_HTML

    def test_banner_and_export_agree_under_masquerade(self):
        """The banner must key off the REAL session user, as the export does.

        masq["is_read_only"] describes the EFFECTIVE user, so an admin
        masquerading as an auditor would be told the download is redacted and
        then handed a full one — a false PASS for exactly the workflow an admin
        would use to verify this control.
        """
        i = ROUTES_SRC.index("async def admin_backup(")
        fn = ROUTES_SRC[i: ROUTES_SRC.index("@dashboard_router.get", i + 10)]
        assert '"export_is_redacted": not user.group.is_admin' in fn
        assert "{% if is_read_only %}" not in BACKUP_HTML.split("Export Card")[1].split("Restore Card")[0]

    def test_audit_entry_matches_house_naming(self):
        """Other config actions use <area>.<verb> with entity_type='config';
        a bespoke entity_type creates a bucket nobody filters on."""
        i = ROUTES_SRC.index("async def admin_backup_export")
        fn = ROUTES_SRC[i: ROUTES_SRC.index("@dashboard_router.post", i)]
        assert 'action="backup.export"' in fn
        assert 'entity_type="config"' in fn


class TestSecretsNotRenderedIntoPageSource:
    """`type="password"` masks a value on screen but not in View Source.

    /admin/voice-config and /admin/search-config are gated on has_admin_read,
    so rendering the key into value= handed the same read-only auditor the very
    secrets the export redaction removes. /admin/settings already had this
    right: placeholder="(key configured)", never value=.
    """

    @pytest.mark.parametrize(
        "template,field",
        [
            ("voice_config.html", "tts_api_key"),
            ("voice_config.html", "stt_api_key"),
            ("search_config.html", "brave_api_key"),
        ],
    )
    def test_key_is_not_round_tripped_into_the_form(self, template, field):
        html = (_APP_DIR / "dashboard" / "templates" / "admin" / template).read_text()
        assert f'value="{{{{ {field} or \'\' }}}}"' not in html, (
            f"{template}: {field} is rendered into value= and is readable in page source"
        )
        assert f'name="{field}"' in html, "the field itself should still exist"
        assert "(key configured)" in html, "the placeholder should show whether one is set"

    @pytest.mark.parametrize(
        "guard,clear_field",
        [
            ("if tts_api_key:", "clear_tts_api_key"),
            ("if stt_api_key:", "clear_stt_api_key"),
            ("if brave_key:", "clear_brave_api_key"),
        ],
    )
    def test_blank_submit_keeps_the_stored_key(self, guard, clear_field):
        """Since the form no longer round-trips the value, a blank field must
        mean 'unchanged' — otherwise saving either page wipes the key. An
        explicit Clear checkbox is the only way to remove one."""
        assert guard in ROUTES_SRC, f"missing keep-current guard: {guard}"
        assert f'form.get("{clear_field}") == "on"' in ROUTES_SRC, (
            f"{clear_field} must be honoured, or a key can never be removed"
        )

    @pytest.mark.parametrize(
        "template,clear_field",
        [
            ("voice_config.html", "clear_tts_api_key"),
            ("voice_config.html", "clear_stt_api_key"),
            ("search_config.html", "clear_brave_api_key"),
        ],
    )
    def test_clear_control_is_offered_when_a_key_is_set(self, template, clear_field):
        html = (_APP_DIR / "dashboard" / "templates" / "admin" / template).read_text()
        assert f'name="{clear_field}"' in html

    def test_restore_never_reflects_exception_text(self):
        """This path inserts uploaded rows: a DBAPIError stringifies its bound
        parameters, including password hashes, into the rendered page."""
        i = ROUTES_SRC.index("async def admin_backup_restore")
        fn = ROUTES_SRC[i: i + 6000]
        assert 'f"Restore failed: {exc}"' not in fn
        assert "Restore failed — see the server log for details." in fn
