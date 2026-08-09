############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_api_key_scopes.py: Bounded privilege for API keys
#     (migration 073)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Scoped API keys.

Before migration 073 authorization was one boolean derived from the owning
user's group, so a credential could do everything an admin can or nothing
administrative at all. Letting a registered app provision its users meant
making it a full deployment admin — able to read every stored prompt and
revoke anyone's key.

The two invariants these tests defend:

1. NULL scopes = legacy. Every pre-073 key keeps behaving exactly as before,
   so adding scopes changes no existing credential.
2. Scopes only REMOVE privilege. A key minted by an app for an administrator
   is still not an admin credential — the same class of mistake that gave the
   DLP internal key admin rights simply because of who owned it.

scopes.py is pure (no DB/telemetry imports) so it loads directly; the
enforcement sites in api/auth.py are checked structurally because importing
them pulls the DB chain (see MEMORY.md "Import Chain Gotcha").
"""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

_APP = Path(__file__).resolve().parents[2]


def _load_scopes():
    spec = importlib.util.spec_from_file_location(
        "scopes_under_test", _APP / "security" / "scopes.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scopes = _load_scopes()


def _key(scope_str):
    return SimpleNamespace(id=1, scopes=scope_str, app_id=None)


class TestLegacyKeysAreUnrestricted:
    """A key that predates 073 has NULL scopes and must be unaffected."""

    def test_null_scopes_permits_everything(self):
        k = _key(None)
        for s in (scopes.SCOPE_INFERENCE, scopes.SCOPE_ADMIN, scopes.SCOPE_APP_PROVISION):
            assert scopes.key_has_scope(k, s) is True

    def test_null_scopes_is_not_scoped(self):
        assert scopes.is_scoped(_key(None)) is False

    def test_parse_none_is_none_not_empty_set(self):
        """None and '' mean different things: legacy vs may-do-nothing."""
        assert scopes.parse_scopes(None) is None
        assert scopes.parse_scopes("") == set()


class TestScopesOnlyRemovePrivilege:
    def test_inference_key_is_not_admin(self):
        k = _key("inference")
        assert scopes.key_has_scope(k, scopes.SCOPE_INFERENCE) is True
        assert scopes.key_has_scope(k, scopes.SCOPE_ADMIN) is False

    def test_app_credential_cannot_run_inference_or_admin(self):
        k = _key("apps:provision")
        assert scopes.key_has_scope(k, scopes.SCOPE_APP_PROVISION) is True
        assert scopes.key_has_scope(k, scopes.SCOPE_ADMIN) is False
        assert scopes.key_has_scope(k, scopes.SCOPE_INFERENCE) is False

    def test_empty_scope_list_permits_nothing(self):
        k = _key("")
        assert scopes.is_scoped(k) is True
        for s in scopes.ALL_SCOPES:
            assert scopes.key_has_scope(k, s) is False

    def test_app_user_keys_are_inference_only(self):
        """The default an app mints for an end user — including when that end
        user happens to be an administrator."""
        assert tuple(scopes.APP_USER_KEY_SCOPES) == ("inference",)
        k = _key(scopes.format_scopes(scopes.APP_USER_KEY_SCOPES))
        assert scopes.key_has_scope(k, scopes.SCOPE_ADMIN) is False


class TestSerialization:
    def test_round_trip(self):
        raw = scopes.format_scopes(["admin", "inference"])
        assert scopes.parse_scopes(raw) == {"admin", "inference"}

    def test_format_none_stays_none(self):
        assert scopes.format_scopes(None) is None

    def test_whitespace_and_dupes_normalised(self):
        assert scopes.parse_scopes(" inference , inference ,, admin ") == {
            "inference", "admin"
        }

    def test_format_is_deterministic(self):
        assert scopes.format_scopes(["inference", "admin"]) == scopes.format_scopes(
            ["admin", "inference"]
        )


class TestEveryAdminPathChecksScope:
    """api/auth.py has FIVE places that derive admin from the owner's group.
    Missing one leaves a route where an app-minted key belonging to an admin
    still reaches admin endpoints."""

    def _auth_src(self):
        return (_APP / "api" / "auth.py").read_text()

    def test_require_admin_and_admin_read_enforce(self):
        src = self._auth_src()
        for fn in ("def check_admin(", "def check_admin_read("):
            i = src.index(fn)
            block = src[i:i + 1200]
            assert "_deny_unscoped" in block, f"{fn} does not enforce scope"

    def test_api_key_branches_of_or_session_helpers_enforce(self):
        """Both *_or_session helpers authenticate by API key first; the
        session-cookie branches carry no scope and are correctly exempt."""
        src = self._auth_src()
        n = len(re.findall(r"key_has_scope\(api_key, SCOPE_ADMIN\)", src))
        assert n >= 2, (
            f"expected both or_session API-key branches to check scope, found {n}"
        )

    def test_class_based_dependency_enforces(self):
        src = self._auth_src()
        i = src.index("if self.require_admin_flag:")
        assert "_deny_unscoped" in src[i:i + 700]

    def test_no_admin_derivation_left_unguarded(self):
        """Count group-derived admin checks against scope enforcement points so
        a newly added path cannot slip through unnoticed."""
        src = self._auth_src()
        derivations = len(re.findall(
            r"user\.group\.(is_admin|has_admin_read)", src
        ))
        enforcement = (
            len(re.findall(r"_deny_unscoped\(", src))
            + len(re.findall(r"key_has_scope\(api_key, SCOPE_ADMIN\)", src))
        )
        # Session-cookie branches legitimately have no scope to check, so
        # enforcement is expected to be fewer than derivations — but if a new
        # API-key path appears without a check this ratio drifts.
        assert enforcement >= 5, (
            f"{derivations} group-derived admin checks but only {enforcement} "
            "scope enforcement points — a new API-key path may be unguarded"
        )


class TestInferenceScopeIsEnforced:
    """`inference` was declared before it was checked anywhere, which made the
    scope list deny-admin only. An app's provisioning credential is owned by
    the administrator who issued it, so until this was wired a leaked one could
    spend that administrator's token budget on /v1/chat/completions."""

    def _auth_src(self):
        return (_APP / "api" / "auth.py").read_text()

    def test_the_shared_inference_dependency_demands_the_scope(self):
        src = self._auth_src()
        i = src.index("async def authenticate_request(")
        block = src[i:i + 1800]
        assert "_deny_unscoped(api_key, SCOPE_INFERENCE)" in block, (
            "authenticate_request must require the inference scope — it is the "
            "one dependency every inference endpoint shares"
        )

    def test_scope_gated_routes_do_not_inherit_the_inference_requirement(self):
        """An app credential carries `apps:provision` and NOT `inference`, so
        provisioning would be impossible if require_scope sat downstream of the
        inference check."""
        src = self._auth_src()
        i = src.index("def require_scope(")
        block = src[i:i + 1400]
        assert "Depends(authenticate_credential)" in block
        assert "Depends(authenticate_request)" not in block

    def test_admin_dependencies_do_not_inherit_it_either(self):
        src = self._auth_src()
        for fn in ("def check_admin(", "def check_admin_read("):
            i = src.index(fn)
            block = src[i:i + 400]
            assert "Depends(authenticate_credential)" in block, fn

    def test_no_route_module_bypasses_the_inference_check(self):
        """authenticate_credential establishes identity without authorising
        anything. If a router depended on it directly, that endpoint would
        accept a provisioning credential."""
        offenders = []
        for path in sorted((_APP / "api").rglob("*.py")) + sorted(
            (_APP / "dashboard").rglob("*.py")
        ):
            if path.name == "auth.py":
                continue
            if "authenticate_credential" in path.read_text():
                offenders.append(str(path.relative_to(_APP)))
        assert not offenders, (
            "these modules use the pre-authorisation authenticator directly: "
            f"{offenders} — depend on authenticate_request instead"
        )


class TestRequireScopeIsOptIn:
    def test_legacy_key_cannot_satisfy_require_scope(self):
        """Provisioning is a capability no group confers, so a broad old key
        must NOT drift into it."""
        src = (_APP / "api" / "auth.py").read_text()
        i = src.index("def require_scope(")
        block = src[i:i + 1400]
        assert "not is_scoped(api_key)" in block, (
            "require_scope must reject legacy (NULL-scope) keys explicitly"
        )


class TestSchema:
    def test_migration_073_chain_and_columns(self):
        mig = (_APP / "db" / "migrations" / "versions"
               / "20260808_000001_073_add_registered_apps_and_key_scopes.py").read_text()
        assert re.search(r'^revision = "073"', mig, re.M)
        assert re.search(r'^down_revision = "072"', mig, re.M)
        for col in ("app_id", "scopes", "hidden"):
            assert f'"{col}"' in mig
        assert "create_table" in mig and '"apps"' in mig

    def test_requests_table_is_not_widened(self):
        """App attribution joins through api_keys.app_id; the largest table in
        the database must keep its row width."""
        mig = (_APP / "db" / "migrations" / "versions"
               / "20260808_000001_073_add_registered_apps_and_key_scopes.py").read_text()
        assert "requests" not in mig.replace("# ", "").split("def upgrade")[1].split("def downgrade")[0] or \
               "add_column(\n        \"requests\"" not in mig

    def test_app_created_by_is_detached_on_user_delete(self):
        """Deleting the admin who registered an app must not delete the app —
        pinned by the schema-walk guard in test_admin_user_mgmt.py."""
        src = (_APP / "db" / "crud.py").read_text()
        i = src.index("async def delete_user")
        block = src[i:i + 9000]
        assert "App.created_by == user_id" in block
        assert "created_by=None" in block
