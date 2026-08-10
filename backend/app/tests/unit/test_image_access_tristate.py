############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_image_access_tristate.py: Image access as a global
#     default with per-user exceptions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Image generation access: global default + tri-state per-user override.

`users.image_generation_enabled` became nullable in migration 075. NULL means
"inherit the `img.enabled_by_default` global"; True and False are explicit
decisions that outrank it in both directions.

THE FAILURE MODE THIS FILE EXISTS TO PREVENT: a nullable boolean is FALSY.
`if not user.image_generation_enabled` in Python, `{% if ... %}` in Jinja and
`col == True` in SQL all read NULL as "no". Before 075 every user had an
explicit value so those tests were correct; afterwards almost every user
inherits, and any read site left un-migrated silently denies access to the
entire user base. There was zero test coverage of image gating before this
file — every one of those defects would have shipped green.

The drift guard is the load-bearing test here: it fails the build if the column
is read anywhere outside the resolver.
"""

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Boolean, Column

_APP = Path(__file__).resolve().parents[2]
_MIGRATIONS = _APP / "db" / "migrations" / "versions"
_TEMPLATES = _APP / "dashboard" / "templates"


def _load_feature_access():
    """Load the resolver without dragging in the db/telemetry import chain."""
    saved = {
        k: sys.modules.get(k)
        for k in (
            "backend", "backend.app", "backend.app.db", "backend.app.db.crud",
            "backend.app.db.session", "backend.app.logging_config",
        )
    }
    sys.modules.setdefault("backend", MagicMock())
    sys.modules.setdefault("backend.app", MagicMock())
    sys.modules.setdefault("backend.app.db", MagicMock())
    sys.modules["backend.app.db.crud"] = MagicMock()
    sys.modules["backend.app.db.session"] = MagicMock()
    sys.modules["backend.app.logging_config"] = MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "feature_access_under_test",
            _APP / "services" / "feature_access.py",
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


fa = _load_feature_access()


class TestResolutionTruthTable:
    """The whole policy in six cases."""

    @pytest.mark.parametrize(
        "override,default,expected",
        [
            (None, True, True),    # inherits ON  — the normal state after 075
            (None, False, False),  # inherits OFF
            (True, True, True),    # redundant grant
            (True, False, True),   # explicit grant beats a global OFF
            (False, True, False),  # explicit denial beats a global ON
            (False, False, False), # redundant denial
        ],
    )
    def test_resolution(self, override, default, expected):
        assert fa.resolve_feature_access(override, default) is expected

    def test_none_is_not_treated_as_false(self):
        """The single assertion the whole feature turns on."""
        assert fa.resolve_feature_access(None, True) is True
        assert fa.resolve_feature_access(False, True) is False


class TestSqlPredicates:
    def _col(self):
        return Column("image_generation_enabled", Boolean)

    def test_effective_access_includes_inheritors_when_default_on(self):
        sql = str(fa.access_filter(self._col(), True))
        assert "IS NULL" in sql, "inheriting users must count as having access"
        assert "IS true" in sql or "IS 1" in sql or "IS true" in sql.lower()

    def test_effective_access_excludes_inheritors_when_default_off(self):
        sql = str(fa.access_filter(self._col(), False))
        assert "IS NULL" not in sql

    def test_exception_selector_matches_the_crud_predicate(self):
        """ONE implementation of the exception rule.

        An earlier revision had feature_access build its own predicate that
        nothing called, while the admin route hand-derived the same rule — so
        these assertions guarded code that never ran. The policy now lives here
        as a selector and the SQL lives in crud.get_users; this pins the seam
        between them by executing BOTH.
        """
        crud_src = (_APP / "db" / "crud.py").read_text()
        i = crud_src.index("if image_override in")
        block = crud_src[i:i + 500]
        # The mapping crud actually applies.
        assert '"on": col.is_(True)' in block
        assert '"off": col.is_(False)' in block
        # With the default ON the exceptions are the denied, and vice versa.
        assert fa.exception_kind(True) == "off"
        assert fa.exception_kind(False) == "on"

    def test_a_redundant_override_is_never_an_exception(self):
        """An override equal to the global resolves identically to inheriting,
        so listing it would be noise the operator has to mentally filter."""
        assert fa.exception_kind(True) != fa.exception_kind(False)

    def test_the_admin_route_uses_the_selector(self):
        """Guards against the route re-deriving the rule inline again."""
        src = (_APP / "dashboard" / "routes.py").read_text()
        assert "feature_access.exception_kind(default_enabled)" in src
        assert '"off" if default_enabled else "on"' not in src


class TestLegacyBackupNormalization:
    """Restoring a pre-075 backup into a FRESH database is the disaster-recovery
    path, and it is the one that must fail closed.

    Every export taken before 075 records `false` for each user who was never
    individually granted access — 206 of 255 in production. Inserted verbatim
    they become explicit force-OFF rows and hard-deny the whole user base with
    no error and nothing in the logs.
    """

    def _pre075(self, users):
        return {"users": users, "app_config": [{"key": "img.enabled"}]}

    def _post075(self, users):
        return {"users": users, "app_config": [
            {"key": "img.enabled"}, {"key": "img.enabled_by_default", "value": "true"},
        ]}

    def test_pre075_denials_become_inherit(self):
        # Distinct dicts: `[{...}] * 3` would repeat one object, so a single
        # mutation would look like three and the count assertion would lie.
        data = self._pre075([{"image_generation_enabled": False} for _ in range(3)])
        assert fa.normalize_legacy_image_access(data) == 3
        assert all(u["image_generation_enabled"] is None for u in data["users"])

    def test_pre075_grants_are_preserved(self):
        data = self._pre075([
            {"image_generation_enabled": True}, {"image_generation_enabled": False},
        ])
        assert fa.normalize_legacy_image_access(data) == 1
        assert data["users"][0]["image_generation_enabled"] is True

    def test_integer_zero_is_treated_as_false(self):
        """JSON round-trips and older exporters both produce 0 rather than false."""
        data = self._pre075([{"image_generation_enabled": 0}])
        assert fa.normalize_legacy_image_access(data) == 1
        assert data["users"][0]["image_generation_enabled"] is None

    def test_post075_export_is_never_rewritten(self):
        """THE FAIL-OPEN THIS GUARDS. A small deployment where an admin has
        explicitly classified everyone contains no NULLs while being perfectly
        post-075. A heuristic that sniffed for NULLs would rewrite every
        deliberate DENIAL to inherit, and on restore those users would silently
        regain access that was revoked on purpose."""
        data = self._post075([
            {"image_generation_enabled": False},
            {"image_generation_enabled": True},
        ])
        assert fa.normalize_legacy_image_access(data) == 0
        assert data["users"][0]["image_generation_enabled"] is False

    def test_post075_export_with_nulls_is_untouched(self):
        data = self._post075([{"image_generation_enabled": None}])
        assert fa.normalize_legacy_image_access(data) == 0

    @pytest.mark.parametrize(
        "data",
        [
            {},
            {"users": []},
            {"users": "not-a-list"},
            {"users": [None, "junk", 7]},
            {"users": [{"username": "x"}]},   # column absent entirely (pre-058)
        ],
    )
    def test_malformed_input_never_raises(self, data):
        assert fa.normalize_legacy_image_access(data) == 0

    def test_discriminator_is_the_config_row_not_the_data_shape(self):
        """Pinned explicitly: the decision must rest on a fact the exporter
        wrote, not on an inference from user values."""
        src = (_APP / "services" / "feature_access.py").read_text()
        i = src.index("def normalize_legacy_image_access")
        block = src[i:i + 3000]
        assert 'r.get("key") == "img.enabled_by_default"' in block


class TestMissingUserFailsClosed:
    @pytest.mark.asyncio
    async def test_no_user_is_denied(self):
        assert await fa.image_generation_allowed(None, None) is False

    def test_nav_helper_handles_anonymous(self):
        assert fa.image_access(None) is False


class TestModelDeclaration:
    """Both defaults must stay gone.

    `default=False` is a PYTHON-side default: crud.create_user never mentions
    the column, so SQLAlchemy would write an explicit 0 on every INSERT and
    every SSO- and app-provisioned account would be born force-OFF.

    `server_default="0"` is the subtle one — with a DB-level default present
    the ORM omits the column from the INSERT even when it is explicitly
    assigned None, so "just pass None" does not rescue it.
    """

    def _line(self):
        src = (_APP / "db" / "models.py").read_text()
        for ln in src.splitlines():
            if ln.strip().startswith("image_generation_enabled:"):
                return ln
        raise AssertionError("image_generation_enabled declaration not found")

    def test_is_nullable(self):
        assert "nullable=True" in self._line()

    def test_carries_no_python_default(self):
        assert "default=False" not in self._line()

    def test_carries_no_server_default(self):
        assert "server_default" not in self._line()

    def test_is_optional_typed(self):
        assert "Mapped[Optional[bool]]" in self._line()


class TestVideoIsUntouched:
    """Video is a parallel opt-in feature and explicitly out of scope. It is
    declared on the adjacent line and is the obvious thing to change by
    accident."""

    def test_video_column_still_not_null(self):
        src = (_APP / "db" / "models.py").read_text()
        line = next(
            ln for ln in src.splitlines()
            if ln.strip().startswith("video_generation_enabled:")
        )
        assert "nullable=False" in line
        assert "server_default" in line

    def test_nav_still_reads_the_video_flag_directly(self):
        base = (_TEMPLATES / "base.html").read_text()
        assert "user.video_generation_enabled" in base
        assert "user.image_generation_enabled" not in base, (
            "the images nav must resolve through image_access(user)"
        )


class TestMigration:
    def _src(self):
        matches = list(_MIGRATIONS.glob("*_075_*.py"))
        assert len(matches) == 1, f"expected exactly one 075 migration, got {matches}"
        return matches[0].read_text()

    def test_revision_chain(self):
        src = self._src()
        assert re.search(r'^revision = "075"', src, re.M)
        assert re.search(r'^down_revision = "074"', src, re.M)

    def test_backfills_zeros_to_null(self):
        """Without this the change is a no-op for every previously-ungranted
        user, and the exception list opens full of phantom denials."""
        assert "SET image_generation_enabled = NULL" in self._src()
        assert "WHERE image_generation_enabled = 0" in self._src()

    def test_preserves_explicit_grants(self):
        """The 1s are the only surviving record of deliberate grants and are
        the allow-list if the global is ever flipped OFF."""
        src = self._src()
        assert "= NULL WHERE image_generation_enabled = 1" not in src.replace("\n", " ")

    def test_drops_the_server_default(self):
        """Passing existing_server_default WITHOUT server_default=None emits
        `BOOL NULL DEFAULT 0`, which keeps the DB default and quietly defeats
        inheritance on every INSERT."""
        assert "server_default=None" in self._src()

    def test_emitted_ddl_actually_drops_the_default(self):
        """Compile the real DDL rather than trusting the spelling.

        This is the whole migration in one assertion: if the emitted statement
        still carries DEFAULT 0, the column keeps a database-level default,
        every INSERT that omits it writes a forced deny, and inheritance never
        happens for a single new account.
        """
        import sqlalchemy as sa
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from sqlalchemy.dialects import mysql

        def emit(**kw):
            ctx = MigrationContext.configure(dialect=mysql.dialect())
            out = []
            ctx.impl._exec = lambda c, *a, **k: out.append(
                str(c.compile(dialect=mysql.dialect()))
            )
            Operations(ctx).alter_column(
                "users", "image_generation_enabled",
                existing_type=sa.Boolean(), nullable=True, existing_nullable=False,
                **kw,
            )
            return " ".join(out)

        as_written = emit(
            server_default=None, existing_server_default=sa.text("0")
        )
        assert "BOOL NULL" in as_written
        assert "DEFAULT" not in as_written, as_written

        # The trap, pinned so nobody "simplifies" the migration into it.
        trap = emit(existing_server_default=sa.text("0"))
        assert "DEFAULT 0" in trap

    def test_seeds_the_global(self):
        assert "img.enabled_by_default" in self._src()

    def test_downgrade_materialises_nulls_before_tightening(self):
        """MariaDB runs STRICT_TRANS_TABLES here and rejects the narrowing
        rather than coercing, so the UPDATE must come first."""
        src = self._src()
        down = src[src.index("def downgrade"):]
        assert "IS NULL" in down
        assert down.index("UPDATE users") < down.index("alter_column")


class TestConfigKeyConsistency:
    """get_config_json returns the CALLER's default when the row is absent, so
    one site with a different key or default is a silent policy fork."""

    def test_single_key_spelling(self):
        hits = set()
        for path in _APP.rglob("*.py"):
            if "tests" in path.parts:
                continue
            for m in re.finditer(r'["\'](img\.enabled_by_default|images\.enabled_by_default)["\']', path.read_text()):
                hits.add(m.group(1))
        assert hits <= {"img.enabled_by_default"}, (
            f"inconsistent config key spellings: {hits} — the img.* namespace "
            "is the established one and a second prefix reads as missing, "
            "which on an access flag denies everyone"
        )

    def test_resolver_defaults_to_enabled(self):
        assert fa.IMAGE_DEFAULT_KEY == "img.enabled_by_default"
        assert fa.IMAGE_DEFAULT_FALLBACK is True


class TestNoDirectColumnReads:
    """THE DRIFT GUARD. Resolution happens in one place; anything else that
    reads the raw column is a NULL-falsy bug waiting to happen."""

    ALLOWED = {
        "db/models.py",                          # the declaration
        "services/feature_access.py",            # the resolver + backup normalizer
        "db/crud.py",                            # the explicit tri-state filter
        "dashboard/routes.py",                   # admin read/write, all null-aware
        "dashboard/templates/admin/images_config.html",
        "dashboard/templates/admin/_image_override_badge.html",
    }

    def _offenders(self):
        offenders = []
        for pattern in ("*.py", "*.html"):
            for path in _APP.rglob(pattern):
                rel = str(path.relative_to(_APP))
                if "tests" in path.parts or "migrations" in path.parts:
                    continue
                if rel in self.ALLOWED:
                    continue
                if "image_generation_enabled" in path.read_text():
                    offenders.append(rel)
        return offenders

    def test_column_is_not_read_outside_the_resolver(self):
        assert not self._offenders(), (
            f"{self._offenders()} read users.image_generation_enabled directly. "
            "It is tri-state and NULL is falsy — use "
            "feature_access.image_generation_allowed(db, user) for gates, or "
            "feature_access.image_access(user) for nav visibility."
        )

    def test_the_gates_use_the_resolver(self):
        for rel in ("api/v1_openai.py", "dashboard/images.py"):
            src = (_APP / rel).read_text()
            assert "image_generation_allowed" in src, f"{rel} must gate via the resolver"


class TestJinjaGlobalRegistered:
    """base.html is rendered by four independent Jinja2Templates envs. A missing
    global raises 'image_access is undefined' at render time on that env only —
    the same drift that has already bitten `branding` here."""

    def test_every_env_registers_image_access(self):
        missing = []
        for path in (_APP / "dashboard").glob("*.py"):
            src = path.read_text()
            if "Jinja2Templates(" not in src:
                continue
            if 'globals["image_access"]' not in src:
                missing.append(path.name)
        assert not missing, f"these template envs lack the image_access global: {missing}"

    def test_nav_calls_the_helper(self):
        assert "image_access(user)" in (_TEMPLATES / "base.html").read_text()


class TestAdminOverrideControl:
    def _src(self):
        return (_APP / "dashboard" / "routes.py").read_text()

    def _code(self):
        """Just the images-config POST handler, comments stripped.

        Scoped deliberately: the VIDEO config handler further down still has a
        legitimate two-state `toggle_user`, because video is out of scope and
        its column is still NOT NULL. Asserting over the whole file would
        either fail on video or force video to change. The prose in this
        handler also quotes the old buggy expression to explain why it is gone,
        hence stripping comments.
        """
        src = self._src()
        start = src.index("async def admin_images_config_post")
        end = src.index("# Admin Video Generation Config", start)
        return "\n".join(
            ln for ln in src[start:end].splitlines()
            if not ln.lstrip().startswith("#")
        )

    def test_the_negating_toggle_is_gone(self):
        """`not target_user.image_generation_enabled` on a tri-state column
        evaluates `not None` to True, so touching an inheriting user wrote a
        silent force-ON that survived the global being turned off. It is the
        only defect in this feature that fails OPEN."""
        assert "not target_user.image_generation_enabled" not in self._code()
        assert 'action == "toggle_user"' not in self._code()

    def test_override_is_set_from_an_explicit_enum(self):
        src = self._src()
        assert 'action == "set_user_override"' in src
        assert '{"allow": True, "deny": False, "inherit": None}' in src

    def test_invalid_values_are_rejected(self):
        src = self._src()
        i = src.index('action == "set_user_override"')
        block = src[i:i + 1600]
        assert '("allow", "deny", "inherit")' in block

    def test_audit_records_the_real_prior_value(self):
        """None must reach the log as null, so "was inheriting" is
        distinguishable from "was explicitly denied"."""
        src = self._src()
        assert "before_val = target_user.image_generation_enabled" in src
        assert 'before_value={"image_generation_enabled": before_val}' in src

    def test_flipping_the_global_never_rewrites_user_rows(self):
        """Keeping a redundant override costs nothing; deleting one silently
        revokes access the next time the global goes OFF."""
        code = self._code()
        i = code.index('if action == "save_config"')
        block = code[i:code.index('elif action == "set_user_override"')]
        assert "image_generation_enabled" not in block, (
            "the global-default handler must not write per-user overrides"
        )


class TestOverrideBadgeUsesNullTest:
    def test_badge_tests_for_none_explicitly(self):
        """A truthiness test would render 'Force OFF' for an inheriting user —
        the confusion that makes an operator revoke access they never granted."""
        src = (_TEMPLATES / "admin" / "_image_override_badge.html").read_text()
        assert "row.override is none" in src
        assert "Inherit" in src and "Force ON" in src and "Force OFF" in src
