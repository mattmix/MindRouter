"""Tests for first-boot / first-admin bootstrap (2.9.8).

Three defects fixed here, all of which only bite a NEW deployment — the
case nobody on an existing install ever exercises:

1. _run_migrations was decorated @asynccontextmanager but awaited, so
   RUN_MIGRATIONS=1 raised TypeError and guaranteed the crash-loop it
   exists to prevent.
2. A supplied ADMIN_API_KEY without the mr2_ prefix was stored happily
   but could never authenticate.
3. RUN_MIGRATIONS had no docker-compose passthrough, so it was inert on
   the default stack.
"""

import ast
import inspect
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[4]
MAIN_SRC = (_APP_DIR / "main.py").read_text()
SEED_SRC = (_REPO / "scripts" / "seed_dev_data.py").read_text()


def _func_node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


class TestRunMigrations:
    def test_not_an_async_context_manager(self):
        """It is awaited directly in lifespan, so it must be a plain
        coroutine function. @asynccontextmanager would make it return an
        _AsyncGeneratorContextManager, which is not awaitable."""
        node = _func_node(MAIN_SRC, "_run_migrations")
        decorators = [ast.unparse(d) for d in node.decorator_list]
        assert "asynccontextmanager" not in " ".join(decorators), (
            f"_run_migrations must not be a context manager; got {decorators}"
        )
        assert isinstance(node, ast.AsyncFunctionDef)

    def test_has_no_yield(self):
        """A context-manager decorator would also require a yield; the
        absence of one is what made the old form fail at call time."""
        node = _func_node(MAIN_SRC, "_run_migrations")
        assert not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))

    def test_lifespan_awaits_it(self):
        assert "await _run_migrations()" in MAIN_SRC

    def test_the_old_shape_really_was_unawaitable(self):
        """Pin the reason: decorating a yield-less coroutine and awaiting
        it fails. If this ever stops being true, the guard above can relax."""

        @asynccontextmanager
        async def broken():
            return

        obj = broken()
        try:
            assert not hasattr(obj, "__await__"), (
                "an _AsyncGeneratorContextManager must not be awaitable — "
                "the 2.9.8 bug depended on exactly this"
            )
        finally:
            # broken() built a coroutine that is never awaited; close it so
            # the test doesn't emit a RuntimeWarning.
            obj.gen.close()

    def test_compose_passthrough_on_both_stacks(self):
        """DEPLOYMENT.md's Option B drives docker-compose.prod.yml, so the
        passthrough must exist there too — it was added only to the default
        stack at first, which made the documented command silently inert."""
        compose = (_REPO / "docker-compose.yml").read_text()
        assert "RUN_MIGRATIONS=${RUN_MIGRATIONS:-" in compose

        import yaml

        prod = yaml.safe_load((_REPO / "docker-compose.prod.yml").read_text())
        env = prod["services"]["app"].get("environment") or []
        entries = env if isinstance(env, list) else [f"{k}={v}" for k, v in env.items()]
        assert any(str(e).split("=")[0] == "RUN_MIGRATIONS" for e in entries), (
            "docker-compose.prod.yml must pass RUN_MIGRATIONS through"
        )
        # Bare form: `environment:` beats `env_file:`, so a `${VAR:-false}`
        # default here would clobber a value set in .env.prod.
        assert "RUN_MIGRATIONS" in entries, (
            "use the bare `- RUN_MIGRATIONS` form on the prod stack so it does "
            "not override .env.prod"
        )

    def test_in_process_upgrade_does_not_reconfigure_logging(self):
        """alembic's env.py calls fileConfig(), which disables existing
        loggers — in-process that would kill app logging permanently."""
        assert 'cfg.attributes["configure_logger"] = False' in MAIN_SRC
        env_src = (_APP_DIR / "db" / "migrations" / "env.py").read_text()
        assert 'config.attributes.get("configure_logger", True)' in env_src

    def test_in_process_upgrade_is_serialized(self):
        """Multiple uvicorn workers must not race concurrent DDL (MariaDB
        DDL is non-transactional, so a partial upgrade needs manual repair)."""
        assert "GET_LOCK" in MAIN_SRC and "RELEASE_LOCK" in MAIN_SRC


class TestSeedAdminApiKey:
    def test_rejects_key_without_expected_prefix(self):
        """verify_api_key() bails on the prefix before any lookup, so a
        badly-prefixed supplied key would 401 forever."""
        assert "API_KEY_PREFIX" in SEED_SRC
        assert "admin_api_key.startswith(API_KEY_PREFIX)" in SEED_SRC
        assert "SystemExit" in SEED_SRC

    def test_prefix_constant_is_imported_not_hardcoded(self):
        """Hardcoding 'mr2_' here would silently drift from the real
        constant the authenticator checks."""
        assert "from backend.app.security.api_keys import API_KEY_PREFIX" in SEED_SRC

    def test_auth_really_rejects_on_prefix_first(self):
        """The premise of the check: authentication rejects before lookup."""
        auth_src = (_APP_DIR / "security" / "api_keys.py").read_text()
        fn = auth_src.split("async def verify_api_key(")[1]
        head = fn.split("digest =")[0]
        assert "if not api_key.startswith(API_KEY_PREFIX):" in head
        assert "return None" in head


class TestBootstrapDocumented:
    """The SSO chicken-and-egg must be written down: SSO can never create
    the first admin, and the fix is order-dependent."""

    def test_deployment_documents_sso_admin_path(self):
        doc = (_REPO / "deploy" / "DEPLOYMENT.md").read_text()
        assert "Making an SSO identity the admin" in doc
        # The order-dependent bit is the whole point.
        assert "before your first SSO login" in doc
        # And the footgun must be named.
        assert "every" in doc.lower() and "*_DEFAULT_GROUP" in doc

    def test_deployment_documents_fresh_db_ordering(self):
        doc = (_REPO / "deploy" / "DEPLOYMENT.md").read_text()
        assert "fresh database" in doc.lower()
        assert "alembic upgrade head" in doc
        assert "RUN_MIGRATIONS=1" in doc

    def test_sso_config_points_at_the_bootstrap(self):
        doc = (_REPO / "docs" / "sso-configuration.md").read_text()
        assert "SSO cannot create your first admin" in doc
        assert "DEPLOYMENT.md" in doc

    def test_admin_api_key_prefix_requirement_documented(self):
        doc = (_REPO / "deploy" / "DEPLOYMENT.md").read_text()
        assert "mr2_" in doc and "ADMIN_API_KEY" in doc
