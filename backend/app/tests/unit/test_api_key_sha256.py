############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_api_key_sha256.py: SHA-256 fast-path API-key verification
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for SHA-256 API-key verification with verify-and-upgrade.

Covers:
- generate_api_key 4-tuple: digest matches key, Argon2 hash kept, prefix format
- Fast path: sha256 lookup hit returns row, no prefix lookup, no Argon2 work
- Belt-and-braces digest mismatch on fast path rejected
- Fallback: prefix + Argon2 verify, key_sha256 backfilled on success
- Wrong key on fallback: rejected, no backfill
- Garbage keys rejected without any DB lookup / Argon2 work
- Revoked-key enforcement stays in the caller (auth.py source contract)
- Argon2 fallback bounded by Semaphore(4) via asyncio.to_thread
- Migration 069 / crud / models / caller source contracts

api_keys.py is spec-loaded with backend.app.db* pre-mocked to avoid the
package import chain — see MEMORY.md "Import Chain Gotcha".
"""

import asyncio
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_SECURITY_DIR = _APP_DIR / "security"
_DB_DIR = _APP_DIR / "db"
_MIGRATION = _DB_DIR / "migrations" / "versions" / "20260801_000000_069_add_api_key_sha256.py"


def _load_api_keys():
    """Spec-load api_keys.py, pre-mocking the backend.app.db package chain."""
    saved = {}
    for name in [
        "backend",
        "backend.app",
        "backend.app.db",
        "backend.app.db.crud",
        "backend.app.db.models",
    ]:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = MagicMock()

    spec = importlib.util.spec_from_file_location(
        "mr2_api_keys_under_test", _SECURITY_DIR / "api_keys.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, saved


@pytest.fixture(scope="module")
def api_keys():
    mod, saved = _load_api_keys()
    yield mod
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def crud_mock(api_keys):
    """Fresh crud mock per test so call assertions don't leak between tests."""
    crud = MagicMock()
    crud.get_api_key_by_sha256 = AsyncMock(return_value=None)
    crud.get_api_key_by_prefix = AsyncMock(return_value=None)
    original = api_keys.crud
    api_keys.crud = crud
    yield crud
    api_keys.crud = original


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _db_key(**kwargs) -> SimpleNamespace:
    defaults = dict(id=1, key_hash="", key_sha256=None, status="active")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ===================================================================
# generate_api_key
# ===================================================================

class TestGenerate:
    def test_returns_four_tuple_with_matching_digests(self, api_keys):
        full_key, key_hash, key_prefix, key_sha256 = api_keys.generate_api_key()
        assert full_key.startswith("mr2_")
        assert key_sha256 == _digest(full_key)
        assert len(key_sha256) == 64
        # Argon2 hash still written (rollback safety)
        assert key_hash.startswith("$argon2")
        assert api_keys._verify_key_hash(full_key, key_hash)
        assert key_prefix == full_key[:12]

    def test_entropy_invariant_pinned(self, api_keys):
        # The plain-SHA-256 column is only safe with token_urlsafe(32) entropy
        src = (_SECURITY_DIR / "api_keys.py").read_text()
        assert "secrets.token_urlsafe(32)" in src
        assert "SECURITY INVARIANT" in src


# ===================================================================
# verify_api_key — fast path
# ===================================================================

class TestFastPath:
    async def test_hit_skips_prefix_lookup_and_argon2(self, api_keys, crud_mock, monkeypatch):
        full_key, key_hash, _, key_sha256 = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=key_sha256)
        crud_mock.get_api_key_by_sha256 = AsyncMock(return_value=row)
        # Any Argon2 verify on the fast path is a failure
        monkeypatch.setattr(
            api_keys, "_verify_key_hash",
            MagicMock(side_effect=AssertionError("Argon2 ran on fast path")),
        )

        result = await api_keys.verify_api_key(MagicMock(), full_key)

        assert result is row
        crud_mock.get_api_key_by_sha256.assert_awaited_once_with(
            crud_mock.get_api_key_by_sha256.await_args.args[0], key_sha256
        )
        crud_mock.get_api_key_by_prefix.assert_not_awaited()

    async def test_stored_digest_mismatch_rejected(self, api_keys, crud_mock):
        full_key, key_hash, _, _ = api_keys.generate_api_key()
        # Belt-and-braces: a row whose stored digest doesn't match is rejected
        row = _db_key(key_hash=key_hash, key_sha256="0" * 64)
        crud_mock.get_api_key_by_sha256 = AsyncMock(return_value=row)

        assert await api_keys.verify_api_key(MagicMock(), full_key) is None


# ===================================================================
# verify_api_key — Argon2 fallback + backfill upgrade
# ===================================================================

class TestFallback:
    async def test_argon2_verify_backfills_sha256(self, api_keys, crud_mock):
        full_key, key_hash, key_prefix, key_sha256 = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)

        result = await api_keys.verify_api_key(MagicMock(), full_key)

        assert result is row
        # Verify-and-upgrade: digest backfilled onto the ORM row
        assert row.key_sha256 == key_sha256
        crud_mock.get_api_key_by_prefix.assert_awaited_once()
        assert crud_mock.get_api_key_by_prefix.await_args.args[1] == key_prefix

    async def test_wrong_key_rejected_no_backfill(self, api_keys, crud_mock):
        _, other_hash, _, _ = api_keys.generate_api_key()
        row = _db_key(key_hash=other_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)

        wrong_key = "mr2_" + "A" * 43
        assert await api_keys.verify_api_key(MagicMock(), wrong_key) is None
        assert row.key_sha256 is None

    def test_argon2_bounded_by_semaphore_and_to_thread(self, api_keys):
        # 64 MiB per verify — the cap prevents RSS blowup under key floods
        assert isinstance(api_keys._argon2_verify_semaphore, asyncio.Semaphore)
        assert api_keys._argon2_verify_semaphore._value == 4
        src = (_SECURITY_DIR / "api_keys.py").read_text()
        assert "asyncio.to_thread(_verify_key_hash" in src


# ===================================================================
# verify_api_key — garbage keys
# ===================================================================

class TestGarbageKeys:
    async def test_wrong_prefix_rejected_without_db(self, api_keys, crud_mock):
        assert await api_keys.verify_api_key(MagicMock(), "sk-not-ours-at-all") is None
        crud_mock.get_api_key_by_sha256.assert_not_awaited()
        crud_mock.get_api_key_by_prefix.assert_not_awaited()

    async def test_unknown_mr2_key_rejected_without_argon2(self, api_keys, crud_mock, monkeypatch):
        monkeypatch.setattr(
            api_keys, "_verify_key_hash",
            MagicMock(side_effect=AssertionError("Argon2 ran with no candidate row")),
        )
        assert await api_keys.verify_api_key(MagicMock(), "mr2_" + "x" * 43) is None
        crud_mock.get_api_key_by_sha256.assert_awaited_once()
        crud_mock.get_api_key_by_prefix.assert_awaited_once()


# ===================================================================
# Revocation/expiry/user-active enforcement stays in the callers
# ===================================================================

class TestCallerEnforcement:
    async def test_verify_returns_revoked_row_for_caller_to_reject(self, api_keys, crud_mock):
        # verify_api_key proves possession only — status is the caller's job
        full_key, key_hash, _, key_sha256 = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=key_sha256, status="revoked")
        crud_mock.get_api_key_by_sha256 = AsyncMock(return_value=row)
        assert await api_keys.verify_api_key(MagicMock(), full_key) is row

    def test_auth_checks_run_after_verify(self):
        # auth.authenticate_request must reject revoked/expired keys and
        # inactive users AFTER verify_api_key — fast path included
        src = (_APP_DIR / "api" / "auth.py").read_text()
        body = src[src.index("async def authenticate_request"):]
        verify_pos = body.index("await verify_api_key(db, api_key_str)")
        assert body.index("api_key.status != ApiKeyStatus.ACTIVE") > verify_pos
        assert body.index("API key has expired") > verify_pos
        assert body.index("not user or not user.is_active or user.deleted_at") > verify_pos


# ===================================================================
# Source contracts: migration, crud, models, generate_api_key callers
# ===================================================================

class TestSourceContracts:
    def test_migration_069(self):
        src = _MIGRATION.read_text()
        assert 'revision = "069"' in src
        assert 'down_revision = "068"' in src
        assert '"api_keys", sa.Column("key_sha256", sa.CHAR(64), nullable=True)' in src
        assert '"uq_api_keys_key_sha256"' in src
        assert "unique=True" in src
        # Working downgrade: index first, then column
        downgrade = src[src.index("def downgrade"):]
        assert 'op.drop_index("uq_api_keys_key_sha256"' in downgrade
        assert 'op.drop_column("api_keys", "key_sha256")' in downgrade

    def test_crud_sha256_lookup_and_create(self):
        src = (_DB_DIR / "crud.py").read_text()
        fn = src[src.index("async def get_api_key_by_sha256"):]
        fn = fn[: fn.index("\n\n\nasync def")]
        assert "selectinload(ApiKey.user).selectinload(User.group)" in fn
        # Unique column: scalar_one_or_none is safe (no MultipleResultsFound)
        assert "scalar_one_or_none" in fn
        assert "ApiKey.key_sha256 == key_sha256" in fn
        create = src[src.index("async def create_api_key"):]
        create = create[: create.index("\n\n\nasync def")]
        assert "key_sha256: Optional[str] = None" in create
        assert "key_sha256=key_sha256" in create

    def test_model_column(self):
        src = (_DB_DIR / "models.py").read_text()
        assert (
            "key_sha256: Mapped[Optional[str]] = mapped_column"
            "(String(64), unique=True, nullable=True, index=True)" in src
        )

    @pytest.mark.parametrize(
        "rel_path",
        [
            "dashboard/routes.py",
            "api/admin_api.py",
            "services/dlp_worker.py",
        ],
    )
    def test_callers_store_both_columns(self, rel_path):
        src = (_APP_DIR / rel_path).read_text()
        assert "full_key, key_hash, key_prefix, key_sha256 = generate_api_key()" in src
        assert "key_sha256=key_sha256" in src
