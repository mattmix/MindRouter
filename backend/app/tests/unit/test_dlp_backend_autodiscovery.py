############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_backend_autodiscovery.py: Unit tests for the
#     authoritative auto-discovery of the off-host GLiNER
#     pool from engine=dlp fleet backends.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for DLP off-host pool auto-discovery in ``_load_dlp_config``.

DLP is a first-class backend engine: a node can host a vLLM backend on one GPU
and a DLP (GLiNER) backend on another. When the off-host GLiNER scanner is
enabled but no endpoints were entered by hand, ``_load_dlp_config`` treats the
HEALTHY ``engine=dlp`` fleet backends as the endpoint pool (authoritative
health). A hand-entered endpoint list always overrides discovery, and any
failure leaves the configured endpoints untouched (never raises).

dlp_worker.py is spec-loaded with its dependencies pre-mocked in sys.modules so
the real function executes — see MEMORY.md "Import Chain Gotcha" (Fix A + Fix B)
and the recipe at the top of test_dlp_worker.py. dlp_scanner.py is a pure-logic
module (its only backend import is logging_config), so it is spec-loaded too to
exercise the REAL parse_remote_endpoints manual/legacy normalization.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_SERVICES_DIR = _APP_DIR / "services"


def _spec_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_with_logging_mock(name, filename):
    """Spec-load a services module with backend.app.logging_config mocked.

    Restores sys.modules exactly afterwards: a leaked MagicMock of
    backend.app.* silently breaks unrelated modules later in the run.
    """
    keys = ("backend", "backend.app", "backend.app.logging_config")
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules.setdefault("backend", MagicMock())
    sys.modules.setdefault("backend.app", MagicMock())
    sys.modules["backend.app.logging_config"] = MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    )
    try:
        return _spec_load(name, _SERVICES_DIR / filename)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def worker():
    """Spec-load dlp_worker.py (module-level imports are only asyncio, time,
    and logging_config, so nothing DB-shaped is pulled in)."""
    return _load_with_logging_mock("dlp_worker_autodiscovery_under_test", "dlp_worker.py")


@pytest.fixture(scope="module")
def scanner():
    """Spec-load dlp_scanner.py for the REAL parse_remote_endpoints +
    GLINER_DEFAULT_MAX_CHARS (pure-logic module, no DB chain)."""
    return _load_with_logging_mock("dlp_scanner_autodiscovery_under_test", "dlp_scanner.py")


# The engine sentinel the core builder ships: BackendEngine.DLP = "dlp".
class _Engines:
    DLP = "dlp"


def _dlp_backend(url):
    b = MagicMock()
    b.url = url
    return b


async def _run_load(
    worker,
    scanner,
    *,
    remote_enabled,
    endpoints,
    url="",
    discovered=None,
    discovery_raises=False,
):
    """Execute the REAL _load_dlp_config with crud + dlp_scanner pre-mocked.

    Returns (config, calls) where calls records whether — and how —
    crud.get_backends_by_engine was invoked.
    """
    calls = {}
    fake_db = MagicMock()

    async def _get_config_json(db, key, default=None):
        overrides = {
            "dlp.gliner.remote.enabled": remote_enabled,
            "dlp.gliner.remote.endpoints": endpoints,
            "dlp.gliner.remote.url": url,
        }
        return overrides.get(key, default)

    async def _get_backends_by_engine(db, engine, healthy_only=False):
        calls["invoked"] = True
        calls["db"] = db
        calls["engine"] = engine
        calls["healthy_only"] = healthy_only
        if discovery_raises:
            raise RuntimeError("discovery boom")
        return discovered or []

    crud_stub = MagicMock(
        get_config_json=_get_config_json,
        get_backends_by_engine=_get_backends_by_engine,
    )

    keys = (
        "backend.app.db",
        "backend.app.db.crud",
        "backend.app.db.models",
        "backend.app.services",
        "backend.app.services.dlp_scanner",
    )
    saved = {k: sys.modules.get(k) for k in keys}
    # Fresh stub modules only — never mutate an attribute on a PRE-EXISTING
    # sys.modules entry (an attribute planted on someone else's module object
    # survives the finally-restore below and breaks unrelated tests).
    sys.modules["backend.app.db"] = MagicMock(crud=crud_stub)
    sys.modules["backend.app.db.crud"] = crud_stub
    sys.modules["backend.app.db.models"] = MagicMock(BackendEngine=_Engines)
    sys.modules.setdefault("backend.app.services", MagicMock())
    sys.modules["backend.app.services.dlp_scanner"] = scanner  # real parse_remote_endpoints
    try:
        config = await worker._load_dlp_config(fake_db)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return config, calls, fake_db


# ===================================================================
# Authoritative auto-discovery
# ===================================================================

class TestDlpPoolAutodiscovery:
    """When off-host GLiNER is enabled and no endpoints were entered by hand,
    the healthy engine=dlp fleet backends become the pool (authoritative)."""

    @pytest.mark.asyncio
    async def test_enabled_and_empty_discovers_healthy_dlp_urls(self, worker, scanner):
        """(a) remote enabled + manual empty -> endpoints become the healthy dlp URLs."""
        backends = [_dlp_backend("https://dlp-a:8001"), _dlp_backend("https://dlp-b:8001")]
        config, calls, fake_db = await _run_load(
            worker, scanner,
            remote_enabled=True, endpoints=[], url="", discovered=backends,
        )
        assert config["gliner.remote.endpoints"] == ["https://dlp-a:8001", "https://dlp-b:8001"]
        # Authoritative + healthy-only, and against the caller's own db session.
        assert calls.get("invoked") is True
        assert calls["engine"] == _Engines.DLP
        assert calls["healthy_only"] is True
        assert calls["db"] is fake_db

    @pytest.mark.asyncio
    async def test_manual_list_overrides_discovery(self, worker, scanner):
        """(b) manual present -> discovery NOT invoked; list preserved verbatim."""
        config, calls, _ = await _run_load(
            worker, scanner,
            remote_enabled=True, endpoints=["https://manual:8001"], url="",
            discovered=[_dlp_backend("https://should-not-be-used:8001")],
        )
        assert calls.get("invoked") is None
        assert config["gliner.remote.endpoints"] == ["https://manual:8001"]

    @pytest.mark.asyncio
    async def test_legacy_url_counts_as_manual_and_overrides(self, worker, scanner):
        """(b, legacy) a legacy single .url is a manual override too — no discovery."""
        config, calls, _ = await _run_load(
            worker, scanner,
            remote_enabled=True, endpoints=[], url="https://legacy:8001",
            discovered=[_dlp_backend("https://should-not-be-used:8001")],
        )
        assert calls.get("invoked") is None
        assert config["gliner.remote.endpoints"] == []

    @pytest.mark.asyncio
    async def test_remote_disabled_never_discovers(self, worker, scanner):
        """(c) remote disabled -> discovery NOT invoked; endpoints left empty."""
        config, calls, _ = await _run_load(
            worker, scanner,
            remote_enabled=False, endpoints=[], url="",
            discovered=[_dlp_backend("https://dlp-a:8001")],
        )
        assert calls.get("invoked") is None
        assert config["gliner.remote.endpoints"] == []

    @pytest.mark.asyncio
    async def test_no_healthy_dlp_backends_yields_empty_pool(self, worker, scanner):
        """(d) no healthy dlp backends -> endpoints become empty (pool drained)."""
        config, calls, _ = await _run_load(
            worker, scanner,
            remote_enabled=True, endpoints=[], url="", discovered=[],
        )
        assert calls.get("invoked") is True
        assert config["gliner.remote.endpoints"] == []

    @pytest.mark.asyncio
    async def test_crud_error_leaves_endpoints_unchanged_and_does_not_raise(self, worker, scanner):
        """(e) crud error -> endpoints unchanged, no raise (DLP keeps running)."""
        # A blank-only entry normalizes to an empty manual list (so discovery
        # runs) while remaining observably the configured value after the error.
        sentinel = ["   "]
        config, calls, _ = await _run_load(
            worker, scanner,
            remote_enabled=True, endpoints=sentinel, url="", discovery_raises=True,
        )
        assert calls.get("invoked") is True          # discovery was attempted
        assert config["gliner.remote.endpoints"] == sentinel   # and left untouched
