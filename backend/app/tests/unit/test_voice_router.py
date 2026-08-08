############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_voice_router.py: Voice backend resolution (2.9.10)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for services/voice_router.resolve_voice_backend.

TTS/STT were the only modalities not routed through the registry — a single
hardcoded app_config URL each, with no health check, circuit breaker or
failover. This resolver is the migration seam: it prefers a registered
backend and falls back to the legacy config key, so voice services can be
registered one at a time with no flag day.

voice_router.py is spec-loaded with its imports deferred inside functions, so
no DB/telemetry package chain is pulled in — see MEMORY.md "Import Chain
Gotcha".
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_SERVICES_DIR = Path(__file__).resolve().parents[2] / "services"


def _load_module():
    saved = {k: sys.modules.get(k) for k in ("backend", "backend.app", "backend.app.logging_config")}
    sys.modules.setdefault("backend", MagicMock())
    sys.modules.setdefault("backend.app", MagicMock())
    sys.modules["backend.app.logging_config"] = MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "voice_router_under_test", _SERVICES_DIR / "voice_router.py",
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


@pytest.fixture(scope="module")
def vr():
    return _load_module()


def _backend(bid=7, name="wintermute-gpu1-kokoro", url="https://wintermute.nkn.uidaho.edu:8003/"):
    b = MagicMock()
    b.id = bid
    b.name = name
    b.url = url
    return b


def _install_registry(backends, available=True, raises=False):
    """Stub the registry module the resolver imports lazily."""
    reg = MagicMock()

    async def _healthy(engine=None):
        if raises:
            raise RuntimeError("registry down")
        return backends

    async def _avail(bid):
        return available

    reg.get_healthy_backends = _healthy
    reg.is_backend_available = _avail

    engines = MagicMock()
    engines.TTS = "tts"
    engines.STT = "stt"

    sys.modules.setdefault("backend.app.core", MagicMock())
    sys.modules.setdefault("backend.app.core.telemetry", MagicMock())
    sys.modules.setdefault("backend.app.db", MagicMock())
    sys.modules["backend.app.core.telemetry.registry"] = MagicMock(
        get_registry=MagicMock(return_value=reg)
    )
    sys.modules["backend.app.db.models"] = MagicMock(BackendEngine=engines)


def _install_crud(url=None, api_key=None):
    async def _get_config_json(db, key, default=None):
        if key.endswith("_url"):
            return url
        if key.endswith("_api_key"):
            return api_key
        return default

    sys.modules.setdefault("backend.app.db", MagicMock())
    crud = MagicMock(get_config_json=_get_config_json)
    sys.modules["backend.app.db.crud"] = crud
    sys.modules["backend.app.db"].crud = crud


_SENTINEL = object()


@pytest.fixture(autouse=True)
def _clean_modules():
    """Restore sys.modules AND the attributes we mutate on shared mocks.

    _install_crud sets `sys.modules["backend.app.db"].crud`, which mutates an
    object other test modules already hold a reference to — restoring only the
    sys.modules keys leaves that attribute pointing at this file's stub and
    silently breaks test_voice_api.py when the two run together.
    """
    keys = ("backend.app.core", "backend.app.core.telemetry",
            "backend.app.core.telemetry.registry", "backend.app.db",
            "backend.app.db.crud", "backend.app.db.models")
    saved = {k: sys.modules.get(k) for k in keys}
    db_mod = sys.modules.get("backend.app.db")
    saved_crud_attr = getattr(db_mod, "crud", _SENTINEL) if db_mod is not None else _SENTINEL
    try:
        yield
    finally:
        db_mod_now = sys.modules.get("backend.app.db")
        if db_mod_now is not None:
            if saved_crud_attr is _SENTINEL:
                try:
                    del db_mod_now.crud
                except Exception:
                    pass
            else:
                db_mod_now.crud = saved_crud_attr
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class TestRegistryPreferred:
    @pytest.mark.asyncio
    async def test_registered_backend_wins_over_config(self, vr):
        _install_registry([_backend()])
        _install_crud(url="https://legacy.example.edu:8003")

        t = await vr.resolve_voice_backend(MagicMock(), "tts")

        assert t is not None
        assert t.backend_id == 7
        assert t.source == "registry"
        assert t.url == "https://wintermute.nkn.uidaho.edu:8003", "trailing slash must be stripped"

    @pytest.mark.asyncio
    async def test_spreads_across_multiple_backends(self, vr):
        _install_registry([_backend(1, url="https://a:8003"),
                           _backend(2, url="https://b:8003"),
                           _backend(3, url="https://c:8003")])
        _install_crud()

        seen = set()
        for _ in range(60):
            t = await vr.resolve_voice_backend(MagicMock(), "tts")
            seen.add(t.backend_id)
        assert len(seen) > 1, "resolution must not pin to a single backend"

    @pytest.mark.asyncio
    async def test_registry_target_still_carries_the_configured_api_key(self, vr):
        """`backends` has no credential column, so an operator-set
        voice.<kind>_api_key must keep applying once a backend is registered —
        otherwise registering silently stops sending a key that was working."""
        _install_registry([_backend()])
        _install_crud(url="https://legacy.example.edu:8003", api_key="sk-upstream")

        t = await vr.resolve_voice_backend(MagicMock(), "tts")

        assert t.source == "registry"
        assert t.api_key == "sk-upstream"

    @pytest.mark.asyncio
    async def test_open_circuit_backends_are_skipped(self, vr):
        _install_registry([_backend()], available=False)
        _install_crud(url="https://legacy.example.edu:8003")

        t = await vr.resolve_voice_backend(MagicMock(), "tts")

        assert t.source == "config_fallback", "a backend with an open circuit must not be used"
        assert t.url == "https://legacy.example.edu:8003"


class TestConfigFallback:
    @pytest.mark.asyncio
    async def test_falls_back_when_nothing_registered(self, vr):
        _install_registry([])
        _install_crud(url="https://neuromancer.nkn.uidaho.edu:8005/", api_key="sk-legacy")

        t = await vr.resolve_voice_backend(MagicMock(), "stt")

        assert t.source == "config_fallback"
        assert t.backend_id is None
        assert t.url == "https://neuromancer.nkn.uidaho.edu:8005"
        assert t.api_key == "sk-legacy"

    @pytest.mark.asyncio
    async def test_registry_failure_does_not_break_voice(self, vr):
        """A registry outage must not take voice down while the legacy path works."""
        _install_registry([], raises=True)
        _install_crud(url="https://neuromancer.nkn.uidaho.edu:8003")

        t = await vr.resolve_voice_backend(MagicMock(), "tts")

        assert t is not None and t.source == "config_fallback"

    @pytest.mark.asyncio
    async def test_none_when_neither_available(self, vr):
        _install_registry([])
        _install_crud(url=None)

        assert await vr.resolve_voice_backend(MagicMock(), "tts") is None


class TestAllRequestPathsUseTheResolver:
    """Every path that DIALS a voice service must go through the resolver.

    Reading ``voice.tts_url`` / ``voice.stt_url`` directly bypasses health
    checks, circuit breakers and failover — it is exactly the bypass this
    change exists to remove. The admin Voice Config page is the one legitimate
    reader: it loads those keys to render the form and writes them on save.
    """

    _APP = Path(__file__).resolve().parents[2]

    def _request_path_sources(self):
        return {
            "api/voice_api.py": (self._APP / "api" / "voice_api.py").read_text(),
            "dashboard/chat.py": (self._APP / "dashboard" / "chat.py").read_text(),
        }

    def test_request_paths_never_read_the_url_config_directly(self):
        offenders = []
        for name, src in self._request_path_sources().items():
            for key in ('"voice.tts_url"', '"voice.stt_url"'):
                if key in src:
                    offenders.append(f"{name} reads {key}")
        assert not offenders, (
            "voice request paths must resolve through voice_router: " + "; ".join(offenders)
        )

    def test_request_paths_call_the_resolver(self):
        for name, src in self._request_path_sources().items():
            assert "resolve_voice_backend" in src, f"{name} does not use the resolver"

    def test_admin_voice_config_page_still_manages_the_legacy_keys(self):
        """The fallback must stay editable until it is retired."""
        src = (self._APP / "dashboard" / "routes.py").read_text()
        assert 'set_config(db, "voice.tts_url"' in src
        assert 'set_config(db, "voice.stt_url"' in src

    def test_voices_endpoint_uses_the_resolver(self):
        src = (self._APP / "dashboard" / "routes.py").read_text()
        i = src.index("/v1/audio/voices")
        window = src[max(0, i - 1500):i]
        assert "resolve_voice_backend" in window, (
            "the voices lookup should ask a registered backend before the config URL"
        )


class TestContract:
    @pytest.mark.asyncio
    async def test_unknown_kind_rejected(self, vr):
        with pytest.raises(ValueError):
            await vr.resolve_voice_backend(MagicMock(), "asr")

    def test_source_label_reflects_origin(self, vr):
        assert vr.VoiceTarget(url="u", backend_id=3).source == "registry"
        assert vr.VoiceTarget(url="u").source == "config_fallback"

    def test_both_kinds_have_legacy_keys(self, vr):
        assert set(vr._LEGACY_KEYS) == {"tts", "stt"}
        assert vr._LEGACY_KEYS["tts"] == ("voice.tts_url", "voice.tts_api_key")
        assert vr._LEGACY_KEYS["stt"] == ("voice.stt_url", "voice.stt_api_key")
