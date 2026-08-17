############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_ws8_logging_privacy.py: security-hardening regression
#     tests for the ws8-logging-privacy workstream.
#
############################################################

"""Focused tests for the ws8 privacy/logging hardening.

Covered:
  * client_ip.get_client_ip trusts the LAST X-Forwarded-For hop (the value
    the bundled nginx appends), not the client-forgeable first one.
  * otel._exporter_insecure derives channel security from the endpoint
    scheme and preserves the historical plaintext default for bare/http.
  * retention._bulk_insert_ignore writes its plaintext-PII dump 0o600 in a
    0o700 dir, removes it on full success, and retains it (still private)
    only on the controlled skip-for-triage path.

The modules are loaded via ``spec_from_file_location`` so importing them does
not drag in the db/settings package ``__init__`` chain (see MEMORY import
gotcha). retention's only heavy top-level import — ``logging_config`` — is
stubbed in ``sys.modules`` first, because it pulls settings → create_engine.
"""

import asyncio
import importlib.util
import stat
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APP = _REPO_ROOT / "backend" / "app"


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, _APP / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


client_ip = _load("ws8_client_ip", "core/client_ip.py")
otel = _load("ws8_otel", "core/otel.py")


class _FakeHeaders:
    """Case-insensitive header lookup, like starlette's Headers."""

    def __init__(self, data: dict):
        self._data = {k.lower(): v for k, v in data.items()}

    def get(self, key, default=None):
        return self._data.get(key.lower(), default)


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, headers=None, client_host="203.0.113.9"):
        self.headers = _FakeHeaders(headers or {})
        self.client = _FakeClient(client_host) if client_host else None


# ------------------------------------------------------------------
# client_ip: trust the closest-proxy-observed (last) XFF hop
# ------------------------------------------------------------------


class TestClientIp:
    def test_last_xff_entry_wins_not_the_forgeable_first(self):
        # Client forged "1.2.3.4"; nginx appended the real 198.51.100.7.
        req = _FakeRequest(
            {"x-forwarded-for": "1.2.3.4, 10.0.0.1, 198.51.100.7"}
        )
        assert client_ip.get_client_ip(req) == "198.51.100.7"

    def test_single_entry_xff(self):
        req = _FakeRequest({"x-forwarded-for": "198.51.100.7"})
        assert client_ip.get_client_ip(req) == "198.51.100.7"

    def test_trailing_empty_and_whitespace_are_skipped(self):
        req = _FakeRequest({"x-forwarded-for": "198.51.100.7, ,  "})
        assert client_ip.get_client_ip(req) == "198.51.100.7"

    def test_falls_back_to_x_real_ip(self):
        req = _FakeRequest({"x-real-ip": "198.51.100.8"})
        assert client_ip.get_client_ip(req) == "198.51.100.8"

    def test_falls_back_to_direct_client(self):
        req = _FakeRequest(client_host="203.0.113.9")
        assert client_ip.get_client_ip(req) == "203.0.113.9"

    def test_no_headers_no_client(self):
        req = _FakeRequest(client_host=None)
        assert client_ip.get_client_ip(req) is None


# ------------------------------------------------------------------
# otel: insecure channel derived from endpoint scheme
# ------------------------------------------------------------------


class TestOtelInsecure:
    def test_https_endpoint_is_secure(self):
        assert otel._exporter_insecure("https://collector:4317") is False

    def test_https_case_insensitive(self):
        assert otel._exporter_insecure("HTTPS://collector:4317") is False

    def test_http_endpoint_stays_insecure(self):
        assert otel._exporter_insecure("http://collector:4317") is True

    def test_bare_hostport_preserves_historical_default(self):
        # The pre-fix behaviour was insecure=True unconditionally; a bare
        # host:port endpoint must keep that so existing collectors work.
        assert otel._exporter_insecure("collector:4317") is True

    def test_empty_endpoint_is_insecure(self):
        assert otel._exporter_insecure("") is True


# ------------------------------------------------------------------
# retention: private dump file + cleanup
# ------------------------------------------------------------------


def _load_retention():
    # Stub logging_config so retention.py loads without the settings chain.
    stub = types.ModuleType("backend.app.logging_config")

    class _NullLogger:
        def __getattr__(self, _name):
            def _noop(*a, **k):
                return None

            return _noop

    stub.get_logger = lambda *a, **k: _NullLogger()
    # Install the stub only for the duration of the load, then restore the real
    # module (retention captured its get_logger reference at exec time and keeps
    # working). Leaving the stub in sys.modules would pollute unrelated tests
    # that import the real backend.app.logging_config.
    _orig = sys.modules.get("backend.app.logging_config")
    sys.modules["backend.app.logging_config"] = stub
    try:
        return _load("ws8_retention", "services/retention.py")
    finally:
        if _orig is not None:
            sys.modules["backend.app.logging_config"] = _orig
        else:
            sys.modules.pop("backend.app.logging_config", None)


retention = _load_retention()


class _FakeSavepoint:
    async def commit(self):
        return None

    async def rollback(self):
        return None


class _FakeResult:
    rowcount = 1


class _FakeArchiveDb:
    """Minimal async DB double; ``execute`` may be told to raise."""

    def __init__(self, fail=False, on_execute=None):
        self.fail = fail
        self.on_execute = on_execute

    async def begin_nested(self):
        return _FakeSavepoint()

    async def execute(self, stmt):
        if self.on_execute is not None:
            self.on_execute()
        if self.fail:
            raise RuntimeError("insert boom")
        return _FakeResult()


class _FakeTable:
    name = "requests"

    def insert(self):
        class _Stmt:
            def prefix_with(self, *_a, **_k):
                return self

            def values(self, **_k):
                return self

        return _Stmt()


class _FakeModel:
    __table__ = _FakeTable()


def _dump_files(dump_dir):
    return [p for p in Path(dump_dir).iterdir() if p.is_file()]


class TestRetentionDumpPerms:
    def test_dump_file_is_private_and_removed_on_success(self, tmp_path, monkeypatch):
        dump_dir = tmp_path / "retention"
        monkeypatch.setattr(retention, "_RETENTION_DUMP_DIR", str(dump_dir))

        captured = {}

        def _capture_mode():
            files = _dump_files(dump_dir)
            assert files, "dump file should exist while inserting"
            f = files[0]
            captured["file_mode"] = stat.S_IMODE(f.stat().st_mode)
            captured["dir_mode"] = stat.S_IMODE(dump_dir.stat().st_mode)

        db = _FakeArchiveDb(fail=False, on_execute=_capture_mode)
        rows = [{"id": 1, "prompt": "top secret user content"}]

        inserted, skipped = asyncio.run(
            retention._bulk_insert_ignore(db, _FakeModel, rows)
        )

        assert inserted == 1
        assert skipped == []
        # File was 0o600 while live, dir 0o700, and cleaned up afterwards.
        assert captured["file_mode"] == 0o600
        assert captured["dir_mode"] == 0o700
        assert _dump_files(str(dump_dir)) == []

    def test_dump_retained_but_still_private_on_skip(self, tmp_path, monkeypatch):
        dump_dir = tmp_path / "retention"
        monkeypatch.setattr(retention, "_RETENTION_DUMP_DIR", str(dump_dir))

        db = _FakeArchiveDb(fail=True)
        rows = [{"id": 7, "prompt": "top secret user content"}]

        inserted, skipped = asyncio.run(
            retention._bulk_insert_ignore(db, _FakeModel, rows)
        )

        assert inserted == 0
        assert len(skipped) == 1
        files = _dump_files(str(dump_dir))
        assert len(files) == 1, "failed-insert dump is retained for triage"
        # ...but never world-readable.
        mode = stat.S_IMODE(files[0].stat().st_mode)
        assert mode == 0o600
        assert not (mode & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH))

    def test_empty_rows_write_nothing(self, tmp_path, monkeypatch):
        dump_dir = tmp_path / "retention"
        monkeypatch.setattr(retention, "_RETENTION_DUMP_DIR", str(dump_dir))
        inserted, skipped = asyncio.run(
            retention._bulk_insert_ignore(_FakeArchiveDb(), _FakeModel, [])
        )
        assert (inserted, skipped) == (0, [])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
