"""Unit tests for HttpVideoWorkerClient auth header + TLS-verify wiring.

Loads the module directly via importlib to avoid the backend package import
chain (db/session -> settings -> pymysql), per the repo's test hygiene rules.
A fake httpx.AsyncClient records the verify kwarg and the headers/params each
request receives, so we assert the client sends X-Worker-Key (and that TLS
verification comes from the module-level _tls_verify() default) without a live
worker. The fetch-cap and _tls_verify behavior is covered in
test_video_worker_client_hardening.py.
"""

import importlib.util
import os
import secrets

import pytest

_CLIENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "services", "video_worker_client.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_vwc_under_test", os.path.abspath(_CLIENT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vwc = _load_module()

# Generated per run — never a hardcoded secret literal in the tree.
_KEY = secrets.token_hex(16)


class _FakeResponse:
    def __init__(self, *, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = ""

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise vwc.httpx.HTTPStatusError("err", request=None, response=None)


class _FakeAsyncClient:
    """Captures constructor + request kwargs into a shared sink."""

    def __init__(self, sink, response):
        self._sink = sink
        self._response = response

    def _factory(self):
        outer = self

        class _Ctx:
            def __init__(self, **kwargs):
                outer._sink["init"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, **kwargs):
                outer._sink["post"] = {"url": url, **kwargs}
                return outer._response

            async def get(self, url, **kwargs):
                outer._sink["get"] = {"url": url, **kwargs}
                return outer._response

            async def delete(self, url, **kwargs):
                outer._sink["delete"] = {"url": url, **kwargs}
                return outer._response

        return _Ctx


@pytest.fixture
def patched(monkeypatch):
    sink = {}
    resp = _FakeResponse(json_data={"id": "wjob-abc123"})
    fake = _FakeAsyncClient(sink, resp)
    monkeypatch.setattr(vwc.httpx, "AsyncClient", fake._factory())
    return sink


@pytest.mark.asyncio
async def test_submit_sends_key(patched):
    client = vwc.HttpVideoWorkerClient(api_key=_KEY)
    job_id = await client.submit("https://node:18300", {"prompt": "x"})
    assert job_id == "wjob-abc123"
    assert patched["post"]["headers"] == {"X-Worker-Key": _KEY}
    # TLS verification comes from _tls_verify() (secure default), not a ctor arg.
    assert patched["init"]["verify"] is True


@pytest.mark.asyncio
async def test_no_key_sends_no_header(patched):
    client = vwc.HttpVideoWorkerClient()  # no key -> no auth header
    await client.poll("https://node:18300", "wjob-1")
    assert patched["get"]["headers"] == {}


@pytest.mark.asyncio
async def test_cancel_sends_key(patched):
    client = vwc.HttpVideoWorkerClient(api_key=_KEY)
    await client.cancel("https://node:18300", "wjob-1")
    assert patched["delete"]["headers"] == {"X-Worker-Key": _KEY}


def test_headers_helper():
    assert vwc.HttpVideoWorkerClient(api_key=_KEY)._headers() == {"X-Worker-Key": _KEY}
    assert vwc.HttpVideoWorkerClient()._headers() == {}
