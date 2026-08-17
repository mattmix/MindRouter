"""Tests for the CSRF origin guard (F12).

backend/app/core/csrf.py is a raw-ASGI middleware importable without the
db package chain (it depends only on settings + logging), so these are
real functional ASGI tests driving the middleware directly.

Contract:
  - same-origin state-changing request (Origin matches Host) -> allowed
  - cross-origin state-changing request -> 403 blocked
  - state-changing request with NO Origin/Referer -> allowed (API clients)
  - GET (and other safe methods) -> always allowed
  - trusted cross-origin (settings.csrf_trusted_origins) -> allowed
"""

import pytest

from backend.app.core import csrf
from backend.app.core.csrf import CSRFOriginMiddleware, _netloc


class _App:
    """Downstream ASGI app that records whether it was reached."""

    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class _Sent:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        return self.messages[0]["status"] if self.messages else None


def _scope(method="POST", host="mindrouter.uidaho.edu", origin=None, referer=None, path="/api/x"):
    headers = []
    if host is not None:
        headers.append((b"host", host.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    if referer is not None:
        headers.append((b"referer", referer.encode()))
    return {"type": "http", "method": method, "path": path, "headers": headers}


async def _run(mw, scope):
    sent = _Sent()

    async def receive():
        return {"type": "http.request"}

    await mw(scope, receive, sent)
    return sent


# --- _netloc helper --------------------------------------------------------

def test_netloc_full_url():
    assert _netloc("https://Example.com/foo") == "example.com"


def test_netloc_with_port():
    assert _netloc("http://localhost:8000") == "localhost:8000"


def test_netloc_bare_host():
    assert _netloc("mindrouter.uidaho.edu") == "mindrouter.uidaho.edu"


def test_netloc_null_origin():
    assert _netloc("null") is None
    assert _netloc("") is None


# --- middleware behavior ---------------------------------------------------

@pytest.mark.asyncio
async def test_same_origin_post_allowed():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin="https://mindrouter.uidaho.edu"))
    assert app.called is True
    assert sent.status == 200


@pytest.mark.asyncio
async def test_cross_origin_post_blocked():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin="https://evil.example.com"))
    assert app.called is False
    assert sent.status == 403


@pytest.mark.asyncio
async def test_no_origin_post_allowed():
    """API-key / server-to-server clients send no Origin and must pass."""
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin=None, referer=None))
    assert app.called is True
    assert sent.status == 200


@pytest.mark.asyncio
async def test_get_always_allowed_even_cross_origin():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(method="GET", origin="https://evil.example.com"))
    assert app.called is True
    assert sent.status == 200


@pytest.mark.asyncio
async def test_referer_fallback_same_origin_allowed():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin=None, referer="https://mindrouter.uidaho.edu/chat"))
    assert app.called is True
    assert sent.status == 200


@pytest.mark.asyncio
async def test_referer_fallback_cross_origin_blocked():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin=None, referer="https://evil.example.com/x"))
    assert app.called is False
    assert sent.status == 403


@pytest.mark.asyncio
async def test_trusted_origin_allowed(monkeypatch):
    class _S:
        csrf_trusted_origins = ["https://app.trusted.example.com"]

    monkeypatch.setattr(csrf, "get_settings", lambda: _S())
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin="https://app.trusted.example.com"))
    assert app.called is True
    assert sent.status == 200


@pytest.mark.asyncio
async def test_null_origin_blocked():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin="null"))
    assert app.called is False
    assert sent.status == 403


@pytest.mark.asyncio
async def test_scheme_ignored_host_matches(monkeypatch):
    """A TLS-terminating proxy forwards over http; an https Origin whose host
    matches the Host header must still be treated as same-origin."""
    class _S:
        csrf_trusted_origins = []

    monkeypatch.setattr(csrf, "get_settings", lambda: _S())
    app = _App()
    mw = CSRFOriginMiddleware(app)
    sent = await _run(mw, _scope(origin="https://mindrouter.uidaho.edu"))
    assert app.called is True
    assert sent.status == 200


@pytest.mark.asyncio
async def test_websocket_scope_passthrough():
    app = _App()
    mw = CSRFOriginMiddleware(app)
    scope = {"type": "websocket", "headers": []}
    await _run(mw, scope)
    assert app.called is True
