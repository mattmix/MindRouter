"""Tests for the session deactivation guard (2.9.5).

A deactivated/deleted user's signed session cookie (valid 7 days) must
stop working immediately across the dashboard — not just at login
boundaries.  backend/app/core/session_guard.py is importable without the
db package chain (db imports live inside methods), so these are real
functional ASGI tests, with _user_is_active patched for flow cases.
"""

import pytest
from itsdangerous import URLSafeTimedSerializer

from backend.app.core.session_guard import (
    SESSION_COOKIE,
    SessionDeactivationMiddleware,
    _cookie_user_id,
)

SECRET = "test-secret-key"


def _make_cookie(user_id: int, secret: str = SECRET, salt: str = "session") -> str:
    value = URLSafeTimedSerializer(secret, salt=salt).dumps(user_id)
    return f"{SESSION_COOKIE}={value}"


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

    def header(self, name: bytes):
        for k, v in self.messages[0].get("headers", []):
            if k == name:
                return v
        return None


def _scope(path="/dashboard", cookie=None, accept=b"text/html"):
    headers = []
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    if accept:
        headers.append((b"accept", accept))
    return {"type": "http", "path": path, "headers": headers}


async def _run(mw, scope):
    sent = _Sent()

    async def receive():
        return {"type": "http.request"}

    await mw(scope, receive, sent)
    return sent


def _mw(active: bool, app=None):
    mw = SessionDeactivationMiddleware(app or _App())

    async def fake_is_active(user_id):
        return active

    mw._user_is_active = fake_is_active
    return mw


def _patch_settings(monkeypatch):
    import backend.app.settings as settings_mod

    class _S:
        secret_key = SECRET

    monkeypatch.setattr(settings_mod, "get_settings", lambda: _S())


class TestCookieDecode:
    def test_valid_cookie_decodes(self):
        assert _cookie_user_id(_make_cookie(42), SECRET) == 42

    def test_tampered_cookie_returns_none(self):
        cookie = _make_cookie(42) + "x"
        assert _cookie_user_id(cookie, SECRET) is None

    def test_wrong_secret_returns_none(self):
        assert _cookie_user_id(_make_cookie(42, secret="other"), SECRET) is None

    def test_wrong_salt_returns_none(self):
        assert _cookie_user_id(_make_cookie(42, salt="not-session"), SECRET) is None

    def test_absent_cookie_returns_none(self):
        assert _cookie_user_id("other=abc", SECRET) is None


class TestFlow:
    async def test_no_cookie_passes_through(self):
        app = _App()
        mw = _mw(active=False, app=app)  # even inactive: no cookie -> no check
        sent = await _run(mw, _scope(cookie=None))
        assert app.called
        assert sent.status == 200

    async def test_active_user_passes_through(self, monkeypatch):
        _patch_settings(monkeypatch)
        app = _App()
        mw = _mw(active=True, app=app)
        sent = await _run(mw, _scope(cookie=_make_cookie(7)))
        assert app.called
        assert sent.status == 200

    async def test_inactive_user_html_redirects_and_clears_cookie(self, monkeypatch):
        _patch_settings(monkeypatch)
        app = _App()
        mw = _mw(active=False, app=app)
        sent = await _run(mw, _scope(cookie=_make_cookie(7), accept=b"text/html,*/*"))
        assert not app.called
        assert sent.status == 302
        assert b"/login" in sent.header(b"location")
        assert b"Max-Age=0" in sent.header(b"set-cookie")

    async def test_inactive_user_api_gets_401_and_cleared_cookie(self, monkeypatch):
        _patch_settings(monkeypatch)
        app = _App()
        mw = _mw(active=False, app=app)
        sent = await _run(mw, _scope(cookie=_make_cookie(7), accept=b"application/json"))
        assert not app.called
        assert sent.status == 401
        assert b"Max-Age=0" in sent.header(b"set-cookie")

    async def test_exempt_paths_pass_through_even_inactive(self, monkeypatch):
        """Real registered routes only — an exempt prefix matching no
        route is dead config that hides nothing."""
        _patch_settings(monkeypatch)
        for path in (
            "/login",
            "/login/azure/authorized",
            "/login/google/authorized",
            "/login/oidc/authorized",
            "/login/saml/acs",
            "/saml/metadata",
            "/logout",
            "/static/app.css",
            "/health",
        ):
            app = _App()
            mw = _mw(active=False, app=app)
            await _run(mw, _scope(path=path, cookie=_make_cookie(7)))
            assert app.called, f"{path} must be exempt"

    async def test_protected_paths_are_not_accidentally_exempt(self, monkeypatch):
        _patch_settings(monkeypatch)
        for path in ("/dashboard", "/chat", "/admin/users", "/v1/chat/completions"):
            app = _App()
            mw = _mw(active=False, app=app)
            sent = await _run(mw, _scope(path=path, cookie=_make_cookie(7)))
            assert not app.called, f"{path} must be guarded"
            assert sent.status in (302, 401)

    async def test_malformed_sibling_cookie_cannot_smuggle_session(self, monkeypatch):
        """Parser-differential regression: a cookie header that a strict
        parser (SimpleCookie) rejects wholesale is still parsed by
        Starlette, so the app would authenticate a session the guard
        never saw.  Both orderings, plus an illegal cookie NAME."""
        _patch_settings(monkeypatch)
        token = _make_cookie(7)
        for header in (
            f"junk=a b; {token}",
            f"{token}; junk=a b",
            f"{{weird}}=1; {token}",
            f'sess_meta="quoted; value"; {token}',
        ):
            app = _App()
            mw = _mw(active=False, app=app)
            sent = await _run(mw, _scope(cookie=header))
            assert not app.called, f"guard bypassed by header: {header}"
            assert sent.status == 302

    async def test_split_cookie_headers_cannot_smuggle_session(self, monkeypatch):
        """Starlette merges every Cookie header; the guard must too."""
        _patch_settings(monkeypatch)
        app = _App()
        mw = _mw(active=False, app=app)
        scope = {
            "type": "http",
            "path": "/dashboard",
            "headers": [
                (b"cookie", b"junk=1"),
                (b"cookie", _make_cookie(7).encode()),
                (b"accept", b"text/html"),
            ],
        }
        sent = await _run(mw, scope)
        assert not app.called, "guard read only the first Cookie header"
        assert sent.status == 302

    async def test_guard_cookie_view_matches_starlette(self, monkeypatch):
        """The guard's parse must agree with request.cookies for every
        header shape we test — the invariant behind both bypasses."""
        from starlette.requests import cookie_parser

        from backend.app.core.session_guard import _cookie_user_id

        token_cookie = _make_cookie(11)
        raw_token = token_cookie.split("=", 1)[1]
        for header in (
            token_cookie,
            f"a=1; {token_cookie}",
            f"junk=a b; {token_cookie}",
            f"{{weird}}=1; {token_cookie}",
        ):
            starlette_view = cookie_parser(header).get(SESSION_COOKIE)
            assert starlette_view == raw_token, f"test premise wrong for {header}"
            assert _cookie_user_id(header, SECRET) == 11, f"guard blind to {header}"

    async def test_undecodable_cookie_passes_through(self, monkeypatch):
        _patch_settings(monkeypatch)
        app = _App()
        mw = _mw(active=False, app=app)
        sent = await _run(mw, _scope(cookie=f"{SESSION_COOKIE}=garbage"))
        assert app.called  # routes handle their own bad-session redirect
        assert sent.status == 200

    async def test_non_http_scope_passes_through(self):
        app = _App()
        mw = _mw(active=False, app=app)
        sent = _Sent()

        async def receive():
            return {}

        await mw({"type": "websocket"}, receive, sent)
        assert app.called


class TestSourceContracts:
    def test_middleware_registered_in_main(self):
        from pathlib import Path

        main_src = (
            Path(__file__).resolve().parents[2] / "main.py"
        ).read_text()
        assert "SessionDeactivationMiddleware" in main_src
        assert "add_middleware(SessionDeactivationMiddleware)" in main_src

    def test_fails_open_on_db_error(self):
        import inspect

        from backend.app.core import session_guard

        src = inspect.getsource(session_guard.SessionDeactivationMiddleware._user_is_active)
        assert "except Exception" in src
        assert "return True" in src  # fail open

    def test_serializer_matches_dashboard_routes(self):
        """The guard must decode the exact cookies routes.py mints."""
        from pathlib import Path

        routes_src = (
            Path(__file__).resolve().parents[2] / "dashboard" / "routes.py"
        ).read_text()
        assert 'salt="session"' in routes_src
        assert "86400 * 7" in routes_src

        from backend.app.core.session_guard import SESSION_MAX_AGE, _session_serializer

        assert SESSION_MAX_AGE == 86400 * 7
        ser = _session_serializer(SECRET)
        assert ser.salt in (b"session", "session")
