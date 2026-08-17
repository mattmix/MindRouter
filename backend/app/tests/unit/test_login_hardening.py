"""Tests for the login/auth hardening in dashboard/routes.py (ws4-auth-cookies).

Covers the F13/F18/F30/F32 fixes:
  * Session/masquerade cookies honor settings.session_cookie_secure.
  * Login brute-force throttle (sliding window, self-expiring, never a
    permanent lockout, per-(user,IP) keyed, bounded).
  * A cached, valid dummy Argon2 hash exists for timing-equalized failed
    logins.
  * Source-contract guards that lock the enumeration/offload properties of
    the /login handler in place so a future refactor can't silently drop
    them.

routes.py imports the full db/telemetry chain, but that chain imports
cleanly in this environment, so these are real behavioral tests plus a few
AST/source-contract assertions for the handler body.
"""

import ast
import inspect
from pathlib import Path

import pytest

from backend.app.dashboard import routes


# ---------------------------------------------------------------------------
# Fake settings
# ---------------------------------------------------------------------------


class _FakeSettings:
    secret_key = "test-secret-key-login-hardening"
    session_cookie_secure = True
    session_cookie_httponly = True
    session_cookie_samesite = "lax"
    login_max_attempts_per_window = 3
    login_attempt_window_seconds = 300


@pytest.fixture
def fake_settings(monkeypatch):
    s = _FakeSettings()
    monkeypatch.setattr(routes, "get_settings", lambda: s)
    return s


@pytest.fixture(autouse=True)
def _clear_throttle():
    routes._login_attempts.clear()
    yield
    routes._login_attempts.clear()


# ---------------------------------------------------------------------------
# Cookie Secure flag
# ---------------------------------------------------------------------------


def test_session_cookie_sets_secure_when_configured(fake_settings):
    from fastapi import Response

    resp = Response()
    routes.set_session_cookie(resp, 42)
    header = resp.headers["set-cookie"].lower()
    assert "secure" in header
    assert "httponly" in header
    assert "samesite=lax" in header


def test_session_cookie_omits_secure_when_disabled(fake_settings):
    from fastapi import Response

    fake_settings.session_cookie_secure = False
    resp = Response()
    routes.set_session_cookie(resp, 42)
    header = resp.headers["set-cookie"].lower()
    assert "secure" not in header
    # Still HttpOnly so plain-HTTP dev keeps its existing protection.
    assert "httponly" in header


# ---------------------------------------------------------------------------
# Brute-force throttle
# ---------------------------------------------------------------------------


def test_throttle_allows_up_to_limit_then_blocks(fake_settings):
    key = "alice|203.0.113.7"
    # max_attempts = 3 -> first three recorded attempts are allowed.
    assert routes._login_throttle_check_and_record(key) is False
    assert routes._login_throttle_check_and_record(key) is False
    assert routes._login_throttle_check_and_record(key) is False
    # Fourth is over the limit.
    assert routes._login_throttle_check_and_record(key) is True
    # And it stays blocked without unbounded growth.
    assert routes._login_throttle_check_and_record(key) is True
    assert len(routes._login_attempts[key]) == 3


def test_throttle_is_per_key(fake_settings):
    a = "alice|203.0.113.7"
    b = "bob|203.0.113.7"
    for _ in range(3):
        routes._login_throttle_check_and_record(a)
    assert routes._login_throttle_check_and_record(a) is True
    # A different user (or IP) is unaffected.
    assert routes._login_throttle_check_and_record(b) is False


def test_throttle_window_expires(fake_settings, monkeypatch):
    """Old attempts fall out of the window -> never a permanent lockout."""
    key = "carol|198.51.100.2"
    base = [1000.0]
    monkeypatch.setattr(routes.time, "monotonic", lambda: base[0])
    for _ in range(3):
        routes._login_throttle_check_and_record(key)
    assert routes._login_throttle_check_and_record(key) is True
    # Advance past the window; the stale stamps are pruned.
    base[0] = 1000.0 + fake_settings.login_attempt_window_seconds + 1
    assert routes._login_throttle_check_and_record(key) is False


# ---------------------------------------------------------------------------
# Dummy Argon2 hash (timing equalization)
# ---------------------------------------------------------------------------


def test_dummy_password_hash_is_valid_and_cached():
    from backend.app.security import verify_password

    h1 = routes._get_dummy_password_hash()
    h2 = routes._get_dummy_password_hash()
    assert h1 is h2  # cached, computed once
    assert h1.startswith("$argon2")
    # A valid hash the wrong password never matches (so it's a pure timing sink).
    assert verify_password("not-the-real-password", h1) is False


# ---------------------------------------------------------------------------
# Source-contract guards for the /login handler
# ---------------------------------------------------------------------------


def _login_source() -> str:
    return inspect.getsource(routes.login)


def test_login_offloads_argon2_and_uses_dummy_hash():
    src = _login_source()
    assert "_verify_password_offloaded" in src
    assert "_get_dummy_password_hash" in src
    # No direct synchronous verify_password call left in the handler body.
    assert "verify_password(" not in src


def test_login_uniform_error_message_no_sso_oracle():
    src = _login_source()
    # One generic rejection string, reused; the old SSO-only oracle is gone.
    assert "generic_error" in src
    assert "no local password" not in src
    assert "please use" not in src


def test_login_throttle_returns_429():
    src = _login_source()
    assert "_login_throttle_check_and_record" in src
    assert "HTTP_429_TOO_MANY_REQUESTS" in src


def test_login_and_credential_mutations_emit_audit():
    src = Path(routes.__file__).read_text()
    # Auth lifecycle events.
    for action in (
        "auth.login_success",
        "auth.login_failure",
        "auth.logout",
        "auth.password_change",
    ):
        assert action in src, f"missing audit action {action}"
    # API-key create/revoke self-service audit.
    assert "apikey.create" in src
    assert "apikey.revoke" in src


def test_masquerade_cookie_honors_secure_setting():
    """The masquerade set_cookie must pass secure= from settings, like the
    session cookie."""
    src = Path(routes.__file__).read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_cookie"
        ):
            kwargs = {kw.arg for kw in node.keywords}
            assert "secure" in kwargs, "a set_cookie call is missing secure="
            found = True
    assert found, "no set_cookie calls found"
