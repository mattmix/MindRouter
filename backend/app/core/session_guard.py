"""Session deactivation guard.

Dashboard/chat sessions are signed cookies valid for up to 7 days, and
until 2.9.5 nothing re-checked ``users.is_active`` after login — a
deactivated (or deleted) user's existing session kept working until the
cookie expired.  This raw-ASGI middleware closes that gap: any request
carrying a session cookie resolves the user and, if the account is
inactive or gone, clears the cookie and refuses the request.

Raw ASGI (not BaseHTTPMiddleware) for the same reason as
RequestIDMiddleware in main.py: no separate handler task, no streaming
interference.  Requests without a session cookie — the entire API-key
inference path — pay only a header scan.
"""

from typing import Optional

from itsdangerous import URLSafeTimedSerializer
from starlette.requests import cookie_parser

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

SESSION_COOKIE = "mindrouter_session"
SESSION_MAX_AGE = 86400 * 7  # keep in sync with dashboard/routes.py

# Paths a deactivated user may still reach: the login page and its
# assets, logout, and the SSO flows — all of which enforce is_active
# themselves at the auth boundary.  Every SSO route lives under
# /login/* (azure|google|oidc|saml, plus their /authorized|/acs
# callbacks) except the SP metadata endpoint, which must serve to the
# IdP regardless of session state.
_EXEMPT_PREFIXES = (
    "/login",
    "/logout",
    "/saml/metadata",
    "/static",
    "/favicon",
    "/health",
)

_CLEAR_COOKIE = (
    f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"
).encode()


def _session_serializer(secret_key: str) -> URLSafeTimedSerializer:
    # Must match _get_session_serializer in dashboard/routes.py exactly.
    return URLSafeTimedSerializer(secret_key, salt="session")


def _cookie_user_id(cookie_header: str, secret_key: str) -> Optional[int]:
    """Decode the session cookie value into a user id, or None.

    Uses Starlette's own ``cookie_parser`` — the parser every route
    reaches through ``request.cookies``.  Anything stricter (e.g.
    ``http.cookies.SimpleCookie``, which aborts the whole header on one
    malformed pair) would let a request smuggle a session past this
    guard that the app then happily authenticates.
    """
    value = cookie_parser(cookie_header).get(SESSION_COOKIE)
    if not value:
        return None
    try:
        return int(_session_serializer(secret_key).loads(value, max_age=SESSION_MAX_AGE))
    except Exception:
        return None


class SessionDeactivationMiddleware:
    """Refuse session-cookie requests from inactive or deleted users."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.startswith(_EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        # ALL cookie headers, joined — Starlette merges every one of them
        # into request.cookies, so stopping at the first would let a
        # request split its session cookie into a second header the guard
        # never sees.
        cookie_header = "; ".join(
            value.decode("latin-1")
            for name, value in scope.get("headers", [])
            if name == b"cookie"
        )
        if not cookie_header or SESSION_COOKIE not in cookie_header:
            await self.app(scope, receive, send)
            return

        from backend.app.settings import get_settings

        user_id = _cookie_user_id(cookie_header, get_settings().secret_key)
        if user_id is None:
            # Undecodable/expired cookie: let the route's own session
            # handling produce its usual redirect.
            await self.app(scope, receive, send)
            return

        if await self._user_is_active(user_id):
            await self.app(scope, receive, send)
            return

        logger.info("session_refused_inactive_user", user_id=user_id, path=path)
        await self._refuse(scope, send)

    @staticmethod
    async def _user_is_active(user_id: int) -> bool:
        """True if the user exists and is active.  Fails OPEN on DB
        errors: the guard is defense-in-depth, and a DB outage must not
        take down routes that would themselves have failed the request
        anyway."""
        try:
            from sqlalchemy import select

            from backend.app.db.models import User
            from backend.app.db.session import get_async_db_context

            async with get_async_db_context() as db:
                result = await db.execute(
                    select(User.is_active).where(User.id == user_id)
                )
                row = result.first()
            return bool(row is not None and row[0])
        except Exception:
            logger.exception("session_guard_db_error", user_id=user_id)
            return True

    @staticmethod
    async def _refuse(scope, send):
        """401 JSON for API-style calls, redirect-to-login for pages.
        Both clear the session cookie."""
        accept = b""
        for name, value in scope.get("headers", []):
            if name == b"accept":
                accept = value
                break

        if b"text/html" in accept:
            await send({
                "type": "http.response.start",
                "status": 302,
                "headers": [
                    (b"location", b"/login?error=Your+account+has+been+deactivated"),
                    (b"set-cookie", _CLEAR_COOKIE),
                ],
            })
            await send({"type": "http.response.body", "body": b""})
        else:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"set-cookie", _CLEAR_COOKIE),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"detail": "Account deactivated"}',
            })
