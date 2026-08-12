############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# csrf.py: Origin/Referer CSRF guard for state-changing requests
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""CSRF origin guard (F12).

State-changing requests (POST/PUT/PATCH/DELETE) issued by a browser carry
an ``Origin`` header (with ``Referer`` as a fallback) identifying the page
that initiated them.  This middleware rejects such a request when that
origin is neither this deployment's own host nor an explicitly trusted
origin, which stops a third-party page from driving a victim's
cookie-authenticated dashboard session (classic CSRF).

Requests with NO Origin and NO Referer are ALLOWED.  That is the shape of
non-browser API traffic — curl, the OpenAI/Ollama/Anthropic SDKs,
server-to-server callers — which authenticate with an API key or bearer
token, not an ambient cookie, and so are not CSRF-able.  This keeps the
``/v1``, ``/api`` and vendor-compatible API surfaces completely unaffected.

Implemented as raw ASGI (not ``BaseHTTPMiddleware``) for the same reason
as ``RequestIDMiddleware`` / ``SessionDeactivationMiddleware`` in
``main.py``: it must not wrap request handling in a separate task, which
would let a client disconnect cancel in-flight DB work on the long-lived
inference streams and leak connections.  A passthrough here is a header
scan and a direct call to the downstream app.

Comparison is by host[:port] only — the scheme is intentionally NOT
compared, because a TLS-terminating reverse proxy typically forwards to
the app over http, so comparing schemes would reject every genuine https
browser request.  Trusted cross-origins are configured via
``settings.csrf_trusted_origins`` (default empty).
"""

from typing import Optional
from urllib.parse import urlsplit

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Paths that legitimately receive a cross-origin browser POST and must NOT be
# subject to the Origin check. The SAML 2.0 HTTP-POST binding has the IdP's
# browser auto-submit a form to the Assertion Consumer Service from the IdP's
# own origin; the SAML assertion is cryptographically signed and carries its
# own replay/InResponseTo protection (see dashboard/sso/saml.py), so ambient-
# cookie CSRF protection is unnecessary here and would break SSO login.
_EXEMPT_PATHS = frozenset({"/login/saml/acs"})

_FORBIDDEN_BODY = (
    b'{"error": {"message": "CSRF origin check failed", "type": "forbidden"}}'
)


def _netloc(value: str) -> Optional[str]:
    """Return ``host[:port]`` (lowercased) from an Origin/Referer/Host value.

    Accepts both a full URL (``https://host:port/path``) and a bare
    ``host[:port]`` such as a raw ``Host`` header.  Returns ``None`` when no
    host can be parsed (e.g. the opaque ``Origin: null``).
    """
    if not value:
        return None
    candidate = value.strip()
    if candidate.lower() == "null":
        # Opaque origin (sandboxed iframe, some cross-scheme redirects) — no
        # host, so it can never be same-origin.
        return None
    if "://" not in candidate:
        # Bare host[:port] (Host header, or a bare config entry).
        candidate = "//" + candidate
    parts = urlsplit(candidate)
    if not parts.hostname:
        return None
    if parts.port:
        return f"{parts.hostname.lower()}:{parts.port}"
    return parts.hostname.lower()


def _header(headers, name: bytes) -> Optional[str]:
    """First value of the given (lowercase) header from an ASGI header list."""
    for key, value in headers:
        if key == name:
            return value.decode("latin-1")
    return None


class CSRFOriginMiddleware:
    """Reject cross-origin browser state-changing requests (F12)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("method", "") not in _UNSAFE_METHODS:
            await self.app(scope, receive, send)
            return

        if scope.get("path", "") in _EXEMPT_PATHS:
            # Legitimate cross-origin signed POST callback (e.g. SAML ACS).
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        origin = _header(headers, b"origin") or _header(headers, b"referer")
        if not origin:
            # No browser-supplied origin: API-key / server-to-server client.
            await self.app(scope, receive, send)
            return

        origin_host = _netloc(origin)
        if origin_host is not None:
            allowed = {_netloc(_header(headers, b"host") or "")}
            for trusted in get_settings().csrf_trusted_origins:
                allowed.add(_netloc(trusted))
            allowed.discard(None)
            if origin_host in allowed:
                await self.app(scope, receive, send)
                return

        logger.warning(
            "csrf_origin_rejected",
            method=scope.get("method"),
            path=scope.get("path"),
            origin=origin,
        )
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": _FORBIDDEN_BODY})
