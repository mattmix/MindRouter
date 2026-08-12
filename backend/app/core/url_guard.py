############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# url_guard.py: SSRF guard for user-supplied image URLs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""SSRF guard for user-supplied image URLs.

User-supplied ``image_url`` values (OpenAI vision format) are forwarded to
vLLM backends that live inside the GPU/HPC network. Without validation a
caller could point ``image_url`` at internal services or the cloud
metadata endpoint (169.254.169.254) and have the backend fetch it,
turning MindRouter into an SSRF pivot.

The public entrypoints are :func:`is_safe_image_url` (bool) and
:func:`image_url_reason` (bool + human-readable reason). Both fail closed:
if a hostname cannot be resolved, or anything looks off, the URL is
rejected for fetching. Rejection only drops the offending image block; it
never fails the whole request.
"""

import concurrent.futures
import ipaddress
import socket
from urllib.parse import urlsplit

import structlog

from backend.app.settings import get_settings

logger = structlog.get_logger(__name__)

# Cloud metadata endpoints — link-local already covers 169.254.0.0/16, but
# call it out explicitly so the intent is unmistakable.
_METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})

# Hostname resolution is a blocking syscall. This guard is invoked from the
# (synchronous) request-translation path, so an attacker-controlled hostname
# backed by a deliberately slow resolver could pin a worker for the full
# system resolver timeout (tens of seconds). Bound it with a small shared pool
# and a short timeout; a resolution that overruns fails closed (rejected).
# NOTE: this caps the worst-case stall but the call is still synchronous —
# a future refactor should make the translator SSRF check async and await
# asyncio.to_thread so DNS never touches the event loop at all.
_DNS_TIMEOUT_SECONDS = 2.0
_RESOLVER_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="url_guard_dns"
)


def _resolve_bounded(host: str):
    """Resolve ``host`` with a hard timeout. Raises on timeout/error (fail closed)."""
    future = _RESOLVER_POOL.submit(
        socket.getaddrinfo, host, None, proto=socket.IPPROTO_TCP
    )
    return future.result(timeout=_DNS_TIMEOUT_SECONDS)


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    """True when an address falls in a range we must never fetch from."""
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def image_url_reason(url: str) -> tuple[bool, str]:
    """Validate a user-supplied image URL for backend fetching.

    Returns ``(ok, reason)``. ``ok`` is True when the URL is safe to forward
    to a backend for fetching; ``reason`` explains a rejection (empty when
    ``ok`` is True).

    Rules:
      * ``data:`` URIs are always allowed — inline base64 images are the
        common legitimate case and involve no network fetch.
      * The scheme must be in ``settings.image_url_allowed_schemes``.
      * For ``http``/``https``, when ``settings.image_url_block_private`` is
        True, every IP the hostname resolves to (and any literal-IP host)
        must be a public address; otherwise the URL is rejected. Resolution
        failures fail closed (rejected).
    """
    if not url or not isinstance(url, str):
        return False, "empty or non-string url"

    settings = get_settings()
    allowed_schemes = getattr(
        settings, "image_url_allowed_schemes", ["http", "https", "data"]
    )
    block_private = getattr(settings, "image_url_block_private", True)

    try:
        parts = urlsplit(url)
    except ValueError as exc:
        return False, f"unparseable url: {exc}"

    scheme = (parts.scheme or "").lower()

    # data: URIs carry the image inline — no fetch, always allowed (subject
    # to the scheme allow-list so an operator can turn even those off).
    if scheme == "data":
        if "data" in allowed_schemes:
            return True, ""
        return False, "data: scheme not permitted"

    if scheme not in allowed_schemes:
        return False, f"scheme not permitted: {scheme or '(none)'}"

    if scheme not in ("http", "https"):
        # Allowed by config but not a fetchable web scheme we understand.
        return False, f"unsupported scheme for fetch: {scheme}"

    host = parts.hostname
    if not host:
        return False, "missing host"

    if not block_private:
        # Private-range blocking disabled by configuration.
        return True, ""

    # Resolve the host to every address it maps to and reject if ANY of them
    # is in a blocked range. A literal-IP host resolves to itself.
    try:
        infos = _resolve_bounded(host)
    except (OSError, UnicodeError) as exc:
        return False, f"host resolution failed: {exc}"
    except concurrent.futures.TimeoutError:
        return False, "host resolution timed out"

    resolved = set()
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            resolved.add(sockaddr[0])

    if not resolved:
        return False, "host did not resolve to any address"

    for addr in resolved:
        # Strip any IPv6 zone id (e.g. fe80::1%eth0) before parsing.
        addr_clean = addr.split("%", 1)[0]
        if addr_clean in _METADATA_IPS:
            return False, f"cloud metadata address blocked: {addr_clean}"
        try:
            ip = ipaddress.ip_address(addr_clean)
        except ValueError:
            return False, f"unparseable resolved address: {addr_clean}"
        if _ip_is_blocked(ip):
            return False, f"resolves to non-public address: {addr_clean}"

    return True, ""


def is_safe_image_url(url: str) -> bool:
    """True when ``url`` is safe to forward to a backend for fetching."""
    ok, _reason = image_url_reason(url)
    return ok
