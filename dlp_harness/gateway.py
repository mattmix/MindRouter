############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/gateway.py: HTTP client layer for end-to-end
# DLP runs — admin-API provisioning (group, throwaway users,
# API keys) and the chat-traffic sender that plants PII on
# either side of a conversation and extracts the gateway's
# request_uuid for exact DB correlation.
#
# Correlation contract (verified against the gateway):
#   non-stream: response body "id" == requests.request_uuid
#   stream:     MOCK-ONLY — the mock backend omits chunk ids,
#               so the gateway stamps request_uuid into every
#               SSE chunk's "id" (first chunk wins here). A
#               real backend stamps its own chatcmpl-* ids,
#               which match no requests row; e2e preflight
#               sends a stream probe to enforce this before
#               any run streams.
#
# Every function that talks to a gateway refuses non-local
# base_urls unless allow_prod=True is passed explicitly.
#
############################################################

"""Provisioning + traffic client for DLP end-to-end evaluation."""

import base64
import json
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

import httpx

from dlp_harness.mock_backend import REPLY_B64_MARKER, REPLY_MARKER  # noqa: F401 (both exported for callers)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def require_local(base_url: str, allow_prod: bool = False, what: str = "gateway") -> None:
    """Refuse any non-localhost target unless allow_prod is explicit."""
    host = urllib.parse.urlsplit(base_url).hostname
    if host is None:
        raise ValueError(f"cannot parse a host from base_url {base_url!r}")
    if host not in _LOCAL_HOSTS and not allow_prod:
        raise RuntimeError(
            f"{what} refuses non-local base_url {base_url!r} without allow_prod=True "
            "(this harness mutates DLP config and generates synthetic PII traffic)"
        )


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# Provisioning (admin API; flow mirrors stress.py's UserProvisioner)
# ---------------------------------------------------------------------------

@dataclass
class ProvisionedUser:
    user_id: int
    username: str
    api_key: str


def ensure_group(
    base_url: str,
    admin_key: str,
    name: str = "dlp-harness",
    token_budget: int = 1_000_000_000,
    rpm_limit: int = 100_000,
    scheduler_weight: int = 1,
    timeout: float = 30.0,
    allow_prod: bool = False,
) -> int:
    """Create (or find) the harness group and return its id."""
    require_local(base_url, allow_prod, what="admin API")
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        r = client.post(
            "/api/admin/groups",
            headers=_headers(admin_key),
            json={
                "name": name,
                "display_name": "DLP Harness",
                "description": "throwaway group for DLP harness runs",
                "token_budget": token_budget,
                "rpm_limit": rpm_limit,
                "scheduler_weight": scheduler_weight,
                "is_admin": False,
                "is_auditor": False,
            },
        )
        if r.status_code == 409:
            g = client.get("/api/admin/groups", headers=_headers(admin_key))
            g.raise_for_status()
            for grp in g.json().get("groups", []):
                if grp.get("name") == name:
                    return int(grp["id"])
            raise RuntimeError(
                f"group {name!r} conflicted on create but is absent from GET /api/admin/groups")
        if r.status_code not in (200, 201):
            raise RuntimeError(f"create group {name!r} failed: {r.status_code} {r.text[:200]}")
        return int(r.json()["id"])


def provision_users(
    base_url: str,
    admin_key: str,
    n: int,
    group_id: int,
    prefix: str = "_dlpharness_",
    timeout: float = 30.0,
    allow_prod: bool = False,
) -> List[ProvisionedUser]:
    """Create n users in the group, each with one API key.

    A 409 means a leftover from a crashed run: look the user up (search
    param), delete, and retry the create once. On any failure, users created
    so far are torn down before the error propagates.
    """
    require_local(base_url, allow_prod, what="admin API")
    users: List[ProvisionedUser] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        try:
            for i in range(n):
                username = f"{prefix}{i:03d}"
                payload = {
                    "username": username,
                    "email": f"{username}@test.local",
                    # Never reused (auth is via API key) and never deterministic
                    # (a seeded password would be a hardcoded credential).
                    "password": secrets.token_urlsafe(18),
                    "role": "student",
                    "group_id": group_id,
                    "full_name": "DLP Harness User",
                }
                r = client.post("/api/admin/users", headers=_headers(admin_key), json=payload)
                if r.status_code == 409:
                    lr = client.get("/api/admin/users", headers=_headers(admin_key),
                                    params={"search": username, "limit": 1000})
                    if lr.status_code == 200:
                        for u in lr.json().get("users", []):
                            if u.get("username") == username:
                                client.delete(f"/api/admin/users/{u['id']}",
                                              headers=_headers(admin_key))
                                break
                    r = client.post("/api/admin/users", headers=_headers(admin_key), json=payload)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"create user {username!r} failed: {r.status_code} {r.text[:200]}")
                user_id = int(r.json()["id"])
                kr = client.post(f"/api/admin/users/{user_id}/api-keys",
                                 headers=_headers(admin_key), json={"name": "dlp-harness"})
                if kr.status_code != 200:
                    raise RuntimeError(
                        f"create API key for {username!r} failed: {kr.status_code} {kr.text[:200]}")
                users.append(ProvisionedUser(user_id=user_id, username=username,
                                             api_key=kr.json()["full_key"]))
        except Exception:
            for u in users:
                try:
                    client.delete(f"/api/admin/users/{u.user_id}", headers=_headers(admin_key))
                except Exception:
                    pass
            raise
    return users


def teardown_users(
    base_url: str,
    admin_key: str,
    users: List[ProvisionedUser],
    retries: int = 3,
    timeout: float = 30.0,
    allow_prod: bool = False,
) -> List[ProvisionedUser]:
    """Delete provisioned users; never raises. Returns the ones NOT deleted."""
    require_local(base_url, allow_prod, what="admin API")
    failed: List[ProvisionedUser] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for u in users:
            for attempt in range(retries):
                try:
                    r = client.delete(f"/api/admin/users/{u.user_id}",
                                      headers=_headers(admin_key))
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if attempt < retries - 1:
                    time.sleep(2.0)   # in-flight requests may still hold FK rows
            else:
                failed.append(u)
    return failed


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------

@dataclass
class SendResult:
    request_uuid: Optional[str]
    status_code: Optional[int]
    error: Optional[str]
    ttfb_ms: Optional[float]
    ttft_ms: Optional[float]
    e2e_ms: float
    ok: bool


def build_messages(text: str, plant_side: str) -> List[dict]:
    """Chat messages that place ``text`` on the requested side of the exchange.

    "prompt" sends the text verbatim as the user turn. "response" sends only
    an opaque base64 payload (invisible to every scanner); the mock backend
    decodes it and returns the plaintext, so the PII exists ONLY in the
    response. "echo" asks the model to repeat the text verbatim, so the
    payload lands on BOTH sides: the prompt carries it and a compliant
    (real, non-mock) model's reply organically contains it — response-side
    testing against real models, where the b64 marker convention does not
    exist.
    """
    if plant_side == "prompt":
        content = text
    elif plant_side == "response":
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        content = "Please echo the encoded payload. " + REPLY_B64_MARKER + " " + payload
    elif plant_side == "echo":
        content = "Repeat the following text exactly, with no commentary: " + text
    else:
        raise ValueError(
            f"plant_side must be 'prompt', 'response' or 'echo', got {plant_side!r}")
    return [{"role": "user", "content": content}]


async def send_chat(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    stream: bool,
    plant_side: str,
    max_tokens: int = 64,
    timeout: float = 120.0,
    allow_prod: bool = False,
) -> SendResult:
    """One chat completion through the gateway; never raises."""
    require_local(base_url, allow_prod)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": build_messages(text, plant_side),
        "max_tokens": max_tokens,
        "stream": stream,
    }
    t0 = time.monotonic()

    def ms() -> float:
        return (time.monotonic() - t0) * 1000.0

    try:
        if not stream:
            r = await client.post(url, headers=_headers(api_key), json=payload,
                                  timeout=timeout)
            e2e = ms()
            if r.status_code != 200:
                return SendResult(None, r.status_code,
                                  f"HTTP {r.status_code}: {r.text[:200]}",
                                  e2e, None, e2e, False)
            body = r.json()
            uuid = body.get("id")
            return SendResult(uuid if uuid else None, r.status_code, None,
                              e2e, None, e2e, True)

        request_uuid: Optional[str] = None
        ttfb: Optional[float] = None
        ttft: Optional[float] = None
        stream_error: Optional[str] = None
        async with client.stream("POST", url, headers=_headers(api_key),
                                 json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                body_bytes = await r.aread()
                return SendResult(None, r.status_code,
                                  f"HTTP {r.status_code}: "
                                  f"{body_bytes.decode('utf-8', 'replace')[:200]}",
                                  ms(), None, ms(), False)
            async for line in r.aiter_lines():
                if ttfb is None:
                    ttfb = ms()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                # The gateway reports mid-stream backend failure as an SSE
                # error event on an HTTP 200 stream and never DLP-scans the
                # (FAILED) request; without this the row would score as
                # "sent OK, no alert" — a fake DLP coverage miss.
                if "error" in obj:
                    stream_error = json.dumps(obj["error"])[:200]
                    continue
                if request_uuid is None and obj.get("id"):
                    request_uuid = obj["id"]
                if ttft is None:
                    for choice in obj.get("choices") or []:
                        if (choice.get("delta") or {}).get("content"):
                            ttft = ms()
                            break
        return SendResult(request_uuid, 200, stream_error, ttfb, ttft, ms(),
                          stream_error is None)
    except Exception as exc:                       # transport/protocol/parse
        return SendResult(None, None, f"{type(exc).__name__}: {exc}",
                          None, None, ms(), False)
