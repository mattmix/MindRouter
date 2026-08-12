############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# sso/oidc.py: OIDC authorization-code driver (Google + generic)
#
############################################################

"""OIDC authorization-code flow, shared by Google and generic OIDC.

The generic provider covers any spec-compliant IdP — Okta, Keycloak, Auth0,
and CILogon (the OIDC gateway to the InCommon federation). Endpoints come
from the issuer's ``/.well-known/openid-configuration`` discovery document
(cached in-process). Identity comes from the userinfo endpoint fetched with
the access token over TLS — same trust model as the Azure driver's Graph
call, no local JWT validation needed for a confidential client.
"""

import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import Request
from fastapi.responses import RedirectResponse

from backend.app.dashboard.sso.base import (
    SSOProfile,
    find_or_create_sso_user,
    finish_login,
    new_signed_state,
    validate_state,
)
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

GOOGLE_ISSUER = "https://accounts.google.com"

# issuer -> (fetched_at, metadata dict); discovery documents rarely change.
_DISCOVERY_CACHE: dict[str, tuple[float, dict]] = {}
_DISCOVERY_TTL = 3600.0


@dataclass
class OIDCConfig:
    """Static per-provider configuration resolved from settings."""

    provider_id: str          # "google" | "oidc"
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: str
    default_group: str
    hosted_domain: Optional[str] = None  # Google Workspace 'hd' restriction


def google_config() -> Optional[OIDCConfig]:
    s = get_settings()
    if not s.google_sso_enabled:
        return None
    return OIDCConfig(
        provider_id="google",
        issuer=GOOGLE_ISSUER,
        client_id=s.google_sso_client_id,
        client_secret=s.google_sso_client_secret,
        redirect_uri=s.google_sso_redirect_uri or "/login/google/authorized",
        scopes="openid profile email",
        default_group=s.google_sso_default_group,
        hosted_domain=s.google_sso_hosted_domain,
    )


def generic_config() -> Optional[OIDCConfig]:
    s = get_settings()
    if not s.oidc_sso_enabled:
        return None
    return OIDCConfig(
        provider_id="oidc",
        issuer=s.oidc_sso_issuer.rstrip("/"),
        client_id=s.oidc_sso_client_id,
        client_secret=s.oidc_sso_client_secret,
        redirect_uri=s.oidc_sso_redirect_uri or "/login/oidc/authorized",
        scopes=s.oidc_sso_scopes,
        default_group=s.oidc_sso_default_group,
    )


def _absolute_redirect_uri(request: Request, redirect_uri: str) -> str:
    """Allow relative redirect URIs in config; resolve against the public base URL.

    request.base_url would report http:// behind a TLS-terminating proxy that
    uvicorn does not trust for X-Forwarded-Proto, and IdPs match the redirect
    URI exactly — so use the configured public URL instead.
    """
    if redirect_uri.startswith("http://") or redirect_uri.startswith("https://"):
        return redirect_uri
    base = (get_settings().app_base_url or "").rstrip("/")
    if not base:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("host", request.url.netloc)
        base = f"{proto}://{host}"
    return base + redirect_uri


async def discover(issuer: str) -> Optional[dict]:
    """Fetch (and cache) the OIDC discovery document for an issuer."""
    cached = _DISCOVERY_CACHE.get(issuer)
    if cached and (time.monotonic() - cached[0]) < _DISCOVERY_TTL:
        return cached[1]
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            meta = resp.json()
    except Exception as e:
        logger.error("oidc_discovery_failed", issuer=issuer, error=str(e))
        return None
    if not meta.get("authorization_endpoint") or not meta.get("token_endpoint"):
        logger.error("oidc_discovery_incomplete", issuer=issuer)
        return None
    _DISCOVERY_CACHE[issuer] = (time.monotonic(), meta)
    return meta


def _state_cookie(provider_id: str) -> str:
    return f"{provider_id}_oauth_state"


async def begin_login(request: Request, cfg: OIDCConfig):
    """Redirect the browser to the IdP's authorization endpoint."""
    meta = await discover(cfg.issuer)
    if not meta:
        return RedirectResponse(url="/login?error=SSO+provider+discovery+failed", status_code=302)

    signed_state = new_signed_state()
    params = {
        "client_id": cfg.client_id,
        "response_type": "code",
        "redirect_uri": _absolute_redirect_uri(request, cfg.redirect_uri),
        "scope": cfg.scopes,
        "state": signed_state,
    }
    if cfg.hosted_domain:
        params["hd"] = cfg.hosted_domain

    response = RedirectResponse(
        url=f"{meta['authorization_endpoint']}?{urlencode(params)}", status_code=302
    )
    response.set_cookie(
        key=_state_cookie(cfg.provider_id),
        value=signed_state,
        httponly=True,
        samesite="lax",
        secure=get_settings().session_cookie_secure,
        max_age=600,
    )
    return response


async def handle_callback(
    request: Request,
    cfg: OIDCConfig,
    db,
    code: Optional[str],
    state: Optional[str],
    error: Optional[str],
    error_description: Optional[str],
):
    """Exchange the code, fetch userinfo, JIT-provision, and log in."""
    if error:
        return RedirectResponse(
            url=f"/login?error=SSO+login+failed:+{error_description or error}", status_code=302
        )
    if not code or not state:
        return RedirectResponse(url="/login?error=Invalid+callback+parameters", status_code=302)
    if not validate_state(state, request.cookies.get(_state_cookie(cfg.provider_id))):
        return RedirectResponse(url="/login?error=Invalid+or+expired+state", status_code=302)

    meta = await discover(cfg.issuer)
    if not meta:
        return RedirectResponse(url="/login?error=SSO+provider+discovery+failed", status_code=302)

    token_data = {
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "code": code,
        "redirect_uri": _absolute_redirect_uri(request, cfg.redirect_uri),
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Client authentication: POST body (client_secret_post) first, then
        # HTTP Basic (client_secret_basic) — Okta and several other IdPs
        # register clients as Basic by default and reject the body form.
        token_response = await client.post(meta["token_endpoint"], data=token_data)
        if token_response.status_code != 200:
            basic_data = {k: v for k, v in token_data.items() if k != "client_secret"}
            basic_response = await client.post(
                meta["token_endpoint"],
                data=basic_data,
                auth=(cfg.client_id, cfg.client_secret),
            )
            if basic_response.status_code == 200:
                logger.info(
                    "oidc_token_exchange_basic_auth",
                    provider=cfg.provider_id,
                    note="IdP rejected client_secret_post; client_secret_basic succeeded",
                )
                token_response = basic_response
            else:
                # Log the body: OIDC error responses carry the actionable
                # detail (invalid_client, redirect_uri mismatch, ...) that a
                # bare status code hides.
                logger.warning(
                    "oidc_token_exchange_failed",
                    provider=cfg.provider_id,
                    status=token_response.status_code,
                    body=token_response.text[:500],
                    basic_status=basic_response.status_code,
                    basic_body=basic_response.text[:500],
                )
                return RedirectResponse(
                    url="/login?error=Failed+to+exchange+authorization+code", status_code=302
                )
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        if not access_token:
            return RedirectResponse(url="/login?error=No+access+token+received", status_code=302)

        # Start from the id_token's claims: some IdPs (notably ADFS) return
        # a minimal userinfo document containing only `sub`, with email and
        # name available solely in the id_token.  Userinfo values win on
        # conflict — it is the fresher source.
        claims: dict = {}
        id_token = tokens.get("id_token")
        if id_token:
            claims.update(_decode_id_token_claims(id_token))

        userinfo_endpoint = meta.get("userinfo_endpoint")
        if userinfo_endpoint:
            userinfo_response = await client.get(
                userinfo_endpoint, headers={"Authorization": f"Bearer {access_token}"}
            )
            if userinfo_response.status_code == 200:
                claims.update(userinfo_response.json())
            else:
                # Always log the failure: continuing on id_token claims
                # alone is a fallback, not a success, and silence here
                # would misreport a broken userinfo endpoint as a
                # scopes/claim-mapping problem further down.
                logger.warning(
                    "oidc_userinfo_failed",
                    provider=cfg.provider_id,
                    status=userinfo_response.status_code,
                    body=userinfo_response.text[:500],
                    continuing_on_id_token=bool(claims),
                )
                if not claims:
                    return RedirectResponse(
                        url="/login?error=Failed+to+fetch+user+profile", status_code=302
                    )
        elif not claims:
            return RedirectResponse(
                url="/login?error=SSO+provider+has+no+userinfo+endpoint", status_code=302
            )

    profile = profile_from_claims(cfg, claims)
    if profile is None:
        # Distinguish the three ways this fails, so the operator knows
        # whether to fix scopes, the IdP's email_verified, or the domain
        # restriction — one generic message sent them down the wrong path.
        if not claims.get("sub") or not claims.get("email"):
            missing = [c for c in ("sub", "email") if not claims.get(c)]
            logger.warning(
                "oidc_profile_missing_claims",
                provider=cfg.provider_id,
                missing=missing,
                claims_seen=sorted(claims.keys())[:20],
            )
            return RedirectResponse(
                url="/login?error=SSO+profile+is+missing+required+claims+"
                    "(check+the+scopes+requested+and+the+IdP+claim+mapping)",
                status_code=302,
            )
        if not _email_is_verified(claims):
            return RedirectResponse(
                url="/login?error=SSO+account+email+is+not+verified+at+the+identity+provider",
                status_code=302,
            )
        return RedirectResponse(
            url="/login?error=SSO+account+is+not+in+an+allowed+domain", status_code=302
        )

    user = await find_or_create_sso_user(db, profile, cfg.default_group)
    return await finish_login(db, user, clear_cookie=_state_cookie(cfg.provider_id))


def _email_is_verified(claims: dict) -> bool:
    """Mirror of the email_verified rule in profile_from_claims: the claim
    is trusted when absent, and string forms ("false"/"0") must not fail
    open."""
    verified = claims.get("email_verified")
    if verified is None:
        return True
    return str(verified).strip().lower() in ("true", "1")


def _decode_id_token_claims(id_token: str) -> dict:
    """Read the claim set out of an id_token WITHOUT verifying it.

    Signature verification is deliberately not done here: these claims are
    only a fallback source for `sub`/`email`/`name` when the IdP's userinfo
    document is thin, and the token arrived over TLS directly from the
    token endpoint in response to our own client-authenticated request —
    the same trust basis as the userinfo response itself.  Nothing
    security-relevant is decided from these values that isn't equally
    decided from userinfo.
    """
    import base64
    import json as _json

    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore base64url padding
        decoded = _json.loads(base64.urlsafe_b64decode(payload))
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}


def profile_from_claims(cfg: OIDCConfig, claims: dict) -> Optional[SSOProfile]:
    """Reduce OIDC userinfo claims to an SSOProfile; None if unusable."""
    subject = claims.get("sub")
    email = claims.get("email")
    if not subject or not email:
        return None
    # Reject unverified emails. Some IdPs send the claim as a JSON string
    # ("false"/"0"), so compare on the normalized value rather than identity —
    # a bare `is False` check would fail open. IdPs that omit it are trusted.
    verified = claims.get("email_verified")
    if verified is not None and str(verified).strip().lower() not in ("true", "1"):
        return None
    if cfg.hosted_domain and claims.get("hd") != cfg.hosted_domain:
        return None
    return SSOProfile(
        provider=cfg.provider_id,
        subject=str(subject),
        email=email,
        display_name=claims.get("name"),
        username_hint=claims.get("preferred_username"),
        raw=claims,
    )
