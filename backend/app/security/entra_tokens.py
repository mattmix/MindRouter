############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# entra_tokens.py: Verify Microsoft Entra id_tokens presented
#     by a registered application
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Verify an Entra id_token forwarded by a registered app.

A first-party app (e.g. VandalChat) authenticates its users against the same
Entra tenant as MindRouter but with its own client registration, so the token
it holds carries ITS audience, not MindRouter's. Entra signs every token in a
tenant with the same keys, so MindRouter can verify such a token completely
without having been party to the original sign-in.

That the token was not minted for MindRouter is why the checks here are
strict. Accepting it is a deliberate, operator-configured trust decision — the
app's client id is recorded on the `apps` row — not an inference from the token
itself.

WHY THIS EXISTS AT ALL: the alternative is trusting the app's word about who
the user is. An app credential would then be an unbounded impersonation
primitive: whoever holds it could provision and act as any user in the tenant.
Requiring a live user token means a compromised app can only act for users who
are actually signing in to it.

DO NOT reuse `dashboard/sso/oidc.py:_decode_id_token_claims` for this — it
decodes claims WITHOUT verifying, which is safe there (the token came straight
from the token endpoint over TLS) and fatal here (the token arrives from a
third party).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx
from jose import jwt
from jose.exceptions import (
    ExpiredSignatureError,
    JWKError,
    JWTClaimsError,
    JWTError,
)

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Tolerance for clock drift between Entra and this host.
LEEWAY_SECONDS = 60

# How long a fetched key set is reused before refetching.
JWKS_CACHE_SECONDS = 3600

# Entra rotates signing keys, so an unknown `kid` may be legitimate. It is also
# what an attacker would send to force unbounded refetches, so refreshes are
# rate limited per tenant.
JWKS_MIN_REFETCH_SECONDS = 60

_JWKS_URL = "https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys"

# tenant -> (fetched_at, {kid: key_dict})
_jwks_cache: Dict[str, Tuple[float, Dict[str, dict]]] = {}


class EntraTokenError(Exception):
    """An id_token was not acceptable. The message is safe to log, NOT to
    return verbatim to a caller — it can describe why validation failed."""


@dataclass(frozen=True)
class EntraIdentity:
    """The verified subject of an id_token."""

    oid: str                  # stable per-user object id — maps to users.azure_oid
    tid: str                  # tenant id
    email: Optional[str]
    display_name: Optional[str]
    raw_claims: Dict[str, Any]


def _acceptable_issuers(tenant_id: str) -> Tuple[str, ...]:
    """Issuers Entra uses for a given tenant.

    v2.0 ONLY, deliberately. The v1.0 issuer (`sts.windows.net/{tid}/`) is the
    token generation whose ACCESS tokens carry the bare client-id GUID as their
    audience — the same value an id_token carries — which is what makes an
    access token for the app's own API hard to tell from a sign-in assertion.
    A registered app controls its own token version, and a new integration has
    no reason to use v1.0, so the ambiguity is refused rather than managed.
    """
    return (f"https://login.microsoftonline.com/{tenant_id}/v2.0",)


async def _fetch_jwks(tenant_id: str) -> Dict[str, dict]:
    """Fetch a tenant's signing keys.

    Every failure leaves as an EntraTokenError. Microsoft's discovery endpoint
    is rate limited and the network is not guaranteed, so httpx and JSON errors
    are ordinary operating conditions here — letting them escape would turn a
    transient upstream problem into an unhandled 500 with no application or
    tenant context in the log, which reads to the calling app as a server fault
    worth retrying hard.
    """
    url = _JWKS_URL.format(tenant=tenant_id)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise EntraTokenError(
            f"could not fetch tenant JWKS: {type(e).__name__}"
        ) from e
    except ValueError as e:  # malformed JSON body
        raise EntraTokenError("tenant JWKS was not valid JSON") from e

    if not isinstance(data, dict):
        raise EntraTokenError("tenant JWKS was not an object")

    # Only RSA signing keys. The algorithms are pinned to RS* below, and
    # python-jose builds the key OUTSIDE its own try block — a non-RSA entry
    # sharing our kid would raise JWKError straight past the handlers here.
    keys = {
        k["kid"]: k
        for k in data.get("keys", [])
        if isinstance(k, dict)
        and k.get("kid")
        and k.get("kty") == "RSA"
        and k.get("use", "sig") == "sig"
    }
    if not keys:
        raise EntraTokenError("tenant JWKS contained no usable keys")
    return keys


async def _key_for_kid(tenant_id: str, kid: str) -> dict:
    """Return the signing key for ``kid``, refetching on a miss.

    A cache miss is refetched at most once per JWKS_MIN_REFETCH_SECONDS so a
    stream of tokens bearing bogus kids cannot turn into a request amplifier
    against Entra.
    """
    now = time.monotonic()
    cached = _jwks_cache.get(tenant_id)

    if cached and (now - cached[0]) < JWKS_CACHE_SECONDS and kid in cached[1]:
        return cached[1][kid]

    if cached and kid not in cached[1] and (now - cached[0]) < JWKS_MIN_REFETCH_SECONDS:
        raise EntraTokenError("token signed by an unknown key")

    try:
        keys = await _fetch_jwks(tenant_id)
    except EntraTokenError:
        raise
    except Exception as e:
        # The contract of this module is that it raises EntraTokenError and
        # nothing else, so the route can turn every failure into a logged 401
        # instead of an unhandled 500. Belt and braces with _fetch_jwks's own
        # handling: this holds even if that function grows a new failure mode.
        raise EntraTokenError(
            f"could not fetch tenant JWKS: {type(e).__name__}"
        ) from e

    _jwks_cache[tenant_id] = (now, keys)

    if kid not in keys:
        raise EntraTokenError("token signed by an unknown key")
    return keys[kid]


async def verify_entra_id_token(
    token: str,
    expected_client_id: str,
    expected_tenant_id: str,
) -> EntraIdentity:
    """Fully verify an id_token and return its subject.

    ``expected_client_id`` MUST be the registered app's own Entra client id.
    Never widen this to "any audience from the tenant": that would let every
    application in the tenant provision MindRouter accounts.
    """
    if not token or not expected_client_id or not expected_tenant_id:
        raise EntraTokenError("missing token or app registration details")

    try:
        header = jwt.get_unverified_header(token)
    except JWTError as e:
        raise EntraTokenError(f"malformed token header: {e}") from e

    kid = header.get("kid")
    if not kid:
        raise EntraTokenError("token header has no kid")
    if header.get("alg") not in ("RS256", "RS384", "RS512"):
        # Pin to asymmetric algorithms: an attacker-chosen `alg` (notably
        # "none" or an HMAC variant keyed on the public cert) is the classic
        # JWT forgery route.
        raise EntraTokenError(f"unacceptable signing algorithm: {header.get('alg')}")

    key = await _key_for_kid(expected_tenant_id, kid)

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256", "RS384", "RS512"],
            audience=expected_client_id,
            issuer=_acceptable_issuers(expected_tenant_id),
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_iss": True,
                "verify_exp": True,
                "verify_nbf": True,
                # id_tokens carry a nonce bound to the app's original sign-in
                # request. MindRouter did not issue that request and cannot
                # validate it; audience+issuer+signature are what bind the
                # token to a known app in a known tenant.
                "verify_at_hash": False,
                "leeway": LEEWAY_SECONDS,
            },
        )
    except ExpiredSignatureError as e:
        raise EntraTokenError("token has expired") from e
    except JWTClaimsError as e:
        # Wrong audience or issuer lands here — the interesting rejection.
        raise EntraTokenError(f"token claims rejected: {e}") from e
    except JWKError as e:
        # NOT a subclass of JWTError, and raised from outside python-jose's own
        # try block, so it would otherwise escape unsanitised.
        raise EntraTokenError("signing key could not be constructed") from e
    except JWTError as e:
        raise EntraTokenError(f"token signature invalid: {e}") from e

    # THIS MUST BE AN ID TOKEN, not an access token.
    #
    # An access token minted for the app's OWN API can carry the same audience
    # as its id_tokens, and the app's backend receives one on every call — so
    # it is far more widely handled, logged, and passed around than a sign-in
    # assertion. Accepting one would mean "somebody was authorised to call
    # VandalChat's API on this user's behalf" is treated as "this user just
    # signed in to VandalChat", which is a much weaker claim.
    #
    # `scp` (delegated) and `roles` (app-only) appear on access tokens and
    # never on an id_token, so their presence is the discriminator.
    for access_only in ("scp", "scope", "roles"):
        if claims.get(access_only) is not None:
            raise EntraTokenError(
                f"token carries the access-token claim '{access_only}';"
                " an id_token is required"
            )

    # When present, the authorized party must be the app we trust. Entra omits
    # `azp` from most v2.0 id_tokens, so this validates rather than requires.
    for party_claim in ("azp", "appid"):
        party = claims.get(party_claim)
        if party is not None and party != expected_client_id:
            raise EntraTokenError(
                f"token '{party_claim}' does not match the app registration"
            )

    # Defense in depth: the issuer check already pins the tenant, but a token
    # whose tid disagrees with where we fetched keys from is malformed enough
    # to refuse outright.
    tid = claims.get("tid")
    if tid and tid != expected_tenant_id:
        raise EntraTokenError("token tenant does not match the app registration")

    oid = claims.get("oid")
    if not oid:
        # Without a stable object id there is nothing to key an account on.
        # `sub` is pairwise per-application and therefore useless to a second
        # relying party.
        raise EntraTokenError("token has no oid claim")

    email = (
        claims.get("email")
        or claims.get("preferred_username")
        or claims.get("upn")
    )

    return EntraIdentity(
        oid=oid,
        tid=tid or expected_tenant_id,
        email=email.lower() if isinstance(email, str) else None,
        display_name=claims.get("name"),
        raw_claims=claims,
    )


def _reset_jwks_cache_for_tests() -> None:
    _jwks_cache.clear()
