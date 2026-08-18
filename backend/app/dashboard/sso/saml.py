############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# sso/saml.py: SAML 2.0 SP driver (Shibboleth/InCommon, ADFS)
#
############################################################

"""SAML 2.0 Service Provider via python3-saml (lazily imported).

Supports a single IdP configured either from a metadata URL (typical for a
campus Shibboleth/InCommon IdP) or from explicit entity-id/SSO-URL/cert
settings. Endpoints:

    GET  /login/saml        -> AuthnRequest redirect to the IdP
    POST /login/saml/acs    -> Assertion Consumer Service
    GET  /saml/metadata     -> SP metadata XML (register this with the IdP)

python3-saml (and its xmlsec system libs) is an optional dependency —
installed in the Docker image, but the module degrades with a clear error
if it is missing. Attribute names default to eduPerson conventions
(mail / displayName / eduPersonPrincipalName) and are overridable via
SAML_ATTR_* settings.
"""

import time
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import RedirectResponse, Response

from backend.app.dashboard.sso.base import (
    STATE_MAX_AGE,
    SSOConfigError,
    SSOProfile,
    find_or_create_sso_user,
    finish_login,
    public_base_url,
    state_serializer,
)
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

# IdP settings parsed from metadata, cached in-process.
_IDP_CACHE: dict[str, tuple[float, dict]] = {}
_IDP_TTL = 3600.0

# Signed cookie carrying our AuthnRequest ID across the IdP round trip.
REQUEST_ID_COOKIE = "saml_request_id"


def _import_onelogin():
    """Lazy import so deployments without SAML never need xmlsec installed."""
    try:
        from onelogin.saml2.auth import OneLogin_Saml2_Auth
        from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser

        return OneLogin_Saml2_Auth, OneLogin_Saml2_IdPMetadataParser
    except ImportError:
        return None, None


def _idp_settings_from_metadata(metadata_url: str) -> Optional[dict]:
    """Fetch + parse IdP metadata (cached)."""
    cached = _IDP_CACHE.get(metadata_url)
    if cached and (time.monotonic() - cached[0]) < _IDP_TTL:
        return cached[1]
    _, parser = _import_onelogin()
    if parser is None:
        return None
    try:
        remote = parser.parse_remote(metadata_url)
        idp = remote.get("idp")
    except Exception as e:
        logger.error("saml_metadata_fetch_failed", url=metadata_url, error=str(e))
        return None
    if not idp:
        return None
    _IDP_CACHE[metadata_url] = (time.monotonic(), idp)
    return idp


def build_saml_settings(request_scheme_host: Optional[str] = None) -> Optional[dict]:
    """Assemble the python3-saml settings dict from MindRouter settings."""
    s = get_settings()
    if not s.saml_sso_enabled:
        return None

    if s.saml_idp_metadata_url:
        # The metadata document carries the IdP signing cert — the sole trust
        # anchor for assertion validation — so it must arrive over TLS.
        if not s.saml_idp_metadata_url.lower().startswith("https://"):
            logger.error("saml_metadata_url_not_https", url=s.saml_idp_metadata_url)
            return None
        idp = _idp_settings_from_metadata(s.saml_idp_metadata_url)
        if not idp:
            return None
    else:
        idp = {
            "entityId": s.saml_idp_entity_id,
            "singleSignOnService": {
                "url": s.saml_idp_sso_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": s.saml_idp_x509_cert,
        }

    acs_url = s.saml_sp_acs_url
    if not acs_url and request_scheme_host:
        acs_url = request_scheme_host.rstrip("/") + "/login/saml/acs"

    sp: dict[str, Any] = {
        "entityId": s.saml_sp_entity_id,
        "assertionConsumerService": {
            "url": acs_url,
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
        },
        "NameIDFormat": "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent",
    }
    sp_cert = _load_pem(s.saml_sp_x509_cert, "SAML_SP_X509_CERT")
    sp_key = _load_pem(s.saml_sp_private_key, "SAML_SP_PRIVATE_KEY")
    if sp_cert:
        # Publishes a <KeyDescriptor> in SP metadata — what a federation
        # registrar (e.g. InCommon) expects and what lets the IdP verify
        # our signatures.  NOTE: python3-saml emits use="signing" only;
        # the use="encryption" descriptor an IdP needs in order to encrypt
        # TO this SP is gated on wantAssertionsEncrypted/wantNameIdEncrypted,
        # not on cert presence — so encryption also needs the flag below.
        sp["x509cert"] = sp_cert
    if sp_key:
        sp["privateKey"] = sp_key

    security: dict[str, Any] = {
        "wantAssertionsSigned": True,
        "wantMessagesSigned": False,
        # Refuse RSA-SHA1/DSA-SHA1 signatures.
        "rejectDeprecatedAlgorithm": True,
    }
    # AuthnRequest signing and encrypted assertions BOTH require the key
    # pair: python3-saml rejects the entire settings object with
    # sp_cert_not_found_and_required if either is on without one, which
    # would take SAML login down completely rather than degrade.
    if sp_cert and sp_key:
        security["authnRequestsSigned"] = bool(s.saml_authn_requests_signed)
        security["wantAssertionsEncrypted"] = bool(s.saml_want_assertions_encrypted)
    elif s.saml_authn_requests_signed or s.saml_want_assertions_encrypted:
        logger.warning(
            "saml_sp_keypair_missing",
            authn_requests_signed=s.saml_authn_requests_signed,
            want_assertions_encrypted=s.saml_want_assertions_encrypted,
            note="both options need SAML_SP_X509_CERT and SAML_SP_PRIVATE_KEY; ignoring them",
        )

    return {
        "strict": True,
        "debug": False,
        "sp": sp,
        "idp": idp,
        "security": security,
        # NOTE: php-saml's "rejectUnsolicitedResponsesWithInResponseTo" does NOT
        # exist in python3-saml (it is accepted into the settings dict but never
        # read). SP-initiated-only is enforced in handle_acs() instead.
    }


def _load_pem(value: Optional[str], setting_name: str) -> Optional[str]:
    """Resolve a PEM setting given either inline or as a file path.

    Inline values may carry literal ``\\n`` escapes, which is how a
    multi-line PEM survives a single-line ``.env`` entry.  Anything without
    a PEM header is treated as a path to a mounted file.  Never logs the
    value — one of these is a private key.
    """
    if not value:
        return None
    raw = value.strip()
    if "-----BEGIN" in raw:
        return raw.replace("\\n", "\n")

    import hashlib
    from pathlib import Path

    try:
        text = Path(raw).read_text(encoding="utf-8").strip()
    except (OSError, ValueError) as exc:
        # `raw` was only ASSUMED to be a path, and we are here precisely
        # because it did not look like PEM — so it may be mangled key
        # material (a base64-packed PEM, a headerless key body).  Never
        # echo it: logs travel much further than the host .env does.
        # ValueError covers UnicodeDecodeError (a DER/PKCS#12 file) and
        # embedded NULs, which would otherwise 500 every SAML endpoint.
        logger.error(
            "saml_sp_pem_unreadable",
            setting=setting_name,
            error=exc.__class__.__name__,
            value_len=len(raw),
            value_sha256_8=hashlib.sha256(raw.encode()).hexdigest()[:8],
            note="expected inline PEM (\\n escapes ok) or a path to a readable PEM file",
        )
        return None
    if "-----BEGIN" not in text:
        # `raw` is a real, readable path here — safe to name.
        logger.error("saml_sp_pem_invalid", setting=setting_name, path=raw)
        return None
    return text


async def _prepare_fastapi_request(request: Request) -> dict[str, Any]:
    """Adapt a FastAPI request to python3-saml's expected dict shape.

    The scheme/host come from the configured public URL, never from request
    headers: python3-saml validates the assertion's Destination/Recipient
    against this value, and a client-supplied X-Forwarded-Host would let an
    attacker relax that check.
    """
    form = {}
    if request.method == "POST":
        form = dict(await request.form())

    # Fail closed on the configured public base URL; never fall back to the
    # attacker-controllable Host header (raises SSOConfigError when APP_BASE_URL
    # is unset — callers turn it into an error redirect).
    scheme, _, host = public_base_url(request).partition("://")

    return {
        "https": "on" if scheme == "https" else "off",
        "http_host": host,
        "script_name": request.url.path,
        "get_data": dict(request.query_params),
        "post_data": form,
    }


def _scheme_host(req_dict: dict[str, Any]) -> str:
    scheme = "https" if req_dict["https"] == "on" else "http"
    return f"{scheme}://{req_dict['http_host']}"


async def begin_login(request: Request):
    """Start the SP-initiated flow: redirect to the IdP with an AuthnRequest."""
    OneLogin_Saml2_Auth, _ = _import_onelogin()
    if OneLogin_Saml2_Auth is None:
        return RedirectResponse(
            url="/login?error=SAML+support+is+not+installed+(python3-saml)", status_code=302
        )
    try:
        # Config/literal-only request dict: the AuthnRequest redirect must not
        # carry any request-derived scheme/host/path/query. public_base_url
        # fails closed (SSOConfigError) when APP_BASE_URL is unset.
        scheme, _, host = public_base_url().partition("://")
    except SSOConfigError:
        return RedirectResponse(url="/login?error=SSO+is+misconfigured", status_code=302)
    req = {
        "https": "on" if scheme == "https" else "off",
        "http_host": host,
        "script_name": "/login/saml",
        "get_data": {},
        "post_data": {},
    }
    saml_settings = build_saml_settings(_scheme_host(req))
    if not saml_settings:
        return RedirectResponse(url="/login?error=SAML+is+not+configured", status_code=302)
    try:
        auth = OneLogin_Saml2_Auth(req, saml_settings)
        redirect_url = auth.login()
        response = RedirectResponse(url=redirect_url, status_code=302)
        # Remember our AuthnRequest ID (signed) so the ACS can require the
        # response's InResponseTo to match it.
        #
        # SameSite MUST be "none" (which requires Secure): the IdP returns the
        # assertion via a cross-site HTTP-POST form submission to our ACS, and
        # browsers do not send SameSite=Lax cookies on cross-site POSTs. With
        # "lax" this cookie would be missing on every login from an IdP that
        # does not share our registrable domain, and handle_acs would reject
        # the response as unsolicited. Consequence: SAML requires HTTPS.
        response.set_cookie(
            key=REQUEST_ID_COOKIE,
            value=state_serializer().dumps(auth.get_last_request_id()),
            httponly=True,
            samesite="none",
            secure=True,
            max_age=STATE_MAX_AGE,
        )
        return response
    except Exception as e:
        logger.error("saml_login_init_failed", error=str(e))
        return RedirectResponse(url="/login?error=SAML+initialization+failed", status_code=302)


async def handle_acs(request: Request, db):
    """Assertion Consumer Service: validate the response, provision, log in."""
    OneLogin_Saml2_Auth, _ = _import_onelogin()
    if OneLogin_Saml2_Auth is None:
        return RedirectResponse(
            url="/login?error=SAML+support+is+not+installed+(python3-saml)", status_code=302
        )
    try:
        req = await _prepare_fastapi_request(request)
    except SSOConfigError:
        return RedirectResponse(url="/login?error=SSO+is+misconfigured", status_code=302)
    saml_settings = build_saml_settings(_scheme_host(req))
    if not saml_settings:
        return RedirectResponse(url="/login?error=SAML+is+not+configured", status_code=302)

    # Recover the AuthnRequest ID from the signed cookie. python3-saml only
    # compares InResponseTo when BOTH a request_id is supplied and the response
    # carries one, so SP-initiated-only has to be enforced here: no cookie means
    # we never issued the request (login-CSRF / IdP-initiated), so refuse.
    signed_request_id = request.cookies.get(REQUEST_ID_COOKIE)
    request_id = None
    if signed_request_id:
        try:
            request_id = state_serializer().loads(signed_request_id, max_age=STATE_MAX_AGE)
        except Exception:
            request_id = None
    if not request_id:
        logger.warning("saml_acs_unsolicited_rejected", reason="no valid AuthnRequest cookie")
        return RedirectResponse(
            url="/login?error=SAML+login+must+start+from+this+site", status_code=302
        )

    try:
        auth = OneLogin_Saml2_Auth(req, saml_settings)
        auth.process_response(request_id=request_id)
        errors = auth.get_errors()
    except Exception as e:
        logger.error("saml_acs_failed", error=str(e))
        return RedirectResponse(url="/login?error=SAML+response+processing+failed", status_code=302)

    if errors or not auth.is_authenticated():
        logger.warning("saml_auth_rejected", errors=errors, reason=auth.get_last_error_reason())
        return RedirectResponse(url="/login?error=SAML+authentication+failed", status_code=302)

    # The library skips its InResponseTo comparison when the response omits the
    # attribute entirely — the defining trait of an unsolicited response — so
    # require the echo explicitly.
    if auth.get_last_response_in_response_to() != request_id:
        logger.warning(
            "saml_acs_in_response_to_mismatch",
            expected=request_id,
            got=auth.get_last_response_in_response_to(),
        )
        return RedirectResponse(
            url="/login?error=SAML+response+did+not+match+the+login+request", status_code=302
        )

    profile = profile_from_assertion(
        auth.get_attributes(), auth.get_nameid(), auth.get_nameid_format()
    )
    if profile is None:
        return RedirectResponse(
            url="/login?error=SAML+assertion+is+missing+a+usable+email+attribute", status_code=302
        )

    user = await find_or_create_sso_user(db, profile, get_settings().saml_default_group)
    return await finish_login(db, user, clear_cookie=REQUEST_ID_COOKIE)


def profile_from_assertion(
    attributes: dict, nameid: Optional[str], nameid_format: Optional[str]
) -> Optional[SSOProfile]:
    """Reduce SAML attributes to an SSOProfile using the configured mapping."""
    s = get_settings()

    def attr(name: str) -> Optional[str]:
        values = attributes.get(name) or []
        return values[0] if values else None

    email = attr(s.saml_attr_email)
    # Fall back to an email-format NameID (common with ADFS).
    if not email and nameid and "@" in nameid:
        email = nameid
    if not email:
        return None

    # Prefer a persistent NameID as the stable subject; fall back to ePPN, then email.
    subject = nameid or attr(s.saml_attr_username) or email

    return SSOProfile(
        provider="saml",
        subject=subject,
        email=email,
        display_name=attr(s.saml_attr_name),
        username_hint=attr(s.saml_attr_username),
        raw={k: v for k, v in attributes.items()},
    )


async def metadata_response(request: Request) -> Response:
    """Serve SP metadata XML for registration with the IdP / federation."""
    try:
        req = await _prepare_fastapi_request(request)
    except SSOConfigError:
        return RedirectResponse(url="/login?error=SSO+is+misconfigured", status_code=302)
    saml_settings = build_saml_settings(_scheme_host(req))
    if not saml_settings:
        return Response(content="SAML is not configured", status_code=404)
    try:
        from onelogin.saml2.settings import OneLogin_Saml2_Settings

        sp_settings = OneLogin_Saml2_Settings(saml_settings, sp_validation_only=True)
        metadata = sp_settings.get_sp_metadata()
        errors = sp_settings.validate_metadata(metadata)
        if errors:
            return Response(content=f"Invalid SP metadata: {errors}", status_code=500)
        return Response(content=metadata, media_type="application/samlmetadata+xml")
    except ImportError:
        return Response(content="SAML support is not installed", status_code=501)
