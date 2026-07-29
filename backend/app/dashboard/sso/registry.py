############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# sso/registry.py: enabled-provider registry + routes
#
############################################################

"""Registry of enabled SSO providers and their FastAPI routes.

``enabled_providers()`` powers the login page: one button per enabled
provider, each with a label and Bootstrap icon. Azure AD keeps its original
routes in ``azure_auth.py``; it appears here only as a descriptor.
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dashboard.sso import oidc as oidc_driver
from backend.app.dashboard.sso import saml as saml_driver
from backend.app.db.session import get_async_db
from backend.app.settings import get_settings

sso_router = APIRouter(tags=["sso"])


@dataclass
class ProviderDescriptor:
    """What the login template needs to render one SSO button."""

    id: str
    label: str          # "Sign in with <label>"
    login_url: str
    icon: str           # Bootstrap icon class


def enabled_providers(org_name: Optional[str] = None) -> list[ProviderDescriptor]:
    """Descriptors for every configured provider, in display order.

    Azure AD (the primary/institutional IdP for existing deployments) is
    labeled with the branded org name when one is set; other providers use
    their own display-name settings.
    """
    s = get_settings()
    providers: list[ProviderDescriptor] = []
    if s.azure_ad_enabled:
        providers.append(ProviderDescriptor(
            id="azure",
            label=org_name or "SSO",
            login_url="/login/azure",
            icon="bi-microsoft",
        ))
    if s.saml_sso_enabled:
        providers.append(ProviderDescriptor(
            id="saml",
            label=(org_name if not s.azure_ad_enabled and s.saml_display_name == "SSO" else None)
                  or s.saml_display_name,
            login_url="/login/saml",
            icon="bi-shield-lock",
        ))
    if s.oidc_sso_enabled:
        providers.append(ProviderDescriptor(
            id="oidc",
            label=(org_name if not providers and s.oidc_sso_display_name == "SSO" else None)
                  or s.oidc_sso_display_name,
            login_url="/login/oidc",
            icon="bi-box-arrow-in-right",
        ))
    if s.google_sso_enabled:
        providers.append(ProviderDescriptor(
            id="google",
            label="Google",
            login_url="/login/google",
            icon="bi-google",
        ))
    return providers


# --- Google ---------------------------------------------------------------

@sso_router.get("/login/google")
async def google_login(request: Request):
    cfg = oidc_driver.google_config()
    if not cfg:
        return _not_configured("Google")
    return await oidc_driver.begin_login(request, cfg)


@sso_router.get("/login/google/authorized")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    cfg = oidc_driver.google_config()
    if not cfg:
        return _not_configured("Google")
    return await oidc_driver.handle_callback(request, cfg, db, code, state, error, error_description)


# --- Generic OIDC -----------------------------------------------------------

@sso_router.get("/login/oidc")
async def oidc_login(request: Request):
    cfg = oidc_driver.generic_config()
    if not cfg:
        return _not_configured("OIDC")
    return await oidc_driver.begin_login(request, cfg)


@sso_router.get("/login/oidc/authorized")
async def oidc_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    cfg = oidc_driver.generic_config()
    if not cfg:
        return _not_configured("OIDC")
    return await oidc_driver.handle_callback(request, cfg, db, code, state, error, error_description)


# --- SAML -------------------------------------------------------------------

@sso_router.get("/login/saml")
async def saml_login(request: Request):
    if not get_settings().saml_sso_enabled:
        return _not_configured("SAML")
    return await saml_driver.begin_login(request)


@sso_router.post("/login/saml/acs")
async def saml_acs(request: Request, db: AsyncSession = Depends(get_async_db)):
    if not get_settings().saml_sso_enabled:
        return _not_configured("SAML")
    return await saml_driver.handle_acs(request, db)


@sso_router.get("/saml/metadata")
async def saml_metadata(request: Request):
    return await saml_driver.metadata_response(request)


def _not_configured(name: str):
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url=f"/login?error={name}+SSO+is+not+configured", status_code=302)
