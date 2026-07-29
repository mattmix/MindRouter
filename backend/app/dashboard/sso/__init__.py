############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# sso: pluggable SSO provider framework (OIDC, Google, SAML)
#
############################################################

"""Pluggable SSO providers.

Azure AD (the original provider) lives in ``dashboard/azure_auth.py`` and is
intentionally untouched; this package adds Google, generic OIDC (Okta,
Keycloak, Auth0, CILogon/InCommon), and SAML 2.0 (Shibboleth/InCommon IdPs,
ADFS). Each provider is enabled purely by its environment configuration —
see ``docs/sso-configuration.md``.
"""

from backend.app.dashboard.sso.registry import enabled_providers, sso_router

__all__ = ["enabled_providers", "sso_router"]
