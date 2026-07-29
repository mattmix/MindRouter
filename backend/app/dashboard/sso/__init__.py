############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# sso: pluggable SSO provider framework (OIDC, Google, SAML)
#
############################################################

"""Pluggable SSO providers.

Azure AD (the original provider) keeps its own routes and ``azure_oid``
identity column in ``dashboard/azure_auth.py``, but as of 2.9.0 it shares this
package's email-linking rule: an account already claimed by an identity
provider is never adopted by an email match. This package adds Google, generic
OIDC (Okta, Keycloak, Auth0, CILogon/InCommon), and SAML 2.0
(Shibboleth/InCommon IdPs, ADFS). Each provider is enabled purely by its
environment configuration — see ``docs/sso-configuration.md``.
"""

from backend.app.dashboard.sso.registry import enabled_providers, sso_router

__all__ = ["enabled_providers", "sso_router"]
