# MindRouter SSO Configuration (Administrator Guide)

**Last updated: 2026-07-29**

This guide covers configuring single sign-on for the MindRouter dashboard. It is
grounded in the actual implementation:

- Settings: `backend/app/settings.py` (the `azure_ad_*`, `google_sso_*`,
  `oidc_sso_*`, `saml_*` fields and the `*_enabled` properties)
- Provider registry + routes: `backend/app/dashboard/sso/registry.py`
- OIDC driver (Google + generic): `backend/app/dashboard/sso/oidc.py`
- SAML 2.0 SP driver: `backend/app/dashboard/sso/saml.py`
- Shared JIT provisioning: `backend/app/dashboard/sso/base.py`
- Legacy Azure AD driver (unchanged): `backend/app/dashboard/azure_auth.py`

## Overview

- **Providers are enabled purely by environment variables.** There is no admin-UI
  toggle. A provider is "on" when its required variables are set (see the
  per-provider enablement rules below); it is "off" when they are unset. No code
  changes or feature flags involved.
- **Any subset can be enabled simultaneously.** Azure AD, Google, generic OIDC,
  and SAML are independent. The login page renders **one button per enabled
  provider**, in this fixed order: Azure AD, SAML, generic OIDC, Google.
- **Local username/password accounts are always available.** SSO never disables
  the local login form; SSO buttons appear alongside it.
- **Config is process-level.** `get_settings()` is `@lru_cache`d — each worker
  reads the environment once at startup. OIDC discovery documents and SAML IdP
  metadata are also cached in-process (1 hour TTL, per worker). After changing
  any SSO env var, **restart the app** (`docker compose up -d` recreates the
  container); do not expect a live reload.

Enablement rules (from the `settings.py` properties):

| Provider | Enabled when |
|---|---|
| Azure AD | `AZURE_AD_CLIENT_ID` **and** `AZURE_AD_TENANT_ID` set (secret is still required for the token exchange to succeed) |
| Google | `GOOGLE_SSO_CLIENT_ID` **and** `GOOGLE_SSO_CLIENT_SECRET` set |
| Generic OIDC | `OIDC_SSO_ISSUER` **and** `OIDC_SSO_CLIENT_ID` **and** `OIDC_SSO_CLIENT_SECRET` set |
| SAML | `SAML_SP_ENTITY_ID` **and** (`SAML_IDP_METADATA_URL` **or** all three of `SAML_IDP_ENTITY_ID` / `SAML_IDP_SSO_URL` / `SAML_IDP_X509_CERT`) set |

---

## Azure AD / Entra ID

The original MindRouter SSO provider. Its behavior is **unchanged** — it keeps
its own routes in `azure_auth.py` and appears in the new registry only as a
login-button descriptor.

**Routes:** `GET /login/azure` (start), `GET /login/azure/authorized` (callback).

**IdP-side setup (Azure portal → App registrations):**

1. Register a web application in your tenant.
2. Add a **Web redirect URI**: `https://<your-mindrouter-host>/login/azure/authorized`.
3. Create a client secret.
4. Grant delegated Microsoft Graph permission `User.Read` (the driver requests
   scopes `openid profile email User.Read` and reads the profile from
   `https://graph.microsoft.com/v1.0/me`).

**MindRouter-side env vars:**

```bash
AZURE_AD_CLIENT_ID=<application (client) id>
AZURE_AD_CLIENT_SECRET=<client secret value>
AZURE_AD_TENANT_ID=<directory (tenant) id>
AZURE_AD_REDIRECT_URI=https://<your-mindrouter-host>/login/azure/authorized
```

Unlike the newer providers, the Azure redirect URI is **not** derived from the
request — `AZURE_AD_REDIRECT_URI` must be set to the full absolute URL and must
match the app registration exactly.

**Group mapping via job title** (Azure-only behavior, in
`_map_job_title_to_group()`): for a brand-new user, the Graph `jobTitle` is
matched case-insensitively — contains "student" → group `students`, contains
"faculty" or "professor" → `faculty`, contains "staff" → `staff`; anything else
(or no title) → `AZURE_AD_DEFAULT_GROUP` (default `other`). The group name also
maps to the user's role (`students`→STUDENT, `faculty`→FACULTY, `staff`→STAFF,
`admin`→ADMIN). Graph `department` and `officeLocation` populate the user's
department/college fields.

---

## Google

OIDC authorization-code flow against `https://accounts.google.com` (hard-coded
issuer), handled by the shared driver in `sso/oidc.py`.

**Routes:** `GET /login/google`, `GET /login/google/authorized`.

**IdP-side setup (Google Cloud console → APIs & Services → Credentials):**

1. Create an **OAuth client ID** of type *Web application*.
2. Add authorized redirect URI: `https://<your-mindrouter-host>/login/google/authorized`.
3. Configure the OAuth consent screen for your organization.

**MindRouter-side env vars:**

```bash
GOOGLE_SSO_CLIENT_ID=<client id>.apps.googleusercontent.com
GOOGLE_SSO_CLIENT_SECRET=<client secret>
# Optional — defaults to <APP_BASE_URL>/login/google/authorized:
GOOGLE_SSO_REDIRECT_URI=https://<your-mindrouter-host>/login/google/authorized
# Optional — restrict sign-in to one Google Workspace domain:
GOOGLE_SSO_HOSTED_DOMAIN=example.edu
GOOGLE_SSO_DEFAULT_GROUP=other
```

`GOOGLE_SSO_HOSTED_DOMAIN` does two things: it passes `hd=<domain>` on the
authorization request (Google pre-filters the account picker) **and** the
callback rejects any profile whose `hd` claim does not match — so it is
enforced server-side, not just cosmetically. Accounts with
`email_verified: false` are always rejected.

The login button is always labeled **"Sign in with Google"**.

---

## Generic OIDC (Okta, Keycloak, Auth0, ...)

Any spec-compliant OIDC IdP works. Endpoints are taken from the issuer's
discovery document at `<issuer>/.well-known/openid-configuration` (fetched at
first login, cached in-process for 1 hour) — you never configure token/authorize
URLs by hand. Identity comes from the IdP's `userinfo` endpoint, so the IdP must
publish one (all mainstream IdPs do).

**Routes:** `GET /login/oidc`, `GET /login/oidc/authorized`.

**IdP-side setup:**

1. Register a confidential **Web** client (authorization-code grant).
2. Redirect/callback URI: `https://<your-mindrouter-host>/login/oidc/authorized`.
3. Ensure the client can request scopes `openid profile email` (or adjust
   `OIDC_SSO_SCOPES`).

**MindRouter-side env vars:**

```bash
OIDC_SSO_ISSUER=https://idp.example.edu/realms/campus   # issuer base URL, no trailing slash needed
OIDC_SSO_CLIENT_ID=<client id>
OIDC_SSO_CLIENT_SECRET=<client secret>
# Optional — defaults to <APP_BASE_URL>/login/oidc/authorized:
OIDC_SSO_REDIRECT_URI=https://<your-mindrouter-host>/login/oidc/authorized
OIDC_SSO_DISPLAY_NAME=Okta          # login button reads "Sign in with <this>"
OIDC_SSO_SCOPES="openid profile email"
OIDC_SSO_DEFAULT_GROUP=other
```

Claims used: `sub` (stable subject), `email` (required; rejected if
`email_verified` is present and not true — string forms like `"false"` count as unverified), `name` (display name),
`preferred_username` (username hint for the generated local username).

## InCommon via CILogon (recommended InCommon path)

MindRouter's recommended way to accept **InCommon federation** logins
(university Shibboleth accounts nationwide) is **CILogon**, which acts as an
OIDC gateway in front of the whole federation. You configure MindRouter's
generic OIDC provider against CILogon — no SAML metadata exchange, no
per-campus registration:

1. Register an OIDC client at **cilogon.org** (CILogon client registration).
   Callback URL: `https://<your-mindrouter-host>/login/oidc/authorized`.
2. Configure:

```bash
OIDC_SSO_ISSUER=https://cilogon.org
OIDC_SSO_CLIENT_ID=cilogon:/client_id/<...>
OIDC_SSO_CLIENT_SECRET=<secret>
OIDC_SSO_DISPLAY_NAME=InCommon
OIDC_SSO_DEFAULT_GROUP=other
```

Users pick their home institution on the CILogon page, authenticate at their
campus IdP, and come back with standard OIDC claims (`sub` is a stable
`http://cilogon.org/serverA/users/...` identifier). If you need direct SAML to a
single campus IdP instead, use the native SAML provider below.

---

## Native SAML 2.0 (Shibboleth IdP, ADFS)

A single-IdP SAML SP built on **python3-saml**, for deployments that must speak
SAML directly (campus Shibboleth IdP, ADFS) rather than going through CILogon.

**Routes:**

| Route | Purpose |
|---|---|
| `GET /login/saml` | SP-initiated AuthnRequest redirect to the IdP (HTTP-Redirect binding) |
| `POST /login/saml/acs` | Assertion Consumer Service (HTTP-POST binding) |
| `GET /saml/metadata` | SP metadata XML — give this URL (or its output) to the IdP admin |

**SP characteristics** (from `build_saml_settings()`): strict mode, assertions
must be signed (`wantAssertionsSigned: true`), requested NameID format is
`urn:oasis:names:tc:SAML:2.0:nameid-format:persistent`. The request adapter
honors `X-Forwarded-Proto` / `X-Forwarded-Host`, so signature/destination
validation works behind the nginx proxy.

**IdP-side setup:**

1. Register MindRouter as an SP using the metadata served at
   `https://<your-mindrouter-host>/saml/metadata`.
2. Release attributes: `mail`, `displayName`, `eduPersonPrincipalName` (or
   whatever you map via `SAML_ATTR_*` below). An email-format NameID also works
   as a fallback for the email (common with ADFS).

**MindRouter-side env vars — metadata-URL style (typical Shibboleth):**

```bash
SAML_SP_ENTITY_ID=https://<your-mindrouter-host>/saml/metadata
SAML_IDP_METADATA_URL=https://idp.example.edu/idp/shibboleth
```

**— or explicit style (no metadata URL available):**

```bash
SAML_SP_ENTITY_ID=https://<your-mindrouter-host>/saml/metadata
SAML_IDP_ENTITY_ID=https://idp.example.edu/idp/shibboleth
SAML_IDP_SSO_URL=https://idp.example.edu/idp/profile/SAML2/Redirect/SSO
SAML_IDP_X509_CERT="MIIC...single-line base64, no PEM headers..."
```

**Optional:**

```bash
# Defaults to <request scheme+host>/login/saml/acs when unset:
SAML_SP_ACS_URL=https://<your-mindrouter-host>/login/saml/acs
SAML_DISPLAY_NAME=Example University
SAML_DEFAULT_GROUP=other
# Attribute mapping (defaults are eduPerson conventions):
SAML_ATTR_EMAIL=mail
SAML_ATTR_NAME=displayName
SAML_ATTR_USERNAME=eduPersonPrincipalName
```

Subject selection for account keying: persistent NameID if the IdP sends one,
else the `SAML_ATTR_USERNAME` attribute (ePPN), else the email.

**Dependency note:** python3-saml and its xmlsec system libraries **ship in the
Docker image** (`Dockerfile` installs `libxmlsec1-dev` + `libxmlsec1-openssl`
and runs `pip install -e .[saml]`). Bare-metal installs need
`apt install libxmlsec1-dev libxmlsec1-openssl` then `pip install .[saml]`.
Without it, SAML routes fail gracefully with a "SAML support is not installed"
error; the other providers are unaffected (the import is lazy).

---

## JIT provisioning and account linking

All providers share the same semantics (`find_or_create_sso_user()` in
`sso/base.py`; the Azure driver implements the same logic with `azure_oid`):

1. **Lookup by `(provider, subject)` first** — the stable IdP identifier
   (OIDC `sub`, SAML persistent NameID, Azure object ID) stored on the user row.
2. **Then lookup by email** (lowercased), but **only unclaimed accounts are
   adopted**. If the matched account already carries an identity from another
   provider (`azure_oid` or a different `sso_provider`), the login is
   **refused** and an `sso_email_link_refused` warning is logged. Email is an
   IdP-supplied attribute, not proof of ownership — without this rule, any
   enabled IdP could assert an existing user's address (including an admin's)
   and inherit that account.

   Unclaimed accounts — notably **local username/password accounts** — are
   linked: the SSO identity is attached and **the local password is kept**, so
   the user can continue to log in either way. Display name, department, and
   college are refreshed from the IdP on every login.

   To move a user between providers, an admin must first clear the old identity
   on the user row.
3. **Otherwise, a new user is created:**
   - Username = local part of the username hint (ePPN /
     `preferred_username`) or email; on collision, `_<first 8 chars of subject>`
     is appended.
   - No password hash (SSO-only account until an admin sets one).
   - Group = the provider's `*_DEFAULT_GROUP` setting (default `other`).
     **The group must already exist** — create it on the admin **Groups** page
     first. If the named group does not exist, the user is created with no
     group and **no quota row**, so keep this pointed at a real group.
   - **Quota is seeded from the group** (`rpm_limit` copied from the group's
     defaults).

Deactivated accounts (`is_active = false`) are refused at login regardless of
provider.

## Login button labels (branding tie-in)

Button labels come from `enabled_providers()` in `sso/registry.py` and tie into
**Admin → Branding → "Institution / organization name"** (`branding.org_name`):

- **Azure AD** is treated as the primary/institutional provider: its button is
  labeled with the branding org name (e.g. "Sign in with University of Idaho");
  falls back to "SSO" when no org name is set.
- **SAML** uses the org name when Azure is not enabled and `SAML_DISPLAY_NAME`
  is left at its default `SSO`; otherwise it shows `SAML_DISPLAY_NAME`.
- **Generic OIDC** uses the org name when it is the first enabled provider and
  `OIDC_SSO_DISPLAY_NAME` is left at its default `SSO`; otherwise it shows
  `OIDC_SSO_DISPLAY_NAME`.
- **Google** is always labeled "Google".

Practical rule: for your institutional IdP, set the org name in Admin →
Branding and leave `*_DISPLAY_NAME` alone; for secondary providers, set an
explicit `OIDC_SSO_DISPLAY_NAME` / `SAML_DISPLAY_NAME`.

## Environment variable reference

| Variable | Default | Required? |
|---|---|---|
| `AZURE_AD_CLIENT_ID` | – | Required for Azure |
| `AZURE_AD_CLIENT_SECRET` | – | Required for Azure |
| `AZURE_AD_TENANT_ID` | – | Required for Azure |
| `AZURE_AD_REDIRECT_URI` | `https://your-domain.example.com/login/azure/authorized` (placeholder) | Required for Azure (absolute URL) |
| `AZURE_AD_DEFAULT_GROUP` | `other` | Optional |
| `GOOGLE_SSO_CLIENT_ID` | – | Required for Google |
| `GOOGLE_SSO_CLIENT_SECRET` | – | Required for Google |
| `GOOGLE_SSO_REDIRECT_URI` | `<APP_BASE_URL>/login/google/authorized` | Optional |
| `GOOGLE_SSO_HOSTED_DOMAIN` | – | Optional (restricts to a Workspace domain) |
| `GOOGLE_SSO_DEFAULT_GROUP` | `other` | Optional |
| `OIDC_SSO_ISSUER` | – | Required for OIDC |
| `OIDC_SSO_CLIENT_ID` | – | Required for OIDC |
| `OIDC_SSO_CLIENT_SECRET` | – | Required for OIDC |
| `OIDC_SSO_REDIRECT_URI` | `<APP_BASE_URL>/login/oidc/authorized` | Optional |
| `OIDC_SSO_DISPLAY_NAME` | `SSO` | Optional |
| `OIDC_SSO_SCOPES` | `openid profile email` | Optional |
| `OIDC_SSO_DEFAULT_GROUP` | `other` | Optional |
| `SAML_SP_ENTITY_ID` | – | Required for SAML |
| `SAML_SP_ACS_URL` | `<request scheme+host>/login/saml/acs` | Optional |
| `SAML_IDP_METADATA_URL` | – | Required for SAML unless the explicit trio below is set |
| `SAML_IDP_ENTITY_ID` | – | Required if no metadata URL |
| `SAML_IDP_SSO_URL` | – | Required if no metadata URL |
| `SAML_IDP_X509_CERT` | – | Required if no metadata URL |
| `SAML_DISPLAY_NAME` | `SSO` | Optional |
| `SAML_DEFAULT_GROUP` | `other` | Optional |
| `SAML_ATTR_EMAIL` | `mail` | Optional |
| `SAML_ATTR_NAME` | `displayName` | Optional |
| `SAML_ATTR_USERNAME` | `eduPersonPrincipalName` | Optional |

## Deployment reminder

The app reads settings from the container environment only — `.env` files are
**not** mounted into the container. Therefore:

- **Every SSO variable must be listed in `docker-compose.yml` under
  `environment:`** using `${VAR:-}` passthrough. All variables above are
  already listed there **except `AZURE_AD_DEFAULT_GROUP`** — it currently rides
  on its in-code default (`other`); add a passthrough line if you ever need to
  override it.
- **Values live in `/opt/mindrouter/.env` on the production host.** Docker
  Compose interpolates them at container start. Never commit client secrets or
  certificates to the repo.
- After editing `/opt/mindrouter/.env`, recreate the container
  (`docker compose up -d`) — a restart is required because settings are cached
  per process (see Overview).

---

## Security notes

Behavior enforced by the shared framework (see `sso/base.py`, `sso/oidc.py`,
`sso/saml.py`):

- **Email linking only adopts unclaimed accounts.** A login is refused (logged
  as `sso_email_link_refused`) when the email matches an account already bound
  to another provider. Prevents a second enabled IdP from asserting an existing
  user's address — including an admin's — and inheriting the account.
- **Unverified emails are rejected** for OIDC/Google. The `email_verified`
  claim is normalized, so string forms (`"false"`, `"0"`) do not pass.
- **CSRF state** is a signed, timed token (10 min) round-tripped through an
  HttpOnly cookie, checked on every OIDC callback.
- **SAML is SP-initiated only.** The ACS requires a signed `saml_request_id`
  cookie from `/login/saml` and requires the response's `InResponseTo` to echo
  that AuthnRequest ID, so unsolicited IdP-initiated POSTs to the ACS are
  refused. (This is enforced in MindRouter, not by a library setting —
  python3-saml has no unsolicited-response option, and it skips its own
  `InResponseTo` comparison when the response omits the attribute.)
  `rejectDeprecatedAlgorithm` blocks SHA-1 signatures, and assertions must be
  signed (`strict` mode). Note: IdP-initiated login (e.g. launching MindRouter
  from a campus app portal tile) is therefore not supported — users must start
  at the MindRouter login page.
- **SAML IdP metadata must be served over HTTPS** — it carries the signing
  certificate, the only trust anchor for assertion validation. A plain-`http`
  `SAML_IDP_METADATA_URL` disables the provider. For a stronger anchor, pin the
  certificate locally with `SAML_IDP_ENTITY_ID` / `SAML_IDP_SSO_URL` /
  `SAML_IDP_X509_CERT` instead of fetching metadata.
- **Public URLs come from `APP_BASE_URL`, not request headers.** OIDC redirect
  URIs and the SAML Destination/Recipient check are derived from the configured
  base URL so a spoofed `X-Forwarded-Host` cannot influence them. Set
  `APP_BASE_URL` to your public HTTPS origin.
