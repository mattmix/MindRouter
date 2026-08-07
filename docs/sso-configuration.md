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
- Legacy Azure AD driver (own routes, shared linking rule as of 2.9.0):
  `backend/app/dashboard/azure_auth.py`

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

The original MindRouter SSO provider. It keeps its own routes in
`azure_auth.py`, its own `azure_oid` identity column, and its `jobTitle` group
mapping, and appears in the new registry only as a login-button descriptor.

**Changed in 2.9.0:** its email-linking fallback now enforces the same
unclaimed-account-only rule as the shared driver (see *JIT provisioning and
account linking*). The one difference an operator can notice: an Azure login
whose email matches an account carrying a **different** `azure_oid` is now
refused (logged as `sso_email_link_refused`) instead of silently rebinding that
account to the new object id. This is rare — the Entra object id is stable.
Genuinely unclaimed accounts (no `azure_oid`, no `sso_provider` — e.g. local
password accounts) still link exactly as before.

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
enforced server-side, not just cosmetically. Sign-in is rejected when the IdP
sends `email_verified` and it is not true (string forms like `"false"` count as
unverified); an IdP that omits the claim entirely is trusted.

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
must be signed (`wantAssertionsSigned: true`) while a message-level signature is
**not** required (`wantMessagesSigned: false`), requested NameID format is
`urn:oasis:names:tc:SAML:2.0:nameid-format:persistent`. The request adapter
derives scheme and host from `APP_BASE_URL` rather than from
`X-Forwarded-Proto` / `X-Forwarded-Host` — precisely so a client-supplied
forwarded host cannot relax the Destination/Recipient validation that
python3-saml performs against that value. (It falls back to the request scheme
and `Host` header only when `APP_BASE_URL` is blank, which is why you should
keep it set behind the nginx proxy.)

> **Native SAML limitations — read before choosing this path.** MindRouter's
> SAML SP holds **no key pair**, which means it cannot decrypt encrypted
> assertions, cannot sign AuthnRequests, and publishes metadata with no
> `<KeyDescriptor>`. Practical consequences:
>
> - **Assertion encryption must be disabled for this SP.** Stock Shibboleth
>   encrypts by default; if it stays on, the login fails at the IdP and no
>   response is ever POSTed to MindRouter, so **nothing appears in MindRouter's
>   logs** — check the IdP's logs for a key-resolution/encryption error.
> - An IdP configured to **require signed AuthnRequests** will refuse the
>   login. Stock Shibboleth does not require them.
> - Key-less SP metadata is awkward to register in a federation such as
>   **InCommon**, whose registrars generally expect a key.
> - There is **no Single Logout (SLO)** endpoint.
>
> **If you are at an InCommon member institution, prefer CILogon over native
> SAML** (see the CILogon section above). It reaches the same campus IdP
> through the OIDC path, needs no metadata exchange or SP key material, and is
> the configuration we recommend and test.

**IdP-side setup:**

1. Register MindRouter as an SP using the metadata served at
   `https://<your-mindrouter-host>/saml/metadata`.
2. **Disable assertion encryption for this SP** (see the limitations note
   above) and enable assertion signing.
3. Release attributes: `mail`, `displayName`, `eduPersonPrincipalName` (or
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
# Defaults to <APP_BASE_URL>/login/saml/acs when unset:
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

**Configure your IdP to release a persistent (or otherwise stable) NameID.**
MindRouter asks for
`urn:oasis:names:tc:SAML:2.0:nameid-format:persistent` in its AuthnRequest but
does **not** verify the format it gets back, so an IdP configured for
**transient** NameIDs will hand MindRouter a new subject on every login. The
consequence is a one-login lockout: the first login provisions the account and
stamps `sso_provider`; the second login misses on subject, falls through to the
email match, sees `sso_provider` already set, and is **refused** permanently
(`sso_email_link_refused`). Clearing the stale identity requires editing the
`users` row directly in the database — there is no admin UI or API for it. If
your IdP cannot emit a persistent NameID, release a stable
`eduPersonPrincipalName` and suppress the NameID.

**SAML requires HTTPS.** The `saml_request_id` cookie that carries the
AuthnRequest ID is set `SameSite=None; Secure`, because the IdP returns the
assertion by a **cross-site HTTP-POST** to the ACS and browsers withhold
`SameSite=Lax` cookies on cross-site POSTs. A `Secure` cookie is not stored
over plain `http`, so SAML cannot be exercised against a plain-http dev URL —
the ACS will reject every response as unsolicited. Test SAML against a TLS
origin.

**Dependency note:** python3-saml and its xmlsec system libraries **ship in the
Docker image** (`Dockerfile` installs `libxmlsec1-dev` + `libxmlsec1-openssl`
and runs `pip install -e .[saml]`). Bare-metal installs need
`apt install libxmlsec1-dev libxmlsec1-openssl` then `pip install .[saml]`.
Without it, SAML routes fail gracefully with a "SAML support is not installed"
error; the other providers are unaffected (the import is lazy). `GET
/saml/metadata` returns **404** (`SAML is not configured`) when the SAML env
vars above are unset, and **501** (`SAML support is not installed`) when
python3-saml is absent — but do **not** rely on the status code to tell the two
apart. `metadata_response()` builds the SAML settings first, and in the common
metadata-URL setup that build needs the python3-saml metadata parser: with
`SAML_IDP_METADATA_URL` set and the library missing, settings construction
fails and you get **404**, not 501. The 501 is only reliably reached in the
explicit entity-id / SSO-URL / cert configuration. Confirm the library
directly (`python -c "import onelogin.saml2"`) rather than inferring it from
the response code.

---

## JIT provisioning and account linking

All providers share the same semantics (`find_or_create_sso_user()` in
`sso/base.py`; the Azure driver implements the same logic with `azure_oid`):

1. **Lookup by `(provider, subject)` first** — the stable IdP identifier
   (OIDC `sub`, SAML persistent NameID, Azure object ID) stored on the user row.
2. **Then lookup by email** (lowercased), but **only unclaimed accounts are
   adopted**. If the matched account already carries **any** IdP identity —
   `azure_oid` set, or `sso_provider` set to *anything*, including the same
   provider — the login is **refused** and an `sso_email_link_refused` warning
   is logged. Email is an IdP-supplied attribute, not proof of ownership —
   without this rule, any enabled IdP could assert an existing user's address
   (including an admin's) and inherit that account.

   Because the refusal keys on `sso_provider` being set at all — not on it
   being a *different* provider — **an IdP that rotates its subject locks the
   user out after one login**: the second login misses on `(provider, subject)`,
   matches by email, sees `sso_provider` already set, and is refused. So the
   IdP must emit a **stable** subject: a persistent NameID for SAML (MindRouter
   requests `nameid-format:persistent` but does not verify what comes back), a
   stable `sub` for OIDC.

   Unclaimed accounts — notably **local username/password accounts** — are
   linked: the SSO identity is attached and **the local password is kept**, so
   the user can continue to log in either way. Display name is refreshed from
   the IdP on every login; **department and college are refreshed for Azure
   only** — `profile_from_claims()` (OIDC/Google) and `profile_from_assertion()`
   (SAML) never populate those fields, so they stay empty for those providers.

   Moving a user between providers, or clearing a stale identity after a
   subject rotation, means editing `azure_oid` / `sso_provider` / `sso_subject`
   on the `users` row **directly in the database**. There is no admin UI or API
   for it.
3. **Otherwise, a new user is created:**
   - Username = local part of the username hint (ePPN /
     `preferred_username`) or email; on collision, `_<first 8 chars of subject>`
     is appended.
   - No password hash. The account is SSO-only: it has no local password and
     cannot use the local login form. There is currently **no admin UI or API
     to add one** — `/dashboard/change-password` returns early when
     `password_hash` is NULL, and the admin user-update endpoint has no password
     field. A local credential means creating a separate local account (admin →
     **Users** → *Create Local User*) or setting the hash directly in the
     database.
   - Group = the provider's `*_DEFAULT_GROUP` setting (default `other`).
     **The group must already exist** — create it on the admin **Groups** page
     first. If the named group does not exist, provisioning is refused: the
     user is bounced to the login page with "Failed to provision user
     account" and the server logs `sso_default_group_missing` naming the
     group. Existing users are unaffected; fix the setting (or create the
     group) and the login succeeds on retry.
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
| `SAML_SP_ACS_URL` | `<APP_BASE_URL>/login/saml/acs` | Optional |
| `SAML_IDP_METADATA_URL` | – | Required for SAML unless the explicit trio below is set |
| `SAML_IDP_ENTITY_ID` | – | Required if no metadata URL |
| `SAML_IDP_SSO_URL` | – | Required if no metadata URL |
| `SAML_IDP_X509_CERT` | – | Required if no metadata URL |
| `SAML_DISPLAY_NAME` | `SSO` | Optional |
| `SAML_DEFAULT_GROUP` | `other` | Optional |
| `SAML_ATTR_EMAIL` | `mail` | Optional |
| `SAML_ATTR_NAME` | `displayName` | Optional |
| `SAML_ATTR_USERNAME` | `eduPersonPrincipalName` | Optional |

**Note on `AZURE_AD_REDIRECT_URI`:** in a Docker Compose deployment you must set
it explicitly. Compose passes it as `${AZURE_AD_REDIRECT_URI:-}`, so an unset
variable arrives in the container as an **empty string** — the placeholder
default in `settings.py` never applies, and the Azure flow will fail with a
redirect-URI mismatch.

## Deployment reminder

The app reads settings from the **container** environment. How values get there
depends on which Compose stack the deployment runs, and the two stacks in this
repo do it differently — check which one you are on before editing anything:

| Stack | Put values in | How they reach the container |
|---|---|---|
| `docker-compose.yml` (host-networked stack; what a bare `docker compose` command starts) | `/opt/mindrouter/.env` on the host | Compose interpolates `${VAR:-}` into the service's `environment:` block. Every variable in the table above — including `AZURE_AD_DEFAULT_GROUP` — is **already wired through**, so setting it in the env file is enough. |
| `docker-compose.prod.yml` (nginx/TLS stack; see [../deploy/DEPLOYMENT.md](../deploy/DEPLOYMENT.md)) | `.env.prod` in the deployment directory | The service declares `env_file:`, so the file is handed to the container wholesale. Any variable you add is picked up as-is. |

- **No compose edit is needed for the variables in the table above** on either
  stack. If you add a variable that is *not* listed there and you run the
  `docker-compose.yml` stack, add a matching `- NEW_VAR=${NEW_VAR:-}` line to its
  `environment:` block; the `env_file:` stack needs no such edit.
- **Secrets stay on the host.** `.env` / `.env.prod` are never committed — no
  client secrets, no certificates, no private keys in the repo.
- **Restart with the same Compose file the deployment was started with.**
  Settings are cached per process (see Overview), so a recreate is required:

  ```bash
  # docker-compose.yml stack
  docker compose up -d

  # docker-compose.prod.yml stack
  docker compose -f docker-compose.prod.yml up -d
  ```

  These are not interchangeable. Running the bare command on a host started with
  `-f docker-compose.prod.yml` does not reload your SSO settings — it starts the
  *other* stack alongside the running one.
- **`APP_BASE_URL` must name this deployment's own public HTTPS origin** before
  any provider will work; redirect URIs and the SAML `Destination` check are
  derived from it. On the `docker-compose.yml` stack it is passed through as
  `${APP_BASE_URL:-}`, so leaving it unset delivers an **empty string** to the
  app — the code then falls back to the request scheme and `Host` header, which
  behind a TLS-terminating proxy yields an `http://` SAML `Destination` and a
  failed validation at the IdP. Set it explicitly. See the Security notes
  below.

---

## Security notes

Behavior enforced by the shared framework (see `sso/base.py`, `sso/oidc.py`,
`sso/saml.py`):

- **Email linking only adopts unclaimed accounts.** A login is refused (logged
  as `sso_email_link_refused`) when the email matches an account already bound
  to any identity provider. Prevents a second enabled IdP from asserting an
  existing user's address — including an admin's — and inheriting the account.
  Since 2.9.0 the Azure driver enforces this too.
- **Unverified emails are rejected** for OIDC/Google **when the IdP sends
  `email_verified` and it is not true**; IdPs that omit the claim entirely are
  trusted. The claim is normalized, so string forms (`"false"`, `"0"`) do not
  pass.
- **CSRF state** is a signed, timed token (10 min) round-tripped through an
  HttpOnly cookie, checked on every OIDC callback.
- **SAML is SP-initiated only.** The ACS requires a signed `saml_request_id`
  cookie from `/login/saml` and requires the response's `InResponseTo` to echo
  that AuthnRequest ID, so unsolicited IdP-initiated POSTs to the ACS are
  refused. (This is enforced in MindRouter, not by a library setting —
  python3-saml has no unsolicited-response option, and it skips its own
  `InResponseTo` comparison when the response omits the attribute.)
  `rejectDeprecatedAlgorithm` blocks SHA-1 signatures, and assertions must be
  signed (`strict` mode). That cookie is `SameSite=None; Secure` so it survives
  the IdP's cross-site POST to the ACS, which makes **HTTPS a hard requirement
  for SAML**. Note: IdP-initiated login (e.g. launching MindRouter from a
  campus app portal tile) is therefore not supported — users must start at the
  MindRouter login page.
- **SAML IdP metadata must be served over HTTPS** — it carries the signing
  certificate, the only trust anchor for assertion validation. A plain-`http`
  `SAML_IDP_METADATA_URL` disables the provider. For a stronger anchor, pin the
  certificate locally with `SAML_IDP_ENTITY_ID` / `SAML_IDP_SSO_URL` /
  `SAML_IDP_X509_CERT` instead of fetching metadata.
- **Public URLs are derived from `APP_BASE_URL` rather than from request
  headers — keep `APP_BASE_URL` set.** OIDC redirect URIs and the SAML
  Destination/Recipient check are built from the configured base URL, so a
  spoofed `X-Forwarded-Host` cannot influence them. If `APP_BASE_URL` is blank
  the code falls back to the request's own scheme/Host headers — the OIDC path
  reads `X-Forwarded-Proto` (`sso/oidc.py`), while the SAML request adapter
  uses `request.url.scheme` plus the `Host` header and never consults
  `X-Forwarded-Proto` (`sso/saml.py`), so behind a TLS-terminating proxy a
  blank `APP_BASE_URL` yields an `http` SAML Destination. Which is exactly why
  leaving it set matters. Point it at your public HTTPS origin.
- **What the IdP must sign (SAML):** the **assertion**
  (`wantAssertionsSigned: true`). A message-level signature on the SAML
  `<Response>` is **not** required (`wantMessagesSigned: false`).
- **What the IdP must NOT do (SAML): encrypt the assertion.** MindRouter's SP
  has no key pair, so it cannot decrypt encrypted assertions and cannot sign
  AuthnRequests — see the limitations note in the SAML section above. Stock
  Shibboleth encrypts assertions by default, so this must be turned off for
  the MindRouter SP or logins fail **at the IdP**, before any response reaches
  MindRouter (meaning nothing appears in MindRouter's logs).
- **`SECRET_KEY` underpins the SSO handshake.** The signed OIDC `state` cookie
  and the SAML `saml_request_id` cookie are both signed with it
  (`state_serializer()` in `sso/base.py`). A weak or leaked `SECRET_KEY`
  therefore weakens OIDC CSRF protection and SAML SP-initiated-only
  enforcement; rotating it invalidates any login already in flight.
