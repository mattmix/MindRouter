# Registered Applications (Administrator + Integrator Guide)

A **registered application** is a first-party app — VandalChat is the first —
that signs its own users in against the same Entra tenant as MindRouter and
then obtains a MindRouter API key *for each of those users*, server-side. Its
users get per-user telemetry, per-user quota, and their own share of the
scheduler, without ever visiting MindRouter or creating a key.

This exists because the alternatives are all worse:

| Alternative | Why it was rejected |
|---|---|
| One shared API key for the whole app | Every user becomes one user. No per-person quota, no fair share, no attribution, and one leaked key exposes everyone. |
| Ask users to create a MindRouter key and paste it into the app | Defeats the point of single sign-on, and users end up with long-lived keys in a second system. |
| Give the app an admin key so it can create users itself | An admin credential can read every stored prompt and revoke anyone's key. Far more privilege than the job needs — the same mistake as the old DLP internal key. |

---

## How it works

The app calls one endpoint, once per user sign-in, with **two** credentials:

```
POST /api/apps/{slug}/sessions
X-API-Key: <the app's provisioning credential>
Content-Type: application/json

{"id_token": "<the user's Entra id_token>"}
```

* The **app credential** proves the caller is that app's server. It carries a
  single scope, `apps:provision`, and can do nothing else — not inference, not
  administration, and not provisioning for a different application.
* The **user's `id_token`** proves that specific person just authenticated.
  MindRouter verifies it in full: signature against the tenant's JWKS, audience
  equal to the app's own client id, issuer pinned to the tenant, and `exp`/`nbf`
  with 60 seconds of clock-drift tolerance.

### Two requirements on the token

**It must come from the v2.0 endpoint.** The issuer must be
`https://login.microsoftonline.com/{tenant}/v2.0`; the v1.0 issuer
(`sts.windows.net`) is refused. That older generation issues *access* tokens
whose audience is the bare client-id GUID — the same value an id_token carries
— so the two become hard to tell apart. Your app registration controls this;
MSAL and the v2.0 authorize endpoint give you v2.0 tokens by default.

**It must be an id_token, not an access token.** A token carrying `scp`,
`scope`, or `roles` is rejected. An access token minted for your own API can
share the audience of your id_tokens, and your backend receives one on every
call — so it is far more widely handled, logged, and forwarded. "Somebody was
authorised to call your API on this user's behalf" is a weaker claim than "this
user just signed in", and only the second is a basis for provisioning.

**The token must carry an email address.** MindRouter reads `email`,
`preferred_username`, or `upn`. The `email` claim is optional in Entra and off
by default, so add it as an optional claim or request the `profile` scope. A
token without one gets a `400` that says so.

Neither is sufficient alone, and that is the security property worth
understanding. Without the token, the app credential would be an unbounded
impersonation primitive — whoever stole it could act as any user in the tenant.
Without the app credential, a leaked token would be usable by anyone who got
hold of it. Requiring both means **a compromised app can only reach users who
are actually signing in to it.**

The response carries that user's key:

```json
{
  "api_key": "mr2_…",
  "key_prefix": "mr2_ESpw",
  "expires_at": "2026-09-08T17:04:11Z",
  "rotated": true,
  "user_id": 412,
  "username": "jdoe"
}
```

### No Entra administrator involvement is required

The token the app already holds was minted for **its** client id, not
MindRouter's. That is fine: Entra signs every token in a tenant with the same
keys, so MindRouter can verify it completely without having been party to the
sign-in. Accepting it is a deliberate operator decision recorded on the
application's row — the client id you enter in the admin panel — not something
inferred from the token.

So this needs **no** app registration change, no exposed API, no
`knownClientApplications`, and no on-behalf-of flow.

---

## Registering an application

**Admin → Applications.**

1. **Register** the app: slug, name, its Entra **client ID** and **tenant ID**,
   and the per-user key lifetime (default 30 days).
   The slug appears in the URL, so it is restricted to lowercase letters,
   digits and hyphens. Both Entra ids must be GUIDs.
2. **Issue a credential.** This is a separate action so it is separately
   audited. The credential is displayed **once** — MindRouter stores only a
   hash. Put it in the app's server-side configuration; it must never reach a
   browser.

That is the whole setup. The app can start provisioning immediately.

### What the app should do

* Call the endpoint **once per user sign-in**, right after it validates its own
  token, and cache the returned key server-side against that user's session.
* Keep using the key it holds until a call returns a new one. A response with
  `"rotated": false` means "keep what you have": MindRouter stores only a hash,
  so the plaintext cannot be re-shown and `api_key` is `null`. Compare
  `key_prefix` against the key you hold to confirm you are in sync.
* **If you lose your key cache — a restart with in-memory storage, a redeploy,
  a wiped Redis — send `"force_rotate": true`.** Without it you would be handed
  back a key you can no longer see, and your users would be stuck until it aged
  past half its lifetime. Do not set it on every call; the ordinary path exists
  so concurrent sessions do not invalidate each other.
* Never send the key to the browser. It is a MindRouter credential for a real
  user account; treat it exactly like a password.

Responses worth handling:

| Status | Meaning | What to do |
|---|---|---|
| `400` | The token carries no email address | Add the optional `email` claim, or request the `profile` scope, in your Entra app registration. |
| `401` | The `id_token` was not accepted | Re-run your own sign-in. The specific reason is logged on the MindRouter side and deliberately not returned — including the v2.0-endpoint and id_token-vs-access-token rules above, so check those first when every call fails. |
| `403` | Your credential does not belong to that app, or the account is inactive | Configuration error, or the user was deactivated. Do not retry. |
| `404` | Unknown or disabled application | Same response either way, so a stolen credential cannot enumerate what is registered here. |
| `409` | The account could not be provisioned | The email is already bound to a different identity provider. A human needs to resolve it. |
| `429` | You forced a rotation moments ago | Only relevant with `force_rotate`. Honour `Retry-After`; minting is deliberately expensive, so it is allowed once a minute per user. |

---

## Group classification

MindRouter assigns a user's group from the Azure `jobTitle` attribute, which it
reads from Microsoft Graph at sign-in. **An `id_token` does not carry
`jobTitle`**, so a user an app creates lands in the default group.

Rather than let the app assert a group — it should not be trusted to — those
accounts are marked unclassified, and their group is settled automatically the
first time they sign in to MindRouter directly, where the full directory
profile is available. Until then they hold default limits.

Two rules keep that from ever *removing* privilege. Only accounts an app
genuinely created are marked unclassified — an account that already existed and
was merely linked by email keeps its group, which matters because the local
bootstrap admin is exactly such an account. And reclassification never touches
a user who is in an admin or auditor group, because `jobTitle` maps to only
four group names and `admin` is not one of them: a deliberate placement
outranks an inferred one. Setting a user's group from Admin → Users also
settles the flag permanently.

**Admin → Applications** lists who is waiting. Note that app users are ordinary
MindRouter users in ordinary groups: many people will be both API users and app
users, and there is deliberately no separate "app users" group.

---

## Revocation and rotation

Per-user keys rotate silently. The app calls the endpoint on every sign-in, and
MindRouter hands back the existing key while it still has more than half its
lifetime left, minting a fresh one otherwise. So a leaked user key stops working
within the application's configured TTL without anyone doing anything, and
concurrent sessions do not invalidate each other.

Everything else is in the admin panel:

* **Rotate credential** — revokes the previous credential *before* minting the
  next, in one transaction. There is never a window with two live credentials.
* **Revoke all session keys** — revokes the per-user keys but leaves the app
  running. Users re-provision transparently on their next sign-in. This is the
  blunt instrument for "something about that app's sessions looks wrong".
* **Disable** — revokes *everything* the app holds: its credential and every
  per-user key. Disabling without revoking would only stop new sessions while
  existing keys kept working for their full TTL, which is a month by default.
* **Deregister** — the same revocation, then the registration is removed. The
  keys themselves are kept and detached, because their request history is what
  explains what the app did.

Every one of these is written to the admin audit log, as is every session the
app provisions. Sessions are logged with no human actor — the audit view shows
them as *application*.

---

## Security notes

* **An app-minted key is inference-only, whoever owns it.** Scopes intersect
  with the owner's group privilege and never extend it, so when an
  administrator uses a first-party app, the key that app receives is not an
  admin credential.
* **Identity comes only from the verified token.** The request body accepts
  nothing but `id_token` — no email, no username, no object id. An app
  asserting who its users are is precisely the attack this design prevents.
* **The audience check is exact.** MindRouter accepts a token only if its `aud`
  equals the client id recorded on that application's row. Accepting any token
  from the tenant would let every application in the tenant provision accounts
  here.
* **Existing accounts are not adopted blindly.** Provisioning runs through the
  same predicate as interactive Azure sign-in, so an app-supplied email can
  only link to a local account that no identity provider has claimed yet. See
  [SSO configuration](sso-configuration.md#jit-provisioning-and-account-linking).
* **Legacy keys cannot drift into this.** `apps:provision` is opt-in by
  construction: a key with no scope list — which is every key created before
  this feature — is refused, rather than treated as unrestricted.
* **`nonce` is not validated, deliberately.** It binds a token to the sign-in
  request that produced it, and MindRouter did not issue that request. Signature,
  exact audience, and pinned issuer are what bind the token to a known
  application in a known tenant.

See also: [SSO configuration](sso-configuration.md) for the tenant-side setup
that MindRouter's own sign-in uses.
