# UI Branding & Theming

MindRouter can be rebranded per deployment so a single installation matches one
institution's visual identity — organization name, logos, favicon, and accent
colors — across the entire web UI, in both light and dark mode.

This is an **installation-wide** setting (not per-user or per-group). Every user
of the deployment sees the same brand.

## Where

Admin menu → **Branding** (`/admin/branding`). Viewing requires admin-read;
saving requires a full admin (`is_admin`). All changes are recorded in the admin
audit log (`branding.*` actions).

## What you can customize

| Field | Applies to |
|-------|-----------|
| Organization / app name | Top navbar, browser tab titles, footer |
| Institution / organization name | Login page SSO button ("Sign in with …") and the SSO helper text |
| Tagline | Footer, beside the name |
| Footer attribution | Small credit line in the footer of **every** page, including the public login and status pages ("Powered by …"). Clear it to remove the line entirely. |
| Footer attribution link | Optional link target for the attribution text (`http://`, `https://`, or a site-relative `/path`); blank renders plain text |
| Accent color — light theme | Buttons, links, focus rings, stat-card accents (light mode) |
| Accent color — dark theme | Same, in dark mode |
| Headline color | Headlines such as the email blog/notification title; defaults to a neutral near-black (light accents read poorly as headline text). Buttons/links keep the accent. |
| Logo — dark theme | Top navbar (always a dark background), plus footer and login card in dark mode |
| Logo — light theme | Footer and login card in light mode |
| Favicon | Browser tab icon |

### App name vs. institution name

The first two rows are **two different fields** and are easy to confuse:

- **Organization / app name** (`branding.app_name`, max 80 chars) is the
  *product* name — what this software is called in the navbar, browser tab
  titles, and footer.
- **Institution / organization name** (`branding.org_name`, optional, max 120
  chars, added in 2.8.48) is the *institution running the deployment*. It
  supplies the sign-in wording: the SSO button reads "Sign in with University of
  Idaho" and the helper text below it reads "Use your University of Idaho
  credentials to sign in." (It also names the provider in the default chat
  assistant system prompt and in the "this account has no local password" login
  message.)

Leave the institution name blank and the login page falls back to generic
wording — **"Sign in with SSO"** and **"Use your organization credentials to
sign in."** Which providers get the org-name label depends on which SSO
providers are enabled; see
[SSO configuration → Login button labels](sso-configuration.md#login-button-labels-branding-tie-in).

### Footer attribution

The footer carries a small credit line above the NSF award notice. Unlike the
other fields it is **not blank by default**: MindRouter ships with
`branding.footer_note` = `Powered by RCDS` and `branding.footer_note_url` =
`https://hpc.uidaho.edu`, the operator credit for the reference deployment at
the University of Idaho, so upgrading an existing install changes nothing.

Any deployment can rewrite both fields (Admin → Branding → *Footer attribution*
/ *Footer attribution link*, max 120 / 300 chars) or **clear the text field to
remove the line altogether** — an explicitly-saved empty value is honored and
is *not* replaced by the default. So a fresh install elsewhere shows
"Powered by RCDS" until an admin changes or removes it; that is the one branding
field a new operator should review before going live.

The link is validated on save: only `http://`, `https://`, and site-relative
`/path` targets are stored, and only such values are ever emitted into the
`href` (so a stored value cannot become a `javascript:` URL). With no link set,
the attribution renders as plain text.

The NSF award credit beside it is **product** attribution for the grant that
funded MindRouter, not deployment branding, and is intentionally not
configurable.

Logos appear in the **navbar (header)**, the **footer**, and the **login card**.
The navbar always has a dark background, so it uses the dark-theme logo (falling
back to the light one); the footer and login card follow the active theme and
swap between the two logo variants automatically.

The **live preview** on the page shows a mock navbar, buttons, link, and stat
card in both light and dark themes, updating as you pick colors. Nothing is
applied site-wide until you click **Save**.

Use **Reset to defaults** to restore the stock MindRouter name, colors, and
remove all uploaded assets. It clears the institution name too, so the login
page returns to generic SSO wording. An install with no branding configured
looks the same way: stock name and colors, no logo on the login card,
"Sign in with SSO" / "Use your organization credentials to sign in." — and the
default "Powered by RCDS" footer attribution, since reset restores shipped
defaults rather than blanking every field.

## Accessible accent colors (contrast handling)

A brand accent is often a light color (e.g. University of Idaho *Pride Gold*
`#F1B300`) that would be illegible with the default white button text. For each
theme's accent, MindRouter derives two accessible companions so light accents
stay readable:

- **`--mr-accent-on`** — the foreground used *on* an accent fill (button text).
  White by default; flips to black when white would fall below a 3.0 contrast
  ratio on the accent (matching Bootstrap's convention: white on blue/red, dark
  on gold/yellow). A gold button gets black, legible text.
- **`--mr-accent-ink`** — the accent used as *text* on the page background
  (links, `.text-primary`, outline-button text, active sidebar item). Darkened
  (light page) or lightened (dark page) only as far as needed to reach a 4.5:1
  WCAG contrast ratio. Fills, borders, and focus rings keep the true brand color.

Stock mid-tone accents (default blue `#0d6efd`) are left untouched — the
derivation only intervenes when the accent would otherwise be unreadable. The
math lives in `_best_fg` / `_accessible_ink` in
`backend/app/services/branding.py`.

## Emails

Outgoing emails (blog-post notifications, admin/bulk notifications, the test
email) follow the same brand — the blue default header is gone. Because email is
a constrained medium, the treatment differs from the web UI:

- **No CSS variables or SVG.** Email clients strip `<style>` and don't render
  SVG, so all styling is inline and the logo must be a **raster** image. Upload a
  PNG/JPG/GIF to the **Email logo** slot on the branding page (separate from the
  navbar SVG logos). A horizontal logo on a transparent/white background works
  best.
- **The logo is embedded, not linked.** It's attached inline via `cid:` (the
  message becomes `multipart/related`), so it displays even when the client
  blocks remote images. If no email logo is set, the header shows the
  organization name as text.
- **Accent colors are contrast-safe.** The header carries a thin accent rule, the
  blog-post button uses the accent as its fill with the accessible foreground
  (black text on gold), and links/headings use the darkened `ink` — so a light
  brand accent stays legible on the white email background.

Email templates live in `backend/app/services/email_service.py` (`_wrap_html`
builds the branded shell; `_send_one` performs the CID embed).

## How it works

- **Text/color values** are stored as `branding.*` rows in the `app_config`
  key/value table (`crud.get_config_json` / `crud.set_config`).
- **Uploaded logos/favicon** are written to `BRANDING_STORAGE_PATH`
  (default `/data/branding`, a persistent `branding_data` volume) and served,
  cache-busted, from the public route `/branding/asset/{filename}` — public so
  logos also render on the login page.
- Accent colors are injected into `base.html` as CSS custom properties
  (`--bs-primary` and friends) scoped to `:root` and `[data-bs-theme="dark"]`,
  layering on top of the existing Bootstrap 5 theming. Only validated hex values
  are ever emitted, so the injection is safe.
- Branding is read on every page from an in-memory cache
  (`backend/app/services/branding.py`) that is loaded at startup and refreshed
  every ~15s, so a save propagates to all uvicorn workers within a few seconds
  without a restart. The saving worker refreshes immediately.

## Deployment notes

`BRANDING_STORAGE_PATH` and `BRANDING_MAX_LOGO_MB` must be present in
`settings.py` **and** `docker-compose.yml` (pydantic-settings only reads env
inside the container). The `branding_data` named volume and the Dockerfile
`mkdir /data/branding` keep uploaded assets across container rebuilds.

## Notes / limits

- Logo/favicon uploads are capped at `BRANDING_MAX_LOGO_MB` (default 4 MB).
  Allowed logo types: PNG, JPG, WebP, SVG, GIF. Favicon: ICO, PNG, SVG.
- The top navbar keeps its dark background by design (safe contrast in both
  themes); the accent color drives buttons, links, and highlights rather than
  the navbar itself.
- The public landing page marketing copy is not templated by this feature —
  branding covers the app chrome (navbar, titles, footer, login, colors, logos).
