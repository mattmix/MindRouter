############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_branding.py: Unit tests for the UI branding service
#
# Research Computing and Data Services (RCDS)
# University of Idaho
#
############################################################

"""Unit tests for backend.app.services.branding.

Covers the pure color/normalization logic, the template-ready view builder,
and on-disk asset save/resolve/delete (traversal-safe). No live DB is used;
the DB-backed loaders are exercised via the render tests elsewhere.
"""

import pytest

from backend.app.services import branding


# ── Color helpers ────────────────────────────────────────────────

def test_is_valid_hex():
    assert branding.is_valid_hex("#0d6efd")
    assert branding.is_valid_hex("#abc")
    assert not branding.is_valid_hex("0d6efd")       # missing #
    assert not branding.is_valid_hex("#12345")       # wrong length
    assert not branding.is_valid_hex("#gggggg")      # non-hex
    assert not branding.is_valid_hex(None)
    assert not branding.is_valid_hex(123)


def test_normalize_hex_expands_and_falls_back():
    assert branding._normalize_hex("#ABC", "#000000") == "#aabbcc"
    assert branding._normalize_hex("#0D6EFD", "#000000") == "#0d6efd"
    assert branding._normalize_hex("nonsense", "#0d6efd") == "#0d6efd"
    assert branding._normalize_hex(None, "#123456") == "#123456"


def test_hex_to_rgb():
    assert branding._hex_to_rgb("#000000") == (0, 0, 0)
    assert branding._hex_to_rgb("#ffffff") == (255, 255, 255)
    assert branding._hex_to_rgb("#0d6efd") == (13, 110, 253)


def test_shade_darken_and_lighten():
    # Darkening black stays black; lightening white stays white (clamped).
    assert branding._shade("#000000", -0.5) == "#000000"
    assert branding._shade("#ffffff", 0.5) == "#ffffff"
    # Darkening reduces channels.
    darker = branding._shade("#808080", -0.5)
    assert darker == "#404040"


# ── View builder ─────────────────────────────────────────────────

def test_build_view_defaults():
    v = branding._build_view({})
    assert v["app_name"] == branding.DEFAULTS["app_name"]
    assert v["tagline"] == branding.DEFAULTS["tagline"]
    assert v["primary_light"] == "#0d6efd"
    assert v["primary_light_rgb"] == "13, 110, 253"
    assert v["logo_light_url"] is None
    assert v["logo_dark_url"] is None
    assert v["favicon_url"] is None
    assert v["has_custom_logo"] is False
    assert v["is_customized"] is False
    # Derived shades present for both themes.
    for k in ("primary_light_hover", "primary_dark_hover",
              "primary_light_active", "primary_dark_active"):
        assert branding.is_valid_hex(v[k]), k


def test_build_view_custom_and_asset_urls():
    v = branding._build_view({
        "app_name": "Acme University",
        "tagline": "Powered by Acme",
        "primary_light": "#8B1E3F",
        "primary_dark": "#e0729a",
        "logo_light": "logo_light-abc123.png",
        "logo_dark": "logo_dark-def456.svg",
        "favicon": "favicon-999.ico",
    })
    assert v["app_name"] == "Acme University"
    assert v["primary_light"] == "#8b1e3f"           # normalized to lowercase
    assert v["logo_light_url"] == "/branding/asset/logo_light-abc123.png"
    assert v["logo_dark_url"] == "/branding/asset/logo_dark-def456.svg"
    assert v["favicon_url"] == "/branding/asset/favicon-999.ico"
    assert v["has_custom_logo"] is True
    assert v["is_customized"] is True


def test_build_view_invalid_color_falls_back_to_default():
    v = branding._build_view({"primary_light": "red", "primary_dark": "#zzz"})
    assert v["primary_light"] == branding.DEFAULTS["primary_light"]
    assert v["primary_dark"] == branding.DEFAULTS["primary_dark"]


def test_headline_color_defaults_neutral_and_is_configurable():
    # Default = neutral near-black (admin-configurable knob).
    assert branding.DEFAULTS["headline_color"] == "#231f20"
    assert branding._build_view({})["headline_color"] == "#231f20"
    # Custom value is normalized and marks the brand customized.
    v = branding._build_view({"headline_color": "#008080"})
    assert v["headline_color"] == "#008080"
    assert v["is_customized"] is True
    # Invalid falls back to the neutral default.
    assert branding._build_view({"headline_color": "teal"})["headline_color"] == "#231f20"


def test_get_branding_never_empty():
    branding._CACHE = {}
    v = branding.get_branding()
    assert v["app_name"] == branding.DEFAULTS["app_name"]


# ── Asset storage (filesystem) ───────────────────────────────────

@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(branding, "storage_dir", lambda: str(tmp_path))
    return tmp_path


def test_save_asset_writes_file_with_slot_prefix(tmp_storage):
    stored = branding.save_asset("logo_light", "mylogo.PNG", b"\x89PNG\r\n\x1a\n")
    assert stored.startswith("logo_light-")
    assert stored.endswith(".png")            # extension lower-cased
    assert (tmp_storage / stored).read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_save_asset_rejects_bad_extension(tmp_storage):
    with pytest.raises(ValueError):
        branding.save_asset("logo_light", "evil.exe", b"data")


def test_save_asset_favicon_allows_ico(tmp_storage):
    stored = branding.save_asset("favicon", "fav.ico", b"icodata")
    assert stored.endswith(".ico")


def test_save_asset_favicon_rejects_webp(tmp_storage):
    # webp is allowed for logos but not favicons.
    with pytest.raises(ValueError):
        branding.save_asset("favicon", "fav.webp", b"data")


def test_asset_path_resolves_and_blocks_traversal(tmp_storage):
    stored = branding.save_asset("logo_dark", "l.svg", b"<svg/>")
    assert branding.asset_path(stored) is not None
    # Traversal / absolute / hidden names are rejected.
    assert branding.asset_path("../../etc/passwd") is None
    assert branding.asset_path("/etc/passwd") is None
    assert branding.asset_path(".hidden") is None
    assert branding.asset_path("does-not-exist.png") is None


def test_delete_asset_removes_file(tmp_storage):
    stored = branding.save_asset("logo_light", "l.png", b"x")
    assert branding.asset_path(stored) is not None
    branding.delete_asset(stored)
    assert branding.asset_path(stored) is None
    # Deleting a missing / None asset is a no-op (no exception).
    branding.delete_asset(None)
    branding.delete_asset("nope.png")


def test_content_type_for():
    assert branding.content_type_for("x.png") == "image/png"
    assert branding.content_type_for("x.svg") == "image/svg+xml"
    assert branding.content_type_for("x.ico") == "image/x-icon"
    assert branding.content_type_for("x.unknown") == "application/octet-stream"


# ── Email logo slot (raster-only, CID reads) ─────────────────────

def test_build_view_exposes_email_logo():
    v = branding._build_view({"email_logo": "email_logo-abc.png"})
    assert v["email_logo_file"] == "email_logo-abc.png"
    assert v["email_logo_url"] == "/branding/asset/email_logo-abc.png"
    assert v["is_customized"] is True
    # Absent by default.
    assert branding._build_view({})["email_logo_url"] is None


def test_email_logo_rejects_svg_and_webp(tmp_storage):
    # Email clients can't render SVG/WebP — the slot must reject them.
    for bad in ("logo.svg", "logo.webp"):
        with pytest.raises(ValueError):
            branding.save_asset("email_logo", bad, b"data")
    # PNG/JPG/GIF are accepted.
    for good in ("logo.png", "logo.jpg", "logo.gif"):
        stored = branding.save_asset("email_logo", good, b"data")
        assert stored.startswith("email_logo-")


def test_read_email_logo_returns_bytes_and_subtype(tmp_storage):
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 32
    stored = branding.save_asset("email_logo", "logo.png", png)
    branding._CACHE = branding._build_view({"email_logo": stored})
    result = branding.read_email_logo()
    assert result is not None
    data, subtype = result
    assert data == png and subtype == "png"
    # Unset → None.
    branding._CACHE = branding._build_view({})
    assert branding.read_email_logo() is None


# ── Accessible foreground / ink derivation ───────────────────────

def test_contrast_bounds():
    assert round(branding._contrast("#000000", "#ffffff"), 1) == 21.0
    assert branding._contrast("#123456", "#123456") == 1.0


def test_best_fg_prefers_white_but_rescues_light_accents():
    # Conventional mid-tone accents keep white text (Bootstrap convention).
    assert branding._best_fg("#0d6efd") == "#ffffff"   # blue
    assert branding._best_fg("#dc3545") == "#ffffff"   # red
    # Light accents flip to black so text stays legible.
    assert branding._best_fg("#f1b300") == "#000000"   # U of I Pride Gold
    assert branding._best_fg("#ffc107") == "#000000"   # warning yellow
    # The chosen fg always clears the 3.0 UI-text threshold on its fill.
    for accent in ("#0d6efd", "#f1b300", "#ffc107", "#dc3545", "#008080"):
        assert branding._contrast(branding._best_fg(accent), accent) >= 3.0, accent


def test_accessible_ink_meets_target_on_body_bg():
    # A light accent on a white page is darkened until it reads as text.
    ink_light = branding._accessible_ink("#f1b300", "#ffffff")
    assert branding._contrast(ink_light, "#ffffff") >= 4.5
    assert ink_light != "#f1b300"                       # was too light, got darkened
    # The same gold on a dark page already passes, so it's left as-is.
    ink_dark = branding._accessible_ink("#f1b300", "#1a1d21")
    assert ink_dark == "#f1b300"
    assert branding._contrast(ink_dark, "#1a1d21") >= 4.5


def test_build_view_exposes_on_and_ink_for_both_themes():
    v = branding._build_view({"primary_light": "#F1B300", "primary_dark": "#F1B300"})
    for k in ("primary_light_on", "primary_dark_on",
              "primary_light_ink", "primary_dark_ink"):
        assert branding.is_valid_hex(v[k]), k
    # Gold button text is black and high-contrast; light-mode links are darkened.
    assert v["primary_light_on"] == "#000000"
    assert branding._contrast(v["primary_light_ink"], "#ffffff") >= 4.5


def test_default_look_unchanged():
    """Unbranded install keeps stock blue with white button text."""
    d = branding._build_view({})
    assert d["primary_light_on"] == "#ffffff"
    assert d["primary_light_ink"] == "#0d6efd"


def test_every_dashboard_template_env_registers_branding_global():
    """base.html calls branding() on line 1, so EVERY Jinja2Templates env that
    renders it must register the global — otherwise authenticated pages (chat,
    images, video) 500 with "'branding' is undefined". Regression guard for the
    bug where chat/images/video had their own env without the global.
    """
    import pathlib

    dash = pathlib.Path(__file__).resolve().parents[2] / "dashboard"
    offenders = []
    for py in sorted(dash.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        if "Jinja2Templates(" in src and 'globals["branding"]' not in src:
            offenders.append(py.name)
    assert not offenders, (
        "These create a Jinja2Templates env but never register the branding() "
        f"global that base.html requires: {offenders}"
    )


# ── Organization name (SSO wording) ──────────────────────────────

def test_build_view_org_name_default_none():
    v = branding._build_view({})
    assert v["org_name"] is None
    assert v["is_customized"] is False


def test_build_view_org_name_set():
    v = branding._build_view({"org_name": "University of Idaho"})
    assert v["org_name"] == "University of Idaho"
    assert v["is_customized"] is True


def test_build_view_org_name_whitespace_is_none():
    assert branding._build_view({"org_name": "   "})["org_name"] is None


def test_org_name_config_key_defined():
    assert branding.KEY_ORG_NAME == "branding.org_name"
    assert branding.DEFAULTS["org_name"] is None


# ── Login page must be org-agnostic (branding-driven) ────────────

def _login_template_src():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    return (root / "dashboard" / "templates" / "public" / "login.html").read_text()


def test_login_template_has_no_hardcoded_institution():
    src = _login_template_src()
    assert "University of Idaho" not in src
    assert "uidaho" not in src.lower()


def test_login_template_uses_branding_org_name():
    src = _login_template_src()
    # Multi-provider loop: labels are resolved server-side (azure gets
    # brand.org_name in sso/registry.py); helper text stays branding-driven.
    assert "Sign in with {{ p.label }}" in src
    assert "{% for p in sso_providers %}" in src
    assert "{{ brand.org_name or 'organization' }} credentials" in src

    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    registry_src = (root / "dashboard" / "sso" / "registry.py").read_text()
    assert 'label=org_name or "SSO"' in registry_src


def test_login_template_local_account_toggle():
    src = _login_template_src()
    assert "Sign in with a local account" in src
    assert "Admin login" not in src


def test_login_route_error_is_org_agnostic():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "dashboard" / "routes.py").read_text()
    assert "Please use University of Idaho sign-in" not in src
    assert 'single sign-on (SSO)' in src


def test_branding_form_has_org_name_field():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "dashboard" / "templates" / "admin" / "branding.html").read_text()
    assert 'name="org_name"' in src
    # save route persists it
    routes_src = (root / "dashboard" / "routes.py").read_text()
    assert "KEY_ORG_NAME" in routes_src


def test_reset_all_clears_org_name():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "dashboard" / "routes.py").read_text()
    reset_branch = src.split('action == "reset_all"')[1].split("return await _finish")[0]
    assert "KEY_ORG_NAME" in reset_branch


# ── Footer attribution (configurable "Powered by …") ─────────────

def test_footer_note_config_keys_and_defaults():
    assert branding.KEY_FOOTER_NOTE == "branding.footer_note"
    assert branding.KEY_FOOTER_NOTE_URL == "branding.footer_note_url"
    # Ships with the reference-deployment credit so existing installs are
    # unchanged by the field becoming configurable.
    assert branding.DEFAULTS["footer_note"] == "Powered by RCDS"
    assert branding.DEFAULTS["footer_note_url"] == "https://hpc.uidaho.edu"


def test_build_view_footer_note_default_is_rcds():
    v = branding._build_view({})
    assert v["footer_note"] == "Powered by RCDS"
    assert v["footer_note_url"] == "https://hpc.uidaho.edu"
    assert v["is_customized"] is False


def test_build_view_footer_note_override():
    v = branding._build_view({
        "footer_note": "Hosted by Acme Research Computing",
        "footer_note_url": "https://rc.acme.edu",
    })
    assert v["footer_note"] == "Hosted by Acme Research Computing"
    assert v["footer_note_url"] == "https://rc.acme.edu"
    assert v["is_customized"] is True


def test_build_view_footer_note_empty_string_removes_line():
    """An explicitly-saved empty value hides the line (NOT replaced by default)."""
    v = branding._build_view({"footer_note": "", "footer_note_url": ""})
    assert v["footer_note"] == ""
    assert v["footer_note_url"] is None
    assert v["is_customized"] is True


def test_build_view_footer_note_text_without_link():
    v = branding._build_view({"footer_note": "Operated by IT Services", "footer_note_url": ""})
    assert v["footer_note"] == "Operated by IT Services"
    assert v["footer_note_url"] is None


def test_safe_link_rejects_dangerous_schemes():
    assert branding.safe_link("https://example.edu") == "https://example.edu"
    assert branding.safe_link("http://example.edu/x") == "http://example.edu/x"
    assert branding.safe_link("/about") == "/about"
    assert branding.safe_link("javascript:alert(1)") is None
    assert branding.safe_link("data:text/html,<script>") is None
    assert branding.safe_link("example.edu") is None
    assert branding.safe_link("") is None
    assert branding.safe_link(None) is None


def test_build_view_footer_note_url_drops_unsafe_value():
    v = branding._build_view({"footer_note": "Credit", "footer_note_url": "javascript:alert(1)"})
    assert v["footer_note_url"] is None


def test_base_template_footer_note_is_branding_driven():
    """base.html is the parent of the login/status/dashboard pages: the footer
    attribution must come from branding, not a hardcoded institution link."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "dashboard" / "templates" / "base.html").read_text()
    assert "hpc.uidaho.edu" not in src
    assert "{% if brand.footer_note %}" in src
    assert "{{ brand.footer_note }}" in src
    assert "{{ brand.footer_note_url }}" in src
    # The NSF award credit is product attribution for the grant — stays put.
    assert "NSF Award #2427549" in src


def test_branding_form_and_routes_handle_footer_note():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    form_src = (root / "dashboard" / "templates" / "admin" / "branding.html").read_text()
    assert 'name="footer_note"' in form_src
    assert 'name="footer_note_url"' in form_src
    routes_src = (root / "dashboard" / "routes.py").read_text()
    save_branch = routes_src.split('action == "save_identity"')[1].split("elif action in")[0]
    assert "KEY_FOOTER_NOTE" in save_branch
    assert "KEY_FOOTER_NOTE_URL" in save_branch


def test_reset_all_restores_footer_note_default():
    """Reset must restore the shipped default, not persist "" (= hidden)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "dashboard" / "routes.py").read_text()
    reset_branch = src.split('action == "reset_all"')[1].split("return await _finish")[0]
    assert "KEY_FOOTER_NOTE" in reset_branch
    assert "KEY_FOOTER_NOTE_URL" in reset_branch
    assert "key, None" in reset_branch  # JSON null → _build_view falls back to DEFAULTS
