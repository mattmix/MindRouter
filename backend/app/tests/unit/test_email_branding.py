############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_email_branding.py: Branding applied to outgoing emails
#
# Research Computing and Data Services (RCDS)
# University of Idaho
#
############################################################

"""Emails must follow the shared branding config: no hardcoded blue, the brand
accent (contrast-adjusted for legibility), the org name, and the raster email
logo embedded via CID. ``aiosmtplib`` is stubbed so the module imports without
the optional SMTP dependency installed."""

import sys
import types

import pytest

# Stub aiosmtplib (optional dep) before importing the email service.
if "aiosmtplib" not in sys.modules:
    _stub = types.ModuleType("aiosmtplib")
    _stub.SMTP = object
    _stub.SMTPException = Exception
    sys.modules["aiosmtplib"] = _stub

from backend.app.services import branding  # noqa: E402
from backend.app.services import email_service as es  # noqa: E402

GOLD = "#F1B300"


@pytest.fixture
def gold_brand_with_logo(tmp_path, monkeypatch):
    """U of I-style gold branding with a raster email logo configured."""
    monkeypatch.setattr(branding, "storage_dir", lambda: str(tmp_path))
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    stored = branding.save_asset("email_logo", "logo.png", png)
    branding._CACHE = branding._build_view(
        {"app_name": "MindRouter", "primary_light": GOLD, "primary_dark": GOLD, "email_logo": stored}
    )
    return branding.get_branding()


def test_wrapper_has_no_bsu_blue(gold_brand_with_logo):
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "003da5" not in html.lower(), "the old blue #003DA5 must be gone"


def test_wrapper_uses_gold_accent_and_cid_logo(gold_brand_with_logo):
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "border-bottom:3px solid #f1b300" in html      # gold accent rule
    assert "cid:brandlogo" in html                        # logo referenced for CID embed
    assert "color:#906900" in html                        # footer link uses accessible gold ink


def test_wrapper_falls_back_to_app_name_without_logo(monkeypatch):
    branding._CACHE = branding._build_view({"app_name": "Acme University", "primary_light": GOLD, "primary_dark": GOLD})
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "cid:brandlogo" not in html
    assert ">Acme University<" in html


def test_content_with_braces_is_not_reformatted(gold_brand_with_logo):
    # Regression: the old _EMAIL_WRAPPER.format() would crash on code containing braces.
    body = '<pre>{"json": true}</pre>'
    html = es._wrap_html(body, base_url="https://x")
    assert '{"json": true}' in html


def test_blog_email_title_and_button_colors(gold_brand_with_logo):
    html = es._render_blog_email("Post", "# Hi\n\ntext", "post", "Luke", "https://x")
    assert "003da5" not in html.lower()
    assert "color:#906900" in html                        # heading uses ink
    assert "background:#f1b300;color:#000000" in html      # gold button, black (legible) text


def test_default_branding_has_no_blue():
    branding._CACHE = branding._build_view({})
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "003da5" not in html.lower()


@pytest.mark.asyncio
async def test_send_one_embeds_cid_logo(gold_brand_with_logo):
    captured = {}

    class FakeSMTP:
        async def send_message(self, msg):
            captured["msg"] = msg

    html = es._wrap_html("<p>hi</p>", base_url="https://x")  # references cid:brandlogo
    await es._send_one(FakeSMTP(), "from@x", "to@y", "Subj", html)
    msg = captured["msg"]
    assert msg.get_content_type() == "multipart/related"
    ctypes = [p.get_content_type() for p in msg.walk()]
    assert "multipart/alternative" in ctypes and "image/png" in ctypes
    img = next(p for p in msg.walk() if p.get_content_type() == "image/png")
    assert img.get("Content-ID") == "<brandlogo>"
    assert "inline" in img.get("Content-Disposition", "")


@pytest.mark.asyncio
async def test_send_one_without_logo_is_plain_alternative(monkeypatch):
    branding._CACHE = branding._build_view({"app_name": "MindRouter"})  # no email logo
    captured = {}

    class FakeSMTP:
        async def send_message(self, msg):
            captured["msg"] = msg

    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    await es._send_one(FakeSMTP(), "a@x", "b@y", "S", html)
    msg = captured["msg"]
    assert msg.get_content_type() == "multipart/alternative"
    assert not any(p.get_content_type().startswith("image/") for p in msg.walk())
