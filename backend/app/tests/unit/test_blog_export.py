"""Unit tests for blog_export (institution-neutral syndication helpers).

The old push-model exporter (mindrouter.ai page shell, navbar, BLOG_CSS,
GitHub filesets) was removed in 2.8.44 in favor of pull-model syndication
feeds; this module now carries only markdown rendering, description
derivation, image-reference collection, and the RSS feed renderer.

Loaded via importlib to bypass the backend.app package __init__ import chain
(see project memory: dashboard/db package inits pull in the DB stack).
"""

import importlib.util
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from xml.dom import minidom

_MODPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "blog_export.py")
)
_spec = importlib.util.spec_from_file_location("blog_export", _MODPATH)
be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(be)


def make_post(**kw):
    """A BlogPost stand-in with just the attributes the feed touches."""
    defaults = dict(
        id=1,
        slug="hello-world",
        title="Hello World",
        content="# Hi\n\nThis is **bold** and a [link](https://example.com).",
        excerpt=None,
        published_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# --- markdown ---------------------------------------------------------------
def test_render_markdown_codehilite_and_tables():
    html = be.render_markdown("```python\nprint('x')\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert 'class="codehilite"' in html
    assert "<table>" in html


def test_render_markdown_handles_none():
    assert be.render_markdown(None) == ""


# --- image collection --------------------------------------------------------
def test_collect_image_paths_from_markdown_and_html_dedup_and_order():
    content = (
        "![a](/blog/images/2026/07/18/ab/one.png) text "
        '<img src="/blog/images/2026/07/18/cd/two.jpg"> '
        "![again](/blog/images/2026/07/18/ab/one.png)"
    )
    assert be.collect_image_paths(content) == [
        "2026/07/18/ab/one.png",
        "2026/07/18/cd/two.jpg",
    ]


def test_collect_image_paths_empty():
    assert be.collect_image_paths("") == []
    assert be.collect_image_paths(None) == []


# --- description --------------------------------------------------------------
def test_derive_description_prefers_excerpt():
    assert be.derive_description("  The excerpt.  ", "# ignored") == "The excerpt."


def test_derive_description_strips_markdown_and_truncates():
    md = "# Title\n\n![img](/blog/images/x.png) Some **bold** [linked](u) text. " + "word " * 60
    desc = be.derive_description(None, md, limit=80)
    assert "![" not in desc and "#" not in desc and "**" not in desc
    assert len(desc) <= 80
    assert desc.endswith("…")


# --- RSS feed -----------------------------------------------------------------
def test_render_feed_xml_valid_items_link_to_app_blog():
    posts = [make_post(slug="a-post", title="A & B"), make_post(slug="second", title="Second")]
    xml = be.render_feed_xml(posts, "https://gateway.example.edu/", site_name="Acme AI")
    minidom.parseString(xml)  # raises if malformed
    assert xml.count("<item>") == 2
    # Items link to THIS installation's blog (pull model), base slash-trimmed.
    assert "<link>https://gateway.example.edu/blog/a-post</link>" in xml
    assert "<title>A &amp; B</title>" in xml               # escaped
    assert "<title>Acme AI Blog</title>" in xml            # brandable channel title
    assert "<pubDate>" in xml


def test_render_feed_xml_empty_feed_is_valid():
    xml = be.render_feed_xml([], "https://gateway.example.edu")
    minidom.parseString(xml)
    assert "<item>" not in xml
    assert "</rss>" in xml


def test_module_is_institution_neutral():
    """No hardcoded external-site URL or push-era symbols survive."""
    src = open(_MODPATH).read()
    assert "https://mindrouter.ai" not in src
    for gone in ("SITE_BASE_URL", "BLOG_CSS", "render_post_html", "export_post",
                 "render_index_html", "post_repo_path", "post_canonical",
                 "fetch_images", "_navbar", "_page"):
        assert not hasattr(be, gone), gone
