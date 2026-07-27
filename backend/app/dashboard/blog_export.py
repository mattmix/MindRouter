############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# blog_export.py: Institution-neutral blog export helpers —
#                 markdown rendering, description derivation,
#                 image-reference collection, and the public
#                 RSS syndication feed.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Blog export helpers for external syndication.

MindRouter's blog can syndicate selected posts to an external site. The old
model PUSHED rendered pages into a site-specific GitHub repo from inside the
gateway (site chrome, canonical URLs, and a write credential all lived here).
That was institution-specific and was removed in favor of a PULL model:

* the gateway exposes read-only feeds — ``GET /blog/feed.xml`` (RSS 2.0) and
  ``GET /api/blog/syndicated`` (JSON with raw markdown + an image manifest) —
  built from the posts an admin has flagged for syndication;
* the external site's own build pulls those feeds, renders pages with ITS OWN
  templates, and rehosts the images. Presentation lives with the site.

This module keeps only the neutral pieces those feeds need.
"""

import html
import re
from datetime import datetime
from typing import Any, List, Optional

import markdown

# Matches blog image references in raw Markdown or rendered HTML:
#   ![alt](/blog/images/2026/07/18/ab/hash_uuid.png)
#   <img src="/blog/images/2026/07/18/ab/hash_uuid.png">
# Capture group 1 is the ArtifactStorage-relative path (after /blog/images/).
_IMG_REF_RE = re.compile(r"/blog/images/([^\s\"')]+)")

# Strip HTML tags / markdown image+link syntax for deriving a plain-text
# description from post content when no excerpt is set.
_TAG_RE = re.compile(r"<[^>]+>")
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def render_markdown(text: str) -> str:
    """Render post Markdown to HTML (mirrors dashboard.blog._render_markdown)."""
    return markdown.markdown(
        text or "",
        extensions=["fenced_code", "codehilite", "tables", "toc"],
        extension_configs={
            "codehilite": {"css_class": "codehilite", "guess_lang": False},
        },
    )


def _esc(value: Any) -> str:
    """HTML-escape a text value (attributes and body text)."""
    return html.escape("" if value is None else str(value), quote=True)


def derive_description(excerpt: Optional[str], content_md: str, limit: int = 160) -> str:
    """Plain-text description for meta/OG tags: excerpt, else start of content."""
    text = excerpt.strip() if excerpt and excerpt.strip() else (content_md or "")
    if not (excerpt and excerpt.strip()):
        text = _MD_IMG_RE.sub("", text)                    # ![alt](url)
        text = _MD_LINK_RE.sub(r"\1", text)                # [text](url) -> text
        text = _TAG_RE.sub("", text)                       # <tags>
        text = re.sub(r"`+", "", text)                     # `code`
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)  # # ATX headings
        text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)       # > blockquotes
        text = re.sub(r"\*{1,3}", "", text)                # *em* **strong**
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------
def collect_image_paths(content: str) -> List[str]:
    """Return the unique ArtifactStorage paths referenced by a post, in order."""
    seen: List[str] = []
    for match in _IMG_REF_RE.finditer(content or ""):
        path = match.group(1)
        if path not in seen:
            seen.append(path)
    return seen


# ---------------------------------------------------------------------------
# RSS syndication feed
# ---------------------------------------------------------------------------
def render_feed_xml(posts, base_url: str, site_name: str = "MindRouter") -> str:
    """Render an RSS 2.0 feed of the syndicated posts (newest first).

    ``base_url`` is this installation's public base URL (config ``app.base_url``);
    items link to the app's own blog pages, which are the canonical source in
    the pull model.
    """
    base = (base_url or "").rstrip("/")

    def rfc822(dt: Optional[datetime]) -> str:
        return dt.strftime("%a, %d %b %Y %H:%M:%S %z") if dt else ""

    entries = []
    for post in posts:
        link = f"{base}/blog/{post.slug}"
        desc = derive_description(getattr(post, "excerpt", None), getattr(post, "content", ""), limit=300)
        pub = rfc822(getattr(post, "published_at", None))
        entries.append(
            "    <item>\n"
            f"      <title>{_esc(post.title)}</title>\n"
            f"      <link>{_esc(link)}</link>\n"
            f"      <guid isPermaLink=\"true\">{_esc(link)}</guid>\n"
            + (f"      <pubDate>{_esc(pub)}</pubDate>\n" if pub else "")
            + f"      <description>{_esc(desc)}</description>\n"
            "    </item>"
        )
    body = "\n".join(entries)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        "  <channel>\n"
        f"    <title>{_esc(site_name)} Blog</title>\n"
        f"    <link>{_esc(base)}/blog</link>\n"
        f"    <description>Posts from the {_esc(site_name)} blog.</description>\n"
        f"{body}\n"
        "  </channel>\n"
        "</rss>\n"
    )
