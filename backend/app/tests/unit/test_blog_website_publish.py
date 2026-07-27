"""Contract tests for blog syndication (pull model since 2.8.44).

Source-inspection tests (no DB): verify the migration, model, routes, and CRUD
that manage the ``website_published`` (syndication) state stay consistent, and
that the push-era GitHub publisher stays deleted (external sites PULL from
/blog/feed.xml and /api/blog/syndicated instead). This mirrors the
migration/source-check style used by the responses-store tests and avoids the
backend.app package import chain that pulls in the DB stack.
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _read(rel):
    with open(os.path.join(ROOT, rel), "r") as f:
        return f.read()


MIGRATION = "backend/app/db/migrations/versions/20260718_000000_064_add_blog_website_publish.py"
MODELS = "backend/app/db/models.py"
CRUD = "backend/app/db/crud.py"
BLOG = "backend/app/dashboard/blog.py"

_COLS = ["website_published", "website_published_at", "website_commit_sha"]


def test_migration_adds_and_drops_the_three_columns():
    src = _read(MIGRATION)
    assert 'revision = "064"' in src and 'down_revision = "063"' in src
    for col in _COLS:
        assert f'add_column(\n        "blog_posts",' in src  # sanity: targets blog_posts
        assert f'"{col}"' in src, f"migration missing add for {col}"
        assert f'op.drop_column("blog_posts", "{col}")' in src, f"downgrade missing drop for {col}"
    # not-null boolean with a server default so existing rows backfill cleanly
    assert "website_published" in src and 'server_default=sa.text("0")' in src
    assert "nullable=False" in src  # website_published


def test_model_has_website_fields():
    src = _read(MODELS)
    block = src[src.index("class BlogPost("):]
    block = block[: block.index("class ", 10)] if "class " in block[10:] else block
    for col in _COLS:
        assert re.search(rf"\b{col}\b\s*:\s*Mapped", block), f"BlogPost missing {col}"
    assert "website_commit_sha" in block and "String(64)" in block


def test_crud_query_filters_selected_published_undeleted():
    src = _read(CRUD)
    fn = src[src.index("async def get_website_published_blog_posts"):]
    fn = fn[: fn.index("\nasync def ")]
    assert "BlogPost.website_published.is_(True)" in fn
    assert "BlogPost.is_published.is_(True)" in fn      # never leak drafts
    assert "BlogPost.deleted_at.is_(None)" in fn         # never leak deleted
    assert "order_by(BlogPost.published_at.desc())" in fn


def test_routes_exist_with_guard_and_state_transitions():
    src = _read(BLOG)
    assert '"/admin/blog/{post_id}/website-publish"' in src
    assert '"/admin/blog/{post_id}/website-unpublish"' in src

    pub = src[src.index("async def admin_blog_website_publish"):]
    pub = pub[: pub.index("\nasync def ")]
    assert "_require_admin(" in pub                       # full admin, mutating
    assert "if not post.is_published:" in pub             # gate: no draft leaks
    assert "website_published=True" in pub
    assert "website_published_at=datetime.now(timezone.utc)" in pub

    unpub = src[src.index("async def admin_blog_website_unpublish"):]
    unpub = unpub[: unpub.index("\nasync def ")]
    assert "website_published=False" in unpub
    assert "website_published_at=None" in unpub
    assert "website_commit_sha=None" in unpub


# --- pull model (2.8.44): push machinery deleted, feeds exposed --------------

def test_push_publisher_is_deleted():
    """The gateway must hold no external-site publisher or write credential."""
    assert not os.path.exists(
        os.path.join(ROOT, "backend/app/services/website_publisher.py")
    ), "website_publisher.py must stay deleted (pull model)"
    src = _read(BLOG)
    assert "website_publisher" not in src
    assert "get_website_publisher" not in src


def test_settings_have_no_push_credentials():
    src = _read("backend/app/settings.py")
    for field in ("website_publish_enabled", "website_publish_repo",
                  "website_publish_branch", "website_publish_github_token"):
        assert field not in src, f"push-era setting {field} must stay removed"


def test_syndication_feed_routes_exist_and_are_public():
    src = _read(BLOG)
    assert '"/blog/feed.xml"' in src
    assert '"/api/blog/syndicated"' in src
    for route_name in ("blog_feed_xml", "blog_syndicated_json"):
        fn = src[src.index(f"async def {route_name}"):]
        fn = fn[: fn.index("\nasync def ") if "\nasync def " in fn else len(fn)]
        # Public: no admin/session guard — feeds expose only syndicated posts.
        assert "_require_admin" not in fn and "get_session_user_id" not in fn
        # Both feeds are built from the selection-filtered CRUD query.
        assert "get_website_published_blog_posts" in fn


def test_flag_routes_no_longer_push():
    """publish/unpublish flip the flag only — no publisher calls, no pushes."""
    src = _read(BLOG)
    for name in ("admin_blog_website_publish", "admin_blog_website_unpublish"):
        fn = src[src.index(f"async def {name}"):]
        fn = fn[: fn.index("\nasync def ")]
        assert "publisher" not in fn, f"{name} must not push"


def test_blog_export_is_institution_neutral():
    src = _read("backend/app/dashboard/blog_export.py")
    assert "https://mindrouter.ai" not in src
    assert "SITE_BASE_URL" not in src
