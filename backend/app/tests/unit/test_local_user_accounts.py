"""Tests for admin-created local user accounts and account-type badges/filtering.

Covers:
- User.account_type property classification (Admin > SSO > Local)
- POST /admin/users/create dashboard route contract (source inspection)
- crud.get_users account_type filter contract (source inspection)
- Badge partial + template wiring (compile + content checks)

models.py is spec-loaded directly (with backend.app.db.base pre-loaded the same
way) to avoid the backend.app.db package import chain — see MEMORY.md
"Import Chain Gotcha".
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DB_DIR = Path(__file__).resolve().parents[2] / "db"
_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"


def _load_models():
    """Spec-load base.py then models.py, bypassing backend.app.db.__init__.

    A FRESH base module is always installed (and the original restored after),
    so a real backend.app.db.models imported earlier in the session can't
    collide on Base.metadata table names, and vice versa.
    """
    saved = {}
    for name in ["backend", "backend.app", "backend.app.db"]:
        if name not in sys.modules:
            saved[name] = None
            sys.modules[name] = MagicMock()

    base_spec = importlib.util.spec_from_file_location(
        "backend.app.db.base", _DB_DIR / "base.py"
    )
    base_mod = importlib.util.module_from_spec(base_spec)
    saved.setdefault("backend.app.db.base", sys.modules.get("backend.app.db.base"))
    sys.modules["backend.app.db.base"] = base_mod
    base_spec.loader.exec_module(base_mod)

    models_spec = importlib.util.spec_from_file_location(
        "mr2_models_under_test", _DB_DIR / "models.py"
    )
    models_mod = importlib.util.module_from_spec(models_spec)
    models_spec.loader.exec_module(models_mod)
    return models_mod, saved


@pytest.fixture(scope="module")
def models():
    mod, saved = _load_models()
    yield mod
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ---------------------------------------------------------------------------
# account_type property
# ---------------------------------------------------------------------------

def _make_user(models, group=None, azure_oid=None):
    user = models.User(username="u", email="u@example.com", azure_oid=azure_oid)
    user.group = group
    return user


def test_account_type_admin_group(models):
    group = models.Group(name="admins", display_name="Admins", is_admin=True)
    assert _make_user(models, group=group).account_type == "admin"


def test_account_type_admin_wins_over_sso(models):
    group = models.Group(name="admins", display_name="Admins", is_admin=True)
    user = _make_user(models, group=group, azure_oid="11111111-1111-1111-1111-111111111111")
    assert user.account_type == "admin"


def test_account_type_sso_with_nonadmin_group(models):
    group = models.Group(name="students", display_name="Students", is_admin=False)
    user = _make_user(models, group=group, azure_oid="22222222-2222-2222-2222-222222222222")
    assert user.account_type == "sso"


def test_account_type_sso_without_group(models):
    user = _make_user(models, azure_oid="33333333-3333-3333-3333-333333333333")
    assert user.account_type == "sso"


def test_account_type_local_with_nonadmin_group(models):
    group = models.Group(name="students", display_name="Students", is_admin=False)
    assert _make_user(models, group=group).account_type == "local"


def test_account_type_local_without_group(models):
    assert _make_user(models).account_type == "local"


# ---------------------------------------------------------------------------
# Dashboard create route contract (source inspection — routes.py is DB-heavy)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routes_src():
    return (_DASHBOARD_DIR / "routes.py").read_text()


def test_create_route_registered(routes_src):
    assert '@dashboard_router.post("/admin/users/create")' in routes_src


def test_create_route_requires_full_admin(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert "admin_user.group.is_admin" in create_src


def test_create_route_hashes_password(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert "password_hash=hash_password(password)" in create_src


def test_create_route_checks_duplicates(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert "get_user_by_username" in create_src
    assert "get_user_by_email" in create_src


def test_create_route_creates_quota_from_group(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert "create_quota" in create_src
    assert "rpm_limit=group.rpm_limit" in create_src


def test_create_route_audit_logged(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert 'action="user.create"' in create_src


def test_create_route_enforces_password_length(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert "len(password) < 8" in create_src


def test_create_route_enforces_field_lengths(routes_src):
    create_src = routes_src.split('"/admin/users/create"')[1]
    assert "len(username) > 100 or len(email) > 255" in create_src


def test_create_route_never_reflects_exception_text(routes_src):
    """DB exception text embeds INSERT parameters (incl. the Argon2 hash) —
    it must never reach the redirect URL or error banner."""
    create_src = routes_src.split('"/admin/users/create"')[1].split("@dashboard_router")[0]
    assert "_reject(str(e))" not in create_src
    assert "except IntegrityError" in create_src
    assert create_src.count("await db.rollback()") >= 2
    assert "logger.exception" in create_src


def test_admin_users_route_passes_account_type(routes_src):
    users_src = routes_src.split('async def admin_users(')[1].split("@dashboard_router")[0]
    assert "account_type=account_type" in users_src
    assert '"admin", "sso", "local"' in users_src


# ---------------------------------------------------------------------------
# crud.get_users account_type filter contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def crud_src():
    return (_DB_DIR / "crud.py").read_text()


def test_get_users_has_account_type_param(crud_src):
    sig = crud_src.split("async def get_users(")[1].split(") ->")[0]
    assert "account_type: Optional[str] = None" in sig


def test_get_users_filter_conditions(crud_src):
    body = crud_src.split("async def get_users(")[1].split("async def ")[0]
    assert "Group.is_admin.is_(True)" in body
    assert "User.azure_oid.isnot(None)" in body
    assert "User.azure_oid.is_(None)" in body
    # Users with no group must still count as non-admin for sso/local filters
    assert "User.group_id.is_(None)" in body


def test_api_keys_eager_load_group(crud_src):
    body = crud_src.split("async def get_all_api_keys(")[1].split("\nasync def ")[0]
    assert "selectinload(ApiKey.user).selectinload(User.group)" in body


def test_pending_quota_requests_eager_load_user_group(crud_src):
    body = crud_src.split("async def get_pending_quota_requests(")[1].split("\nasync def ")[0]
    assert "selectinload(QuotaRequest.user).selectinload(User.group)" in body


def test_top_active_users_include_account_type(crud_src):
    body = crud_src.split("async def get_top_active_users(")[1].split("\nasync def ")[0]
    assert '"account_type": user.account_type' in body


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

BADGE_TEMPLATES = [
    "admin/_user_badge.html",
    "admin/users.html",
    "admin/user_detail.html",
    "admin/api_keys.html",
    "admin/requests.html",
    "admin/dashboard.html",
]


def _template_env():
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True
    )
    env.filters["localtime"] = lambda v, fmt=None: v
    env.filters["fromjson"] = lambda s: []
    return env


@pytest.mark.parametrize("template", BADGE_TEMPLATES)
def test_templates_compile(template):
    _template_env().get_template(template)


def test_badge_partial_covers_all_types():
    src = (_TEMPLATES_DIR / "admin/_user_badge.html").read_text()
    assert "badge_user.account_type == 'admin'" in src
    assert "badge_user.account_type == 'sso'" in src
    assert "> Admin</span>" in src
    assert "> SSO</span>" in src
    assert "> Local</span>" in src


def test_users_page_has_account_type_filter():
    src = (_TEMPLATES_DIR / "admin/users.html").read_text()
    assert 'name="account_type"' in src
    for opt in ('value="admin"', 'value="sso"', 'value="local"'):
        assert opt in src


def test_users_page_pagination_preserves_account_type():
    src = (_TEMPLATES_DIR / "admin/users.html").read_text()
    assert src.count("&account_type={{ account_type }}") >= 3


def test_users_page_has_create_form():
    src = (_TEMPLATES_DIR / "admin/users.html").read_text()
    assert 'action="/admin/users/create"' in src
    assert 'minlength="8"' in src
    assert 'name="group_id"' in src and "required" in src
    # hidden from read-only auditors
    assert "{% if not is_read_only %}" in src


def test_users_page_shows_badge():
    src = (_TEMPLATES_DIR / "admin/users.html").read_text()
    assert 'include "admin/_user_badge.html"' in src


@pytest.mark.parametrize(
    "template",
    ["admin/user_detail.html", "admin/api_keys.html", "admin/requests.html"],
)
def test_badge_included(template):
    src = (_TEMPLATES_DIR / template).read_text()
    assert 'include "admin/_user_badge.html"' in src


def test_dashboard_top_users_badge():
    src = (_TEMPLATES_DIR / "admin/dashboard.html").read_text()
    assert "u.account_type" in src
