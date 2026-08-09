############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# apps_routes.py: Admin registered-application management
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Admin UI for registered applications.

A registered app holds a credential that can create MindRouter accounts and
mint keys for them. That is real privilege, so the operator surface has to make
three things visible and reversible: which apps exist, what credential each one
holds, and how to take it away.

DISABLING AN APP REVOKES ITS KEYS. `apps.status` alone only stops NEW sessions;
every key the app already minted keeps working until it lapses, which for a
30-day TTL means a month of access after the operator believed they had cut it
off. The revoke is the point of the button.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dashboard.routes import (
    _admin_masquerade_context,
    get_client_ip,
    get_session_user_id,
    templates,
)
from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger
from backend.app.security.api_keys import generate_api_key
from backend.app.security.scopes import APP_CREDENTIAL_SCOPES, format_scopes

logger = get_logger(__name__)

apps_admin_router = APIRouter(tags=["apps-admin"])

# The slug appears in the provisioning URL, so keep it to what is unambiguous
# in a path segment.
#
# Anchored with \Z, not $: in Python `$` also matches immediately before a
# trailing newline, so "<valid>\n" would pass a `$`-anchored check.
_SLUG_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{0,62}[a-z0-9]\Z")

# Both Entra ids are interpolated into the JWKS URL and into the pinned issuer
# string. Anything but a GUID there is either a typo or an attempt to steer key
# retrieval elsewhere, so it is validated rather than trusted.
_GUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)

MIN_KEY_TTL_DAYS = 1
MAX_KEY_TTL_DAYS = 365
DEFAULT_CREDENTIAL_TTL_DAYS = 365
MAX_NAME_CHARS = 200
MAX_DESCRIPTION_CHARS = 2000


def _err(message: str) -> RedirectResponse:
    """Redirect back to the apps page with a static, URL-encoded error.

    Never interpolate exception text: a SQLAlchemy error stringifies its
    statement and bound parameters, which is how secrets reach browser history
    and access logs.
    """
    return RedirectResponse(f"/admin/apps?error={quote_plus(message)}", status_code=302)


def _ok(message: str) -> RedirectResponse:
    return RedirectResponse(
        f"/admin/apps?success={quote_plus(message)}", status_code=302
    )


async def _require_admin_read(request: Request, db: AsyncSession):
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.is_active or not user.group or not user.group.has_admin_read:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


async def _require_admin(request: Request, db: AsyncSession):
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.is_active or not user.group or not user.group.is_admin:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


def _clean_ttl(raw: Optional[str], default: int) -> Optional[int]:
    """Parse a day count, returning None when it is not usable."""
    try:
        value = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return None
    if not (MIN_KEY_TTL_DAYS <= value <= MAX_KEY_TTL_DAYS):
        return None
    return value


@apps_admin_router.get("/admin/apps", response_class=HTMLResponse)
async def admin_apps_page(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """List registered apps with their credential and usage state."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    apps = await crud.get_apps(db)
    stats = await crud.get_app_key_stats(db)
    credentials = await crud.get_app_provision_keys(db)
    unclassified, unclassified_total = await crud.get_unclassified_users(db, limit=50)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/apps.html",
        {
            "request": request,
            "user": user,
            **masq,
            "apps": apps,
            "stats": stats,
            "credentials": credentials,
            "unclassified": unclassified,
            "unclassified_total": unclassified_total,
            "now_utc": datetime.now(timezone.utc),
            "default_credential_ttl": DEFAULT_CREDENTIAL_TTL_DAYS,
            "success": success,
            "error": error,
            "active": "apps",
        },
    )


@apps_admin_router.post("/admin/apps/create")
async def admin_create_app(
    request: Request,
    slug: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    entra_client_id: str = Form(""),
    entra_tenant_id: str = Form(""),
    key_ttl_days: str = Form("30"),
    db: AsyncSession = Depends(get_async_db),
):
    """Register an application. No credential is minted here — that is a
    separate, separately audited action."""
    admin, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    slug = (slug or "").strip().lower()
    name = (name or "").strip()
    description = (description or "").strip()
    entra_client_id = (entra_client_id or "").strip()
    entra_tenant_id = (entra_tenant_id or "").strip()

    if not _SLUG_RE.match(slug):
        return _err("Slug must be lowercase letters, digits and hyphens")
    if not name or len(name) > MAX_NAME_CHARS:
        return _err(f"Name is required and must be under {MAX_NAME_CHARS} characters")
    if len(description) > MAX_DESCRIPTION_CHARS:
        return _err("Description is too long")
    if entra_client_id and not _GUID_RE.match(entra_client_id):
        return _err("Entra client ID must be a GUID")
    if entra_tenant_id and not _GUID_RE.match(entra_tenant_id):
        return _err("Entra tenant ID must be a GUID")

    ttl = _clean_ttl(key_ttl_days, 30)
    if ttl is None:
        return _err(f"Key lifetime must be {MIN_KEY_TTL_DAYS}-{MAX_KEY_TTL_DAYS} days")

    if await crud.get_app_by_slug(db, slug) is not None:
        return _err("An application with that slug already exists")

    app = await crud.create_app(
        db,
        slug=slug,
        name=name,
        description=description or None,
        entra_client_id=entra_client_id or None,
        entra_tenant_id=entra_tenant_id or None,
        key_ttl_days=ttl,
        created_by=admin.id,
    )
    await crud.log_admin_action(
        db, user_id=admin.id, action="app.create", entity_type="app",
        entity_id=str(app.id),
        after_value={"slug": slug, "name": name, "key_ttl_days": ttl},
        ip_address=get_client_ip(request),
    )
    await db.commit()
    return _ok(f"Registered {slug}")


@apps_admin_router.post("/admin/apps/{app_id}/update")
async def admin_update_app(
    request: Request,
    app_id: int,
    name: str = Form(...),
    description: str = Form(""),
    entra_client_id: str = Form(""),
    entra_tenant_id: str = Form(""),
    key_ttl_days: str = Form("30"),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit an app's registration details."""
    admin, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    app = await crud.get_app_by_id(db, app_id)
    if app is None:
        return _err("Application not found")

    name = (name or "").strip()
    description = (description or "").strip()
    entra_client_id = (entra_client_id or "").strip()
    entra_tenant_id = (entra_tenant_id or "").strip()

    if not name or len(name) > MAX_NAME_CHARS:
        return _err(f"Name is required and must be under {MAX_NAME_CHARS} characters")
    if len(description) > MAX_DESCRIPTION_CHARS:
        return _err("Description is too long")
    if entra_client_id and not _GUID_RE.match(entra_client_id):
        return _err("Entra client ID must be a GUID")
    if entra_tenant_id and not _GUID_RE.match(entra_tenant_id):
        return _err("Entra tenant ID must be a GUID")

    ttl = _clean_ttl(key_ttl_days, app.key_ttl_days)
    if ttl is None:
        return _err(f"Key lifetime must be {MIN_KEY_TTL_DAYS}-{MAX_KEY_TTL_DAYS} days")

    before = {
        "name": app.name,
        "entra_client_id": app.entra_client_id,
        "entra_tenant_id": app.entra_tenant_id,
        "key_ttl_days": app.key_ttl_days,
    }
    app.name = name
    app.description = description or None
    app.entra_client_id = entra_client_id or None
    app.entra_tenant_id = entra_tenant_id or None
    app.key_ttl_days = ttl

    await crud.log_admin_action(
        db, user_id=admin.id, action="app.update", entity_type="app",
        entity_id=str(app.id), before_value=before,
        after_value={
            "name": name, "entra_client_id": entra_client_id or None,
            "entra_tenant_id": entra_tenant_id or None, "key_ttl_days": ttl,
        },
        ip_address=get_client_ip(request),
    )
    await db.commit()
    return _ok(f"Updated {app.slug}")


@apps_admin_router.post("/admin/apps/{app_id}/credential", response_class=HTMLResponse)
async def admin_issue_app_credential(
    request: Request,
    app_id: int,
    ttl_days: str = Form(str(DEFAULT_CREDENTIAL_TTL_DAYS)),
    db: AsyncSession = Depends(get_async_db),
):
    """Mint the app's provisioning credential, revoking any previous one.

    Rotation is revoke-then-issue and both happen in one transaction: an app
    holding two live provisioning credentials means a leaked one stays useful
    after the operator believes they replaced it.

    The credential carries ONLY `apps:provision`. It is owned by the admin who
    minted it because api_keys.user_id is NOT NULL, but that ownership confers
    nothing: the scope list intersects with group privilege and never extends
    it, so this key cannot act as an admin even though its owner is one.
    """
    admin, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    app = await crud.get_app_by_id(db, app_id)
    if app is None:
        return _err("Application not found")
    if not app.entra_client_id or not app.entra_tenant_id:
        return _err("Set the Entra client and tenant IDs before issuing a credential")

    ttl = _clean_ttl(ttl_days, DEFAULT_CREDENTIAL_TTL_DAYS)
    if ttl is None:
        return _err(f"Credential lifetime must be {MIN_KEY_TTL_DAYS}-{MAX_KEY_TTL_DAYS} days")

    revoked = await crud.revoke_app_provision_keys(db, app.id)

    expires_at = datetime.now(timezone.utc) + timedelta(days=ttl)
    full_key, key_hash, key_prefix, key_sha256 = generate_api_key()
    key_row = await crud.create_api_key(
        db,
        user_id=admin.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=f"{app.slug} provisioning",
        expires_at=expires_at,
        key_sha256=key_sha256,
    )
    key_row.app_id = app.id
    key_row.scopes = format_scopes(APP_CREDENTIAL_SCOPES)
    # Hidden from the owning admin's own dashboard — it is managed here, not
    # there. Deliberately NOT flagged is_service: that flag drives the service
    # key promotion and revocation-request workflow, which does not apply to a
    # credential an operator issues and rotates directly.
    key_row.hidden = True
    await db.flush()

    await crud.log_admin_action(
        db, user_id=admin.id, action="app.credential_issue", entity_type="app",
        entity_id=str(app.id),
        detail=f"key_id={key_row.id} revoked_previous={revoked} ttl_days={ttl}",
        ip_address=get_client_ip(request),
    )
    await db.commit()

    logger.info(
        "app_credential_issued",
        app_id=app.id, key_id=key_row.id, admin_id=admin.id, revoked_previous=revoked,
    )

    # Rendered directly rather than redirected: the plaintext must never enter
    # a URL, and this is the only time it exists.
    return templates.TemplateResponse(
        "admin/app_credential.html",
        {
            "request": request,
            "user": admin,
            "app": app,
            "api_key": full_key,
            "expires_at": expires_at,
            "revoked_previous": revoked,
        },
    )


@apps_admin_router.post("/admin/apps/{app_id}/status")
async def admin_set_app_status(
    request: Request,
    app_id: int,
    status_value: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Enable or disable an app.

    Disabling revokes every key the app holds — its own provisioning credential
    and every per-user key it minted. Without that, `status` would only stop
    new sessions while existing keys kept working for their full TTL.
    """
    admin, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    if status_value not in ("active", "disabled"):
        return _err("Unknown status")

    app = await crud.get_app_by_id(db, app_id)
    if app is None:
        return _err("Application not found")

    before = app.status
    app.status = status_value
    revoked = 0
    if status_value != "active":
        revoked = await crud.revoke_app_keys(db, app.id)

    await crud.log_admin_action(
        db, user_id=admin.id,
        action="app.enable" if status_value == "active" else "app.disable",
        entity_type="app", entity_id=str(app.id),
        before_value={"status": before}, after_value={"status": status_value},
        detail=f"revoked_keys={revoked}",
        ip_address=get_client_ip(request),
    )
    await db.commit()

    if status_value == "active":
        return _ok(f"Enabled {app.slug} — issue a new credential to restore access")
    return _ok(f"Disabled {app.slug} and revoked {revoked} key(s)")


@apps_admin_router.post("/admin/apps/{app_id}/revoke-user-keys")
async def admin_revoke_app_user_keys(
    request: Request,
    app_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Revoke the per-user keys an app minted, leaving the app enabled.

    Users re-provision transparently on their next sign-in, so this is the
    blunt instrument for "something about that app's sessions looks wrong"
    without taking the integration down.
    """
    admin, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    app = await crud.get_app_by_id(db, app_id)
    if app is None:
        return _err("Application not found")

    revoked = await crud.revoke_app_user_keys(db, app.id)
    await crud.log_admin_action(
        db, user_id=admin.id, action="app.revoke_user_keys", entity_type="app",
        entity_id=str(app.id), detail=f"revoked_keys={revoked}",
        ip_address=get_client_ip(request),
    )
    await db.commit()
    return _ok(f"Revoked {revoked} session key(s) for {app.slug}")


@apps_admin_router.post("/admin/apps/{app_id}/delete")
async def admin_delete_app(
    request: Request,
    app_id: int,
    confirm_slug: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Deregister an app after typing its slug to confirm.

    Keys are revoked and detached rather than deleted: they carry request
    history and revoking them is what actually ends access, while deleting the
    rows would orphan the telemetry that explains what the app did.
    """
    admin, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    app = await crud.get_app_by_id(db, app_id)
    if app is None:
        return _err("Application not found")
    if (confirm_slug or "").strip() != app.slug:
        return _err("Type the application slug exactly to confirm deletion")

    slug = app.slug
    revoked = await crud.revoke_app_keys(db, app.id)
    detached = await crud.detach_app_keys(db, app.id)
    await crud.log_admin_action(
        db, user_id=admin.id, action="app.delete", entity_type="app",
        entity_id=str(app.id),
        before_value={"slug": slug, "name": app.name},
        detail=f"revoked_keys={revoked} detached_keys={detached}",
        ip_address=get_client_ip(request),
    )
    await crud.delete_app(db, app.id)
    await db.commit()
    return _ok(f"Deregistered {slug} and revoked {revoked} key(s)")
