############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_routes.py: Admin DLP configuration and alerts routes
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Admin DLP (Data Loss Prevention) routes for MindRouter."""

import json
import re
from typing import Optional
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_client_ip, get_session_user_id, _admin_masquerade_context, templates
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

dlp_router = APIRouter(tags=["dlp"])

VALID_SEVERITIES = ("minor", "moderate", "major")
VALID_SCANNERS = ("regex", "gliner", "llm")

# Caps on admin-supplied config.  Every one of these bounds work the DLP worker
# performs per scanned request, so an unbounded value is a self-inflicted DoS.
MAX_PATTERNS = 100
MAX_PATTERN_CHARS = 500
MAX_KEYWORDS = 500
MAX_CATEGORIES = 60
MAX_SEVERITY_RULES = 200
MAX_PROMPT_CHARS = 20_000
MAX_RECIPIENTS = 25

_EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[A-Za-z]{2,}$")


def _err(message: str) -> RedirectResponse:
    """Redirect back to the DLP page with a static, URL-encoded error.

    Never interpolate exception text: a SQLAlchemy error stringifies its
    statement and bound parameters, which is how secrets reach browser history
    and access logs.
    """
    return RedirectResponse(f"/admin/dlp?error={quote_plus(message)}", status_code=302)


async def _require_admin_read(request: Request, db: AsyncSession):
    """Helper to require admin or auditor access (read-only admin)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.has_admin_read:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


async def _require_admin(request: Request, db: AsyncSession):
    """Helper to require full admin access (mutating actions)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.is_admin:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


@dlp_router.get("/admin/dlp", response_class=HTMLResponse)
async def admin_dlp_page(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    severity: Optional[str] = None,
    scanner: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin DLP configuration and alerts page."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    # FastAPI binds a present-but-empty query param ("?severity=") to "", not
    # None.  The filter form always submits all three controls, so treating ""
    # as a filter matched `severity = ''` and returned zero rows on every use.
    severity = (severity or "").strip() or None
    scanner = (scanner or "").strip() or None
    search = (search or "").strip() or None
    if severity not in VALID_SEVERITIES:
        severity = None
    if scanner not in VALID_SCANNERS:
        scanner = None
    page = max(1, page)

    # Load DLP config
    config = {
        "enabled": await crud.get_config_json(db, "dlp.enabled", False),
        "regex_enabled": await crud.get_config_json(db, "dlp.regex.enabled", True),
        "regex_patterns": await crud.get_config_json(db, "dlp.regex.patterns", []),
        "regex_keywords": await crud.get_config_json(db, "dlp.regex.keywords", []),
        "gliner_enabled": await crud.get_config_json(db, "dlp.gliner.enabled", False),
        "gliner_threshold": await crud.get_config_json(db, "dlp.gliner.threshold", 0.5),
        "gliner_categories": await crud.get_config_json(db, "dlp.gliner.categories", []),
        "llm_enabled": await crud.get_config_json(db, "dlp.llm.enabled", False),
        "llm_model": await crud.get_config_json(db, "dlp.llm.model", ""),
        "llm_system_prompt": await crud.get_config_json(db, "dlp.llm.system_prompt", ""),
        "severity_rules": await crud.get_config_json(db, "dlp.severity_rules", {}),
        "email_minor": await crud.get_config_json(db, "dlp.email.minor_recipients", ""),
        "email_moderate": await crud.get_config_json(db, "dlp.email.moderate_recipients", ""),
        "email_major": await crud.get_config_json(db, "dlp.email.major_recipients", ""),
        "dedup_enabled": await crud.get_config_json(db, "dlp.dedup.enabled", True),
        "dedup_window_seconds": await crud.get_config_json(db, "dlp.dedup.window_seconds", 300),
    }

    # Load alerts with pagination
    per_page = 25
    skip = (page - 1) * per_page
    alerts, total = await crud.get_dlp_alerts(
        db, severity=severity, scanner=scanner, search=search,
        skip=skip, limit=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Load stats
    stats = await crud.get_dlp_stats(db)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/dlp.html",
        {
            "request": request,
            "user": user,
            **masq,
            "config": config,
            "alerts": alerts,
            "stats": stats,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "severity_filter": severity,
            "scanner_filter": scanner,
            "search": search or "",
            "success": success,
            "error": error,
            "active": "dlp",
        },
    )


@dlp_router.post("/admin/dlp/config")
async def save_dlp_config(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Save DLP configuration (requires admin)."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return _err("Unauthorized")

    form = await request.form()

    # ---- Validate everything BEFORE the first write -------------------
    # A half-applied config is worse than a rejected one: the previous code
    # silently skipped malformed fields and still reported success, so an
    # admin could believe a pattern set was saved when it had been discarded.

    llm_enabled = form.get("llm_enabled") == "on"

    # Alert de-duplication window (seconds). Bounded so a typo can't set an
    # absurd retention-length suppression window; 0 disables via the toggle.
    dedup_enabled = form.get("dedup_enabled") == "on"
    try:
        dedup_window = int(float(form.get("dedup_window_seconds", "300")))
    except (TypeError, ValueError):
        return _err("De-duplication window must be a whole number of seconds")
    if not (0 <= dedup_window <= 86400):
        return _err("De-duplication window must be between 0 and 86400 seconds")

    # GLiNER confidence threshold
    try:
        threshold = float(form.get("gliner_threshold", "0.5"))
    except (TypeError, ValueError):
        return _err("GLiNER threshold must be a number between 0.1 and 0.95")
    if not (0.1 <= threshold <= 0.95):  # also rejects nan (all comparisons False)
        return _err("GLiNER threshold must be between 0.1 and 0.95")

    # GLiNER categories — normalized, deduped, and saved even when empty so an
    # admin can clear the list (the old `if categories:` guard made that
    # impossible, silently keeping the previous list).
    categories = sorted({
        c.strip().lower()[:100] for c in form.getlist("gliner_categories") if c and c.strip()
    })
    if len(categories) > MAX_CATEGORIES:
        return _err(f"Too many GLiNER categories (max {MAX_CATEGORIES})")

    # LLM scanner
    llm_model = (form.get("llm_model") or "").strip()[:200]
    if llm_enabled and not llm_model:
        return _err("Enter a model name to enable the LLM scanner")
    llm_prompt = (form.get("llm_system_prompt") or "").strip()
    if len(llm_prompt) > MAX_PROMPT_CHARS:
        return _err(f"System prompt is too long (max {MAX_PROMPT_CHARS} characters)")
    if llm_enabled and not llm_prompt:
        return _err("The LLM scanner needs a system prompt telling the model what to return")

    # severity_rules and regex_patterns are serialized by page JavaScript from
    # the rendered rows.  If that script did not run, the browser posts the
    # empty defaults — writing those would silently wipe the admin's rules and
    # patterns while reporting success, so treat them as absent instead.
    json_fields_authoritative = form.get("_json_ready") == "1"

    # Severity rules — must be a flat {category: level} map
    try:
        severity_rules = json.loads(form.get("severity_rules") or "{}")
    except (ValueError, TypeError, RecursionError):
        return _err("Severity rules were not valid JSON — check the form and retry")
    if not isinstance(severity_rules, dict):
        return _err("Severity rules must be a category-to-level mapping")
    if len(severity_rules) > MAX_SEVERITY_RULES:
        return _err(f"Too many severity rules (max {MAX_SEVERITY_RULES})")
    clean_rules = {}
    for cat, level in severity_rules.items():
        if not isinstance(cat, str) or not isinstance(level, str):
            return _err("Severity rules must map category names to level names")
        if level not in VALID_SEVERITIES:
            return _err(f"Unknown severity level '{level[:20]}' — use minor, moderate, or major")
        clean_rules[cat.strip().lower()[:100]] = level

    # Custom regex patterns — compile every one now, so a broken pattern is
    # rejected here instead of silently never matching at scan time.
    try:
        regex_patterns = json.loads(form.get("regex_patterns") or "[]")
    except (ValueError, TypeError, RecursionError):
        return _err("Custom patterns were not valid JSON — check the form and retry")
    if not isinstance(regex_patterns, list):
        return _err("Custom patterns must be a list")
    if len(regex_patterns) > MAX_PATTERNS:
        return _err(f"Too many custom patterns (max {MAX_PATTERNS})")
    clean_patterns = []
    for entry in regex_patterns:
        if not isinstance(entry, dict):
            return _err("Each custom pattern must have a name and a pattern")
        name = str(entry.get("name") or "").strip()[:100]
        pattern = str(entry.get("pattern") or "").strip()
        if not name or not pattern:
            return _err("Each custom pattern needs both a name and a pattern")
        if len(pattern) > MAX_PATTERN_CHARS:
            return _err(f"Pattern '{name}' is too long (max {MAX_PATTERN_CHARS} characters)")
        try:
            re.compile(pattern)
        except (re.error, OverflowError, RecursionError, ValueError):
            # re.compile raises OverflowError on an oversized repeat count
            # (\d{9999999999}) and RecursionError on deep nesting — neither is
            # an re.error, and either would escape as a bare 500.
            return _err(f"Pattern '{name}' is not a valid regular expression")
        level = entry.get("severity")
        clean_patterns.append({
            "name": name,
            "pattern": pattern,
            "category": str(entry.get("category") or name).strip().lower()[:100],
            "severity": level if level in VALID_SEVERITIES else "moderate",
        })

    # Keywords — one per line.  Single characters match on nearly every request
    # and would bury real findings, so require two.
    keywords = []
    for raw in (form.get("regex_keywords") or "").split("\n"):
        kw = raw.strip()
        if not kw:
            continue
        if len(kw) < 2:
            return _err(f"Keyword '{kw}' is too short — use at least 2 characters")
        keywords.append(kw[:200])
    if len(keywords) > MAX_KEYWORDS:
        return _err(f"Too many keywords (max {MAX_KEYWORDS})")

    # Email recipients, per severity
    recipients = {}
    for field, label in (("email_minor", "minor"), ("email_moderate", "moderate"), ("email_major", "major")):
        raw = (form.get(field) or "").replace("\r", " ").replace("\n", " ")
        addrs = [a.strip() for a in raw.split(",") if a.strip()]
        if len(addrs) > MAX_RECIPIENTS:
            return _err(f"Too many {label} recipients (max {MAX_RECIPIENTS})")
        for a in addrs:
            if not _EMAIL_RE.match(a):
                return _err(f"'{a[:40]}' is not a valid email address")
        recipients[label] = ", ".join(addrs)

    # ---- Everything validated; apply ---------------------------------
    try:
        await crud.set_config(db, "dlp.enabled", form.get("enabled") == "on")
        await crud.set_config(db, "dlp.dedup.enabled", dedup_enabled)
        await crud.set_config(db, "dlp.dedup.window_seconds", dedup_window)
        await crud.set_config(db, "dlp.regex.enabled", form.get("regex_enabled") == "on")
        await crud.set_config(db, "dlp.gliner.enabled", form.get("gliner_enabled") == "on")
        await crud.set_config(db, "dlp.llm.enabled", llm_enabled)
        await crud.set_config(db, "dlp.gliner.threshold", threshold)
        await crud.set_config(db, "dlp.gliner.categories", categories)
        await crud.set_config(db, "dlp.llm.model", llm_model)
        await crud.set_config(db, "dlp.llm.system_prompt", llm_prompt)
        if json_fields_authoritative:
            await crud.set_config(db, "dlp.severity_rules", clean_rules)
            await crud.set_config(db, "dlp.regex.patterns", clean_patterns)
        else:
            logger.warning(
                "dlp_config_json_fields_skipped",
                user_id=user.id,
                reason="page script did not serialize severity_rules/regex_patterns",
            )
        await crud.set_config(db, "dlp.regex.keywords", keywords)
        await crud.set_config(db, "dlp.email.minor_recipients", recipients["minor"])
        await crud.set_config(db, "dlp.email.moderate_recipients", recipients["moderate"])
        await crud.set_config(db, "dlp.email.major_recipients", recipients["major"])

        await crud.log_admin_action(
            db, user.id, "update", "dlp_config",
            detail="DLP configuration updated",
            ip_address=get_client_ip(request),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("dlp_config_save_failed", user_id=user.id)
        return _err("Could not save the DLP configuration — see the server log")

    return RedirectResponse("/admin/dlp?success=DLP+configuration+saved", status_code=302)


@dlp_router.post("/admin/dlp/acknowledge/{alert_id}")
async def acknowledge_alert(
    request: Request,
    alert_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Acknowledge a DLP alert (requires admin)."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    alert = await crud.acknowledge_dlp_alert(db, alert_id, user.id)
    if alert is None:
        return JSONResponse({"error": "Alert not found"}, status_code=404)

    await crud.log_admin_action(
        db, user.id, "acknowledge", "dlp_alert",
        entity_id=str(alert_id),
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return JSONResponse({"ok": True, "alert_id": alert_id})


@dlp_router.get("/admin/dlp/stats")
async def dlp_stats(
    request: Request,
    hours: int = 24,
    db: AsyncSession = Depends(get_async_db),
):
    """JSON stats endpoint for DLP dashboard cards."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    stats = await crud.get_dlp_stats(db, hours=hours)
    return JSONResponse(stats)
