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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_client_ip, get_session_user_id, _admin_masquerade_context, templates
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

dlp_router = APIRouter(tags=["dlp"])

VALID_SEVERITIES = ("minor", "moderate", "major")
# Rule levels an admin may assign to a category.  "ignore" is a rule level
# only: findings in an ignored category are dropped before any alert exists,
# so no alert row ever carries it and the alert filter keeps VALID_SEVERITIES.
VALID_RULE_LEVELS = VALID_SEVERITIES + ("ignore",)
VALID_SCANNERS = ("regex", "gliner", "llm")
# Alert export: formats and a hard row cap so one click can't stream the whole
# table into memory. The bundle is a zip regardless of the inner format.
EXPORT_FORMATS = ("json", "jsonl", "csv")
EXPORT_MAX_ROWS = 100_000

# Caps on admin-supplied config.  Every one of these bounds work the DLP worker
# performs per scanned request, so an unbounded value is a self-inflicted DoS.
MAX_PATTERNS = 100
# Global scan window (dlp.max_scan_chars): default = scanner constant; 0 = no limit.
DEFAULT_MAX_SCAN_CHARS = 200_000
MIN_MAX_SCAN_CHARS = 1_000
MAX_PATTERN_CHARS = 500
MAX_KEYWORDS = 500
MAX_CATEGORIES = 60
MAX_SEVERITY_RULES = 200
MAX_PROMPT_CHARS = 20_000
MAX_RECIPIENTS = 25
MAX_REMOTE_ENDPOINTS = 16   # off-host GLiNER services in the failover pool

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



# ---------------------------------------------------------------------------
# Regex rule list (textarea) <-> stored config
# ---------------------------------------------------------------------------

RULE_SEP = " | "


def _scanner():
    """The scanner module, resolved at call time.

    importlib (not ``from ... import``) so a test that spec-loads dlp_scanner
    and registers it in sys.modules gets that instance; the dashboard package
    import chain stays untouched at module load.
    """
    import importlib
    return importlib.import_module("backend.app.services.dlp_scanner")


def builtin_patterns():
    return _scanner().builtin_patterns()


def builtin_validator_for(name: str):
    return _scanner().builtin_validator_for(name)


def format_regex_rule(p: dict) -> str:
    """One pattern dict -> one textarea line: ``Name | category | regex``."""
    return f"{p.get('name', '')}{RULE_SEP}{p.get('category', '')}{RULE_SEP}{p.get('pattern', '')}"


def render_regex_rules(patterns, keywords, *, builtins_in_list: bool) -> str:
    """Build the textarea contents.

    Until the admin saves from the new page the stored list holds only custom
    patterns and the built-ins are implicit, so they are shown first; after
    that the stored list is the whole truth.  Keywords follow as bare lines.
    """
    rows = []
    if not builtins_in_list:
        rows.extend(builtin_patterns())
    rows.extend(p for p in (patterns or []) if isinstance(p, dict) and p.get("pattern"))
    lines = [format_regex_rule(p) for p in rows]
    lines.extend(str(k) for k in (keywords or []) if str(k).strip())
    return "\n".join(lines)


def parse_regex_rules(text: str):
    """Textarea contents -> (patterns, keywords).  Raises ValueError with an
    admin-readable message on the first bad line; nothing is written then."""
    patterns, keywords = [], []
    for lineno, raw in enumerate((text or "").replace("\r", "").split("\n"), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            # keyword — single characters match on nearly every request
            if len(line) < 2:
                raise ValueError(f"Line {lineno}: keyword '{line}' is too short — use at least 2 characters")
            keywords.append(line[:200])
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Line {lineno}: a pattern line is 'Name | category | regex' "
                f"(a keyword line must not contain '|')"
            )
        name = parts[0].strip()[:100]
        category = parts[1].strip().lower()[:100]
        pattern = parts[2].strip()
        if not name or not pattern:
            raise ValueError(f"Line {lineno}: a pattern line needs both a name and a regex")
        if not category:
            category = name.lower()
        if len(pattern) > MAX_PATTERN_CHARS:
            raise ValueError(f"Line {lineno}: pattern '{name}' is too long (max {MAX_PATTERN_CHARS} characters)")
        try:
            re.compile(pattern)
        except (re.error, OverflowError, RecursionError, ValueError):
            # re.compile raises OverflowError on an oversized repeat count
            # (\d{9999999999}) and RecursionError on deep nesting — neither is
            # an re.error, and either would escape as a bare 500.
            raise ValueError(f"Line {lineno}: pattern '{name}' is not a valid regular expression")
        entry = {"name": name, "pattern": pattern, "category": category}
        validator = builtin_validator_for(name)
        if validator:
            entry["validator"] = validator
        patterns.append(entry)
    if len(patterns) > MAX_PATTERNS:
        raise ValueError(f"Too many patterns (max {MAX_PATTERNS})")
    if len(keywords) > MAX_KEYWORDS:
        raise ValueError(f"Too many keywords (max {MAX_KEYWORDS})")
    return patterns, keywords

@dlp_router.get("/admin/dlp", response_class=HTMLResponse)
async def admin_dlp_page(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    severity: Optional[str] = None,
    scanner: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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
        "gliner_max_scan_chars": await crud.get_config_json(db, "dlp.gliner.max_scan_chars", 10000),
        # Global scan window; 0 = no limit.  Default mirrors the scanner constant.
        "max_scan_chars": await crud.get_config_json(db, "dlp.max_scan_chars", DEFAULT_MAX_SCAN_CHARS),
        "remote_enabled": await crud.get_config_json(db, "dlp.gliner.remote.enabled", False),
        "remote_endpoints": "\n".join(
            await crud.get_config_json(db, "dlp.gliner.remote.endpoints", [])
            or ([await crud.get_config_json(db, "dlp.gliner.remote.url", "")]
                if await crud.get_config_json(db, "dlp.gliner.remote.url", "") else [])
        ),
        "remote_url": await crud.get_config_json(db, "dlp.gliner.remote.url", ""),
        "remote_key": await crud.get_config_json(db, "dlp.gliner.remote.key", ""),
        "remote_timeout": await crud.get_config_json(db, "dlp.gliner.remote.timeout", 10.0),
        "remote_fallback": await crud.get_config_json(db, "dlp.gliner.remote.fallback", "local"),
        "remote_verify_tls": await crud.get_config_json(db, "dlp.gliner.remote.verify_tls", True),
        "llm_enabled": await crud.get_config_json(db, "dlp.llm.enabled", False),
        "llm_model": await crud.get_config_json(db, "dlp.llm.model", ""),
        "llm_system_prompt": await crud.get_config_json(db, "dlp.llm.system_prompt", ""),
        "severity_rules": await crud.get_config_json(db, "dlp.severity_rules", {}),
        "regex_rules_text": render_regex_rules(
            await crud.get_config_json(db, "dlp.regex.patterns", []),
            await crud.get_config_json(db, "dlp.regex.keywords", []),
            builtins_in_list=bool(
                await crud.get_config_json(db, "dlp.regex.builtins_in_list", False)
            ),
        ),
        "builtin_rule_lines": [
            format_regex_rule(p) for p in builtin_patterns()
        ],
        "email_minor": await crud.get_config_json(db, "dlp.email.minor_recipients", ""),
        "email_moderate": await crud.get_config_json(db, "dlp.email.moderate_recipients", ""),
        "email_major": await crud.get_config_json(db, "dlp.email.major_recipients", ""),
        "dedup_enabled": await crud.get_config_json(db, "dlp.dedup.enabled", True),
        "dedup_window_seconds": await crud.get_config_json(db, "dlp.dedup.window_seconds", 300),
        "email_minor_mode": await crud.get_config_json(db, "dlp.email.minor.mode", "immediate"),
        "email_moderate_mode": await crud.get_config_json(db, "dlp.email.moderate.mode", "immediate"),
        "email_major_mode": await crud.get_config_json(db, "dlp.email.major.mode", "immediate"),
        "digest_frequency": await crud.get_config_json(db, "dlp.digest.frequency", "daily"),
        "digest_recipients": await crud.get_config_json(db, "dlp.digest.recipients", ""),
        # Pre-send DLP screening of web-search queries (dlp.websearch.*).
        "websearch_dlp_enabled": await crud.get_config_json(db, "dlp.websearch.enabled", False),
        "websearch_dlp_min_severity": await crud.get_config_json(db, "dlp.websearch.min_severity", "moderate"),
        "websearch_dlp_on_error": await crud.get_config_json(db, "dlp.websearch.on_scanner_error", "block"),
        "websearch_store_original": await crud.get_config_json(db, "search.audit.store_original_query", True),
        # Per-severity Detection Action (block and/or alert) + block scope.
        "block_scope": await crud.get_config_json(db, "dlp.block.scope", "prompt"),
        "block_minor": await crud.get_config_json(db, "dlp.action.minor.block", False),
        "block_moderate": await crud.get_config_json(db, "dlp.action.moderate.block", False),
        "block_major": await crud.get_config_json(db, "dlp.action.major.block", False),
        "alert_minor": await crud.get_config_json(db, "dlp.action.minor.alert", True),
        "alert_moderate": await crud.get_config_json(db, "dlp.action.moderate.alert", True),
        "alert_major": await crud.get_config_json(db, "dlp.action.major.alert", True),
        "redact_minor": await crud.get_config_json(db, "dlp.action.minor.redact", False),
        "redact_moderate": await crud.get_config_json(db, "dlp.action.moderate.redact", False),
        "redact_major": await crud.get_config_json(db, "dlp.action.major.redact", False),
        "notify_user_minor": await crud.get_config_json(db, "dlp.email.minor.notify_user", False),
        "notify_user_moderate": await crud.get_config_json(db, "dlp.email.moderate.notify_user", False),
        "notify_user_major": await crud.get_config_json(db, "dlp.email.major.notify_user", False),
    }

    # Load alerts with pagination
    per_page = 25
    skip = (page - 1) * per_page
    start_at = _parse_export_date(start_date)
    end_at = _parse_export_date(end_date, end=True)
    alerts, total = await crud.get_dlp_alerts(
        db, severity=severity, scanner=scanner, search=search,
        start_at=start_at, end_at=end_at,
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
            "start_date": (start_date or "").strip(),
            "end_date": (end_date or "").strip(),
            "success": success,
            "error": error,
            "active": "dlp",
        },
    )


def _parse_export_date(raw, *, end=False):
    """Parse a YYYY-MM-DD (or full ISO) filter bound into an aware UTC datetime.

    A bare date as the END bound becomes the START of the NEXT day so the whole
    day is included (the query uses scanned_at < end_at). Returns None on empty
    or unparseable input (the filter is simply not applied)."""
    from datetime import datetime, timedelta, timezone
    raw = (raw or "").strip()
    if not raw:
        return None
    dt = None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        return None
    if end and len(raw) == 10:  # date-only end bound -> include the full day
        dt = dt + timedelta(days=1)
    return dt.replace(tzinfo=timezone.utc)


def _alert_to_record(a) -> dict:
    """Flatten one DlpAlert (+ joined user/request) into a JSON-safe dict.

    Matched snippets are already masked at scan time, so an export carries the
    same masked metadata the console shows — never the verbatim sensitive value.
    """
    return {
        "id": a.id,
        "scanned_at": a.scanned_at.isoformat() if a.scanned_at else None,
        "severity": a.severity,
        "scanner": a.scanner,
        "categories": list(a.categories or []),
        "entities": list(a.entities or []),
        "confidence": a.confidence,
        "scan_latency_ms": a.scan_latency_ms,
        "detail": a.detail,
        "acknowledged": bool(a.acknowledged),
        "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
        "user_id": a.user_id,
        "user_email": (a.user.email if a.user else None),
        "request_id": a.request_id,
        "request_uuid": (a.request.request_uuid if a.request else None),
    }


def _render_export(records, fmt: str) -> bytes:
    """Serialize the flattened records to the requested format (bytes)."""
    if fmt == "jsonl":
        return ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else "")).encode("utf-8")
    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        cols = ["id", "scanned_at", "severity", "scanner", "categories", "entities",
                "confidence", "scan_latency_ms", "detail", "acknowledged",
                "acknowledged_at", "user_id", "user_email", "request_id", "request_uuid"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            # JSON-encode the nested list columns so a cell stays one field.
            row["categories"] = json.dumps(row.get("categories") or [], ensure_ascii=False)
            row["entities"] = json.dumps(row.get("entities") or [], ensure_ascii=False)
            w.writerow(row)
        return buf.getvalue().encode("utf-8")
    # default: pretty JSON array
    return json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")


@dlp_router.get("/admin/dlp/alerts/export")
async def export_dlp_alerts(
    request: Request,
    severity: Optional[str] = None,
    scanner: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    format: str = "json",
    db: AsyncSession = Depends(get_async_db),
):
    """Export filtered DLP alerts as a zipped bundle (admins + auditors).

    The zip contains the data file (dlp_alerts.<fmt>) plus a manifest.json
    recording the filters, row count, generator, and UTC timestamp so an
    offline analyst knows exactly what the extract represents.
    """
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    fmt = (format or "json").strip().lower()
    if fmt not in EXPORT_FORMATS:
        return _err(f"Unknown export format '{fmt[:20]}' — choose json, jsonl, or csv")

    sev = (severity or "").strip() or None
    scn = (scanner or "").strip() or None
    q = (search or "").strip() or None
    if sev not in VALID_SEVERITIES:
        sev = None
    if scn not in VALID_SCANNERS:
        scn = None
    start_at = _parse_export_date(start_date)
    end_at = _parse_export_date(end_date, end=True)

    alerts, total = await crud.get_dlp_alerts(
        db, severity=sev, scanner=scn, search=q,
        start_at=start_at, end_at=end_at,
        skip=0, limit=EXPORT_MAX_ROWS,
    )
    records = [_alert_to_record(a) for a in alerts]
    truncated = total > EXPORT_MAX_ROWS

    data_bytes = _render_export(records, fmt)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%SZ")
    manifest = {
        "generated_at": now.isoformat(),
        "generated_by": getattr(user, "email", None),
        "source": "MindRouter DLP alert export",
        "format": fmt,
        "filters": {
            "severity": sev, "scanner": scn, "search": q,
            "start_date": start_date or None, "end_date": end_date or None,
        },
        "row_count": len(records),
        "total_matched": total,
        "truncated": truncated,
        "max_rows": EXPORT_MAX_ROWS,
        "note": "Matched snippets are masked at scan time; this export never contains verbatim sensitive values.",
    }

    import io
    import zipfile
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"dlp_alerts.{fmt}", data_bytes)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    zbuf.seek(0)

    logger.info(
        "dlp_alerts_exported",
        user_id=getattr(user, "id", None),
        rows=len(records), total=total, format=fmt, truncated=truncated,
    )
    filename = f"dlp_alerts_{stamp}.zip"
    return StreamingResponse(
        zbuf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@dlp_router.get("/admin/dlp/alerts/partial", response_class=HTMLResponse)
async def dlp_alerts_partial(
    request: Request,
    severity: Optional[str] = None,
    scanner: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Render just the alerts table + pagination for the AJAX filter swap.

    Same auth and filters as the full page; returns the shared partial so the
    two never drift. Admins and auditors only."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        # A fetch() can't follow a login redirect usefully; signal with 401/403.
        code = 401 if isinstance(redirect, RedirectResponse) and "login" in str(redirect.headers.get("location", "")) else 403
        return HTMLResponse("", status_code=code)

    severity = (severity or "").strip() or None
    scanner = (scanner or "").strip() or None
    search = (search or "").strip() or None
    if severity not in VALID_SEVERITIES:
        severity = None
    if scanner not in VALID_SCANNERS:
        scanner = None
    start_at = _parse_export_date(start_date)
    end_at = _parse_export_date(end_date, end=True)

    per_page = 25
    page = max(1, page)
    skip = (page - 1) * per_page
    alerts, total = await crud.get_dlp_alerts(
        db, severity=severity, scanner=scanner, search=search,
        start_at=start_at, end_at=end_at, skip=skip, limit=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)
    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/_dlp_alerts_table.html",
        {
            "request": request,
            **masq,
            "alerts": alerts,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "severity_filter": severity,
            "scanner_filter": scanner,
            "search": search or "",
            "start_date": (start_date or "").strip(),
            "end_date": (end_date or "").strip(),
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

    # Per-severity email delivery mode (immediate / digest / off) and the
    # central digest schedule.
    email_modes = {}
    for sev in ("minor", "moderate", "major"):
        m = (form.get(f"email_{sev}_mode") or "immediate").strip()
        if m not in ("immediate", "digest", "off"):
            return _err(f"Invalid delivery mode for {sev} alerts")
        email_modes[sev] = m

    # Web-search screening. Validated against the gate module's own tuples so
    # the form can never store a level the gate would silently reject.
    from backend.app.services.search.dlp_gate import (
        VALID_MIN_SEVERITIES as WS_SEVERITIES,
        VALID_ON_ERROR as WS_ON_ERROR,
    )

    websearch_dlp_enabled = form.get("websearch_dlp_enabled") == "on"
    ws_min_sev = (form.get("websearch_dlp_min_severity") or "moderate").strip()
    if ws_min_sev not in WS_SEVERITIES:
        return _err("Web search screening severity must be minor, moderate, or major")
    ws_on_error = (form.get("websearch_dlp_on_error") or "block").strip()
    if ws_on_error not in WS_ON_ERROR:
        return _err("Web search scanner-error behavior must be block or allow")
    ws_store_original = form.get("websearch_store_original") == "on"

    # Detection Action per severity: block and/or alert, plus notify-the-user.
    block_scope = (form.get("block_scope") or "prompt").strip()
    if block_scope not in ("prompt", "response", "both"):
        return _err("Block scope must be prompt, response, or both")
    actions = {}
    for sev in ("minor", "moderate", "major"):
        actions[sev] = {
            "block": form.get(f"block_{sev}") == "on",
            "redact": form.get(f"redact_{sev}") == "on",
            "alert": form.get(f"alert_{sev}") == "on",
            "notify_user": form.get(f"notify_user_{sev}") == "on",
        }

    digest_frequency = (form.get("digest_frequency") or "daily").strip()
    if digest_frequency not in ("hourly", "6h", "12h", "daily"):
        return _err("Digest frequency must be hourly, 6h, 12h, or daily")

    raw_digest = (form.get("digest_recipients") or "").replace("\r", " ").replace("\n", " ")
    digest_addrs = [a.strip() for a in raw_digest.split(",") if a.strip()]
    if len(digest_addrs) > MAX_RECIPIENTS:
        return _err(f"Too many digest recipients (max {MAX_RECIPIENTS})")
    for a in digest_addrs:
        if not _EMAIL_RE.match(a):
            return _err(f"'{a[:40]}' is not a valid email address")
    digest_recipients = ", ".join(digest_addrs)
    # If any severity routes to the digest, the digest needs somewhere to go.
    if "digest" in email_modes.values() and not digest_recipients:
        return _err("Add digest recipients — a severity is set to Digest but no digest address is configured")

    # GLiNER confidence threshold
    try:
        threshold = float(form.get("gliner_threshold", "0.5"))
    except (TypeError, ValueError):
        return _err("GLiNER threshold must be a number between 0.1 and 0.95")
    if not (0.1 <= threshold <= 0.95):  # also rejects nan (all comparisons False)
        return _err("GLiNER threshold must be between 0.1 and 0.95")

    # Global scan window.  "No limit" posts the checkbox and we store 0, which
    # the scanner reads as "scan the entire document" — the defence against a
    # long prompt pushing sensitive text past the window.  A positive value
    # must be at least MIN_MAX_SCAN_CHARS so a typo can't blind the scanner.
    if form.get("max_scan_chars_unlimited") == "on":
        max_scan_chars = 0
    else:
        try:
            max_scan_chars = int(float(form.get("max_scan_chars", str(DEFAULT_MAX_SCAN_CHARS))))
        except (TypeError, ValueError):
            return _err("Max scan characters must be a whole number")
        if max_scan_chars == 0:
            pass  # explicit 0 typed in the box also means no limit
        elif max_scan_chars < MIN_MAX_SCAN_CHARS:
            return _err(f"Max scan characters must be 0 (no limit) or at least {MIN_MAX_SCAN_CHARS}")

    # GLiNER max scan chars — bounds CPU cost (scan time scales with length).
    try:
        gliner_max_chars = int(float(form.get("gliner_max_scan_chars", "10000")))
    except (TypeError, ValueError):
        return _err("GLiNER max scan characters must be a whole number")
    if not (500 <= gliner_max_chars <= 200000):
        return _err("GLiNER max scan characters must be between 500 and 200000")

    # GLiNER categories — normalized, deduped, and saved even when empty so an
    # admin can clear the list (the old `if categories:` guard made that
    # impossible, silently keeping the previous list).
    categories = sorted({
        c.strip().lower()[:100] for c in form.getlist("gliner_categories") if c and c.strip()
    })
    if len(categories) > MAX_CATEGORIES:
        return _err(f"Too many GLiNER categories (max {MAX_CATEGORIES})")

    # Off-host GLiNER service (optional).  The url may be https with a
    # self-signed cert on the cluster, so a verify-TLS toggle rides alongside.
    remote_enabled = form.get("remote_enabled") == "on"
    remote_verify_tls = form.get("remote_verify_tls") == "on"
    # One or more endpoint URLs (textarea, one per line or comma-separated) for
    # scale-out + failover across several GPU DLP services.
    import re as _re
    raw_eps = _re.split(r"[\s,]+", (form.get("remote_endpoints") or "").strip())
    remote_endpoints: list = []
    for ep in raw_eps:
        u = ep.strip().rstrip("/")
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            return _err("Each off-host GLiNER endpoint must start with http:// or https://")
        if u not in remote_endpoints:
            remote_endpoints.append(u)
    if len(remote_endpoints) > MAX_REMOTE_ENDPOINTS:
        return _err(f"Too many off-host GLiNER endpoints (max {MAX_REMOTE_ENDPOINTS})")
    # First endpoint doubles as the legacy single-URL value for compatibility.
    remote_url = remote_endpoints[0] if remote_endpoints else ""
    # An empty endpoint list is intentional, not an error: with off-host enabled
    # and no explicit URLs, the worker auto-discovers every registered
    # engine=dlp backend into the pool (health-authoritative). Listing URLs here
    # only overrides that auto-discovery for a fixed/pinned set.
    remote_key = (form.get("remote_key") or "").strip()
    if len(remote_key) > 200:
        return _err("Off-host GLiNER worker key is too long (max 200 characters)")
    remote_fallback = (form.get("remote_fallback") or "local").strip()
    if remote_fallback not in ("local", "skip"):
        return _err("Off-host GLiNER fallback must be 'local' or 'skip'")
    try:
        remote_timeout = float(form.get("remote_timeout", "10"))
    except (TypeError, ValueError):
        return _err("Off-host GLiNER timeout must be a number between 1 and 120 seconds")
    if not (1 <= remote_timeout <= 120):  # also rejects nan (all comparisons False)
        return _err("Off-host GLiNER timeout must be between 1 and 120 seconds")

    # LLM scanner
    llm_model = (form.get("llm_model") or "").strip()[:200]
    if llm_enabled and not llm_model:
        return _err("Enter a model name to enable the LLM scanner")
    llm_prompt = (form.get("llm_system_prompt") or "").strip()
    if len(llm_prompt) > MAX_PROMPT_CHARS:
        return _err(f"System prompt is too long (max {MAX_PROMPT_CHARS} characters)")
    if llm_enabled and not llm_prompt:
        return _err("The LLM scanner needs a system prompt telling the model what to return")

    # severity_rules is serialized by page JavaScript from the rendered rows.
    # If that script did not run, the browser posts the empty default —
    # writing it would silently wipe the admin's rules while reporting
    # success, so treat it as absent instead.  (The regex rule list is a plain
    # textarea and needs no script.)
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
        if level not in VALID_RULE_LEVELS:
            return _err(f"Unknown severity level '{level[:20]}' — use minor, moderate, major, or ignore")
        clean_rules[cat.strip().lower()[:100]] = level

    # Regex rule list — ONE textarea, one rule per line, built-ins included:
    #   Name | category | regex     -> pattern (regex is the LAST field, so it
    #                                  may itself contain "|")
    #   bare text (no " | ")        -> keyword (literal, case-insensitive)
    # The saved list is authoritative: whatever the admin left in the box is
    # exactly what the scanner runs, so a built-in can be edited or removed.
    try:
        clean_patterns, keywords = parse_regex_rules(form.get("regex_rules") or "")
    except ValueError as e:
        return _err(str(e))

    # Email recipients, per severity
    recipients = {}
    for field, label in (("email_minor", "minor"), ("email_moderate", "moderate"), ("email_major", "major")):
        raw = form.get(field) or ""
        addrs = [a.strip() for a in re.split(r"[,\r\n]+", raw) if a.strip()]
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
        await crud.set_config(db, "dlp.gliner.max_scan_chars", gliner_max_chars)
        await crud.set_config(db, "dlp.max_scan_chars", max_scan_chars)
        await crud.set_config(db, "dlp.gliner.categories", categories)
        await crud.set_config(db, "dlp.gliner.remote.enabled", remote_enabled)
        await crud.set_config(db, "dlp.gliner.remote.endpoints", remote_endpoints)
        await crud.set_config(db, "dlp.gliner.remote.url", remote_url)
        await crud.set_config(db, "dlp.gliner.remote.key", remote_key)
        await crud.set_config(db, "dlp.gliner.remote.timeout", remote_timeout)
        await crud.set_config(db, "dlp.gliner.remote.fallback", remote_fallback)
        await crud.set_config(db, "dlp.gliner.remote.verify_tls", remote_verify_tls)
        await crud.set_config(db, "dlp.llm.model", llm_model)
        await crud.set_config(db, "dlp.llm.system_prompt", llm_prompt)
        if json_fields_authoritative:
            await crud.set_config(db, "dlp.severity_rules", clean_rules)
        else:
            logger.warning(
                "dlp_config_json_fields_skipped",
                user_id=user.id,
                reason="page script did not serialize severity_rules",
            )
        await crud.set_config(db, "dlp.regex.patterns", clean_patterns)
        await crud.set_config(db, "dlp.regex.keywords", keywords)
        await crud.set_config(db, "dlp.regex.builtins_in_list", True)
        await crud.set_config(db, "dlp.email.minor_recipients", recipients["minor"])
        await crud.set_config(db, "dlp.email.moderate_recipients", recipients["moderate"])
        await crud.set_config(db, "dlp.email.major_recipients", recipients["major"])
        await crud.set_config(db, "dlp.email.minor.mode", email_modes["minor"])
        await crud.set_config(db, "dlp.email.moderate.mode", email_modes["moderate"])
        await crud.set_config(db, "dlp.email.major.mode", email_modes["major"])
        await crud.set_config(db, "dlp.digest.frequency", digest_frequency)
        await crud.set_config(db, "dlp.digest.recipients", digest_recipients)
        await crud.set_config(db, "dlp.block.scope", block_scope)
        await crud.set_config(db, "dlp.websearch.enabled", websearch_dlp_enabled)
        await crud.set_config(db, "dlp.websearch.min_severity", ws_min_sev)
        await crud.set_config(db, "dlp.websearch.on_scanner_error", ws_on_error)
        await crud.set_config(db, "search.audit.store_original_query", ws_store_original)
        for sev in ("minor", "moderate", "major"):
            await crud.set_config(db, f"dlp.action.{sev}.block", actions[sev]["block"])
            await crud.set_config(db, f"dlp.action.{sev}.redact", actions[sev]["redact"])
            await crud.set_config(db, f"dlp.action.{sev}.alert", actions[sev]["alert"])
            await crud.set_config(db, f"dlp.email.{sev}.notify_user", actions[sev]["notify_user"])

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

    # Drop the cached block gate so a Block toggle takes effect immediately
    # rather than after the enforcement TTL. Best-effort: never fail a saved
    # config over cache invalidation.
    try:
        from backend.app.services.dlp_enforcement import invalidate_gate_cache
        invalidate_gate_cache()
    except Exception:
        logger.warning("dlp_gate_cache_invalidate_skipped")

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
