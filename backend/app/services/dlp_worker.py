############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_worker.py: Background DLP scanning worker
#
# Drains an asyncio queue of request IDs, loads request
# data from DB, runs DLP scanners, and creates alerts.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Background DLP worker: async queue consumer for post-hoc scanning."""

import asyncio
import time

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Module-level queue — bounded to 10,000 to prevent unbounded memory growth
_dlp_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

# Cap on masked snippets stored per alert (see _process_one).
MAX_STORED_ENTITIES = 50

# Alert-email flood guard: at most EMAIL_BURST messages per severity per
# EMAIL_WINDOW_SECONDS, per process.  severity -> (window_start, count, suppressed)
EMAIL_BURST = 10
EMAIL_WINDOW_SECONDS = 300.0
_email_budget: dict = {}

# Alert de-duplication (admin-toggleable — dlp.dedup.enabled / .window_seconds).
# The same sensitive value re-appears on every conversation turn and across a
# client's auxiliary model calls (web-search classifier, title generation), so
# one exposure of one SSN can mint a dozen alerts+emails in seconds.  Suppress
# a repeat alert whose (user, masked value-set) matches one already raised
# within the window.  Per-process, best-effort — a flood guard, not a ledger.
DEDUP_DEFAULT_WINDOW = 300.0
_DEDUP_MAX_KEYS = 20_000
_dedup_seen: dict = {}  # key -> first-seen monotonic time

# Scanner-failure surfacing (F53): a scanner that errors must never look like
# clean traffic.  Log at ERROR every time, and raise a visible dashboard alert
# at most once per scanner per window so a permanently-broken scanner does not
# flood while an operator is still told DLP is degraded.
SCANNER_ERROR_ALERT_WINDOW = 3600.0
_scanner_error_seen: dict = {}


def _dedup_is_duplicate(key, window: float) -> bool:
    """True if ``key`` was first seen within ``window`` seconds (a duplicate).

    First (or post-expiry) sighting records the time and returns False; repeats
    inside the window return True.  One alert per value per window.
    """
    now = time.monotonic()
    if len(_dedup_seen) > _DEDUP_MAX_KEYS:
        # Bound memory: drop expired keys, then hard-clear if still oversized.
        cutoff = now - window
        for k in [k for k, t in list(_dedup_seen.items()) if t < cutoff]:
            _dedup_seen.pop(k, None)
        if len(_dedup_seen) > _DEDUP_MAX_KEYS:
            _dedup_seen.clear()
    first = _dedup_seen.get(key)
    if first is not None and (now - first) < window:
        return True
    _dedup_seen[key] = now
    return False


def get_dlp_queue() -> asyncio.Queue:
    """Return the module-level DLP queue."""
    return _dlp_queue


async def enqueue_for_dlp(request_id: int) -> None:
    """Enqueue a request ID for DLP scanning. Non-blocking, drops if full."""
    try:
        _dlp_queue.put_nowait(request_id)
    except asyncio.QueueFull:
        logger.warning("dlp_queue_full", dropped_request_id=request_id)


async def dlp_worker_loop() -> None:
    """Main DLP worker loop. Runs as a background task during app lifespan."""
    logger.info("dlp_worker_started")
    while True:
        try:
            request_id = await _dlp_queue.get()
            try:
                await _process_one(request_id)
            except Exception:
                logger.exception("dlp_process_failed", request_id=request_id)
            finally:
                _dlp_queue.task_done()
        except asyncio.CancelledError:
            logger.info("dlp_worker_cancelled")
            break
        except Exception:
            logger.exception("dlp_worker_error")
            await asyncio.sleep(1)


async def _process_one(request_id: int) -> None:
    """Process a single request for DLP scanning."""
    from backend.app.db import crud
    from backend.app.db.session import get_async_db_context
    from backend.app.services.dlp_scanner import (
        extract_scannable_text,
        run_dlp_scan,
    )

    async with get_async_db_context() as db:
        # Load DLP master toggle
        enabled = await crud.get_config_json(db, "dlp.enabled", False)
        if not enabled:
            return

        # Load the request
        from backend.app.db.models import Request, Response
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await db.execute(
            select(Request).where(Request.id == request_id)
        )
        req = result.scalar_one_or_none()
        if req is None:
            return

        # No self-loop guard is needed: the LLM scanner dispatches straight to a
        # backend (_internal_chat) and never re-enters the gateway, so a scan
        # creates no request row of its own.

        # Load response content
        resp_result = await db.execute(
            select(Response).where(Response.request_id == request_id)
        )
        resp = resp_result.scalar_one_or_none()
        response_content = resp.content if resp else None

        # Extract scannable text
        text = extract_scannable_text(
            messages=req.messages,
            prompt=req.prompt,
            response_content=response_content,
            modality=req.modality if hasattr(req, "modality") else None,
        )
        if not text:
            return

        # Build config dict from DB
        config = await _load_dlp_config(db)

        # Run the scan
        scan_result = await run_dlp_scan(text, config)
        if scan_result is None:
            return

        # F53: a scanner that errored must not masquerade as clean traffic —
        # surface it (loud log + rate-limited dashboard alert) so an operator
        # knows DLP may be MISSING sensitive data.
        if scan_result.scanner_errors:
            await _surface_scanner_errors(db, req, request_id, scan_result)
            if not scan_result.findings:
                return

        # Build entities list.  Snippets are MASKED, not merely truncated: an
        # alert row is metadata about sensitive data and must not become a
        # second, longer-lived copy of it.  Cap the count so a pathological
        # match (a 1-char keyword against a large prompt) can't write a
        # multi-megabyte JSON column.
        from backend.app.services.dlp_scanner import mask_snippet

        entities = []
        categories_set = set()
        max_confidence = 0.0
        for f in scan_result.findings:
            if len(entities) < MAX_STORED_ENTITIES:
                entities.append({
                    "scanner": f.scanner,
                    "category": f.category,
                    "text": mask_snippet(f.text),
                    "confidence": f.confidence,
                })
            categories_set.add(f.category)
            if f.confidence > max_confidence:
                max_confidence = f.confidence
        if len(scan_result.findings) > MAX_STORED_ENTITIES:
            entities.append({
                "scanner": "-",
                "category": "(truncated)",
                "text": f"+{len(scan_result.findings) - MAX_STORED_ENTITIES} more",
                "confidence": 0.0,
            })

        # De-duplicate: the same user re-sending the same masked value-set within
        # the window (conversation history, or a client's classifier/title calls)
        # collapses to a single alert+email.  Admin-toggleable; window <= 0 or the
        # toggle off restores the pre-dedup one-alert-per-request behavior.
        if config.get("dedup.enabled", True) and entities:
            window = config.get("dedup.window_seconds", DEDUP_DEFAULT_WINDOW)
            try:
                window = float(window)
            except (TypeError, ValueError):
                window = DEDUP_DEFAULT_WINDOW
            if window > 0:
                masked = tuple(sorted(e["text"] for e in entities if e.get("text")))
                if masked and _dedup_is_duplicate((req.user_id, masked), window):
                    logger.info(
                        "dlp_alert_deduplicated",
                        request_id=request_id,
                        user_id=req.user_id,
                        categories=sorted(categories_set),
                    )
                    return

        # Create DLP alert
        alert = await crud.create_dlp_alert(
            db,
            request_id=request_id,
            user_id=req.user_id,
            severity=scan_result.severity,
            scanner=scan_result.scanner,
            categories=list(categories_set),
            entities=entities,
            confidence=max_confidence,
            scan_latency_ms=scan_result.scan_latency_ms,
            detail=scan_result.detail,
        )
        await db.commit()

        logger.info(
            "dlp_alert_created",
            alert_id=alert.id,
            request_id=request_id,
            severity=scan_result.severity,
            findings=len(scan_result.findings),
            latency_ms=scan_result.scan_latency_ms,
        )

        # Send email notification if configured
        await _maybe_send_email(db, alert, scan_result)


async def _load_dlp_config(db) -> dict:
    """Load all DLP configuration from the database."""
    from backend.app.db import crud

    config = {}
    config["regex.enabled"] = await crud.get_config_json(db, "dlp.regex.enabled", True)
    config["regex.patterns"] = await crud.get_config_json(db, "dlp.regex.patterns", [])
    config["regex.keywords"] = await crud.get_config_json(db, "dlp.regex.keywords", [])
    config["gliner.enabled"] = await crud.get_config_json(db, "dlp.gliner.enabled", False)
    config["gliner.threshold"] = await crud.get_config_json(db, "dlp.gliner.threshold", 0.5)
    config["gliner.categories"] = await crud.get_config_json(db, "dlp.gliner.categories", None)
    config["llm.enabled"] = await crud.get_config_json(db, "dlp.llm.enabled", False)
    config["llm.model"] = await crud.get_config_json(db, "dlp.llm.model", "")
    config["llm.system_prompt"] = await crud.get_config_json(db, "dlp.llm.system_prompt", "")
    config["severity_rules"] = await crud.get_config_json(db, "dlp.severity_rules", {})

    # Alert de-duplication (admin -> DLP panel toggle).
    config["dedup.enabled"] = await crud.get_config_json(db, "dlp.dedup.enabled", True)
    config["dedup.window_seconds"] = await crud.get_config_json(
        db, "dlp.dedup.window_seconds", int(DEDUP_DEFAULT_WINDOW)
    )

    # The LLM scanner dispatches through this callable rather than holding an
    # API key.  Nothing credential-shaped is stored for DLP anywhere.
    config["llm.complete"] = _internal_chat if config["llm.enabled"] else None

    return config


async def _surface_scanner_errors(db, req, request_id, scan_result) -> None:
    """Make a DLP scanner failure VISIBLE (F53).

    Logs at ERROR every time (for monitoring), and raises a dashboard alert at
    most once per scanner per SCANNER_ERROR_ALERT_WINDOW so an operator learns
    DLP is degraded — rather than a broken scanner silently passing sensitive
    data — without a permanently-broken scanner flooding the alert list.
    """
    from backend.app.db import crud

    now = time.monotonic()
    for err in scan_result.scanner_errors:
        scanner = (err.split(":", 1)[0] or "scanner").strip()
        logger.error(
            "dlp_scanner_failed_open",
            scanner=scanner,
            request_id=request_id,
            detail=err[:300],
        )
        last = _scanner_error_seen.get(scanner)
        if last is not None and (now - last) < SCANNER_ERROR_ALERT_WINDOW:
            continue
        _scanner_error_seen[scanner] = now
        try:
            await crud.create_dlp_alert(
                db,
                request_id=request_id,
                user_id=req.user_id,
                severity="major",
                scanner=scanner,
                categories=["dlp_scanner_error"],
                entities=[],
                confidence=0.0,
                scan_latency_ms=scan_result.scan_latency_ms,
                detail=(
                    f"DLP scanner '{scanner}' failed — scans may be MISSING "
                    f"sensitive data until this is resolved. {err[:200]}"
                ),
            )
            await db.commit()
        except Exception:
            logger.exception("dlp_scanner_error_alert_failed", scanner=scanner)


async def _internal_chat(model: str, messages: list) -> str:
    """Run one chat completion for the LLM scanner, straight to a backend.

    Deliberately bypasses the gateway's own /v1 endpoint: routing internally
    means DLP needs no API key, creates no request/response rows (which would
    re-store the very content it flagged), and consumes no user's quota.
    Mirrors services/image_policy._call_judge, which does the same on the
    image-moderation path.

    Trade-off: no scheduler admission, no retry/failover.  Scans are post-hoc
    and best-effort, so a failed dispatch costs one scan, not a request.
    """
    import random

    import httpx

    from backend.app.core.telemetry.registry import get_registry
    from backend.app.db.models import BackendEngine

    registry = get_registry()
    resolved, _ = registry.resolve_alias(model)

    # get_backends_with_model already filters to HEALTHY in SQL.
    backends = await registry.get_backends_with_model(resolved)
    available = []
    for b in backends:
        try:
            if await registry.is_backend_available(b.id):
                available.append(b)
        except Exception:
            continue
    if not available:
        logger.warning("dlp_llm_no_backend", model=model, resolved=resolved)
        raise RuntimeError(f"no healthy backend serving DLP model {resolved!r}")

    backend = random.choice(available)

    payload = {
        "model": resolved,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    # Reasoning is off fleet-wide on the normal inference path; dispatching
    # directly skips that policy, so re-apply it here or the scanner's JSON
    # answer arrives wrapped in <think> blocks.  gpt-oss rejects the kwarg.
    if backend.engine != BackendEngine.OLLAMA and "gpt-oss" not in resolved.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{backend.url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


def _email_allowed(severity: str) -> tuple:
    """Token-bucket guard: at most EMAIL_BURST alerts per severity per window.

    DLP scans every completed request, so a single chatty user can trip the same
    rule hundreds of times a minute.  Without a cap, enabling notifications for
    a low severity mails the admin into oblivion.  Per-process state (each
    uvicorn worker keeps its own), which is fine — this is a flood guard, not
    an exactly-once ledger.

    Returns (allowed, suppressed_since_last_send).
    """
    now = time.monotonic()
    window_start, count, suppressed = _email_budget.get(severity, (now, 0, 0))
    if now - window_start >= EMAIL_WINDOW_SECONDS:
        # Reset the window's send count but CARRY the suppressed tally: it is
        # cleared only by an actual send, so the next email that goes out can
        # tell the admin how many alerts were dropped while the budget was
        # exhausted.  Zeroing it here made that report permanently unreachable.
        window_start, count = now, 0
    if count >= EMAIL_BURST:
        _email_budget[severity] = (window_start, count, suppressed + 1)
        return False, suppressed + 1
    _email_budget[severity] = (window_start, count + 1, 0)
    return True, suppressed


async def _maybe_send_email(db, alert, scan_result) -> None:
    """Send email notification if configured for this severity level."""
    from backend.app.db import crud

    severity = scan_result.severity
    key = f"dlp.email.{severity}_recipients"
    recipients_str = await crud.get_config_json(db, key, "")
    if not recipients_str:
        return

    recipients = [r.strip() for r in str(recipients_str).split(",") if r.strip()]
    if not recipients:
        return

    allowed, suppressed = _email_allowed(severity)
    if not allowed:
        logger.info("dlp_email_suppressed", severity=severity, suppressed=suppressed)
        return

    try:
        from backend.app.services import email_service

        smtp_config = await email_service.get_smtp_config(db)
        if not email_service.is_smtp_configured(smtp_config):
            logger.warning("dlp_email_smtp_not_configured")
            return

        categories = ", ".join(sorted(alert.categories or [])) or "unknown"
        subject = f"[MindRouter DLP] {severity.upper()} alert — {categories}"

        # Body carries metadata only.  The matched values stay out of mail:
        # an inbox is the last place a DLP alert should reproduce them.
        base_url = await email_service.get_base_url(db)
        suppressed_note = (
            f"<p><em>{suppressed} further {severity} alert(s) were suppressed by the "
            f"notification rate limit.</em></p>" if suppressed else ""
        )
        body_html = (
            f"<p><strong>DLP Alert: {severity.upper()}</strong></p>"
            f"<p>Scanner: {scan_result.scanner}<br>"
            f"Categories: {categories}<br>"
            f"Findings: {len(scan_result.findings)}<br>"
            f"Request ID: {alert.request_id if alert.request_id else 'n/a'}<br>"
            f"Scan latency: {scan_result.scan_latency_ms}ms</p>"
            f"{suppressed_note}"
            f'<p><a href="{base_url}/admin/dlp">Review in Admin &rarr; DLP</a></p>'
            f"<p><small>Matched values are not included in this email. "
            f"Open the alert in the admin console to review.</small></p>"
        )

        sent = await email_service.send_notification_email(
            config=smtp_config,
            recipients=recipients,
            subject=subject,
            body_html=body_html,
            base_url=base_url,
        )
        logger.info("dlp_email_sent", severity=severity, recipients=len(recipients), sent=sent)
    except Exception:
        logger.exception("dlp_email_failed")


# NOTE: ensure_internal_api_key() was removed in 2.9.9.  It minted a
# never-expiring API key owned by whichever row `SELECT id FROM users LIMIT 1`
# returned (in practice the bootstrap admin, so the key carried admin rights)
# and stored the RAW key in app_config.dlp.internal_api_key_raw — the only
# unhashed key in a system that otherwise persists Argon2 + SHA-256 only.
# The LLM scanner now dispatches straight to a backend via _internal_chat and
# needs no credential at all.  Migration 071 revokes any key it created and
# drops both config rows.
