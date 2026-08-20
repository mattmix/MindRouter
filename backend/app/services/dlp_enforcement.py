############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_enforcement.py: Synchronous, inline DLP blocking.
#
# The post-hoc async worker (dlp_worker.py) scans AFTER the response is
# delivered and never affects latency. Blocking is different: to reject a
# request with a 4xx, the scan must run INLINE, before the content is
# released. That path lives here.
#
# It is strictly opt-in and gated: unless the DLP master switch is on AND at
# least one severity has a Block action configured, evaluate_prompt_block /
# evaluate_response_block return immediately without scanning — so alert-only
# deployments keep the zero-latency post-hoc behavior unchanged.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Inline (synchronous) DLP blocking, gated so alert-only stays zero-latency."""

import time
from typing import Any, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)

SEVERITIES = ("major", "moderate", "minor")

# The block gate (master toggle, scope, which severities block) is read on the
# request hot path. Cache it for a few seconds so an alert-only deployment —
# where the gate says "inactive" — never touches the DB per request. Admin
# config changes take effect within this window.
_GATE_TTL_SECONDS = 5.0
_gate_cache: dict = {"ts": -1e9, "value": None}


class DlpBlockedError(Exception):
    """Raised when inline DLP scanning blocks a request/response.

    Carries only category NAMES and the severity — never the matched values —
    so the 4xx surfaced to the caller explains what tripped the policy without
    echoing the sensitive data back.
    """

    def __init__(self, categories: List[str], severity: str, side: str):
        self.categories = categories
        self.severity = severity
        self.side = side  # "prompt" or "response"
        super().__init__(
            f"DLP blocked {side} at severity={severity}: {', '.join(categories)}"
        )

    def client_message(self) -> str:
        """A clear, non-leaking explanation for the API error body."""
        cats = ", ".join(self.categories) if self.categories else "sensitive data"
        where = "your prompt" if self.side == "prompt" else "the response"
        return (
            f"This request was blocked by the data-loss-prevention policy: "
            f"{where} appears to contain {cats}. Remove the sensitive data and "
            f"try again. If you believe this is an error, contact your administrator."
        )


def invalidate_gate_cache() -> None:
    """Drop the cached block gate so the next request re-reads it.

    Called after an admin saves DLP config, so a Block toggle takes effect
    immediately rather than after the TTL.
    """
    _gate_cache["ts"] = -1e9
    _gate_cache["value"] = None


async def _load_gate(db) -> dict:
    """Return the cached block gate {enabled, scope, block_severities}."""
    now = time.monotonic()
    cached = _gate_cache["value"]
    if cached is not None and (now - _gate_cache["ts"]) < _GATE_TTL_SECONDS:
        return cached

    from backend.app.db import crud

    enabled = bool(await crud.get_config_json(db, "dlp.enabled", False))
    scope = await crud.get_config_json(db, "dlp.block.scope", "prompt")
    if scope not in ("prompt", "response", "both"):
        scope = "prompt"
    block_severities: Set[str] = set()
    for sev in SEVERITIES:
        if await crud.get_config_json(db, f"dlp.action.{sev}.block", False):
            block_severities.add(sev)

    value = {"enabled": enabled, "scope": scope, "block_severities": block_severities}
    _gate_cache["value"] = value
    _gate_cache["ts"] = now
    return value


def _active(gate: dict, side: str) -> bool:
    """True when inline blocking should run for this side (prompt/response)."""
    if not gate["enabled"] or not gate["block_severities"]:
        return False
    scope = gate["scope"]
    if side == "prompt":
        return scope in ("prompt", "both")
    if side == "response":
        return scope in ("response", "both")
    return False


async def _evaluate(db, text: Optional[str], side: str) -> Optional[DlpBlockedError]:
    """Scan `text` inline and return a DlpBlockedError if it must be blocked.

    Returns None when: blocking is inactive for this side, the text is empty,
    the scanners find nothing that maps to a block-configured severity, OR the
    scan could not run (fail-open — availability over enforcement, per config).
    """
    gate = await _load_gate(db)
    if not _active(gate, side) or not text or not text.strip():
        return None

    # Reuse the worker's config builder (auto-discovers the off-host DLP pool)
    # and the shared scan orchestrator (regex + GLiNER, dispatched to the pool).
    try:
        from backend.app.services.dlp_worker import _load_dlp_config
        from backend.app.services.dlp_scanner import run_dlp_scan

        config = await _load_dlp_config(db)
        result = await run_dlp_scan(text, config)
    except Exception:
        # Fail-open: a scanner outage must not take down the gateway. Log loudly
        # so the coverage gap is visible; allow the request through.
        logger.exception("dlp_block_scan_failed_open", side=side)
        return None

    if result is None or not result.findings:
        # A degraded scan (pool down, on-host fallback also failed) yields no
        # findings -> fail-open. Make the gap observable.
        if result is not None and result.scanner_errors:
            logger.error(
                "dlp_block_scan_degraded_fail_open",
                side=side,
                errors=[e[:120] for e in result.scanner_errors],
            )
        return None

    rules = config.get("severity_rules", {}) or {}
    from backend.app.services.dlp_scanner import _SEVERITY_ORDER

    triggered: dict = {}  # severity -> set(category names)
    for f in result.findings:
        sev = rules.get(f.category, "moderate")
        if sev in gate["block_severities"]:
            triggered.setdefault(sev, set()).add(f.category)

    if not triggered:
        return None

    top = max(triggered.keys(), key=lambda s: _SEVERITY_ORDER.get(s, 1))
    categories = sorted(triggered[top])
    logger.info("dlp_blocked", side=side, severity=top, categories=categories)
    return DlpBlockedError(categories=categories, severity=top, side=side)


async def prompt_blocking_active(db) -> bool:
    """Cheap gate: is inline prompt blocking on? (No scan, cached gate.)

    Lets the hot path skip text extraction entirely when blocking is off.
    """
    return _active(await _load_gate(db), "prompt")


async def response_blocking_active(db) -> bool:
    """Cheap gate: is inline response blocking on? (No scan, cached gate.)"""
    return _active(await _load_gate(db), "response")


async def evaluate_prompt_block(db, text: Optional[str]) -> Optional[DlpBlockedError]:
    """Inline block decision for a request prompt (pre-dispatch). None = allow."""
    return await _evaluate(db, text, "prompt")


async def evaluate_response_block(db, text: Optional[str]) -> Optional[DlpBlockedError]:
    """Inline block decision for a model response (pre-release). None = allow."""
    return await _evaluate(db, text, "response")
