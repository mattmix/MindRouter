############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_enforcement.py: Synchronous, inline DLP actions (block + redact).
#
# The post-hoc async worker (dlp_worker.py) scans AFTER the response is
# delivered and never affects latency. Two actions instead run INLINE, before
# content is released, and live here:
#
#   Block   — reject the request/response with a 422.
#   Redact  — replace the offending spans (e.g. an SSN) in the outbound prompt
#             and/or completion with a placeholder, and let the request proceed.
#
# Both are strictly opt-in and gated: unless the DLP master switch is on AND at
# least one severity has a Block or Redact action configured for the relevant
# side, the evaluate_* helpers return immediately without scanning — so an
# alert-only deployment keeps the zero-latency post-hoc behavior unchanged.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Inline (synchronous) DLP actions — block and redact — gated for zero-cost off."""

import time
from typing import Any, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)

SEVERITIES = ("major", "moderate", "minor")

# The inline gate (master toggle, scope, which severities block/redact) is read
# on the request hot path. Cache it for a few seconds so an alert-only
# deployment — where the gate says "inactive" — never touches the DB per
# request. Admin config changes take effect within this window.
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


class InlineAction:
    """The outcome of an inline DLP scan for one side (prompt or response).

    Exactly one of these applies at a time, block taking precedence over redact
    (a rejected request has nothing left to redact):
      - block: a DlpBlockedError to raise (→ 422), or None.
      - redactions: (value, category) pairs to strip from the outbound text.
    """

    def __init__(
        self,
        block: Optional[DlpBlockedError] = None,
        redactions: Optional[List[Tuple[str, str]]] = None,
    ):
        self.block = block
        self.redactions = redactions or []


def invalidate_gate_cache() -> None:
    """Drop the cached inline gate so the next request re-reads it.

    Called after an admin saves DLP config, so a Block/Redact toggle takes
    effect immediately rather than after the TTL.
    """
    _gate_cache["ts"] = -1e9
    _gate_cache["value"] = None


async def _load_gate(db) -> dict:
    """Return the cached inline gate {enabled, scope, block_/redact_severities}."""
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
    redact_severities: Set[str] = set()
    for sev in SEVERITIES:
        if await crud.get_config_json(db, f"dlp.action.{sev}.block", False):
            block_severities.add(sev)
        if await crud.get_config_json(db, f"dlp.action.{sev}.redact", False):
            redact_severities.add(sev)

    value = {
        "enabled": enabled,
        "scope": scope,
        "block_severities": block_severities,
        "redact_severities": redact_severities,
    }
    _gate_cache["value"] = value
    _gate_cache["ts"] = now
    return value


def _scope_covers(gate: dict, side: str) -> bool:
    scope = gate["scope"]
    if side == "prompt":
        return scope in ("prompt", "both")
    if side == "response":
        return scope in ("response", "both")
    return False


def _active(gate: dict, side: str) -> bool:
    """True when any inline action (block or redact) should run for this side."""
    if not gate["enabled"]:
        return False
    if not (gate["block_severities"] or gate["redact_severities"]):
        return False
    return _scope_covers(gate, side)


async def _evaluate(db, text: Optional[str], side: str) -> Optional[InlineAction]:
    """Scan `text` inline and return the InlineAction to take, or None.

    Returns None when: inline action is inactive for this side, the text is
    empty, nothing maps to a block-/redact-configured severity, OR the scan
    could not run (fail-open — availability over enforcement).
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
        # so the coverage gap is visible; allow the request through unredacted.
        logger.exception("dlp_inline_scan_failed_open", side=side)
        return None

    if result is None or not result.findings:
        if result is not None and result.scanner_errors:
            logger.error(
                "dlp_inline_scan_degraded_fail_open",
                side=side,
                errors=[e[:120] for e in result.scanner_errors],
            )
        return None

    rules = config.get("severity_rules", {}) or {}
    from backend.app.services.dlp_scanner import _SEVERITY_ORDER

    block_triggers: dict = {}  # severity -> set(category names)
    redactions: List[Tuple[str, str]] = []
    seen_values: Set[str] = set()
    for f in result.findings:
        sev = rules.get(f.category, "moderate")
        if sev in gate["block_severities"]:
            block_triggers.setdefault(sev, set()).add(f.category)
        elif sev in gate["redact_severities"]:
            value = (f.text or "").strip()
            if value and value not in seen_values:
                seen_values.add(value)
                redactions.append((value, f.category))

    # Block wins: if anything must be blocked, redaction is moot.
    if block_triggers:
        top = max(block_triggers.keys(), key=lambda s: _SEVERITY_ORDER.get(s, 1))
        categories = sorted(block_triggers[top])
        logger.info("dlp_blocked", side=side, severity=top, categories=categories)
        return InlineAction(block=DlpBlockedError(categories, top, side))

    if redactions:
        logger.info(
            "dlp_redacted", side=side, count=len(redactions),
            categories=sorted({c for _, c in redactions}),
        )
        return InlineAction(redactions=redactions)

    return None


def redact_text(text: Optional[str], redactions: List[Tuple[str, str]]) -> Optional[str]:
    """Replace each offending value with a category-labeled placeholder.

    Value-based (not offset-based) so it stays correct when applied to the
    individual message fields rather than the concatenated scan text.
    """
    if not text or not redactions:
        return text
    out = text
    for value, category in redactions:
        if value:
            out = out.replace(value, f"[REDACTED: {category}]")
    return out


async def prompt_inline_active(db) -> bool:
    """Cheap gate: is inline prompt action (block or redact) on? (No scan.)"""
    return _active(await _load_gate(db), "prompt")


async def response_inline_active(db) -> bool:
    """Cheap gate: is inline response action (block or redact) on? (No scan.)"""
    return _active(await _load_gate(db), "response")


async def evaluate_prompt_inline(db, text: Optional[str]) -> Optional[InlineAction]:
    """Inline action decision for a request prompt (pre-dispatch). None = allow."""
    return await _evaluate(db, text, "prompt")


async def evaluate_response_inline(db, text: Optional[str]) -> Optional[InlineAction]:
    """Inline action decision for a model response (pre-release). None = allow."""
    return await _evaluate(db, text, "response")
