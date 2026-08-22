############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/dlp_gate.py: DLP screening of a web-search query
#     BEFORE it is sent to a third-party provider.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Screen a web-search query with DLP before it leaves the building.

A web search is the one outbound path where user text reaches a third party
verbatim, and unlike an inference request there is no contract with the
provider about what happens to it. So when the operator turns this on, the
query is scanned FIRST and only a clean (or successfully redacted) query is
ever sent.

THE SEQUENCE (each step exists for a reason):

  1. Scan the query with the global DLP settings — the same scanners,
     thresholds, severity rules and ignore list the rest of DLP uses. The
     web-search toggle governs WHETHER this runs, not HOW: an admin should
     not have to keep two rule sets in sync.

  2. If the worst finding reaches the web-search severity threshold, redact
     every non-ignored finding — not merely the ones at that level. Once a
     query is judged risky, leaving a "minor" email in it because it scored
     below the bar would be an odd place to draw the line.

  3. Re-scan the REDACTED query. Redaction is value-based substitution, so a
     match that overlapped another, or a value the scanner reported in a
     normalized form, can survive it. The second pass is what turns "we tried
     to redact" into "it is actually clean".

  4. If the second pass still reaches the threshold, the search is BLOCKED.
     Nothing goes out. The caller reports it gracefully.

SCANNER OUTAGE is deliberately fail-CLOSED here, unlike the inline DLP gate on
inference (dlp_enforcement.py), which fails open so a DLP outage cannot take
down the gateway. The trade is different: a web search is optional enrichment,
and the entire point of this feature is that unscanned text must not reach a
third party. It is configurable for operators who weigh it the other way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Config keys (seeded by migration 082).
ENABLED_KEY = "dlp.websearch.enabled"
MIN_SEVERITY_KEY = "dlp.websearch.min_severity"
ON_ERROR_KEY = "dlp.websearch.on_scanner_error"

DEFAULT_ENABLED = False
DEFAULT_MIN_SEVERITY = "moderate"
DEFAULT_ON_ERROR = "block"

VALID_MIN_SEVERITIES = ("minor", "moderate", "major")
VALID_ON_ERROR = ("block", "allow")

# Actions recorded on the web-search audit row.
ACTION_NONE = "none"          # scanned, nothing met the threshold
ACTION_REDACTED = "redacted"  # redacted and the second pass cleared it
ACTION_BLOCKED = "blocked"    # not sent

# Cap on masked evidence snippets kept per screening (the audit row is
# metadata about sensitive data, never a second copy of it).
MAX_MASKED_SNIPPETS = 20


class WebSearchBlockedError(ValueError):
    """The query was withheld from the provider by DLP.

    Subclasses ValueError on purpose: every existing call site already handles
    ValueError from a provider (missing key, disabled search) and turns it
    into a clean user-facing message, so a block degrades gracefully
    everywhere without a single caller having to change.
    """

    def __init__(self, message: str, *, screen: "QueryScreen" | None = None):
        super().__init__(message)
        self.screen = screen


@dataclass
class QueryScreen:
    """Outcome of screening one query."""

    allowed: bool = True
    query: str = ""                      # what should actually be sent
    scanned: bool = False
    action: str = ACTION_NONE
    severity: Optional[str] = None         # worst severity, first pass
    second_severity: Optional[str] = None  # worst severity, after redaction
    categories: List[str] = field(default_factory=list)
    masked: List[dict] = field(default_factory=list)
    degraded: bool = False                 # a scanner failed to run
    reason: Optional[str] = None

    def audit_detail(self) -> Optional[dict]:
        """The JSON blob stored on the web-search audit row."""
        if not self.scanned:
            return None
        return {
            "action": self.action,
            "severity": self.severity,
            "second_severity": self.second_severity,
            "categories": self.categories,
            "masked": self.masked,
            "degraded": self.degraded,
            "reason": self.reason,
        }


async def load_gate_config(db) -> dict:
    """Read the web-search DLP settings, tolerating missing/garbage values."""
    from backend.app.db import crud

    try:
        enabled = bool(await crud.get_config_json(db, ENABLED_KEY, DEFAULT_ENABLED))
        min_sev = await crud.get_config_json(db, MIN_SEVERITY_KEY, DEFAULT_MIN_SEVERITY)
        on_error = await crud.get_config_json(db, ON_ERROR_KEY, DEFAULT_ON_ERROR)
    except Exception:
        # A config read failure must not silently disable screening; but with
        # no readable config we also cannot know it was ever enabled. Off is
        # the honest answer, and it is logged loudly.
        logger.error("websearch_dlp_config_unreadable", exc_info=True)
        return {"enabled": False, "min_severity": DEFAULT_MIN_SEVERITY,
                "on_scanner_error": DEFAULT_ON_ERROR}

    if min_sev not in VALID_MIN_SEVERITIES:
        min_sev = DEFAULT_MIN_SEVERITY
    if on_error not in VALID_ON_ERROR:
        on_error = DEFAULT_ON_ERROR
    return {"enabled": enabled, "min_severity": min_sev, "on_scanner_error": on_error}


def _meets_threshold(severity: Optional[str], threshold: str) -> bool:
    """True when ``severity`` is at or above the configured bar."""
    from backend.app.services.dlp_scanner import _SEVERITY_ORDER

    if not severity:
        return False
    return _SEVERITY_ORDER.get(severity, 1) >= _SEVERITY_ORDER.get(threshold, 1)


def _worst_severity(findings, severity_rules) -> Optional[str]:
    """Highest severity among findings, honouring the Ignore rule level."""
    from backend.app.services.dlp_scanner import classify_severity, drop_ignored_findings

    kept = drop_ignored_findings(findings, severity_rules)
    if not kept:
        return None
    return classify_severity(kept, severity_rules)


def _redactions(findings, severity_rules) -> List[Tuple[str, str]]:
    """(value, category) pairs for every non-ignored finding, deduplicated.

    Longest value first: redacting a short value that is a substring of a
    longer one would corrupt the longer match before it is replaced.
    """
    from backend.app.services.dlp_scanner import is_ignored_category

    seen = set()
    out: List[Tuple[str, str]] = []
    for f in findings or []:
        value = (getattr(f, "text", "") or "").strip()
        category = getattr(f, "category", "") or "sensitive"
        if not value or value in seen:
            continue
        if is_ignored_category(category, severity_rules):
            continue
        seen.add(value)
        out.append((value, category))
    out.sort(key=lambda pair: len(pair[0]), reverse=True)
    return out


def _masked_evidence(findings, severity_rules) -> List[dict]:
    """Masked snippets for the audit row — never the verbatim value."""
    from backend.app.services.dlp_scanner import is_ignored_category, mask_snippet

    out: List[dict] = []
    for f in findings or []:
        if len(out) >= MAX_MASKED_SNIPPETS:
            break
        category = getattr(f, "category", "") or "sensitive"
        if is_ignored_category(category, severity_rules):
            continue
        out.append({
            "scanner": getattr(f, "scanner", "?"),
            "category": category,
            "text": mask_snippet(getattr(f, "text", "") or ""),
        })
    return out


# The placeholder substituted for a redacted value.
#
# NEUTRAL ON PURPOSE — no category name. The inference-side redaction
# (dlp_enforcement.redact_text) writes "[REDACTED: social security number]",
# which is right there: the model benefits from knowing what was removed, and
# that text is never re-scanned. Here it is actively harmful, because step 3
# re-scans the redacted query and GLiNER flags the words "social security
# number" INSIDE the placeholder as a social security number. Verified against
# the production scanners:
#
#   "ssn [REDACTED: social security number] what documents..." -> major
#   "ssn [REDACTED] what documents..."                         -> clean
#
# A labelled placeholder therefore makes redaction self-defeating: every
# redacted query would block on the second pass. The category is preserved in
# the audit row's dlp_detail, so nothing is lost to the auditor.
REDACTION_PLACEHOLDER = "[REDACTED]"

# Matches both this module's neutral placeholder and the labelled inference-side
# form, so the "is anything searchable left?" test is correct either way.
_PLACEHOLDER_RE = re.compile(r"\[REDACTED(?::[^\]]*)?\]")


def _apply_redactions(text: str, redactions: List[Tuple[str, str]]) -> str:
    """Replace each offending value with the neutral placeholder.

    Value-based substitution, longest value first (the caller sorts), so a
    short match that is a substring of a longer one cannot corrupt it.
    """
    out = text or ""
    for value, _category in redactions:
        if value:
            out = out.replace(value, REDACTION_PLACEHOLDER)
    return out


def _has_content(text: str) -> bool:
    """Whether anything searchable survived redaction."""
    stripped = _PLACEHOLDER_RE.sub(" ", text or "")
    return any(ch.isalnum() for ch in stripped)


async def screen_query(db, query: str, *, dlp_config: Optional[dict] = None) -> QueryScreen:
    """Scan → redact → re-scan a web-search query. Never raises.

    Returns a QueryScreen whose ``allowed``/``query`` the caller must honour:
    when allowed, ``query`` is what should be sent (redacted or not); when not,
    the search must be abandoned.
    """
    screen = QueryScreen(allowed=True, query=query or "")

    try:
        gate = await load_gate_config(db)
        if not gate["enabled"]:
            return screen

        from backend.app.services.dlp_scanner import run_dlp_scan
        from backend.app.services.dlp_worker import _load_dlp_config

        if dlp_config is None:
            dlp_config = await _load_dlp_config(db)
        severity_rules = dlp_config.get("severity_rules") or {}
        threshold = gate["min_severity"]
        screen.scanned = True

        # --- Pass 1 ---------------------------------------------------
        result = await run_dlp_scan(query, dlp_config)
        if result is not None and result.scanner_errors:
            screen.degraded = True
            logger.error(
                "websearch_dlp_scan_degraded",
                errors=[e[:120] for e in result.scanner_errors],
                policy=gate["on_scanner_error"],
            )
            if gate["on_scanner_error"] == "block":
                screen.allowed = False
                screen.action = ACTION_BLOCKED
                screen.reason = "DLP scanners are unavailable and the policy is to block"
                return screen

        findings = list(result.findings) if result is not None else []
        screen.severity = _worst_severity(findings, severity_rules)
        screen.categories = sorted({
            getattr(f, "category", "") for f in findings if getattr(f, "category", "")
        })

        if not _meets_threshold(screen.severity, threshold):
            # Clean, or only findings below the bar: send the query unchanged.
            return screen

        screen.masked = _masked_evidence(findings, severity_rules)

        # --- Redact ---------------------------------------------------
        redactions = _redactions(findings, severity_rules)
        redacted = _apply_redactions(query, redactions)
        if not redactions or redacted == query:
            # Nothing actionable to remove (e.g. GLiNER flagged the phrase
            # rather than a value): there is no safe query to send.
            screen.allowed = False
            screen.action = ACTION_BLOCKED
            screen.reason = "Sensitive content could not be redacted from the query"
            return screen

        if not _has_content(redacted):
            screen.allowed = False
            screen.action = ACTION_BLOCKED
            screen.reason = "Nothing searchable remained after redaction"
            return screen

        # --- Pass 2: prove the redaction actually worked ---------------
        second = await run_dlp_scan(redacted, dlp_config)
        if second is not None and second.scanner_errors and gate["on_scanner_error"] == "block":
            screen.degraded = True
            screen.allowed = False
            screen.action = ACTION_BLOCKED
            screen.reason = "DLP scanners became unavailable while verifying the redaction"
            return screen

        second_findings = list(second.findings) if second is not None else []
        screen.second_severity = _worst_severity(second_findings, severity_rules)
        if _meets_threshold(screen.second_severity, threshold):
            screen.allowed = False
            screen.action = ACTION_BLOCKED
            screen.reason = "Redacted query still contains sensitive data"
            return screen

        screen.allowed = True
        screen.action = ACTION_REDACTED
        screen.query = redacted
        screen.reason = "Sensitive content redacted before sending"
        return screen

    except Exception:
        # Screening itself broke (not a scanner — a bug or an import failure).
        # Honour the configured posture rather than silently sending the query.
        logger.error("websearch_dlp_screen_failed", exc_info=True)
        try:
            policy = (await load_gate_config(db)).get("on_scanner_error", DEFAULT_ON_ERROR)
        except Exception:
            policy = DEFAULT_ON_ERROR
        if policy == "block" and screen.scanned:
            screen.allowed = False
            screen.action = ACTION_BLOCKED
            screen.degraded = True
            screen.reason = "DLP screening failed and the policy is to block"
        return screen
