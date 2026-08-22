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
# Whether the audit trail keeps the caller's ORIGINAL (pre-redaction) query.
# On by default: an auditor investigating a block needs to see what was
# actually typed. Turn it off for a stricter posture — the row then keeps the
# masked evidence, the per-pass verdicts, and the outbound text, but not the
# original. Lives in search.audit.* with the other audit-storage switches.
STORE_ORIGINAL_KEY = "search.audit.store_original_query"

DEFAULT_ENABLED = False
DEFAULT_MIN_SEVERITY = "moderate"
DEFAULT_ON_ERROR = "block"
DEFAULT_STORE_ORIGINAL = True

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
    rounds: int = 0                        # mask/rescan rounds actually used
    widened: bool = False                  # a mask had to grow to cover a label
    original_query: str = ""               # exactly what the caller submitted
    # One entry per DLP pass, in order, each recording the text that was
    # scanned and whether it passed. This is the provenance trail: pass 1's
    # text is the original, the last entry's text is what went out (or the
    # furthest-masked form, when the search was refused).
    passes: List[dict] = field(default_factory=list)
    reason: Optional[str] = None

    def outbound_text(self) -> str:
        """The furthest-processed form of the query.

        When allowed, this is exactly what was sent. When blocked, it is the
        most-masked form reached before the gate gave up — useful to see how
        far masking got, and unambiguous because the row's status and NULL
        request_url record that nothing actually went out.
        """
        if self.allowed:
            return self.query or self.original_query
        for entry in reversed(self.passes):
            if entry.get("text_stored") and entry.get("text") is not None:
                return entry["text"]
        return self.original_query

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
            "rounds": self.rounds,
            "widened": self.widened,
            "passes": self.passes,
            "reason": self.reason,
        }


async def load_gate_config(db) -> dict:
    """Read the web-search DLP settings, tolerating missing/garbage values."""
    from backend.app.db import crud

    try:
        enabled = bool(await crud.get_config_json(db, ENABLED_KEY, DEFAULT_ENABLED))
        min_sev = await crud.get_config_json(db, MIN_SEVERITY_KEY, DEFAULT_MIN_SEVERITY)
        on_error = await crud.get_config_json(db, ON_ERROR_KEY, DEFAULT_ON_ERROR)
        store_original = bool(
            await crud.get_config_json(db, STORE_ORIGINAL_KEY, DEFAULT_STORE_ORIGINAL)
        )
    except Exception:
        # A config read failure must not silently disable screening; but with
        # no readable config we also cannot know it was ever enabled. Off is
        # the honest answer, and it is logged loudly.
        logger.error("websearch_dlp_config_unreadable", exc_info=True)
        return {"enabled": False, "min_severity": DEFAULT_MIN_SEVERITY,
                "on_scanner_error": DEFAULT_ON_ERROR,
                "store_original": DEFAULT_STORE_ORIGINAL}

    if min_sev not in VALID_MIN_SEVERITIES:
        min_sev = DEFAULT_MIN_SEVERITY
    if on_error not in VALID_ON_ERROR:
        on_error = DEFAULT_ON_ERROR
    return {"enabled": enabled, "min_severity": min_sev, "on_scanner_error": on_error,
            "store_original": store_original}


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


# Redaction masks the flagged text with ASTERISKS rather than substituting a
# placeholder token, and it masks the WHOLE span the scanner reported — the
# label word included, not just the value.
#
# Both choices are forced by step 3, which re-scans the redacted query. Measured
# against the production scanners:
#
#   "...or ssn [REDACTED] for records"     -> flagged (span "ssn [REDACTED]")
#   "...or ssn *********** for records"    -> flagged (span "ssn ***********")
#   "...or *** *********** for records"    -> CLEAN
#
# A word-shaped placeholder re-primes GLiNER, and so does the user's own label
# word left standing next to a masked value. Only masking the entire reported
# span removes the thing the scanner is reacting to.
#
# Whitespace is preserved so the query keeps its shape: "ssn 456-78-9012"
# becomes "*** ***********", not one undifferentiated run.
MASK_CHAR = "*"


def mask_span(text: str) -> str:
    """Asterisk every non-space character, preserving the whitespace layout."""
    return "".join(MASK_CHAR if not ch.isspace() else ch for ch in text or "")


# Rounds of mask-then-rescan before giving up. Each round is one more scan, so
# this is a latency ceiling as well as a safety one — but only for queries that
# actually carry sensitive data; a clean query is scanned exactly once.
# Widening (see _widen) can need two or three rounds to walk a mask outwards
# past a label, which is what this allows room for. The caller's contract is
# "redact, verify, and fail if it will not clear"; a bounded loop is that same
# contract when one round of masking exposes a NEW span (the label word beside
# the value it used to sit next to). It always terminates: a round that does
# not change the text stops immediately, which is also what happens when the
# scanner flags an already-masked run like "***********".
MAX_REDACTION_ROUNDS = 5


def _is_all_mask(value: str) -> bool:
    """True when a reported span is already nothing but mask characters.

    This is the signal that the scanner is reacting to CONTEXT rather than to
    anything still readable: it flagged the placeholder we just wrote. Masking
    it again would change nothing, so the mask has to grow outwards instead.
    """
    stripped = (value or "").strip()
    return bool(stripped) and all(ch == MASK_CHAR or ch.isspace() for ch in stripped)


# Function words carry no label meaning, so widening steps over them to reach
# the word that does. Without this, "my ssn is ***********" would mask "is" —
# burning a round and a word without removing what the scanner reacts to.
# Deliberately tiny: this is a "skip the connective" list, not a stopword
# corpus, and anything not on it is treated as a potential label.
_FUNCTION_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "my", "of", "on", "or", "our", "the", "their", "there", "this",
    "to", "was", "were", "with", "your",
})

# How many words to step back over while looking for the label.
_WIDEN_LOOKBACK_WORDS = 3


def _word_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of every whitespace-delimited token, in order."""
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _widen(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """The span of the neighbouring LABEL word to mask as well, or None.

    Returns a separate token span rather than a merged range, so the words in
    between are left readable: "my ssn is ***********" becomes "my *** is
    ***********", not "my *** ** ***********".

    Left first — a label overwhelmingly precedes the value it names ("ssn
    123-45-6789", "DOB: ...") — stepping over function words to reach the word
    that actually carries the meaning. Only if nothing usable sits to the left
    does it look right.
    """
    words = _word_spans(text)
    before = [w for w in words if w[1] <= start]
    after = [w for w in words if w[0] >= end]

    for candidate in reversed(before[-_WIDEN_LOOKBACK_WORDS:]):
        word = text[candidate[0]:candidate[1]].strip(".,:;!?").lower()
        if not word or _is_all_mask(word):
            continue
        if word in _FUNCTION_WORDS:
            continue
        return candidate

    for candidate in after[:_WIDEN_LOOKBACK_WORDS]:
        word = text[candidate[0]:candidate[1]].strip(".,:;!?").lower()
        if not word or _is_all_mask(word) or word in _FUNCTION_WORDS:
            continue
        return candidate
    return None


def _locate(text: str, finding) -> Optional[Tuple[int, int]]:
    """Offsets of a finding in ``text``, trusting them only if they check out."""
    value = getattr(finding, "text", "") or ""
    if not value:
        return None
    start = getattr(finding, "start", 0) or 0
    end = getattr(finding, "end", 0) or 0
    if 0 <= start < end <= len(text) and text[start:end] == value:
        return start, end
    idx = text.find(value)
    return (idx, idx + len(value)) if idx >= 0 else None


def _mask_findings(text: str, findings, severity_rules, *, widen: bool = False) -> str:
    """Mask every non-ignored finding in ``text`` with asterisks.

    Offsets are used when they demonstrably line up with the reported text —
    that is the precise way to remove a multi-word span like "ssn 456-78-9012".
    A finding whose offsets do not check out (the LLM scanner reports none, and
    a truncated scan can shift them) falls back to value substitution, so a
    finding is never skipped merely because its offsets were unusable.
    """
    from backend.app.services.dlp_scanner import is_ignored_category

    spans: List[Tuple[int, int]] = []
    values: List[str] = []
    for f in findings or []:
        category = getattr(f, "category", "") or "sensitive"
        if is_ignored_category(category, severity_rules):
            continue
        value = getattr(f, "text", "") or ""
        if not value:
            continue
        located = _locate(text, f)
        if located is None:
            values.append(value)
            continue
        start, end = located
        spans.append((start, end))
        # Widening applies only to spans the scanner reports that are ALREADY
        # fully masked: masking those again is a no-op, so the label beside
        # them is what has to go. A span with readable text left in it is
        # masked exactly as reported.
        if widen and _is_all_mask(value):
            extra = _widen(text, start, end)
            if extra is not None:
                spans.append(extra)

    out = text
    # Right-to-left so each replacement leaves earlier offsets valid.
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + mask_span(out[start:end]) + out[end:]
    # Longest first: a short value that is a substring of a longer one would
    # otherwise corrupt the longer match before it is replaced.
    for value in sorted(set(values), key=len, reverse=True):
        out = out.replace(value, mask_span(value))
    return out


def _record_pass(
    screen: "QueryScreen", *, number: int, text: str, result, severity_rules,
    threshold: str, store_text: bool,
) -> None:
    """Append one DLP pass to the provenance trail.

    ``store_text`` is False for pass 1 when the operator has turned off
    storage of the original query: every later pass is already masked, so
    only the first one can carry sensitive text.
    """
    from backend.app.services.dlp_scanner import is_ignored_category, mask_snippet

    findings = list(result.findings) if result is not None else []
    kept = [
        f for f in findings
        if not is_ignored_category(getattr(f, "category", "") or "", severity_rules)
    ]
    severity = _worst_severity(findings, severity_rules)
    entry = {
        "pass": number,
        "text": text if store_text else None,
        "text_stored": bool(store_text),
        "text_chars": len(text or ""),
        "severity": severity,
        # The verdict an auditor reads: did THIS text clear the configured bar?
        "verdict": "fail" if _meets_threshold(severity, threshold) else "pass",
        "categories": sorted({
            getattr(f, "category", "") for f in kept if getattr(f, "category", "")
        }),
        "findings": [
            {
                "scanner": getattr(f, "scanner", "?"),
                "category": getattr(f, "category", "") or "sensitive",
                "masked": mask_snippet(getattr(f, "text", "") or ""),
            }
            for f in kept[:MAX_MASKED_SNIPPETS]
        ],
        "degraded": bool(result is not None and getattr(result, "scanner_errors", None)),
        "scanner_errors": [e[:160] for e in (getattr(result, "scanner_errors", None) or [])],
    }
    screen.passes.append(entry)


def _degraded_block(screen: "QueryScreen", result, gate: dict, reason: str) -> bool:
    """Record a degraded scan; True when the policy says to refuse the search."""
    if result is None or not getattr(result, "scanner_errors", None):
        return False
    screen.degraded = True
    logger.error(
        "websearch_dlp_scan_degraded",
        errors=[e[:120] for e in result.scanner_errors],
        policy=gate.get("on_scanner_error"),
    )
    if gate.get("on_scanner_error") != "block":
        return False
    screen.allowed = False
    screen.action = ACTION_BLOCKED
    screen.reason = reason
    return True


def _has_content(text: str) -> bool:
    """Whether anything searchable survived redaction.

    Masks are pure punctuation, so this reduces to "is there an alphanumeric
    character left" — a query that is nothing but asterisks is not a search
    worth sending, and is refused rather than sent as noise.
    """
    return any(ch.isalnum() for ch in text or "")


async def screen_query(db, query: str, *, dlp_config: Optional[dict] = None) -> QueryScreen:
    """Scan → redact → re-scan a web-search query. Never raises.

    Returns a QueryScreen whose ``allowed``/``query`` the caller must honour:
    when allowed, ``query`` is what should be sent (redacted or not); when not,
    the search must be abandoned.
    """
    screen = QueryScreen(allowed=True, query=query or "", original_query=query or "")

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

        store_original = gate.get("store_original", DEFAULT_STORE_ORIGINAL)

        # --- Pass 1 ---------------------------------------------------
        result = await run_dlp_scan(query, dlp_config)
        _record_pass(screen, number=1, text=query, result=result,
                     severity_rules=severity_rules, threshold=threshold,
                     store_text=store_original)
        if _degraded_block(screen, result, gate,
                           "DLP scanners are unavailable and the policy is to block"):
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

        # --- Mask, verify, repeat -------------------------------------
        # Each round masks what the last scan flagged and re-scans the result.
        # A second round is often what clears a query: masking the value can
        # expose the user's own label word ("ssn ***********") as a new span,
        # and only masking that too satisfies the scanner. The loop stops the
        # moment a round changes nothing, which is also what happens when the
        # scanner flags an already-masked run — there is no progress to be had,
        # so the search is refused rather than retried forever.
        current = query
        current_findings = findings
        for round_no in range(1, MAX_REDACTION_ROUNDS + 1):
            masked_text = _mask_findings(current, current_findings, severity_rules)
            if masked_text == current:
                # The scanner flagged something already masked: it is reacting
                # to the words AROUND the mask (the label), so grow outwards
                # rather than declaring defeat. This is what turns "my ssn is
                # ***********" — which the scanners still read as an SSN —
                # into "my *** is ***********", which they do not.
                masked_text = _mask_findings(
                    current, current_findings, severity_rules, widen=True
                )
                if masked_text != current:
                    screen.widened = True
            screen.rounds = round_no

            if masked_text == current:
                screen.allowed = False
                screen.action = ACTION_BLOCKED
                screen.reason = (
                    "Sensitive content could not be redacted from the query"
                    if round_no == 1
                    else "Redacted query still contains sensitive data that cannot be masked further"
                )
                return screen

            if not _has_content(masked_text):
                screen.allowed = False
                screen.action = ACTION_BLOCKED
                screen.reason = "Nothing searchable remained after redaction"
                return screen

            current = masked_text
            verify = await run_dlp_scan(current, dlp_config)
            # Every later pass scans already-masked text, so it is always safe
            # to store — this is the "after" half of the before/after trail.
            _record_pass(screen, number=round_no + 1, text=current, result=verify,
                         severity_rules=severity_rules, threshold=threshold,
                         store_text=True)
            if _degraded_block(screen, verify, gate,
                               "DLP scanners became unavailable while verifying the redaction"):
                return screen

            current_findings = list(verify.findings) if verify is not None else []
            screen.second_severity = _worst_severity(current_findings, severity_rules)
            if not _meets_threshold(screen.second_severity, threshold):
                screen.allowed = True
                screen.action = ACTION_REDACTED
                screen.query = current
                screen.reason = (
                    f"Sensitive content masked before sending "
                    f"({round_no} round{'s' if round_no != 1 else ''})"
                )
                return screen

        # Still flagged after the last permitted round.
        screen.allowed = False
        screen.action = ACTION_BLOCKED
        screen.reason = "Redacted query still contains sensitive data"
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
