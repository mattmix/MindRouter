############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_scanner.py: Data Loss Prevention scanning logic
#
# Pure logic module — no DB imports. Contains regex, GLiNER,
# and LLM-based scanners for detecting sensitive data.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""DLP scanner: regex, GLiNER NER, and LLM-based sensitive data detection."""

import asyncio
import concurrent.futures
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Signature of the completion callable injected into scan_llm.  The scanner
# never holds a credential or speaks HTTP itself — see dlp_worker._internal_chat.
CompleteFn = Callable[[str, List[Dict[str, str]]], Awaitable[str]]

# Upper bound on the text handed to any scanner.  Caps both regex backtracking
# cost and GLiNER/LLM token use on a pathological request.
MAX_SCAN_CHARS = 200_000

# GLiNER is CPU-bound torch inference whose cost scales with text length, so it
# gets its OWN, much smaller cap than the cheap regex scanner: a long chat
# history could otherwise pin a CPU thread for tens of seconds.  Admin-tunable
# via dlp.gliner.max_scan_chars; this bounds a scan to ~1-2s at the default.
GLINER_DEFAULT_MAX_CHARS = 10_000


class DlpScannerError(Exception):
    """A DLP scanner failed to RUN (model load / dispatch / unparseable output).

    Distinct from a scan that ran cleanly and found nothing: a scanner that
    errors must never be treated as "clean" (that is the silent fail-open the
    audit flagged).  run_dlp_scan collects these so the worker can surface a
    degraded scanner instead of letting it pass sensitive data unnoticed.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScanFinding:
    """A single sensitive-data finding from any scanner."""
    scanner: str          # "regex", "gliner", or "llm"
    category: str         # e.g. "social security number", "credit card number"
    text: str             # the matched text snippet
    confidence: float     # 0.0–1.0
    start: int = 0        # character offset in source text
    end: int = 0          # character offset end


@dataclass
class ScanResult:
    """Aggregated result of a DLP scan across all scanners."""
    findings: List[ScanFinding] = field(default_factory=list)
    severity: str = "minor"
    scan_latency_ms: int = 0
    scanner: str = "regex"       # primary scanner that produced the result
    detail: Optional[str] = None
    # Scanners that failed to run this pass (e.g. "gliner: model load failed").
    # Non-empty means the scan was DEGRADED — sensitive data may have been
    # missed — so a result is returned (not None) even with zero findings.
    scanner_errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in regex patterns (always available)
# ---------------------------------------------------------------------------

def _luhn_ok(text: str) -> bool:
    """Luhn check over the digits in a candidate card number.

    The card regex alone matches ANY 13-19 digit run (barcodes, order ids,
    tracking numbers) — measured span precision 0.36 on a labeled corpus.
    Real card numbers carry a Luhn check digit, so validating here removes
    ~90% of those false positives while keeping recall at 1.0.
    """
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# "validator" names a post-match check applied ONLY to built-in patterns —
# admin-supplied custom patterns keep raw regex semantics.
_VALIDATORS = {"luhn": _luhn_ok}

_BUILTIN_PATTERNS = [
    {"name": "SSN", "pattern": r"\b\d{3}-\d{2}-\d{4}\b", "category": "social security number", "severity": "major"},
    {"name": "Credit Card", "pattern": r"\b(?:\d[ -]*?){13,19}\b", "category": "credit card number", "severity": "major", "validator": "luhn"},
    {"name": "Email Address", "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "category": "email", "severity": "minor"},
    {"name": "Phone (US)", "pattern": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "category": "phone number", "severity": "minor"},
    {"name": "Date of Birth", "pattern": r"\b(?:DOB|date of birth|born on)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "category": "date of birth", "severity": "moderate"},
]


# ---------------------------------------------------------------------------
# Regex / keyword scanner
# ---------------------------------------------------------------------------

def scan_regex(
    text: str,
    custom_patterns: Optional[List[Dict[str, str]]] = None,
    keywords: Optional[List[str]] = None,
) -> List[ScanFinding]:
    """Scan text with regex patterns and keyword matching.

    Returns a list of ScanFinding objects for each match.
    """
    findings: List[ScanFinding] = []

    all_patterns = list(_BUILTIN_PATTERNS)
    if custom_patterns:
        # Admin-supplied patterns are validated at save time, but a row written
        # before that validation existed (or by a direct DB edit) can still be
        # the wrong shape.  One bad entry must never abort the built-in patterns
        # or the scanners that run after this one.
        all_patterns.extend(p for p in custom_patterns if isinstance(p, dict) and p.get("pattern"))

    for pat in all_patterns:
        validate = _VALIDATORS.get(pat.get("validator", ""))
        try:
            compiled = re.compile(pat["pattern"], re.IGNORECASE)
            pos = 0
            while True:
                m = compiled.search(text, pos)
                if m is None:
                    break
                if validate is not None and not validate(m.group()):
                    # The greedy card pattern glues short digit prefixes onto a
                    # real number ("cvv 123 4111...") and the glued span fails
                    # Luhn. Skipping past the span would suppress the real
                    # card, so re-attempt INSIDE it from the next character.
                    pos = m.start() + 1
                    continue
                findings.append(ScanFinding(
                    scanner="regex",
                    category=pat.get("category", pat.get("name", "unknown")),
                    text=m.group(),
                    confidence=1.0,
                    start=m.start(),
                    end=m.end(),
                ))
                pos = m.end()
        except Exception:
            # re.error for a bad pattern; anything else means a malformed entry
            # reached us despite the shape filter above.
            logger.warning("dlp_regex_invalid", pattern=str(pat.get("name", "?"))[:80])

    if keywords:
        text_lower = text.lower()
        for kw in keywords:
            if not kw:
                continue
            kw_lower = kw.strip().lower()
            if not kw_lower:
                continue
            idx = 0
            while True:
                pos = text_lower.find(kw_lower, idx)
                if pos == -1:
                    break
                findings.append(ScanFinding(
                    scanner="regex",
                    category="keyword",
                    text=text[pos:pos + len(kw_lower)],
                    confidence=0.9,
                    start=pos,
                    end=pos + len(kw_lower),
                ))
                idx = pos + 1

    return findings


# ---------------------------------------------------------------------------
# GLiNER NER scanner (lazy model load)
# ---------------------------------------------------------------------------

_gliner_model = None
_gliner_lock = asyncio.Lock()

# DLP-private thread pool. Scans used to run on the DEFAULT executor, which
# the inference hot path also uses (tiktoken estimation, Argon2 verification,
# OCR, watermarking) — N concurrent multi-second GLiNER predicts could starve
# all of it. Sized to the worker's max consumer count.
_DLP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="dlp-scan")


async def _load_gliner():
    """Lazily load the GLiNER PII model. Thread-safe via asyncio lock."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    async with _gliner_lock:
        if _gliner_model is not None:
            return _gliner_model

        logger.info("dlp_gliner_loading", model="urchade/gliner_multi_pii-v1")
        t0 = time.monotonic()

        try:
            from gliner import GLiNER
            loop = asyncio.get_event_loop()
            _gliner_model = await loop.run_in_executor(
                _DLP_EXECUTOR, GLiNER.from_pretrained, "urchade/gliner_multi_pii-v1"
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("dlp_gliner_loaded", elapsed_ms=elapsed)
        except ImportError:
            logger.error("dlp_gliner_not_installed", hint="pip install gliner")
            raise
        except Exception:
            logger.exception("dlp_gliner_load_failed")
            raise

    return _gliner_model


async def scan_gliner(
    text: str,
    categories: Optional[List[str]] = None,
    threshold: float = 0.5,
    max_chars: Optional[int] = None,
) -> List[ScanFinding]:
    """Scan text using GLiNER NER model for PII entities.

    Args:
        text: Text to scan.
        categories: Entity categories to detect.
        threshold: Minimum confidence threshold.
        max_chars: Cap the text fed to the model (CPU cost scales with length).
            Defaults to GLINER_DEFAULT_MAX_CHARS; None keeps the default.

    Returns:
        List of ScanFinding objects.
    """
    cap = GLINER_DEFAULT_MAX_CHARS if max_chars is None else max_chars
    if cap and cap > 0 and len(text) > cap:
        logger.info("dlp_gliner_text_capped", original_chars=len(text), kept=cap)
        text = text[:cap]

    if categories is None:
        # "person" is deliberately NOT a default: measured precision 0.34 —
        # the model tags section headers and greetings ("Chief complaint",
        # "CONTACT", "hey") as people. Admins can still opt in via
        # dlp.gliner.categories.
        categories = [
            "phone number", "email", "credit card number",
            "social security number", "date of birth", "driver license number",
            "passport number", "bank account number",
        ]
    elif not categories:
        # An explicitly-empty admin list means "scan no categories" — honor
        # it rather than silently substituting the defaults.
        return []

    try:
        model = await _load_gliner()
    except Exception as e:
        # Model unavailable (not installed / download failed). Surface it —
        # a scan that cannot run is not a clean scan.
        raise DlpScannerError(f"gliner model unavailable: {type(e).__name__}") from e

    loop = asyncio.get_event_loop()

    def _predict():
        return model.predict_entities(text, categories, threshold=threshold)

    try:
        entities = await loop.run_in_executor(_DLP_EXECUTOR, _predict)
    except Exception as e:
        logger.error("dlp_gliner_predict_failed", error=type(e).__name__)
        raise DlpScannerError(f"gliner predict failed: {type(e).__name__}") from e

    findings = []
    for ent in entities:
        findings.append(ScanFinding(
            scanner="gliner",
            category=ent.get("label", "unknown"),
            text=ent.get("text", ""),
            confidence=ent.get("score", threshold),
            start=ent.get("start", 0),
            end=ent.get("end", 0),
        ))

    return findings


# ---------------------------------------------------------------------------
# LLM contextual scanner (dispatched by an injected callable — no credential)
# ---------------------------------------------------------------------------

# Reasoning models wrap their answer in <think>…</think>.  The gateway strips
# this on the normal inference path; the DLP scanner dispatches straight to a
# backend, so it must strip it here or every parse fails.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


async def scan_llm(
    text: str,
    system_prompt: str,
    model: str,
    complete: CompleteFn,
) -> List[ScanFinding]:
    """Scan text using an LLM for contextual sensitive data detection.

    ``complete(model, messages) -> content`` is injected by the caller so this
    module stays free of DB, registry, and HTTP imports — and so the scanner
    holds no API key.  See dlp_worker._internal_chat for the production
    implementation, which dispatches directly to a healthy backend.
    """
    # Truncate very long text to avoid excessive token usage
    max_chars = 8000
    scan_text = text[:max_chars] if len(text) > max_chars else text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this text for sensitive data:\n\n{scan_text}"},
    ]

    try:
        content = await complete(model, messages)
    except Exception as e:
        # Dispatch failed (no backend, timeout, HTTP error). Surface it — the
        # scan did not run, so the request is unscanned, not clean.
        logger.error("dlp_llm_scan_failed", model=model, error=type(e).__name__)
        raise DlpScannerError(f"llm dispatch failed: {type(e).__name__}") from e

    # Extract JSON from the response (may have reasoning or fences)
    json_str = _THINK_RE.sub("", content or "").strip()
    if json_str.startswith("```"):
        # Strip markdown code fences
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines)

    # An empty answer or an explicit empty array is a legitimate CLEAN result.
    if json_str in ("", "[]"):
        return []

    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # The model returned non-empty, non-JSON output: a DEGRADED scan, not a
        # clean one.  Never log the response body — it quotes the very content
        # we are scanning.
        logger.warning("dlp_llm_parse_failed", model=model, content_chars=len(content or ""))
        raise DlpScannerError("llm returned unparseable output") from e

    if not isinstance(parsed, list):
        raise DlpScannerError("llm did not return a JSON array")

    findings = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        findings.append(ScanFinding(
            scanner="llm",
            category=str(item.get("category", "unknown")),
            text=str(item.get("text", "")),
            confidence=float(item.get("confidence", 0.7)),
        ))

    return findings


# ---------------------------------------------------------------------------
# Snippet masking
# ---------------------------------------------------------------------------

def mask_snippet(text: str, keep: int = 2) -> str:
    """Mask a matched snippet for storage in an alert row.

    A DLP alert is metadata about sensitive data — it must not become a second
    copy of it.  Alerts are long-lived and admin-readable, so the verbatim match
    never leaves the scanner; the full context stays in the audit tables, where
    the capture toggles and retention policy govern it.

    Keeps the first and last ``keep`` characters so an admin can still recognise
    a false positive (``4111********1111``), and preserves length for shape.
    """
    if not text:
        return ""
    s = str(text)
    if len(s) > 64:
        s = s[:64]
    if len(s) <= keep * 2:
        return "*" * len(s)
    return f"{s[:keep]}{'*' * (len(s) - keep * 2)}{s[-keep:]}"


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"minor": 0, "moderate": 1, "major": 2}


def classify_severity(
    findings: List[ScanFinding],
    severity_rules: Optional[Dict[str, str]] = None,
) -> str:
    """Classify the overall severity from a list of findings.

    Uses severity_rules mapping (category → severity). The highest severity
    among all findings wins. Unknown categories default to "moderate".
    """
    if not findings:
        return "minor"

    if severity_rules is None:
        severity_rules = {}

    highest = "minor"
    for f in findings:
        cat_severity = severity_rules.get(f.category, "moderate")
        if _SEVERITY_ORDER.get(cat_severity, 1) > _SEVERITY_ORDER.get(highest, 0):
            highest = cat_severity

    return highest


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_scannable_text(
    messages: Optional[Any] = None,
    prompt: Optional[str] = None,
    response_content: Optional[str] = None,
    modality: Optional[str] = None,
) -> Optional[str]:
    """Extract scannable text from request/response data.

    Skips image/multimodal content. Concatenates all text parts.
    """
    parts: List[str] = []

    # Extract from chat messages
    if messages:
        msg_list = messages if isinstance(messages, list) else messages.get("messages", [])
        for msg in msg_list:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Multipart content — only extract text parts
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif part.get("type") in ("image_url", "image"):
                                continue  # skip images

    # Extract from raw prompt
    if prompt:
        parts.append(prompt)

    # Extract from response
    if response_content:
        parts.append(response_content)

    combined = "\n".join(p for p in parts if p)
    return combined if combined.strip() else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_dlp_scan(
    text: str,
    config: Dict[str, Any],
) -> Optional[ScanResult]:
    """Run all enabled DLP scanners on the given text.

    Args:
        text: The text to scan.
        config: DLP configuration dict with keys like:
            - regex.enabled, regex.patterns, regex.keywords
            - gliner.enabled, gliner.threshold, gliner.categories
            - llm.enabled, llm.model, llm.system_prompt, llm.api_key, llm.base_url
            - severity_rules

    Returns:
        ScanResult if any findings, None if clean.
    """
    t0 = time.monotonic()
    all_findings: List[ScanFinding] = []
    scanners_used: List[str] = []
    scanner_errors: List[str] = []

    if len(text) > MAX_SCAN_CHARS:
        logger.info("dlp_scan_text_truncated", original_chars=len(text), kept=MAX_SCAN_CHARS)
        text = text[:MAX_SCAN_CHARS]

    # --- Regex scanner (always fast, run first) ---
    if config.get("regex.enabled", True):
        # Off the event loop: `re` is a backtracking engine with no timeout, and
        # admin-supplied patterns meet attacker-controlled text here.  A
        # pathological pair burns a thread-pool thread instead of stalling the
        # whole worker, which also serves the inference API.
        loop = asyncio.get_event_loop()
        regex_findings = await loop.run_in_executor(
            _DLP_EXECUTOR,
            scan_regex,
            text,
            config.get("regex.patterns"),
            config.get("regex.keywords"),
        )
        all_findings.extend(regex_findings)
        if regex_findings:
            scanners_used.append("regex")

    # --- GLiNER NER scanner ---
    # A scanner that raises DlpScannerError is recorded and the OTHER scanners
    # still run: one broken scanner must not blind the others, and it must not
    # be silently swallowed (the fail-open the audit flagged).
    if config.get("gliner.enabled", False):
        try:
            gliner_findings = await scan_gliner(
                text,
                categories=config.get("gliner.categories"),
                threshold=config.get("gliner.threshold", 0.5),
                max_chars=config.get("gliner.max_scan_chars"),
            )
            all_findings.extend(gliner_findings)
            if gliner_findings:
                scanners_used.append("gliner")
        except DlpScannerError as e:
            scanner_errors.append(f"gliner: {e}")
            logger.error("dlp_scanner_failed", scanner="gliner", error=str(e))

    # --- LLM contextual scanner ---
    if config.get("llm.enabled", False):
        complete = config.get("llm.complete")
        if complete is not None:
            try:
                llm_findings = await scan_llm(
                    text,
                    system_prompt=config.get("llm.system_prompt", ""),
                    model=config.get("llm.model", ""),
                    complete=complete,
                )
                all_findings.extend(llm_findings)
                if llm_findings:
                    scanners_used.append("llm")
            except DlpScannerError as e:
                scanner_errors.append(f"llm: {e}")
                logger.error("dlp_scanner_failed", scanner="llm", error=str(e))

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Genuinely clean only when a scan ran AND found nothing AND nothing errored.
    if not all_findings and not scanner_errors:
        return None

    severity = classify_severity(
        all_findings,
        severity_rules=config.get("severity_rules"),
    )

    # Build detail summary
    categories = list(set(f.category for f in all_findings))
    if all_findings:
        detail_parts = [f"{len(all_findings)} finding(s) across {', '.join(scanners_used)}"]
        detail_parts.append(f"Categories: {', '.join(categories)}")
    else:
        detail_parts = ["no findings"]
    if scanner_errors:
        detail_parts.append(f"scanner errors: {'; '.join(scanner_errors)}")
    detail = "; ".join(detail_parts)

    primary_scanner = scanners_used[-1] if scanners_used else "regex"

    return ScanResult(
        findings=all_findings,
        severity=severity,
        scan_latency_ms=elapsed_ms,
        scanner=primary_scanner,
        detail=detail,
        scanner_errors=scanner_errors,
    )
