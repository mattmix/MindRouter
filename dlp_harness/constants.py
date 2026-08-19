############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/constants.py: Category taxonomy, scanner
# capability maps, and DLP config-key inventory shared by
# every harness module.
#
# The harness measures the DLP subsystem in backend/app/
# (dlp_scanner.py + dlp_worker.py). Everything here mirrors
# facts in that code; each mirror notes its source so drift
# is auditable.
#
############################################################

"""Shared constants for the DLP evaluation harness."""

# ---------------------------------------------------------------------------
# Canonical entity categories
#
# Ground truth is labeled with these canonical keys. Scanner findings arrive
# with scanner-specific labels (regex categories, GLiNER labels, admin-defined
# custom categories); ALIASES maps every observed label back to a canonical
# key so metrics aggregate correctly across scanners.
# ---------------------------------------------------------------------------

SSN = "ssn"
CREDIT_CARD = "credit_card"
EMAIL = "email"
PHONE = "phone"
DATE_OF_BIRTH = "date_of_birth"
PERSON = "person"
DRIVER_LICENSE = "driver_license"
PASSPORT = "passport"
BANK_ACCOUNT = "bank_account"
MEDICAL_RECORD = "medical_record"
STUDENT_ID = "student_id"
KEYWORD = "keyword"

CANONICAL_CATEGORIES = [
    SSN, CREDIT_CARD, EMAIL, PHONE, DATE_OF_BIRTH, PERSON,
    DRIVER_LICENSE, PASSPORT, BANK_ACCOUNT, MEDICAL_RECORD, STUDENT_ID,
    KEYWORD,
]

# Scanner label -> canonical category. Keys are lowercased/stripped before
# lookup. Sources: _BUILTIN_PATTERNS categories (dlp_scanner.py) and the
# GLiNER default label list (dlp_scanner.py scan_gliner) plus the two extra
# labels the admin UI offers (dlp.html).
ALIASES = {
    "social security number": SSN,
    "ssn": SSN,
    "credit card number": CREDIT_CARD,
    "credit card": CREDIT_CARD,
    "email": EMAIL,
    "email address": EMAIL,
    "phone number": PHONE,
    "phone": PHONE,
    "date of birth": DATE_OF_BIRTH,
    "dob": DATE_OF_BIRTH,
    "person": PERSON,
    "person name": PERSON,
    "name": PERSON,
    "driver license number": DRIVER_LICENSE,
    "driver's license number": DRIVER_LICENSE,
    "drivers license": DRIVER_LICENSE,
    "passport number": PASSPORT,
    "bank account number": BANK_ACCOUNT,
    "bank account": BANK_ACCOUNT,
    "medical record number": MEDICAL_RECORD,
    "student id": STUDENT_ID,
    "keyword": KEYWORD,
}


def canonicalize(label: str):
    """Map a scanner-emitted category label to a canonical key (or None)."""
    if not label:
        return None
    return ALIASES.get(str(label).strip().lower())


# ---------------------------------------------------------------------------
# Scanner capability maps
#
# Which canonical categories each scanner is CAPABLE of detecting with the
# production default configuration. Used to split "missed because the scanner
# cannot see this category at all" from "missed despite being in scope" —
# recall against in-scope categories is the fair per-scanner number; recall
# against everything is the honest system number.
# ---------------------------------------------------------------------------

# Mirrors _BUILTIN_PATTERNS in dlp_scanner.py (5 built-in regexes).
REGEX_BUILTIN_SCOPE = {SSN, CREDIT_CARD, EMAIL, PHONE, DATE_OF_BIRTH}

# Which VARIANTS of each in-scope category the built-in regexes can actually
# match (mirrors _BUILTIN_PATTERNS in dlp_scanner.py against the generator
# variants in generators.py). None = every variant is visible. Categories
# absent from this map are never regex-visible (see REGEX_BUILTIN_SCOPE).
REGEX_VISIBLE_VARIANTS = {
    SSN: {"dashed"},              # \b\d{3}-\d{2}-\d{4}\b misses spaced/bare9
    DATE_OF_BIRTH: {"prefixed"},  # pattern requires the DOB/born-on prefix
    CREDIT_CARD: None,
    EMAIL: None,
    PHONE: None,
}


def regex_can_see(category, generator: str) -> bool:
    """True when the built-in regexes can possibly match this planted entity.

    ``generator`` is the ground-truth generator string ("<category>.<variant>",
    e.g. "ssn.dashed"). A bare generator with no variant suffix is treated as
    not visible unless the category's map value is None (all variants visible).
    """
    if category not in REGEX_BUILTIN_SCOPE:
        return False
    visible = REGEX_VISIBLE_VARIANTS.get(category)
    if visible is None:
        return True
    if not generator or "." not in generator:
        return False
    return generator.split(".", 1)[1] in visible


# Mirrors the default label list in scan_gliner() (dlp_scanner.py).
# "person" was dropped from the defaults 2026-08-19 (precision 0.34 — fires
# on section headers/greetings); it remains admin-configurable, so PERSON
# entities in the corpus are measured as out-of-scope for default configs.
GLINER_DEFAULT_SCOPE = {
    PHONE, EMAIL, CREDIT_CARD, SSN, DATE_OF_BIRTH,
    DRIVER_LICENSE, PASSPORT, BANK_ACCOUNT,
}

# Canonical -> the label string to pass to GLiNER for that category
# (inverse of ALIASES restricted to GLiNER's vocabulary).
GLINER_LABELS = {
    PERSON: "person",
    PHONE: "phone number",
    EMAIL: "email",
    CREDIT_CARD: "credit card number",
    SSN: "social security number",
    DATE_OF_BIRTH: "date of birth",
    DRIVER_LICENSE: "driver license number",
    PASSPORT: "passport number",
    BANK_ACCOUNT: "bank account number",
    MEDICAL_RECORD: "medical record number",
    STUDENT_ID: "student id",
}

# ---------------------------------------------------------------------------
# Severity model
#
# Mirrors _BUILTIN_PATTERNS severities + classify_severity() default
# ("moderate" for any category without a rule; "minor" for zero findings).
# ---------------------------------------------------------------------------

DEFAULT_SEVERITY_BY_CATEGORY = {
    SSN: "major",
    CREDIT_CARD: "major",
    EMAIL: "minor",
    PHONE: "minor",
    DATE_OF_BIRTH: "moderate",
}
SEVERITY_FALLBACK = "moderate"       # classify_severity default for unknowns
SEVERITY_ORDER = {"minor": 0, "moderate": 1, "major": 2}


def expected_severity(categories):
    """Severity the worker would assign for a set of canonical categories."""
    best = "minor"
    for c in categories:
        sev = DEFAULT_SEVERITY_BY_CATEGORY.get(c, SEVERITY_FALLBACK)
        if SEVERITY_ORDER[sev] > SEVERITY_ORDER[best]:
            best = sev
    return best


# Raw-label -> severity, in the exact shape run_dlp_scan/classify_severity
# expect for config["severity_rules"]: every alias a scanner can emit mapped
# to the severity the worker would assign its canonical category (unlisted
# labels fall back to SEVERITY_FALLBACK inside classify_severity).
SEVERITY_RULES_OVERRIDE = {
    raw: DEFAULT_SEVERITY_BY_CATEGORY[canonical]
    for raw, canonical in ALIASES.items()
    if canonical in DEFAULT_SEVERITY_BY_CATEGORY
}


# ---------------------------------------------------------------------------
# DLP configuration inventory (app_config keys)
#
# The complete key set written by POST /admin/dlp/config (dlp_routes.py),
# plus the worker-owned digest watermark. snapshot/restore must cover all of
# these; SAFE_RUN_OVERRIDES is applied during harness runs so a test never
# emails a human or silently loses alert rows to dedup.
# ---------------------------------------------------------------------------

DLP_CONFIG_KEYS = [
    "dlp.enabled",
    "dlp.regex.enabled",
    "dlp.regex.patterns",
    "dlp.regex.keywords",
    "dlp.gliner.enabled",
    "dlp.gliner.threshold",
    "dlp.gliner.categories",
    "dlp.gliner.max_scan_chars",
    "dlp.llm.enabled",
    "dlp.llm.model",
    "dlp.llm.system_prompt",
    "dlp.severity_rules",
    "dlp.dedup.enabled",
    "dlp.dedup.window_seconds",
    "dlp.email.minor_recipients",
    "dlp.email.moderate_recipients",
    "dlp.email.major_recipients",
    "dlp.email.minor.mode",
    "dlp.email.moderate.mode",
    "dlp.email.major.mode",
    "dlp.digest.frequency",
    "dlp.digest.recipients",
    "dlp.digest.last_sent_at",
]

# Applied on top of whatever scanner matrix a run selects.
SAFE_RUN_OVERRIDES = {
    "dlp.dedup.enabled": False,          # dedup suppresses alert rows -> breaks coverage math
    "dlp.email.minor.mode": "off",
    "dlp.email.moderate.mode": "off",
    "dlp.email.major.mode": "off",
    "dlp.email.minor_recipients": "",
    "dlp.email.moderate_recipients": "",
    "dlp.email.major_recipients": "",
    "dlp.digest.recipients": "",
}

# Scanner matrices used by e2e/load runs. Each entry fully determines the
# scanner-enable keys (a partial write could inherit stale state).
SCANNER_MODES = {
    "off":    {"dlp.enabled": False},
    "regex":  {"dlp.enabled": True, "dlp.regex.enabled": True,
               "dlp.gliner.enabled": False, "dlp.llm.enabled": False},
    "gliner": {"dlp.enabled": True, "dlp.regex.enabled": False,
               "dlp.gliner.enabled": True, "dlp.llm.enabled": False},
    "regex+gliner": {"dlp.enabled": True, "dlp.regex.enabled": True,
                     "dlp.gliner.enabled": True, "dlp.llm.enabled": False},
}

# ---------------------------------------------------------------------------
# Pipeline facts the harness depends on (sources in backend/app/)
# ---------------------------------------------------------------------------

# dlp_worker.py: asyncio.Queue(maxsize=10_000), put_nowait drops on full.
DLP_QUEUE_MAXSIZE = 10_000
# dlp_scanner.py: global truncation before any scanner runs.
MAX_SCAN_CHARS = 200_000
# dlp_scanner.py: GLiNER-specific default prefix cap.
GLINER_DEFAULT_MAX_CHARS = 10_000
# dlp_worker.py: stored entities per alert row are capped, then masked.
MAX_STORED_ENTITIES = 50
# Alert rows for scanner failures carry this category (exclude from accuracy).
SCANNER_ERROR_CATEGORY = "dlp_scanner_error"
