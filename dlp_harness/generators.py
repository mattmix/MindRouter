############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/generators.py: Deterministic entity and
# hard-negative text generators for the synthetic DLP
# corpus.
#
############################################################

"""Entity and hard-negative generators for the DLP corpus.

Every generator takes an explicit ``random.Random`` and returns
``(text, attrs)``. ``attrs["variant"]`` names the format so corpus code can
tag ground truth as ``"<category>.<variant>"``. Variant coverage mirrors what
the production scanners can and cannot see (the _BUILTIN_PATTERNS regexes in
backend/app/services/dlp_scanner.py, and GLiNER's semantic NER): e.g. only a
dashed SSN is regex-visible, while spaced/bare SSNs are GLiNER-only cases.

Hard negatives live in NEGATIVE_GENERATORS: PII lookalikes (a Luhn-invalid
16-digit number still matches the built-in credit-card regex) used to
attribute false positives to the trap shape that caused them. Each negative's
attrs carry {"mimics": <canonical category>}.
"""

import random
from typing import Any, Dict, Optional, Tuple

from dlp_harness import constants

_UPPER = "ABCDEFGHJKLMNPRSTUVWXYZ"
_ALNUM = _UPPER + "0123456789"

FIRST_NAMES = (
    "James", "Maria", "Wei", "Aisha", "Carlos", "Yuki", "Priya", "Liam",
    "Sofia", "Dmitri", "Fatima", "Kwame", "Ingrid", "Hiroshi", "Amara",
    "Diego", "Chen", "Leila", "Marcus", "Anya", "Tariq", "Elena", "Raj",
    "Nadia", "Owen", "Zainab", "Pedro", "Mei", "Kofi", "Astrid", "Omar",
    "Lucia", "Sven", "Rosa", "Jamal", "Hana", "Viktor", "Esme", "Andre",
    "Keiko",
)
LAST_NAMES = (
    "Smith", "Garcia", "Chen", "Okafor", "Patel", "Kim", "Nguyen",
    "Johansson", "Alvarez", "Kowalski", "Tanaka", "Osei", "Haddad", "Silva",
    "Novak", "Ivanov", "Mbeki", "Larsen", "Romero", "Singh", "Yamamoto",
    "Abara", "Costa", "Petrov", "Nakamura", "Diallo", "Berg", "Moreau",
    "Santos", "Volkov", "Kaur", "Eriksen", "Ortiz", "Hassan", "Lindqvist",
    "Vega", "Popov", "Adeyemi", "Fujita", "Weiss",
)


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(n))


# ---------------------------------------------------------------------------
# Luhn (credit cards) and ABA (routing numbers) checksums
# ---------------------------------------------------------------------------

def luhn_check_digit(partial: str) -> str:
    """Check digit that makes ``partial + digit`` Luhn-valid."""
    total = 0
    for i, ch in enumerate(reversed(partial)):
        d = int(ch)
        if i % 2 == 0:  # doubled positions once the check digit is appended
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_ABA_WEIGHTS = (3, 7, 1, 3, 7, 1, 3, 7, 1)


def aba_valid(digits: str) -> bool:
    if len(digits) != 9 or not digits.isdigit():
        return False
    return sum(int(c) * w for c, w in zip(digits, _ABA_WEIGHTS)) % 10 == 0


def _aba_check_digit(eight: str) -> str:
    total = sum(int(c) * w for c, w in zip(eight, _ABA_WEIGHTS[:8]))
    return str((10 - total % 10) % 10)


# ---------------------------------------------------------------------------
# Entity generators (one per canonical category)
# ---------------------------------------------------------------------------

_SSN_VARIANTS = ("dashed", "spaced", "bare9")


def gen_ssn(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(_SSN_VARIANTS)
    # Realistic area numbers: never 000, 666, or 900-999.
    area = rng.randint(1, 899)
    while area == 666:
        area = rng.randint(1, 899)
    a, g, s = f"{area:03d}", f"{rng.randint(1, 99):02d}", f"{rng.randint(1, 9999):04d}"
    if variant == "dashed":
        text = f"{a}-{g}-{s}"
    elif variant == "spaced":
        text = f"{a} {g} {s}"
    else:
        text = a + g + s
    return text, {"variant": variant}


_CC_NETWORKS = ("visa", "mastercard", "amex", "discover")
_CC_FORMATS = ("plain", "spaced", "hyphen")


def gen_credit_card(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    if variant:
        network, fmt = variant.rsplit("_", 1)
    else:
        network, fmt = rng.choice(_CC_NETWORKS), rng.choice(_CC_FORMATS)
    if network == "visa":
        prefix, length = "4", 16
    elif network == "mastercard":
        prefix, length = "5" + str(rng.randint(1, 5)), 16
    elif network == "amex":
        prefix, length = rng.choice(("34", "37")), 15
    else:
        prefix, length = "6011", 16
    body = prefix + _digits(rng, length - len(prefix) - 1)
    digits = body + luhn_check_digit(body)
    if fmt == "plain":
        text = digits
    else:
        sep = " " if fmt == "spaced" else "-"
        if network == "amex":  # 4-6-5 grouping
            text = sep.join((digits[:4], digits[4:10], digits[10:]))
        else:                  # 4-4-4-4 grouping
            text = sep.join((digits[:4], digits[4:8], digits[8:12], digits[12:]))
    return text, {"variant": f"{network}_{fmt}", "network": network, "luhn_valid": True}


_EMAIL_DOMAINS = ("example.com", "gmail.com", "outlook.com", "protonmail.com")
_EMAIL_VARIANTS = ("plain", "dotted", "subaddressed", "uidaho", "mixed_case")


def gen_email(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(_EMAIL_VARIANTS)
    first = rng.choice(FIRST_NAMES).lower()
    last = rng.choice(LAST_NAMES).lower()
    domain = rng.choice(_EMAIL_DOMAINS)
    if variant == "plain":
        text = f"{first[0]}{last}@{domain}"
    elif variant == "dotted":
        text = f"{first}.{last}@{domain}"
    elif variant == "subaddressed":
        text = f"{first}.{last}+{rng.choice(('news', 'work', 'alerts', 'lists'))}@{domain}"
    elif variant == "uidaho":
        text = f"{first[0]}{last}@uidaho.edu"
    else:
        text = f"{first.capitalize()}{last.capitalize()}@{domain.upper()}"
    return text, {"variant": variant}


_PHONE_VARIANTS = ("parens", "dashed", "dotted", "plus1", "bare10")
_AREA_CODES = (208, 206, 212, 303, 406, 425, 509, 541, 602, 702)


def gen_phone(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(_PHONE_VARIANTS)
    a = str(rng.choice(_AREA_CODES))
    e = str(rng.randint(200, 999))
    l = f"{rng.randint(0, 9999):04d}"
    if variant == "parens":
        text = f"({a}) {e}-{l}"
    elif variant == "dashed":
        text = f"{a}-{e}-{l}"
    elif variant == "dotted":
        text = f"{a}.{e}.{l}"
    elif variant == "plus1":
        text = f"+1-{a}-{e}-{l}"
    else:
        text = a + e + l
    return text, {"variant": variant}


_DOB_PREFIXES = ("DOB: ", "date of birth: ", "born on ")


def gen_date_of_birth(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Prefixed variants mirror the built-in regex, which matches prefix+date;
    the bare variant is a date alone (GLiNER-only case), span = just the date."""
    variant = variant or rng.choice(("prefixed", "bare"))
    sep = rng.choice(("/", "-"))
    pad = rng.random() < 0.5
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    year = rng.randint(1940, 2007)
    y = f"{year % 100:02d}" if rng.random() < 0.3 else str(year)
    date = f"{m:02d}{sep}{d:02d}{sep}{y}" if pad else f"{m}{sep}{d}{sep}{y}"
    if variant == "prefixed":
        prefix = rng.choice(_DOB_PREFIXES)
        return prefix + date, {"variant": "prefixed", "prefix": prefix.strip(": ")}
    return date, {"variant": "bare"}


_PERSON_VARIANTS = ("first_last", "last_first", "titled")


def gen_person(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(_PERSON_VARIANTS)
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    if variant == "last_first":
        text = f"{last}, {first}"
    elif variant == "titled":
        text = f"{rng.choice(('Dr.', 'Prof.', 'Ms.', 'Mr.'))} {first} {last}"
    else:
        text = f"{first} {last}"
    return text, {"variant": variant}


def gen_driver_license(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(("idaho", "california", "newyork", "washington"))
    if variant == "idaho":
        text, state, fmt = rng.choice(_UPPER) + _digits(rng, 9), "ID", "L9"
    elif variant == "california":
        text, state, fmt = rng.choice(_UPPER) + _digits(rng, 7), "CA", "L7"
    elif variant == "newyork":
        text, state, fmt = str(rng.randint(1, 9)) + _digits(rng, 8), "NY", "9"
    else:
        text = "WDL" + "".join(rng.choice(_ALNUM) for _ in range(9))
        state, fmt = "WA", "WDL+9AN"
    return text, {"variant": variant, "state": state, "format": fmt}


def gen_passport(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(("digits9", "letter8"))
    if variant == "digits9":
        text, fmt = str(rng.randint(1, 9)) + _digits(rng, 8), "9"
    else:
        text, fmt = rng.choice(_UPPER) + _digits(rng, 8), "L8"
    return text, {"variant": variant, "format": fmt}


_ABA_PREFIXES = tuple(f"{n:02d}" for n in list(range(1, 13)) + list(range(21, 33)))


def gen_bank_account(rng: random.Random, variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    variant = variant or rng.choice(("digits", "digits", "aba_routing"))
    if variant == "aba_routing":
        body = rng.choice(_ABA_PREFIXES) + _digits(rng, 6)
        return body + _aba_check_digit(body), {"variant": variant, "format": "aba9", "aba_valid": True}
    n = rng.randint(8, 12)
    text = str(rng.randint(1, 9)) + _digits(rng, n - 1)
    return text, {"variant": "digits", "format": f"digits{n}"}


GENERATORS = {
    constants.SSN: gen_ssn,
    constants.CREDIT_CARD: gen_credit_card,
    constants.EMAIL: gen_email,
    constants.PHONE: gen_phone,
    constants.DATE_OF_BIRTH: gen_date_of_birth,
    constants.PERSON: gen_person,
    constants.DRIVER_LICENSE: gen_driver_license,
    constants.PASSPORT: gen_passport,
    constants.BANK_ACCOUNT: gen_bank_account,
}


# ---------------------------------------------------------------------------
# Hard negatives (false-positive traps)
# ---------------------------------------------------------------------------

def neg_luhn_invalid_16(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    """16 digits that FAIL Luhn — still matches the built-in CC regex."""
    body = "4" + _digits(rng, 14)
    bad = str((int(luhn_check_digit(body)) + rng.randint(1, 9)) % 10)
    digits = body + bad
    if rng.random() < 0.5:
        digits = " ".join((digits[:4], digits[4:8], digits[8:12], digits[12:]))
    return digits, {"mimics": constants.CREDIT_CARD, "luhn_valid": False}


def _ean_check_digit(body: str) -> str:
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(body))
    return str((10 - total % 10) % 10)


def neg_isbn13(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    d = "978" + _digits(rng, 9)
    d += _ean_check_digit(d)
    text = f"{d[:3]}-{d[3]}-{d[4:7]}-{d[7:12]}-{d[12]}"
    return text, {"mimics": constants.CREDIT_CARD, "kind": "isbn13"}


def neg_ean13_barcode(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    body = _digits(rng, 12)
    return body + _ean_check_digit(body), {"mimics": constants.CREDIT_CARD, "kind": "ean13"}


def neg_order_id_16digit(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    return "ORD-" + _digits(rng, 16), {"mimics": constants.CREDIT_CARD, "kind": "order_id"}


def neg_tracking_number(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    if rng.random() < 0.5:
        text = "1Z" + "".join(rng.choice(_ALNUM) for _ in range(6)) + _digits(rng, 10)
        kind = "ups"
    else:
        text, kind = _digits(rng, 15), "fedex15"
    return text, {"mimics": constants.CREDIT_CARD, "kind": kind}


def neg_version_string(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    text = f"{rng.randint(1, 12)}.{rng.randint(0, 30)}.{rng.randint(0, 9999)}"
    return text, {"mimics": constants.PHONE, "kind": "semver"}


def neg_uuid4(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    def hx(n):
        return "".join(rng.choice("0123456789abcdef") for _ in range(n))
    text = f"{hx(8)}-{hx(4)}-4{hx(3)}-{rng.choice('89ab')}{hx(3)}-{hx(12)}"
    return text, {"mimics": constants.BANK_ACCOUNT, "kind": "uuid4"}


def neg_git_sha(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    text = "".join(rng.choice("0123456789abcdef") for _ in range(40))
    return text, {"mimics": constants.BANK_ACCOUNT, "kind": "git_sha"}


def neg_iso_timestamp(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    text = (f"{rng.randint(2019, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z")
    return text, {"mimics": constants.DATE_OF_BIRTH, "kind": "iso8601"}


def neg_ip_address(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    text = f"{rng.choice((10, 172, 192))}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return text, {"mimics": constants.PHONE, "kind": "ipv4"}


def neg_us_zip_plus4(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    return f"{rng.randint(10000, 99999)}-{rng.randint(0, 9999):04d}", {"mimics": constants.SSN, "kind": "zip+4"}


def neg_part_number_dashed(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    # SSN-ish shape but 2-3-4 digit groups instead of 3-2-4.
    text = f"{rng.randint(10, 99)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    return text, {"mimics": constants.SSN, "kind": "part_number"}


def neg_fraction_string(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    return f"{rng.randint(1, 31)}/{rng.randint(2, 32)}", {"mimics": constants.DATE_OF_BIRTH, "kind": "fraction"}


def neg_phone_like_1900(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    """13-digit international-looking number (non-US) — also 13 digits, so it
    lands inside the credit-card regex's 13-19 window."""
    cc = rng.choice(("86", "44", "971", "353"))
    return "+" + cc + _digits(rng, 13 - len(cc)), {"mimics": constants.PHONE, "kind": "intl13"}


NEGATIVE_GENERATORS = {
    "luhn_invalid_16": neg_luhn_invalid_16,
    "isbn13": neg_isbn13,
    "ean13_barcode": neg_ean13_barcode,
    "order_id_16digit": neg_order_id_16digit,
    "tracking_number": neg_tracking_number,
    "version_string": neg_version_string,
    "uuid4": neg_uuid4,
    "git_sha": neg_git_sha,
    "iso_timestamp": neg_iso_timestamp,
    "ip_address": neg_ip_address,
    "us_zip_plus4": neg_us_zip_plus4,
    "part_number_dashed": neg_part_number_dashed,
    "fraction_string": neg_fraction_string,
    "phone_like_1900": neg_phone_like_1900,
}
