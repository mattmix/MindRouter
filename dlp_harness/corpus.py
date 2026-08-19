############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/corpus.py: Synthetic labeled-corpus builder —
# DocBuilder (span-exact assembly), carrier templates,
# obfuscation transforms, corpus profiles, and generate().
#
############################################################

"""Synthetic corpus generation for the DLP harness.

Documents are assembled by DocBuilder, which records entity char offsets at
append time (never recovered post-hoc with str.find, which would mis-span
duplicate substrings). Carrier templates wrap planted entities in realistic
prose with natural lead-ins ("my ssn is", "card:") because GLiNER is
context-sensitive; all carrier prose and filler is deliberately free of
names, emails, and phone-like digit runs so ground truth stays complete —
an unlabeled real-looking name would turn a correct GLiNER hit into a
phantom false positive.

Profiles: "accuracy" (mixed dirty/hard/clean/trap), "adversarial"
(obfuscated + boundary placements), "load" (chat-sized docs with unique
entity values), "scale" (length sweep with entities planted near the start
and at ~95% depth so scanner truncation blindness is measurable).
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dlp_harness import constants, generators
from dlp_harness.schemas import GroundTruthEntity, LabeledDocument, write_jsonl


# ---------------------------------------------------------------------------
# DocBuilder
# ---------------------------------------------------------------------------

class DocBuilder:
    """Assembles document text while tracking exact entity spans."""

    def __init__(self, doc_id: str, profile: str, carrier: str, seed: int):
        self.doc_id = doc_id
        self.profile = profile
        self.carrier = carrier
        self.seed = seed
        self._parts: List[str] = []
        self._pos = 0
        self._entities: List[GroundTruthEntity] = []
        self._negatives: List[GroundTruthEntity] = []

    @property
    def length(self) -> int:
        return self._pos

    def append(self, text: str) -> None:
        self._parts.append(text)
        self._pos += len(text)

    def append_entity(self, category: str, text: str, generator: str = "",
                      difficulty: str = "plain", obfuscation: str = "",
                      attrs: Optional[Dict[str, Any]] = None) -> GroundTruthEntity:
        e = GroundTruthEntity(category=category, text=text, start=self._pos,
                              end=self._pos + len(text), generator=generator,
                              difficulty=difficulty, obfuscation=obfuscation,
                              attrs=dict(attrs or {}))
        self._entities.append(e)
        self.append(text)
        return e

    def append_negative(self, category: str, text: str, generator: str = "",
                        attrs: Optional[Dict[str, Any]] = None) -> GroundTruthEntity:
        e = GroundTruthEntity(category=category, text=text, start=self._pos,
                              end=self._pos + len(text), generator=generator,
                              difficulty="plain", attrs=dict(attrs or {}))
        self._negatives.append(e)
        self.append(text)
        return e

    def build(self, meta: Optional[Dict[str, Any]] = None) -> LabeledDocument:
        text = "".join(self._parts)
        for e in self._entities + self._negatives:
            assert text[e.start:e.end] == e.text, (
                f"span corruption in {self.doc_id}: [{e.start}:{e.end}] != {e.text!r}")
        return LabeledDocument(doc_id=self.doc_id, text=text,
                               entities=list(self._entities),
                               negatives=list(self._negatives),
                               profile=self.profile, carrier=self.carrier,
                               seed=self.seed, meta=dict(meta or {}))


# ---------------------------------------------------------------------------
# Obfuscation transforms (adversarial profile)
# ---------------------------------------------------------------------------

_FULLWIDTH = {ord(str(i)): chr(0xFF10 + i) for i in range(10)}
_DIGIT_WORDS = ("zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine")
# Cyrillic lookalikes, written as escapes so the source is unambiguous.
_HOMOGLYPHS = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "c": "\u0441",
               "p": "\u0440", "x": "\u0445", "A": "\u0410", "E": "\u0415",
               "O": "\u041e", "C": "\u0421", "P": "\u0420", "X": "\u0425"}


def ob_spaced_digits(text: str, rng: random.Random) -> str:
    return " ".join(text)


def ob_fullwidth_digits(text: str, rng: random.Random) -> str:
    return text.translate(_FULLWIDTH)


def ob_zero_width_space(text: str, rng: random.Random) -> str:
    n = rng.randint(1, min(3, len(text) - 1))
    for p in sorted(rng.sample(range(1, len(text)), n), reverse=True):
        text = text[:p] + "\u200b" + text[p:]
    return text


def ob_at_dot_spelled(text: str, rng: random.Random) -> str:
    return text.replace("@", " [at] ").replace(".", " [dot] ")


def ob_digit_words(text: str, rng: random.Random) -> str:
    out, replaced = [], 0
    for ch in text:
        if ch.isdigit() and replaced < 4:
            out.append(" " + _DIGIT_WORDS[int(ch)] + " ")
            replaced += 1
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def ob_newline_split(text: str, rng: random.Random) -> str:
    p = rng.randint(1, len(text) - 1)
    return text[:p] + "\n" + text[p:]


def ob_homoglyph(text: str, rng: random.Random) -> str:
    hits = [i for i, ch in enumerate(text) if ch in _HOMOGLYPHS]
    if not hits:
        return text
    chars = list(text)
    for i in rng.sample(hits, max(1, len(hits) // 3)):
        chars[i] = _HOMOGLYPHS[chars[i]]
    return "".join(chars)


OBFUSCATIONS: Dict[str, Callable[[str, random.Random], str]] = {
    "spaced_digits": ob_spaced_digits,
    "fullwidth_digits": ob_fullwidth_digits,
    "zero_width_space": ob_zero_width_space,
    "at_dot_spelled": ob_at_dot_spelled,
    "digit_words": ob_digit_words,
    "newline_split": ob_newline_split,
    "homoglyph": ob_homoglyph,
}

_ALL_ENTITY_CATEGORIES = sorted(generators.GENERATORS)
_DIGIT_CATEGORIES = {constants.SSN, constants.CREDIT_CARD, constants.PHONE,
                     constants.DATE_OF_BIRTH, constants.BANK_ACCOUNT,
                     constants.PASSPORT, constants.DRIVER_LICENSE}

OBFUSCATION_APPLICABLE = {
    "spaced_digits": _DIGIT_CATEGORIES,
    "fullwidth_digits": _DIGIT_CATEGORIES,
    "zero_width_space": set(_ALL_ENTITY_CATEGORIES),
    "at_dot_spelled": {constants.EMAIL},
    "digit_words": _DIGIT_CATEGORIES,
    "newline_split": set(_ALL_ENTITY_CATEGORIES),
    "homoglyph": {constants.EMAIL, constants.PERSON},
}


def obfuscations_for(category: str) -> List[str]:
    return sorted(n for n, cats in OBFUSCATION_APPLICABLE.items() if category in cats)


# ---------------------------------------------------------------------------
# Slots (items a carrier plants) and lead-in phrasing
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    """One item for a carrier to plant: a labeled entity or a negative trap."""
    kind: str                # "entity" | "negative"
    category: str            # canonical category (for negatives: what it mimics)
    text: str
    generator: str = ""
    difficulty: str = "plain"
    obfuscation: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)


def _entity_slot(rng: random.Random, category: str, variant: Optional[str] = None,
                 difficulty: str = "plain", obfuscation: str = "",
                 extra_attrs: Optional[Dict[str, Any]] = None) -> Slot:
    text, attrs = generators.GENERATORS[category](rng, variant)
    if extra_attrs:
        attrs.update(extra_attrs)
    if obfuscation:
        text = OBFUSCATIONS[obfuscation](text, rng)
        difficulty = "obfuscated"
    return Slot("entity", category, text, f"{category}.{attrs['variant']}",
                difficulty, obfuscation, attrs)


_NEGATIVE_NAMES = sorted(generators.NEGATIVE_GENERATORS)


def _negative_slot(rng: random.Random, name: Optional[str] = None) -> Slot:
    name = name or rng.choice(_NEGATIVE_NAMES)
    text, attrs = generators.NEGATIVE_GENERATORS[name](rng)
    return Slot("negative", attrs["mimics"], text, f"negative.{name}", attrs=attrs)


def _plant(b: DocBuilder, slot: Slot) -> GroundTruthEntity:
    if slot.kind == "entity":
        return b.append_entity(slot.category, slot.text, generator=slot.generator,
                               difficulty=slot.difficulty,
                               obfuscation=slot.obfuscation, attrs=slot.attrs)
    return b.append_negative(slot.category, slot.text,
                             generator=slot.generator, attrs=slot.attrs)


# Natural lead-ins per category: GLiNER is context-sensitive, so entities are
# introduced the way a real chat/document would introduce them.  DOB leads
# deliberately avoid the regex prefixes (DOB / date of birth / born on) —
# the "prefixed" DOB variant carries its own regex-triggering prefix.
_LEAD_INS = {
    constants.SSN: ("my ssn is ", "my social security number is ", "SSN ", "social security no. "),
    constants.CREDIT_CARD: ("card: ", "my card number is ", "charge it to ", "the visa on file is "),
    constants.EMAIL: ("email me at ", "reach me at ", "contact email: ", "send it to "),
    constants.PHONE: ("call me at ", "phone: ", "my cell is ", "you can reach the desk at "),
    constants.DATE_OF_BIRTH: ("birthday ", "birthdate ", "patient birthdate ", "for verification, "),
    constants.PERSON: ("this is regarding ", "please contact ", "prepared for ", "on behalf of "),
    constants.DRIVER_LICENSE: ("driver's license ", "my license number is ", "DL# "),
    constants.PASSPORT: ("passport number ", "my passport is ", "passport no. "),
    constants.BANK_ACCOUNT: ("account number ", "deposit to account ", "acct "),
}

_NEG_LEADS = {
    "luhn_invalid_16": ("test card ", "sandbox card number ", "use dummy card "),
    "isbn13": ("the textbook is ISBN ", "cite ISBN ", "ISBN "),
    "ean13_barcode": ("shelf barcode ", "EAN ", "scanned code "),
    "order_id_16digit": ("order ", "order id ", "purchase reference "),
    "tracking_number": ("tracking ", "shipment ", "package tracking id "),
    "version_string": ("running version ", "pinned at ", "upgraded to "),
    "uuid4": ("request id ", "trace ", "correlation id "),
    "git_sha": ("commit ", "deployed sha ", "built from "),
    "iso_timestamp": ("first seen at ", "logged at ", "timestamp "),
    "ip_address": ("host ", "from ip ", "gateway "),
    "us_zip_plus4": ("zip ", "mailing zip ", "postal code "),
    "part_number_dashed": ("part ", "part number ", "SKU "),
    "fraction_string": ("torque spec ", "drill size ", "measured "),
    "phone_like_1900": ("international support line ", "overseas desk ", "dial "),
}


def _lead_for(rng: random.Random, slot: Slot) -> str:
    if slot.kind == "negative":
        name = slot.generator.split(".", 1)[-1]
        return rng.choice(_NEG_LEADS.get(name, ("for reference ",)))
    return rng.choice(_LEAD_INS.get(slot.category, ("",)))


# Filler prose: deliberately digit-free and person-free (see module docstring).
_FILLER = (
    "The migration finished ahead of schedule and the dashboards look stable.",
    "We should circle back on the storage quota discussion after the maintenance window.",
    "The team agreed to keep the rollout behind a feature flag for now.",
    "Documentation for the new pipeline is still a work in progress.",
    "Latency has been flat all week, which matches what the cache metrics show.",
    "The vendor call went well and the renewal terms look reasonable.",
    "Please review the draft policy before the next standup.",
    "The lab instruments were recalibrated and results are consistent again.",
    "Nothing unusual showed up in the overnight batch run.",
    "The training session covered backup procedures and incident response.",
    "Budget planning for next quarter starts after the retreat.",
    "The greenhouse sensors were replaced and readings look normal.",
)


def _filler(rng: random.Random) -> str:
    return rng.choice(_FILLER)


# Config-style key names for code/json carriers, per category.
_CFG_KEYS = {
    constants.SSN: "user_ssn",
    constants.CREDIT_CARD: "payment_card",
    constants.EMAIL: "contact_email",
    constants.PHONE: "owner_phone",
    constants.DATE_OF_BIRTH: "dob",
    constants.PERSON: "account_holder",
    constants.DRIVER_LICENSE: "license_no",
    constants.PASSPORT: "passport_no",
    constants.BANK_ACCOUNT: "account_no",
}


def _cfg_key(slot: Slot) -> str:
    if slot.kind == "negative":
        return slot.generator.split(".", 1)[-1]
    return _CFG_KEYS.get(slot.category, "value")


# ---------------------------------------------------------------------------
# Carrier templates
# ---------------------------------------------------------------------------

def carrier_casual_chat(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append(rng.choice(("hey, quick question. ", "hi! hope your week is going ok. ",
                         "hello - following up from earlier. ")))
    b.append(_filler(rng) + " ")
    for slot in slots:
        b.append(_lead_for(rng, slot))
        _plant(b, slot)
        b.append(rng.choice((". ", " - thanks. ", ", if that helps. ")))
    b.append(rng.choice(("let me know if you need anything else.",
                         "appreciate the help!", "talk soon.")))


def carrier_support_ticket(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("Subject: " + rng.choice(("account access issue", "billing question",
                                       "password reset loop", "duplicate charge")) + "\n")
    b.append("Priority: " + rng.choice(("low", "normal", "high")) + "\n\n")
    b.append("Customer reports: " + _filler(rng) + "\n")
    for slot in slots:
        b.append("Customer provided " + _lead_for(rng, slot))
        _plant(b, slot)
        b.append(" for account verification.\n")
    b.append("Next step: " + rng.choice(("escalate to tier two.", "await customer reply.",
                                         "close as resolved.")))


def carrier_hr_email(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("From: benefits office\nTo: payroll team\n")
    b.append("Subject: " + rng.choice(("onboarding paperwork", "direct deposit update",
                                       "benefits enrollment")) + "\n\n")
    b.append("Hello,\n\n" + _filler(rng) + " ")
    for slot in slots:
        b.append("For the enrollment record, " + _lead_for(rng, slot))
        _plant(b, slot)
        b.append(". ")
    b.append("\n\nThanks,\nHR Operations")


def carrier_medical_note(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("CLINIC INTAKE NOTE\n")
    b.append("Chief complaint: " + rng.choice(("persistent cough", "follow-up visit",
                                               "annual physical", "knee pain")) + "\n")
    b.append("History: " + _filler(rng) + "\n")
    for slot in slots:
        b.append("Intake field - " + _lead_for(rng, slot))
        _plant(b, slot)
        b.append("\n")
    b.append("Plan: " + rng.choice(("routine labs ordered.", "follow up in six weeks.",
                                    "referred to physical therapy.")))


def carrier_code_snippet(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    style = rng.choice(("env", "yaml", "python"))
    if style == "env":
        b.append("# service configuration\nAPP_ENV=production\nLOG_LEVEL=info\n")
        for slot in slots:
            b.append(_cfg_key(slot).upper() + "=")
            _plant(b, slot)
            b.append("\n")
        b.append("FEATURE_FLAGS=default\n")
    elif style == "yaml":
        b.append("service:\n  name: gateway\n  tier: standard\n")
        for slot in slots:
            b.append("  " + _cfg_key(slot) + ": \"")
            _plant(b, slot)
            b.append("\"\n")
        b.append("  retries: none\n")
    else:
        b.append("# fixture data for the staging environment\nconfig = {\n")
        for slot in slots:
            b.append("    \"" + _cfg_key(slot) + "\": \"")
            _plant(b, slot)
            b.append("\",\n")
        b.append("}\n")


def carrier_log_excerpt(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("[info] service=gateway msg=\"request accepted\"\n")
    b.append("[info] " + _filler(rng) + "\n")
    for slot in slots:
        b.append("[warn] scrub_miss field=" + _cfg_key(slot) + " value=")
        _plant(b, slot)
        b.append("\n")
    b.append("[info] service=gateway msg=\"request completed\"\n")


def carrier_resume(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("OBJECTIVE\n" + rng.choice((
        "Seeking a systems administration role with a research focus.",
        "Looking to grow into a data engineering position.",
        "Interested in laboratory operations and compliance work.")) + "\n\nCONTACT\n")
    for slot in slots:
        b.append(_lead_for(rng, slot))
        _plant(b, slot)
        b.append("\n")
    b.append("\nEXPERIENCE\n" + _filler(rng) + "\n" + _filler(rng))


def carrier_meeting_notes(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("Meeting notes - " + rng.choice(("infrastructure sync", "security review",
                                              "quarterly planning")) + "\n")
    b.append("Attendees: platform team, security team\n\n")
    b.append("- " + _filler(rng) + "\n")
    for slot in slots:
        b.append("- action item: record " + _lead_for(rng, slot))
        _plant(b, slot)
        b.append("\n")
    b.append("- " + _filler(rng) + "\n")


def carrier_csv_dump(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("record_id,field,value,notes\n")
    row = 1
    for slot in slots:
        b.append(f"r{row:03d}," + _cfg_key(slot) + ",")
        _plant(b, slot)
        b.append("," + rng.choice(("imported", "verified", "pending review")) + "\n")
        row += 1
    for _ in range(rng.randint(1, 3)):   # benign rows so clean dumps look real
        b.append(f"r{row:03d},status," + rng.choice(("ok", "pending", "archived"))
                 + "," + rng.choice(("no action", "routine", "checked")) + "\n")
        row += 1


def carrier_json_payload(b: DocBuilder, rng: random.Random, slots: List[Slot]) -> None:
    b.append("{\n  \"kind\": \"" + rng.choice(("profile_update", "intake_form",
                                               "billing_event")) + "\",\n")
    b.append("  \"note\": \"" + _filler(rng) + "\",\n  \"fields\": {\n")
    for i, slot in enumerate(slots):
        b.append("    \"" + _cfg_key(slot) + "\": \"")
        _plant(b, slot)
        b.append("\"" + ("," if i < len(slots) - 1 else "") + "\n")
    b.append("  }\n}")


CARRIERS: Dict[str, Callable[[DocBuilder, random.Random, List[Slot]], None]] = {
    "casual_chat": carrier_casual_chat,
    "support_ticket": carrier_support_ticket,
    "hr_email": carrier_hr_email,
    "medical_note": carrier_medical_note,
    "code_snippet": carrier_code_snippet,
    "log_excerpt": carrier_log_excerpt,
    "resume": carrier_resume,
    "meeting_notes": carrier_meeting_notes,
    "csv_dump": carrier_csv_dump,
    "json_payload": carrier_json_payload,
}

_CARRIER_NAMES = sorted(CARRIERS)
# Prose carriers give obfuscated entities (which may contain newlines) a home
# where a line break is not structurally illegal, unlike csv/json/code.
_PROSE_CARRIERS = ("casual_chat", "support_ticket", "hr_email", "medical_note",
                   "resume", "meeting_notes")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

PROFILES = ("accuracy", "adversarial", "load", "scale")
SCALE_LENGTHS = [100, 300, 1000, 3000, 10000, 30000, 100000, 250000]

# Variants the built-in regexes cannot see (plus GLiNER-only categories):
# the "dirty-hard" bucket of the accuracy profile.
_HARD_KINDS = (
    (constants.SSN, "spaced"), (constants.SSN, "bare9"),
    (constants.DATE_OF_BIRTH, "bare"), (constants.PERSON, None),
    (constants.DRIVER_LICENSE, None), (constants.PASSPORT, None),
    (constants.BANK_ACCOUNT, None),
)

# Load profile plants regex-visible kinds only, so an "expected_alert" doc is
# detectable by every scanner mode the load run may select.
_LOAD_KINDS = ((constants.SSN, "dashed"), (constants.CREDIT_CARD, None),
               (constants.EMAIL, None), (constants.PHONE, None))


def _doc_id(profile: str, seed: int, index: int) -> str:
    return f"{profile}-{seed}-{index:06d}"


def _pad_prose(b: DocBuilder, rng: random.Random, target: int) -> None:
    while b.length < target:
        b.append(_filler(rng) + " ")


def _gen_accuracy(size: int, seed: int) -> List[LabeledDocument]:
    rng = random.Random(seed)
    docs = []
    for i in range(size):
        bucket = i % 10  # 0-3 dirty, 4-5 dirty-hard, 6-7 clean, 8-9 clean+traps
        carrier = rng.choice(_CARRIER_NAMES)
        b = DocBuilder(_doc_id("accuracy", seed, i), "accuracy", carrier, seed)
        if bucket <= 3:
            cats = rng.sample(_ALL_ENTITY_CATEGORIES, rng.randint(1, 3))
            slots = [_entity_slot(rng, c) for c in cats]
        elif bucket <= 5:
            kinds = [rng.choice(_HARD_KINDS) for _ in range(rng.randint(1, 2))]
            slots = [_entity_slot(rng, c, variant=v) for c, v in kinds]
        elif bucket <= 7:
            slots = []
        else:
            slots = [_negative_slot(rng) for _ in range(rng.randint(1, 3))]
        CARRIERS[carrier](b, rng, slots)
        docs.append(b.build())
    return docs


_ADVERSARIAL_MODES = ("obfuscated", "doc_start", "doc_end", "adjacent", "embedded")


def _gen_adversarial(size: int, seed: int) -> List[LabeledDocument]:
    rng = random.Random(seed)
    docs = []
    for i in range(size):
        mode = _ADVERSARIAL_MODES[i % len(_ADVERSARIAL_MODES)]
        if mode == "obfuscated":
            carrier = rng.choice(_PROSE_CARRIERS)
            b = DocBuilder(_doc_id("adversarial", seed, i), "adversarial", carrier, seed)
            slots = []
            for _ in range(rng.randint(1, 2)):
                cat = rng.choice(_ALL_ENTITY_CATEGORIES)
                slots.append(_entity_slot(rng, cat,
                                          obfuscation=rng.choice(obfuscations_for(cat))))
            CARRIERS[carrier](b, rng, slots)
        elif mode == "embedded":
            carrier = rng.choice(("code_snippet", "json_payload"))
            b = DocBuilder(_doc_id("adversarial", seed, i), "adversarial", carrier, seed)
            slots = [_entity_slot(rng, rng.choice(_ALL_ENTITY_CATEGORIES),
                                  difficulty="boundary",
                                  extra_attrs={"placement": "embedded"})
                     for _ in range(rng.randint(1, 2))]
            CARRIERS[carrier](b, rng, slots)
        else:
            b = DocBuilder(_doc_id("adversarial", seed, i), "adversarial", "boundary", seed)
            if mode == "doc_start":
                slot = _entity_slot(rng, rng.choice(_ALL_ENTITY_CATEGORIES),
                                    difficulty="boundary",
                                    extra_attrs={"placement": "doc_start"})
                _plant(b, slot)   # entity is the very first character of the doc
                b.append(" - " + _filler(rng) + " " + _filler(rng))
            elif mode == "doc_end":
                slot = _entity_slot(rng, rng.choice(_ALL_ENTITY_CATEGORIES),
                                    difficulty="boundary",
                                    extra_attrs={"placement": "doc_end"})
                b.append(_filler(rng) + " " + _lead_for(rng, slot))
                _plant(b, slot)   # entity is the very last character of the doc
            else:
                b.append(_filler(rng) + " ")
                for cat in rng.sample(_ALL_ENTITY_CATEGORIES, 2):
                    _plant(b, _entity_slot(rng, cat, difficulty="boundary",
                                           extra_attrs={"placement": "adjacent"}))
                b.append(" " + _filler(rng))
        docs.append(b.build(meta={"mode": mode}))
    return docs


def _gen_load(size: int, seed: int, dirty_rate: Optional[float]) -> List[LabeledDocument]:
    if dirty_rate is None:
        dirty_rate = 0.2
    rng = random.Random(seed)
    seen: set = set()
    docs = []
    for i in range(size):
        dirty = rng.random() < dirty_rate
        target = rng.randint(200, 2000)
        b = DocBuilder(_doc_id("load", seed, i), "load", "casual_chat", seed)
        b.append(rng.choice(("hey, quick question. ", "hi! following up on our chat. ",
                             "hello - a couple of updates. ")))
        _pad_prose(b, rng, target - 120)
        if dirty:
            slot = None
            for _ in range(200):
                cat, variant = rng.choice(_LOAD_KINDS)
                candidate = _entity_slot(rng, cat, variant=variant)
                if candidate.text not in seen:
                    slot = candidate
                    break
            if slot is None:
                raise RuntimeError("could not generate a unique entity value")
            seen.add(slot.text)
            b.append(_lead_for(rng, slot))
            _plant(b, slot)
            b.append(". ")
        b.append(rng.choice(("let me know if you need anything else.",
                             "appreciate the help!", "talk soon.")))
        docs.append(b.build(meta={"expected_alert": dirty}))
    return docs


def _gen_scale(size: int, seed: int) -> List[LabeledDocument]:
    rng = random.Random(seed)
    docs = []
    idx = 0
    for target in SCALE_LENGTHS:
        for _ in range(size):
            b = DocBuilder(_doc_id("scale", seed, idx), "scale", "scale_sweep", seed)
            b.append(rng.choice(("synthetic scale probe. ", "filler document follows. ")))
            first = _entity_slot(rng, constants.SSN, variant="dashed")
            b.append(_lead_for(rng, first))
            _plant(b, first)
            b.append(". ")
            _pad_prose(b, rng, int(target * 0.95))
            deep = _entity_slot(rng, constants.EMAIL, variant="dotted")
            b.append(_lead_for(rng, deep))
            _plant(b, deep)
            b.append(". ")
            _pad_prose(b, rng, target)
            doc = b.build(meta={"target_chars": target})
            # Positions recorded so truncation blindness (GLiNER 10k cap,
            # global 200k cap) can be attributed to entity depth.
            for e in doc.entities:
                e.attrs["char_pos"] = e.start
                e.attrs["doc_chars"] = len(doc.text)
                e.attrs["depth"] = round(e.start / len(doc.text), 4)
            docs.append(doc)
            idx += 1
    return docs


def generate(profile: str, size: int, seed: int,
             dirty_rate: Optional[float] = None) -> List[LabeledDocument]:
    """Generate a deterministic labeled corpus for one profile.

    ``size`` is the document count, except for "scale" where it is the number
    of documents per length step (total = size * len(SCALE_LENGTHS)).
    ``dirty_rate`` applies to the "load" profile only (default 0.2).
    """
    if profile == "accuracy":
        return _gen_accuracy(size, seed)
    if profile == "adversarial":
        return _gen_adversarial(size, seed)
    if profile == "load":
        return _gen_load(size, seed, dirty_rate)
    if profile == "scale":
        return _gen_scale(size, seed)
    raise ValueError(f"unknown profile {profile!r} (expected one of {PROFILES})")


# ---------------------------------------------------------------------------
# Corpus persistence
# ---------------------------------------------------------------------------

_HIST_EDGES = (200, 500, 1000, 2000, 5000, 10000, 30000, 100000, 300000)


def _hist_bucket(n: int) -> str:
    for edge in _HIST_EDGES:
        if n <= edge:
            return f"<={edge}"
    return f">{_HIST_EDGES[-1]}"


def build_manifest(docs: List[LabeledDocument]) -> Dict[str, Any]:
    def bump(d: Dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    m: Dict[str, Any] = {
        "docs": len(docs),
        "dirty_docs": sum(1 for d in docs if d.entities),
        "clean_docs": sum(1 for d in docs if not d.entities),
        "entities": sum(len(d.entities) for d in docs),
        "negatives": sum(len(d.negatives) for d in docs),
        "by_profile": {}, "by_carrier": {}, "by_category": {},
        "by_difficulty": {}, "by_generator": {}, "char_histogram": {},
    }
    for d in docs:
        bump(m["by_profile"], d.profile)
        bump(m["by_carrier"], d.carrier)
        bump(m["char_histogram"], _hist_bucket(len(d.text)))
        for e in d.entities:
            bump(m["by_category"], e.category)
            bump(m["by_difficulty"], e.difficulty)
            bump(m["by_generator"], e.generator)
        for e in d.negatives:
            bump(m["by_generator"], e.generator)
    return m


def write_corpus(docs: List[LabeledDocument], out_dir: str) -> Dict[str, Any]:
    """Write corpus.jsonl + manifest.json to ``out_dir``; return the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "corpus.jsonl"), docs)
    manifest = build_manifest(docs)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest
