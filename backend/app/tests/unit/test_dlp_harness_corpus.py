############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_harness_corpus.py: Unit tests for the DLP harness
# synthetic corpus generator (dlp_harness/generators.py +
# dlp_harness/corpus.py).
#
############################################################

"""Corpus generator tests: span integrity, determinism, Luhn/ABA checksums,
regex-detectability against the ACTUAL built-in scanner patterns, load-profile
uniqueness, and scale-profile position attributes."""

import json
import os
import random
import re
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dlp_harness import constants, corpus, generators  # noqa: E402
from dlp_harness.schemas import LabeledDocument, read_jsonl, to_dict  # noqa: E402

# The five built-in patterns, copied VERBATIM from _BUILTIN_PATTERNS in
# backend/app/services/dlp_scanner.py (compiled with re.IGNORECASE as the
# scanner does). Copied rather than imported: importing the scanner module
# pulls the app logging/db chain, which unit tests must never do.
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b", re.IGNORECASE)
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", re.IGNORECASE)
DOB_RE = re.compile(r"\b(?:DOB|date of birth|born on)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                    re.IGNORECASE)

_PROFILE_SIZES = (("accuracy", 60), ("adversarial", 40), ("load", 60), ("scale", 2))


def _overlaps(regex, text, start, end):
    return any(m.start() < end and m.end() > start for m in regex.finditer(text))


def test_span_integrity_all_profiles():
    for profile, size in _PROFILE_SIZES:
        docs = corpus.generate(profile, size, seed=11)
        assert docs
        for i, doc in enumerate(docs):
            assert doc.doc_id == f"{profile}-11-{i:06d}"
            assert doc.profile == profile
            assert doc.carrier
            assert doc.seed == 11
            for e in doc.entities + doc.negatives:
                assert doc.text[e.start:e.end] == e.text, (doc.doc_id, e.generator)
                assert 0 <= e.start < e.end <= len(doc.text)


def test_determinism_byte_identical():
    for profile, size in _PROFILE_SIZES:
        a = corpus.generate(profile, size, seed=99)
        b = corpus.generate(profile, size, seed=99)
        assert (json.dumps([to_dict(d) for d in a], ensure_ascii=False)
                == json.dumps([to_dict(d) for d in b], ensure_ascii=False)), profile


def test_credit_card_entities_are_luhn_valid():
    rng = random.Random(42)
    for _ in range(60):
        text, attrs = generators.gen_credit_card(rng)
        digits = re.sub(r"\D", "", text)
        assert generators.luhn_valid(digits), text
        assert attrs["luhn_valid"] is True
        assert attrs["network"] in ("visa", "mastercard", "amex", "discover")
        assert len(digits) == (15 if attrs["network"] == "amex" else 16)
    # And every credit_card entity in a generated corpus.
    checked = 0
    for doc in corpus.generate("accuracy", 300, seed=3):
        for e in doc.entities:
            if e.category == constants.CREDIT_CARD:
                assert generators.luhn_valid(re.sub(r"\D", "", e.text)), e.text
                checked += 1
    assert checked > 0


def test_luhn_invalid_negatives_fail_luhn():
    rng = random.Random(42)
    for _ in range(60):
        text, attrs = generators.neg_luhn_invalid_16(rng)
        digits = re.sub(r"\D", "", text)
        assert len(digits) == 16
        assert not generators.luhn_valid(digits), text
        assert attrs["mimics"] == constants.CREDIT_CARD
    checked = 0
    for doc in corpus.generate("accuracy", 300, seed=3):
        for e in doc.negatives:
            if e.generator == "negative.luhn_invalid_16":
                assert not generators.luhn_valid(re.sub(r"\D", "", e.text)), e.text
                checked += 1
    assert checked > 0


def test_aba_routing_checksum_valid():
    rng = random.Random(7)
    seen = 0
    for _ in range(80):
        text, attrs = generators.gen_bank_account(rng, variant="aba_routing")
        assert generators.aba_valid(text), text
        assert attrs["aba_valid"] is True
        seen += 1
    assert seen == 80


def test_negatives_declare_what_they_mimic():
    rng = random.Random(5)
    for name, fn in generators.NEGATIVE_GENERATORS.items():
        text, attrs = fn(rng)
        assert text
        assert attrs["mimics"] in constants.CANONICAL_CATEGORIES, name


def test_regex_detectability_smoke():
    """Plain regex-visible entities must be hit by the corresponding built-in
    pattern; bare-9 SSNs must be invisible to both the SSN pattern (dashed
    only) and the CC pattern (needs >= 13 digits)."""
    docs = corpus.generate("accuracy", 200, seed=123)
    hit_counts = {"ssn.dashed": 0, "credit_card": 0, "email": 0,
                  "phone": 0, "dob.prefixed": 0, "ssn.bare9": 0}
    for doc in docs:
        for e in doc.entities:
            if e.difficulty != "plain":
                continue
            span = (e.start, e.end)
            if e.generator == "ssn.dashed":
                assert _overlaps(SSN_RE, doc.text, *span), (doc.doc_id, e.text)
                hit_counts["ssn.dashed"] += 1
            elif e.generator == "ssn.bare9":
                assert not _overlaps(SSN_RE, doc.text, *span), (doc.doc_id, e.text)
                assert not _overlaps(CC_RE, doc.text, *span), (doc.doc_id, e.text)
                hit_counts["ssn.bare9"] += 1
            elif e.category == constants.CREDIT_CARD:
                assert _overlaps(CC_RE, doc.text, *span), (doc.doc_id, e.text)
                hit_counts["credit_card"] += 1
            elif e.category == constants.EMAIL:
                assert _overlaps(EMAIL_RE, doc.text, *span), (doc.doc_id, e.text)
                hit_counts["email"] += 1
            elif e.category == constants.PHONE:
                assert _overlaps(PHONE_RE, doc.text, *span), (doc.doc_id, e.text)
                hit_counts["phone"] += 1
            elif e.generator == "date_of_birth.prefixed":
                assert _overlaps(DOB_RE, doc.text, *span), (doc.doc_id, e.text)
                hit_counts["dob.prefixed"] += 1
    # Guard against a vacuous pass: every asserted kind must actually occur.
    for kind, n in hit_counts.items():
        assert n > 0, f"no {kind} entities generated in 200 accuracy docs"


def test_load_profile_unique_values_and_meta():
    docs = corpus.generate("load", 120, seed=5)
    values = []
    for doc in docs:
        assert 100 <= len(doc.text) <= 2400, len(doc.text)
        assert doc.meta["expected_alert"] == bool(doc.entities)
        assert len(doc.entities) <= 1
        values.extend(e.text for e in doc.entities)
    assert values, "no dirty docs at default dirty_rate"
    assert len(values) == len(set(values)), "duplicate entity values in load corpus"
    # dirty_rate default is 0.2 — allow wide slack, just not degenerate.
    assert 5 <= len(values) <= 60

    all_dirty = corpus.generate("load", 60, seed=5, dirty_rate=1.0)
    texts = [d.entities[0].text for d in all_dirty]
    assert all(len(d.entities) == 1 and d.meta["expected_alert"] for d in all_dirty)
    assert len(texts) == len(set(texts))


def test_scale_profile_positions_and_attrs():
    per_step = 2
    docs = corpus.generate("scale", per_step, seed=9)
    assert len(docs) == per_step * len(corpus.SCALE_LENGTHS)
    seen_targets = {}
    for doc in docs:
        target = doc.meta["target_chars"]
        assert target in corpus.SCALE_LENGTHS
        seen_targets[target] = seen_targets.get(target, 0) + 1
        assert len(doc.text) >= target
        assert len(doc.entities) == 2
        near, deep = sorted(doc.entities, key=lambda e: e.start)
        assert near.start < 100, "first entity must be near the doc start"
        assert deep.start >= int(target * 0.9), "second entity must sit at ~95% depth"
        for e in doc.entities:
            assert e.attrs["char_pos"] == e.start
            assert e.attrs["doc_chars"] == len(doc.text)
            assert 0.0 <= e.attrs["depth"] <= 1.0
    assert all(seen_targets[t] == per_step for t in corpus.SCALE_LENGTHS)


def test_adversarial_profile_all_dirty_and_marked():
    docs = corpus.generate("adversarial", 40, seed=21)
    modes = set()
    for doc in docs:
        assert doc.entities, doc.doc_id
        modes.add(doc.meta["mode"])
        for e in doc.entities:
            assert e.difficulty in ("obfuscated", "boundary"), (doc.doc_id, e)
            if e.difficulty == "obfuscated":
                assert e.obfuscation in corpus.OBFUSCATIONS
            else:
                assert e.attrs["placement"] in ("doc_start", "doc_end",
                                                "adjacent", "embedded")
                if e.attrs["placement"] == "doc_start":
                    assert e.start == 0
                elif e.attrs["placement"] == "doc_end":
                    assert e.end == len(doc.text)
    assert modes == set(corpus._ADVERSARIAL_MODES)


def test_write_corpus_round_trip(tmp_path):
    docs = corpus.generate("accuracy", 50, seed=17)
    manifest = corpus.write_corpus(docs, str(tmp_path))
    assert manifest["docs"] == 50
    assert manifest["dirty_docs"] + manifest["clean_docs"] == 50
    with open(tmp_path / "manifest.json", encoding="utf-8") as f:
        assert json.load(f) == manifest
    loaded = read_jsonl(str(tmp_path / "corpus.jsonl"), cls=LabeledDocument)
    assert len(loaded) == 50
    for orig, back in zip(docs, loaded):
        assert to_dict(orig) == to_dict(back)
        for e in back.entities + back.negatives:
            assert back.text[e.start:e.end] == e.text
# ---------------------------------------------------------------------------
# constants: regex variant visibility + severity rule override map
# ---------------------------------------------------------------------------

def test_regex_visible_variants_shape():
    assert constants.REGEX_VISIBLE_VARIANTS == {
        constants.SSN: {"dashed"},
        constants.DATE_OF_BIRTH: {"prefixed"},
        constants.CREDIT_CARD: None,
        constants.EMAIL: None,
        constants.PHONE: None,
    }
    # Variant map keys mirror the regex scope exactly.
    assert set(constants.REGEX_VISIBLE_VARIANTS) == constants.REGEX_BUILTIN_SCOPE


def test_regex_can_see_variant_rules():
    see = constants.regex_can_see
    assert see(constants.SSN, "ssn.dashed") is True
    assert see(constants.SSN, "ssn.spaced") is False
    assert see(constants.SSN, "ssn.bare9") is False
    assert see(constants.SSN, "ssn") is False           # bare generator, gated map
    assert see(constants.DATE_OF_BIRTH, "date_of_birth.prefixed") is True
    assert see(constants.DATE_OF_BIRTH, "date_of_birth.bare") is False
    # None map value -> every variant (even a bare generator) is visible.
    assert see(constants.CREDIT_CARD, "credit_card.visa_spaced") is True
    assert see(constants.EMAIL, "email") is True
    assert see(constants.PHONE, "") is True
    # Out-of-scope categories are never regex-visible.
    assert see(constants.PERSON, "person.titled") is False
    assert see(constants.MEDICAL_RECORD, "medical_record.plain") is False


def test_regex_can_see_agrees_with_builtin_patterns_on_corpus():
    # Ground-truth cross-check for the variant-gated categories: on plain
    # (un-obfuscated) planted entities the predicate must agree with the
    # ACTUAL built-in pattern the scanner runs.
    pattern_by_cat = {constants.SSN: SSN_RE, constants.DATE_OF_BIRTH: DOB_RE}
    checked = 0
    for doc in corpus.generate("accuracy", 200, seed=17):
        for e in doc.entities:
            regex = pattern_by_cat.get(e.category)
            if regex is None or e.difficulty != "plain":
                continue
            assert (constants.regex_can_see(e.category, e.generator)
                    == _overlaps(regex, doc.text, e.start, e.end)), \
                (doc.doc_id, e.generator)
            checked += 1
    assert checked > 10


def test_severity_rules_override_matches_alias_severity_model():
    # Built from ALIASES x DEFAULT_SEVERITY_BY_CATEGORY: every alias of a
    # severity-ruled category is present, mapped to its canonical severity,
    # and nothing else is.
    for raw, sev in constants.SEVERITY_RULES_OVERRIDE.items():
        canonical = constants.ALIASES[raw]
        assert sev == constants.DEFAULT_SEVERITY_BY_CATEGORY[canonical]
    expected_keys = {raw for raw, c in constants.ALIASES.items()
                     if c in constants.DEFAULT_SEVERITY_BY_CATEGORY}
    assert set(constants.SEVERITY_RULES_OVERRIDE) == expected_keys
    assert constants.SEVERITY_RULES_OVERRIDE["social security number"] == "major"
    assert constants.SEVERITY_RULES_OVERRIDE["dob"] == "moderate"
    assert constants.SEVERITY_RULES_OVERRIDE["phone"] == "minor"
