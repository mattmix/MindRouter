############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_harness_matching.py: Unit tests for the DLP
# harness finding normalization + ground-truth span
# matching (dlp_harness/matching.py).
#
############################################################

"""Unit tests for dlp_harness.matching.

Covers:
- normalize_findings: dict input, object input, unmapped category,
  malformed span collapse, Finding passthrough, empty input
- match_document: lenient vs strict matching, multi-scanner overlap,
  adjacency double-count, duplicate findings, boundary entities,
  zero-width findings, iou_threshold gating
- Spanless LLM findings: all three match branches + non-match -> FP
- False positives + negative-trap attribution (dedup, sorted)
- Mislabeled findings (overlapped an entity, category-matched none of the
  entities overlapped) recorded on DocEval.mislabeled_findings, not as FPs
- Severity prediction (incl. unmapped -> moderate, none -> minor)
  and severity_expected
- DocEval bookkeeping fields, scan_error passthrough, match_corpus
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Locate the repo root so `dlp_harness` (a repo-root package) is importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dlp_harness import constants
from dlp_harness.matching import match_corpus, match_document, normalize_findings
from dlp_harness.schemas import Finding, GroundTruthEntity, LabeledDocument


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _ent(category, text, start, end, generator=""):
    return GroundTruthEntity(category=category, text=text, start=start,
                             end=end, generator=generator)


def _doc(text, entities=(), negatives=(), doc_id="d1",
         profile="accuracy", carrier="note"):
    return LabeledDocument(doc_id=doc_id, text=text, entities=list(entities),
                           negatives=list(negatives), profile=profile,
                           carrier=carrier)


def _fd(scanner="regex", category="ssn", text="", confidence=0.9,
        start=0, end=0):
    """Raw finding in the JSONL dict shape."""
    return {"scanner": scanner, "category": category, "text": text,
            "confidence": confidence, "start": start, "end": end}


# ----------------------------------------------------------------
# normalize_findings
# ----------------------------------------------------------------

class TestNormalizeFindings:
    def test_dict_input(self):
        [f] = normalize_findings([{
            "scanner": "regex", "category": "Social Security Number",
            "text": "123-45-6789", "confidence": 0.95, "start": 10, "end": 21,
        }])
        assert isinstance(f, Finding)
        assert f.scanner == "regex"
        assert f.category_raw == "Social Security Number"   # verbatim
        assert f.category == constants.SSN                  # canonicalized
        assert (f.start, f.end) == (10, 21)
        assert f.confidence == pytest.approx(0.95)

    def test_object_input(self):
        # Mimics the ScanFinding dataclass from dlp_scanner.py.
        raw = SimpleNamespace(scanner="gliner", category="credit card number",
                              text="4111 1111 1111 1111", confidence=0.8,
                              start=5, end=24)
        [f] = normalize_findings([raw])
        assert f.category_raw == "credit card number"
        assert f.category == constants.CREDIT_CARD
        assert (f.start, f.end) == (5, 24)

    def test_unmapped_category_is_none(self):
        [f] = normalize_findings([_fd(category="quantum flux id")])
        assert f.category is None
        assert f.category_raw == "quantum flux id"

    def test_end_before_start_collapses_to_zero_width(self):
        [f] = normalize_findings([_fd(start=30, end=12)])
        assert (f.start, f.end) == (30, 30)

    def test_finding_passthrough(self):
        original = Finding(scanner="regex", category_raw="Credit Card Number",
                           category=constants.CREDIT_CARD, text="4111...",
                           confidence=0.9, start=1, end=8)
        [f] = normalize_findings([original])
        assert f is original
        assert f.category == constants.CREDIT_CARD

    def test_empty_input(self):
        assert normalize_findings([]) == []
        assert normalize_findings(None) == []


# ----------------------------------------------------------------
# Lenient vs strict matching
# ----------------------------------------------------------------

class TestLenientStrict:
    DOC = _doc("SSN: 123-45-6789 end",
               [_ent(constants.SSN, "123-45-6789", 5, 16)])

    def test_lenient_match_wrong_category(self):
        ev = match_document(self.DOC, normalize_findings(
            [_fd(scanner="gliner", category="person", start=5, end=16)]))
        [m] = ev.entity_matches
        assert m.matched is True            # lenient: category ignored
        assert m.category_correct is False  # strict: category wrong
        assert m.matched_by == ["gliner"]

    def test_strict_match_right_category(self):
        ev = match_document(self.DOC, normalize_findings([
            _fd(scanner="gliner", category="person", start=5, end=16),
            _fd(scanner="regex", category="social security number",
                start=5, end=16),
        ]))
        [m] = ev.entity_matches
        assert m.matched is True
        assert m.category_correct is True

    def test_no_overlap_no_match(self):
        ev = match_document(self.DOC, normalize_findings(
            [_fd(start=17, end=20)]))
        [m] = ev.entity_matches
        assert m.matched is False
        assert m.matched_by == []
        assert m.best_confidence == 0.0
        assert len(ev.false_positives) == 1


class TestMultiScannerAndDuplicates:
    def test_multi_scanner_overlap(self):
        doc = _doc("x" * 30, [_ent(constants.SSN, "123-45-6789", 10, 21)])
        ev = match_document(doc, normalize_findings([
            _fd(scanner="regex", confidence=0.99, start=10, end=21),
            _fd(scanner="gliner", category="social security number",
                confidence=0.70, start=12, end=18),
        ]))
        [m] = ev.entity_matches
        assert m.matched_by == ["gliner", "regex"]   # sorted unique
        assert m.best_confidence == pytest.approx(0.99)
        assert ev.false_positives == []

    def test_duplicate_identical_findings(self):
        doc = _doc("x" * 30, [_ent(constants.SSN, "123-45-6789", 10, 21)])
        ev = match_document(doc, normalize_findings(
            [_fd(start=10, end=21), _fd(start=10, end=21)]))
        [m] = ev.entity_matches
        assert m.matched_by == ["regex"]     # deduped scanners
        assert ev.findings_count == 2        # both still counted
        assert ev.false_positives == []      # both matched, neither FP

    def test_one_finding_spans_two_adjacent_entities(self):
        doc = _doc("xxxxxAAAAABBBBBxxx", [
            _ent(constants.EMAIL, "AAAAA", 5, 10),
            _ent(constants.PHONE, "BBBBB", 10, 15),
        ])
        ev = match_document(doc, normalize_findings(
            [_fd(category="email", start=4, end=16)]))
        assert [m.matched for m in ev.entity_matches] == [True, True]
        assert ev.false_positives == []


class TestBoundariesAndZeroWidth:
    def test_entities_at_position_zero_and_len_text(self):
        text = "ABCDE middle VWXYZ"          # len 18
        doc = _doc(text, [
            _ent(constants.EMAIL, "ABCDE", 0, 5),
            _ent(constants.PHONE, "VWXYZ", 13, len(text)),
        ])
        ev = match_document(doc, normalize_findings([
            _fd(category="email", start=0, end=3),
            _fd(category="phone", start=len(text) - 1, end=len(text)),
        ]))
        assert [m.matched for m in ev.entity_matches] == [True, True]

    def test_zero_width_non_llm_finding_is_fp(self):
        doc = _doc("x" * 30, [_ent(constants.SSN, "123-45-6789", 10, 21)])
        # end < start collapses to zero-width at 15 (inside the entity):
        # empty ranges intersect nothing.
        ev = match_document(doc, normalize_findings([_fd(start=15, end=12)]))
        assert ev.entity_matches[0].matched is False
        assert len(ev.false_positives) == 1


# ----------------------------------------------------------------
# IoU gating
# ----------------------------------------------------------------

class TestIouThreshold:
    def test_one_char_overlap_passes_iou_zero_fails_half(self):
        doc = _doc("x" * 40, [_ent(constants.SSN, "s" * 10, 10, 20)])
        findings = normalize_findings([_fd(start=19, end=30)])  # IoU = 1/20
        ev0 = match_document(doc, findings, iou_threshold=0.0)
        assert ev0.entity_matches[0].matched is True
        ev5 = match_document(doc, findings, iou_threshold=0.5)
        assert ev5.entity_matches[0].matched is False
        assert len(ev5.false_positives) == 1   # rejected finding becomes FP

    def test_exact_span_passes_high_iou(self):
        doc = _doc("x" * 40, [_ent(constants.SSN, "s" * 10, 10, 20)])
        ev = match_document(doc, normalize_findings(
            [_fd(start=10, end=20)]), iou_threshold=0.99)
        assert ev.entity_matches[0].matched is True


# ----------------------------------------------------------------
# False positives + trap attribution
# ----------------------------------------------------------------

class TestFalsePositivesAndTraps:
    def test_fp_trap_attribution_and_dedup(self):
        text = "123-45-6789 then 555-0100-FAKE tail"
        doc = _doc(text,
                   entities=[_ent(constants.SSN, "123-45-6789", 0, 11)],
                   negatives=[_ent(constants.PHONE, "555-0100-FAKE", 17, 30,
                                   generator="phone.lookalike")])
        ev = match_document(doc, normalize_findings([
            _fd(start=0, end=11),                                    # TP
            _fd(category="phone", start=17, end=30),                 # FP on trap
            _fd(scanner="gliner", category="phone", start=18, end=25),  # FP same trap
            _fd(category="email", start=31, end=35),                 # FP, no trap
        ]))
        assert ev.entity_matches[0].matched is True
        assert len(ev.false_positives) == 3
        assert ev.fp_trap_hits == ["phone.lookalike"]   # deduped

    def test_multiple_traps_sorted(self):
        doc = _doc("x" * 50, negatives=[
            _ent(constants.PHONE, "b", 30, 40, generator="zzz.trap"),
            _ent(constants.SSN, "a", 10, 20, generator="aaa.trap"),
        ])
        ev = match_document(doc, normalize_findings([
            _fd(start=32, end=38), _fd(start=12, end=18),
        ]))
        assert ev.is_clean is True
        assert len(ev.false_positives) == 2
        assert ev.fp_trap_hits == ["aaa.trap", "zzz.trap"]

    def test_clean_doc_no_findings(self):
        ev = match_document(_doc("all clean"), [])
        assert ev.is_clean is True
        assert ev.doc_flagged is False
        assert ev.false_positives == []
        assert ev.fp_trap_hits == []


# ----------------------------------------------------------------
# Mislabeled findings (strict-mode FP bookkeeping — finding [7])
# ----------------------------------------------------------------

class TestMislabeledFindings:
    DOC = _doc("SSN: 123-45-6789 end",
               [_ent(constants.SSN, "123-45-6789", 5, 16)])

    def test_wrong_category_overlap_is_mislabeled_not_fp(self):
        ev = match_document(self.DOC, normalize_findings(
            [_fd(scanner="gliner", category="person", start=5, end=16)]))
        assert ev.false_positives == []
        [m] = ev.mislabeled_findings
        assert m.category == constants.PERSON
        assert ev.entity_matches[0].matched is True
        assert ev.entity_matches[0].category_correct is False

    def test_correct_category_finding_is_not_mislabeled(self):
        ev = match_document(self.DOC, normalize_findings(
            [_fd(category="social security number", start=5, end=16)]))
        assert ev.mislabeled_findings == []
        assert ev.false_positives == []

    def test_finding_matching_one_of_two_overlapped_categories(self):
        # Spans two adjacent entities and category-matches ONE of them ->
        # not mislabeled (it matched an entity it overlaps).
        doc = _doc("xxxxxAAAAABBBBBxxx", [
            _ent(constants.EMAIL, "AAAAA", 5, 10),
            _ent(constants.PHONE, "BBBBB", 10, 15),
        ])
        ev = match_document(doc, normalize_findings(
            [_fd(category="email", start=4, end=16)]))
        assert ev.mislabeled_findings == []

    def test_finding_matching_neither_overlapped_category(self):
        doc = _doc("xxxxxAAAAABBBBBxxx", [
            _ent(constants.EMAIL, "AAAAA", 5, 10),
            _ent(constants.PHONE, "BBBBB", 10, 15),
        ])
        ev = match_document(doc, normalize_findings(
            [_fd(category="person", start=4, end=16)]))
        [m] = ev.mislabeled_findings
        assert m.category == constants.PERSON
        assert ev.false_positives == []
        assert [e.matched for e in ev.entity_matches] == [True, True]

    def test_unmatched_finding_stays_fp_not_mislabeled(self):
        ev = match_document(self.DOC, normalize_findings(
            [_fd(category="person", start=17, end=20)]))
        assert len(ev.false_positives) == 1
        assert ev.mislabeled_findings == []

    def test_spanless_llm_mislabel(self):
        doc = _doc("id: 123-45-6789 ok",
                   [_ent(constants.SSN, "123-45-6789", 4, 15)])
        ev = match_document(doc, normalize_findings(
            [_fd(scanner="llm", category="person",
                 text="contains 123-45-6789", confidence=0.6,
                 start=0, end=0)]))
        [m] = ev.mislabeled_findings
        assert m.scanner == "llm"
        assert ev.false_positives == []

    def test_unmapped_category_finding_can_be_mislabeled(self):
        ev = match_document(self.DOC, normalize_findings(
            [_fd(category="quantum flux id", start=5, end=16)]))
        [m] = ev.mislabeled_findings
        assert m.category is None
        assert m.category_raw == "quantum flux id"


# ----------------------------------------------------------------
# Spanless LLM findings
# ----------------------------------------------------------------

class TestSpanlessLlm:
    def _llm(self, text, category="person", confidence=0.6):
        return _fd(scanner="llm", category=category, text=text,
                   confidence=confidence, start=0, end=0)

    def test_entity_text_inside_finding_text(self):
        doc = _doc("id: 123-45-6789 ok",
                   [_ent(constants.SSN, "123-45-6789", 4, 15)])
        ev = match_document(doc, normalize_findings(
            [self._llm("document contains SSN 123-45-6789")]))
        [m] = ev.entity_matches
        assert m.matched is True
        assert m.matched_by == ["llm"]
        assert m.category_correct is False   # llm said "person"
        assert ev.false_positives == []

    def test_finding_text_inside_entity_text(self):
        doc = _doc("name: Jonathan Q. Public-Smith",
                   [_ent(constants.PERSON, "Jonathan Q. Public-Smith", 6, 30)])
        ev = match_document(doc, normalize_findings([self._llm("Q. Public")]))
        assert ev.entity_matches[0].matched is True

    def test_category_match_when_doc_has_exactly_one_of_category(self):
        doc = _doc("id: 123-45-6789 ok",
                   [_ent(constants.SSN, "123-45-6789", 4, 15)])
        ev = match_document(doc, normalize_findings(
            [self._llm("a redacted identifier",
                       category="social security number")]))
        [m] = ev.entity_matches
        assert m.matched is True
        assert m.category_correct is True

    def test_category_branch_requires_exactly_one(self):
        doc = _doc("123-45-6789 and 987-65-4321", [
            _ent(constants.SSN, "123-45-6789", 0, 11),
            _ent(constants.SSN, "987-65-4321", 16, 27),
        ])
        ev = match_document(doc, normalize_findings(
            [self._llm("a redacted identifier",
                       category="social security number")]))
        assert [m.matched for m in ev.entity_matches] == [False, False]
        assert len(ev.false_positives) == 1   # ambiguous spanless -> FP

    def test_unmatched_spanless_is_fp_without_trap_hit(self):
        doc = _doc("id: 123-45-6789 ok",
                   [_ent(constants.SSN, "123-45-6789", 4, 15)],
                   negatives=[_ent(constants.PHONE, "555", 0, 3,
                                   generator="phone.trap")])
        ev = match_document(doc, normalize_findings(
            [self._llm("nothing relevant", category="passport number")]))
        assert ev.entity_matches[0].matched is False
        assert len(ev.false_positives) == 1
        assert ev.fp_trap_hits == []          # spanless never hits trap spans

    def test_llm_finding_with_span_uses_overlap(self):
        doc = _doc("id: 123-45-6789 ok",
                   [_ent(constants.SSN, "123-45-6789", 4, 15)])
        ev = match_document(doc, normalize_findings(
            [_fd(scanner="llm", category="person", text="zzz",
                 start=4, end=15)]))
        assert ev.entity_matches[0].matched is True


# ----------------------------------------------------------------
# Severity
# ----------------------------------------------------------------

class TestSeverity:
    @pytest.mark.parametrize("categories,expected", [
        ([], "minor"),                                # no findings
        (["email"], "minor"),
        (["date of birth"], "moderate"),
        (["ssn"], "major"),
        (["email", "ssn"], "major"),                  # highest wins
        (["quantum flux id"], "moderate"),            # unmapped -> fallback
        (["person"], "moderate"),                     # mapped but no rule
    ])
    def test_severity_predicted(self, categories, expected):
        findings = normalize_findings([
            _fd(category=c, start=i * 3, end=i * 3 + 2)
            for i, c in enumerate(categories)
        ])
        ev = match_document(_doc("x" * 50), findings)
        assert ev.severity_predicted == expected

    @pytest.mark.parametrize("entity_categories,expected", [
        ([constants.SSN], "major"),
        ([constants.EMAIL], "minor"),
        ([constants.EMAIL, constants.DATE_OF_BIRTH], "moderate"),
        ([constants.PERSON], "moderate"),             # no rule -> fallback
    ])
    def test_severity_expected_dirty(self, entity_categories, expected):
        entities = [_ent(c, "t", i * 5, i * 5 + 3)
                    for i, c in enumerate(entity_categories)]
        ev = match_document(_doc("x" * 50, entities), [])
        assert ev.severity_expected == expected

    def test_severity_expected_clean_is_minor(self):
        ev = match_document(_doc("clean"), [])
        assert ev.severity_expected == "minor"


# ----------------------------------------------------------------
# DocEval bookkeeping + match_corpus
# ----------------------------------------------------------------

class TestDocEvalFields:
    def test_fields_populated(self):
        text = "SSN: 123-45-6789 end"
        doc = _doc(text, [_ent(constants.SSN, "123-45-6789", 5, 16)],
                   doc_id="docX", profile="stress", carrier="ticket")
        ev = match_document(doc, normalize_findings(
            [_fd(start=5, end=16)]), scan_latency_ms=12.5)
        assert ev.doc_id == "docX"
        assert ev.profile == "stress"
        assert ev.carrier == "ticket"
        assert ev.is_clean is False
        assert ev.scan_error is None
        assert ev.text_chars == len(text)
        assert ev.findings_count == 1
        assert ev.scan_latency_ms == pytest.approx(12.5)
        assert ev.doc_flagged is True

    def test_scan_error_still_records_findings(self):
        doc = _doc("SSN: 123-45-6789 end",
                   [_ent(constants.SSN, "123-45-6789", 5, 16)])
        ev = match_document(doc, normalize_findings(
            [_fd(start=5, end=16)]), scan_error="gliner timeout")
        assert ev.scan_error == "gliner timeout"
        assert ev.entity_matches[0].matched is True
        assert ev.findings_count == 1

    def test_doc_flagged_by_fp_only(self):
        ev = match_document(_doc("clean text"), normalize_findings(
            [_fd(start=0, end=3)]))
        assert ev.doc_flagged is True
        assert len(ev.false_positives) == 1


class TestMatchCorpus:
    def test_corpus_wrapper(self):
        d1 = _doc("SSN: 123-45-6789 end",
                  [_ent(constants.SSN, "123-45-6789", 5, 16)], doc_id="a")
        d2 = _doc("clean text", doc_id="b")
        d3 = _doc("clean too", doc_id="c")
        evs = match_corpus(
            [d1, d2, d3],
            {"a": [_fd(start=5, end=16)]},        # raw dicts normalized inside
            latencies_by_doc_id={"a": 5.0},
            errors_by_doc_id={"b": "timeout"},
        )
        assert [e.doc_id for e in evs] == ["a", "b", "c"]
        assert evs[0].entity_matches[0].matched is True
        assert evs[0].scan_latency_ms == pytest.approx(5.0)
        assert evs[1].scan_error == "timeout"
        assert evs[1].findings_count == 0         # missing id -> no findings
        assert evs[1].doc_flagged is False
        assert evs[2].scan_error is None
        assert evs[2].scan_latency_ms == 0.0

    def test_corpus_iou_threshold_forwarded(self):
        doc = _doc("x" * 40, [_ent(constants.SSN, "s" * 10, 10, 20)],
                   doc_id="a")
        evs = match_corpus([doc], {"a": [_fd(start=19, end=30)]},
                           iou_threshold=0.5)
        assert evs[0].entity_matches[0].matched is False
