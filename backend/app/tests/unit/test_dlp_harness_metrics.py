############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# backend/app/tests/unit/test_dlp_harness_metrics.py:
# Unit tests for the DLP harness statistics engine
# (dlp_harness/metrics.py) against hand-computed values.
#
############################################################

"""Tests for dlp_harness.metrics: confusion math, bootstrap, sweeps."""

import os
import sys

import pytest

# Import dlp_harness from the repo root, not the app package (import-chain rule).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from dlp_harness import metrics
from dlp_harness.constants import (
    CREDIT_CARD, EMAIL, MEDICAL_RECORD, PERSON, PHONE, REGEX_BUILTIN_SCOPE,
    SSN,
)
from dlp_harness.schemas import (
    DocEval, EntityMatch, Finding, GroundTruthEntity, LabeledDocument,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def ent(cat, difficulty="plain", obfuscation="", generator=""):
    return GroundTruthEntity(category=cat, text="x", start=0, end=1,
                             generator=generator, difficulty=difficulty,
                             obfuscation=obfuscation)


def em(cat, matched, correct=None, **entity_kwargs):
    if correct is None:
        correct = matched
    return EntityMatch(entity=ent(cat, **entity_kwargs), matched=matched,
                       category_correct=correct)


def fp(cat, conf=0.9):
    return Finding(scanner="regex", category_raw=cat or "mystery-label",
                   category=cat, text="fp", confidence=conf)


def doc(doc_id="d", is_clean=False, flagged=True, matches=(), fps=(),
        mislabeled=(), scan_error=None, sev_pred="minor", sev_exp="minor",
        carrier="note"):
    return DocEval(doc_id=doc_id, profile="accuracy", carrier=carrier,
                   is_clean=is_clean, scan_error=scan_error,
                   entity_matches=list(matches), false_positives=list(fps),
                   mislabeled_findings=list(mislabeled),
                   doc_flagged=flagged, severity_predicted=sev_pred,
                   severity_expected=sev_exp)


# ---------------------------------------------------------------------------
# span_confusion
# ---------------------------------------------------------------------------

def _span_fixture():
    d1 = doc("d1", matches=[em(SSN, True), em(SSN, True), em(EMAIL, True)],
             fps=[fp(SSN)])
    d2 = doc("d2", matches=[em(EMAIL, False)], fps=[fp(None)])
    return [d1, d2]


def test_span_confusion_micro_and_per_category():
    out = metrics.span_confusion(_span_fixture())
    overall = out["overall"]
    assert (overall["tp"], overall["fn"], overall["fp"]) == (3, 1, 2)
    assert overall["precision"] == pytest.approx(3 / 5)
    assert overall["recall"] == pytest.approx(3 / 4)
    assert overall["f1"] == pytest.approx(2 / 3)

    per = out["per_category"]
    assert per[SSN]["precision"] == pytest.approx(2 / 3)
    assert per[SSN]["recall"] == pytest.approx(1.0)
    assert per[SSN]["f1"] == pytest.approx(0.8)
    assert per[EMAIL] == {"tp": 1, "fn": 1, "fp": 0, "precision": 1.0,
                          "recall": 0.5, "f1": pytest.approx(2 / 3)}
    # Unmapped FP bucket: precision defined (0), recall/f1 undefined.
    assert per["unmapped"]["fp"] == 1
    assert per["unmapped"]["precision"] == 0.0
    assert per["unmapped"]["recall"] is None
    assert per["unmapped"]["f1"] is None


def test_span_confusion_macro_averages():
    overall = metrics.span_confusion(_span_fixture())["overall"]
    # Eligible categories: ssn, email, unmapped. Undefined ratios skipped.
    assert overall["macro_precision"] == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)
    assert overall["macro_recall"] == pytest.approx((1.0 + 0.5) / 2)
    assert overall["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_span_confusion_strict_demotes_mislabeled_to_fn():
    d = doc("d1", matches=[em(SSN, True, correct=False), em(SSN, True),
                           em(EMAIL, True)])
    loose = metrics.span_confusion([d], strict=False)["overall"]
    strict = metrics.span_confusion([d], strict=True)["overall"]
    assert (loose["tp"], loose["fn"]) == (3, 0)
    assert (strict["tp"], strict["fn"]) == (2, 1)
    assert metrics.span_confusion([d], strict=True)["per_category"][SSN]["fn"] == 1


def test_span_confusion_strict_counts_mislabels_as_fp():
    # Finding [7] hand-computed regression: two entities, both detected, one
    # by a wrong-category finding (gliner said "person"); zero spurious
    # findings. Pre-fix strict read tp=1 fn=1 fp=0 -> precision 1.0 despite
    # half the emitted findings carrying a wrong label.
    mis = Finding(scanner="gliner", category_raw="person", category=PERSON,
                  text="123-45-6789", confidence=0.8)
    d = doc("d1", matches=[em(SSN, True), em(EMAIL, True, correct=False)],
            mislabeled=[mis])

    loose = metrics.span_confusion([d], strict=False)
    # Lenient unchanged: mislabels are neither FN nor FP.
    o = loose["overall"]
    assert (o["tp"], o["fn"], o["fp"]) == (2, 0, 0)
    assert PERSON not in loose["per_category"]

    strict = metrics.span_confusion([d], strict=True)
    # Strict: mislabeled entity -> FN AND the mislabeling finding -> FP
    # under its claimed category. tp=1 fn=1 fp=1.
    o = strict["overall"]
    assert (o["tp"], o["fn"], o["fp"]) == (1, 1, 1)
    assert o["precision"] == pytest.approx(0.5)
    assert o["recall"] == pytest.approx(0.5)
    assert strict["per_category"][PERSON]["fp"] == 1
    assert strict["per_category"][EMAIL]["fn"] == 1


def test_span_confusion_strict_mislabel_unmapped_category():
    mis = Finding(scanner="gliner", category_raw="mystery-label",
                  category=None, text="x", confidence=0.7)
    d = doc("d1", matches=[em(SSN, True, correct=False)], mislabeled=[mis])
    strict = metrics.span_confusion([d], strict=True)
    assert strict["per_category"]["unmapped"]["fp"] == 1
    assert strict["overall"]["fp"] == 1
    assert metrics.span_confusion([d], strict=False)["overall"]["fp"] == 0


def test_span_confusion_excludes_scan_error_docs():
    bad = doc("err", matches=[em(SSN, False)], scan_error="boom")
    out = metrics.span_confusion([bad])
    assert out["overall"]["tp"] == 0 and out["overall"]["fn"] == 0
    assert out["per_category"] == {}


# ---------------------------------------------------------------------------
# doc_confusion
# ---------------------------------------------------------------------------

def _doc_fixture():
    dirty_hit = [doc(f"tp{i}") for i in range(2)]                       # tp=2
    dirty_miss = [doc("fn0", flagged=False)]                            # fn=1
    clean_hit = [doc("fp0", is_clean=True, flagged=True)]               # fp=1
    clean_ok = [doc(f"tn{i}", is_clean=True, flagged=False) for i in range(3)]  # tn=3
    errored = [doc("err", scan_error="scanner exploded")]
    return dirty_hit + dirty_miss + clean_hit + clean_ok + errored


def test_doc_confusion_hand_computed():
    out = metrics.doc_confusion(_doc_fixture())
    assert (out["tp"], out["fp"], out["tn"], out["fn"]) == (2, 1, 3, 1)
    assert out["scan_errors"] == 1
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["specificity"] == pytest.approx(3 / 4)
    assert out["accuracy"] == pytest.approx(5 / 7)
    assert out["balanced_accuracy"] == pytest.approx(17 / 24)
    assert out["f1"] == pytest.approx(2 / 3)
    # MCC = (2*3 - 1*1) / sqrt(3 * 3 * 4 * 4) = 5/12
    assert out["mcc"] == pytest.approx(5 / 12)
    assert out["fpr"] == pytest.approx(1 / 4)
    assert out["fnr"] == pytest.approx(1 / 3)


def test_doc_confusion_all_clean_returns_none_ratios():
    out = metrics.doc_confusion([doc(f"c{i}", is_clean=True, flagged=False)
                                 for i in range(4)])
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (0, 0, 0, 4)
    assert out["precision"] is None
    assert out["recall"] is None
    assert out["f1"] is None
    assert out["mcc"] is None
    assert out["balanced_accuracy"] is None
    assert out["fnr"] is None
    assert out["specificity"] == 1.0
    assert out["accuracy"] == 1.0
    assert out["fpr"] == 0.0


def test_doc_confusion_all_dirty_returns_none_specificity():
    out = metrics.doc_confusion([doc("d0"), doc("d1", flagged=False)])
    assert out["specificity"] is None
    assert out["fpr"] is None
    assert out["precision"] == 1.0
    assert out["recall"] == 0.5


def test_doc_confusion_empty_input():
    out = metrics.doc_confusion([])
    assert out["accuracy"] is None and out["mcc"] is None
    assert out["scan_errors"] == 0


# ---------------------------------------------------------------------------
# severity_accuracy
# ---------------------------------------------------------------------------

def test_severity_accuracy():
    evals = [
        doc("a", sev_pred="major", sev_exp="major"),
        doc("b", sev_pred="minor", sev_exp="moderate"),
        doc("c", sev_pred="moderate", sev_exp="moderate"),
        doc("skip-clean", is_clean=True, flagged=True),
        doc("skip-unflagged", flagged=False),
        doc("skip-error", scan_error="boom"),
    ]
    out = metrics.severity_accuracy(evals)
    assert out["n"] == 3
    assert out["exact_match_rate"] == pytest.approx(2 / 3)
    assert out["matrix"]["major"]["major"] == 1
    assert out["matrix"]["moderate"]["minor"] == 1
    assert out["matrix"]["moderate"]["moderate"] == 1
    assert out["matrix"]["minor"]["major"] == 0


def test_severity_accuracy_no_eligible_docs():
    out = metrics.severity_accuracy([doc("c", is_clean=True, flagged=False)])
    assert out["n"] == 0
    assert out["exact_match_rate"] is None


# ---------------------------------------------------------------------------
# recall_by / scope_split
# ---------------------------------------------------------------------------

def test_recall_by_difficulty():
    evals = [doc("d1", matches=[em(SSN, True), em(SSN, True),
                                em(SSN, False),
                                em(EMAIL, True, difficulty="obfuscated"),
                                em(EMAIL, False, difficulty="obfuscated")])]
    out = metrics.recall_by(evals, "difficulty")
    assert out["plain"] == {"tp": 2, "fn": 1, "recall": pytest.approx(2 / 3), "n": 3}
    assert out["obfuscated"] == {"tp": 1, "fn": 1, "recall": 0.5, "n": 2}


def test_recall_by_carrier_reads_doc_attribute():
    evals = [doc("d1", carrier="ticket", matches=[em(SSN, True)]),
             doc("d2", carrier="resume", matches=[em(SSN, False)])]
    out = metrics.recall_by(evals, "carrier")
    assert out["ticket"]["recall"] == 1.0
    assert out["resume"]["recall"] == 0.0


def test_recall_by_rejects_unknown_key():
    with pytest.raises(ValueError):
        metrics.recall_by([], "profile")


def test_scope_split_regex_scope():
    evals = [doc("d1", matches=[em(SSN, True), em(PHONE, False),
                                em(MEDICAL_RECORD, False)])]
    out = metrics.scope_split(evals, REGEX_BUILTIN_SCOPE)
    assert out["in_scope"] == {"tp": 1, "fn": 1, "recall": 0.5, "n": 2}
    assert out["out_of_scope"] == {"tp": 0, "fn": 1, "recall": 0.0, "n": 1}


def test_scope_split_empty_bucket_recall_is_none():
    out = metrics.scope_split([doc("d1", matches=[em(SSN, True)])],
                              REGEX_BUILTIN_SCOPE)
    assert out["out_of_scope"] == {"tp": 0, "fn": 0, "recall": None, "n": 0}


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def _recall_metric(evals):
    return metrics.doc_confusion(evals)["recall"]


def _bootstrap_fixture():
    return ([doc(f"tp{i}") for i in range(10)] +
            [doc(f"fn{i}", flagged=False) for i in range(5)] +
            [doc(f"tn{i}", is_clean=True, flagged=False) for i in range(5)])


def test_bootstrap_ci_reproducible_and_contains_point():
    evals = _bootstrap_fixture()
    a = metrics.bootstrap_ci(evals, _recall_metric, n_boot=300, seed=7)
    b = metrics.bootstrap_ci(evals, _recall_metric, n_boot=300, seed=7)
    assert a == b
    assert a["degenerate"] is False
    assert a["point"] == pytest.approx(2 / 3)
    assert a["lo"] <= a["point"] <= a["hi"]
    assert 0.0 <= a["lo"] <= a["hi"] <= 1.0


def test_bootstrap_ci_different_seed_differs():
    evals = _bootstrap_fixture()
    a = metrics.bootstrap_ci(evals, _recall_metric, n_boot=300, seed=7)
    c = metrics.bootstrap_ci(evals, _recall_metric, n_boot=300, seed=8)
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_bootstrap_ci_degenerate_when_metric_always_none():
    clean = [doc(f"c{i}", is_clean=True, flagged=False) for i in range(5)]
    out = metrics.bootstrap_ci(clean, _recall_metric, n_boot=50, seed=1)
    assert out == {"lo": None, "hi": None, "point": None, "degenerate": True}


def test_bootstrap_ci_empty_input_is_degenerate():
    out = metrics.bootstrap_ci([], _recall_metric)
    assert out["degenerate"] is True and out["lo"] is None


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------

def _labeled(doc_id, cats):
    entities = [GroundTruthEntity(category=c, text="x", start=0, end=1)
                for c in cats]
    return LabeledDocument(doc_id=doc_id, text="body", entities=entities)


def _finding(cat, conf, scanner="gliner"):
    # gliner by default: the sweep gates ONLY gliner findings by confidence.
    return Finding(scanner=scanner, category_raw=cat, category=cat,
                   text="x", confidence=conf)


def _stub_matcher(docs, findings_by_doc_id):
    """Category-equality matcher: an entity is matched iff any surviving
    finding in its doc has the same canonical category."""
    evals = []
    for d in docs:
        fs = findings_by_doc_id.get(d.doc_id, [])
        gt_cats = {e.category for e in d.entities}
        matches = [EntityMatch(entity=e,
                               matched=any(f.category == e.category for f in fs),
                               category_correct=any(f.category == e.category for f in fs))
                   for e in d.entities]
        evals.append(DocEval(doc_id=d.doc_id, profile=d.profile, carrier=d.carrier,
                             is_clean=d.is_clean, entity_matches=matches,
                             false_positives=[f for f in fs if f.category not in gt_cats],
                             doc_flagged=bool(fs), findings_count=len(fs)))
    return evals


def _sweep_fixture():
    docs = [_labeled("A", [SSN]), _labeled("B", [EMAIL]),
            _labeled("C", []), _labeled("D", [])]
    findings = {
        "A": [_finding(SSN, 0.95), _finding(EMAIL, 0.3)],   # hit + FP
        "B": [_finding(EMAIL, 0.6)],                        # hit
        "C": [_finding(PHONE, 0.4)],                        # FP on a clean doc
        "D": [],
    }
    return docs, findings


def test_threshold_sweep_recall_monotone_and_hand_values():
    docs, findings = _sweep_fixture()
    out = metrics.threshold_sweep(docs, findings, [0.0, 0.5, 0.7, 1.0],
                                  _stub_matcher)
    pts = out["points"]
    assert [pt["threshold"] for pt in pts] == [0.0, 0.5, 0.7, 1.0]

    recalls = [pt["span_recall"] for pt in pts]
    assert recalls == [1.0, 1.0, 0.5, 0.0]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))  # non-increasing

    assert pts[0]["span_precision"] == pytest.approx(0.5)   # 2 tp / (2 tp + 2 fp)
    assert pts[0]["doc_fpr"] == pytest.approx(0.5)          # clean C flagged
    assert pts[1]["span_f1"] == pytest.approx(1.0)          # perfect at 0.5
    assert pts[1]["doc_fpr"] == 0.0
    assert pts[3]["span_precision"] is None                 # no findings survive
    assert pts[3]["doc_recall"] == 0.0

    assert out["best_f1"] == {"threshold": 0.5, "f1": pytest.approx(1.0)}


def test_threshold_sweep_aucs():
    docs, findings = _sweep_fixture()
    out = metrics.threshold_sweep(docs, findings, [0.0, 0.5, 0.7, 1.0],
                                  _stub_matcher)
    # PR points (recall, precision): (1.0, 0.5), (1.0, 1.0), (0.5, 1.0); the
    # t=1.0 point has None precision and is dropped. Sorted by recall:
    # (0.5,1.0)->(1.0,0.5) trapezoid = 0.5*(1.5)/2 = 0.375; the vertical pair
    # at recall=1.0 adds zero area.
    assert out["pr_auc_partial"] == pytest.approx(0.375)
    # ROC points (fpr, tpr): (0.5,1.0), (0.0,1.0), (0.0,0.5), (0.0,0.0) ->
    # only the segment fpr 0.0->0.5 at tpr 1.0 has width: area 0.5.
    assert out["roc_auc_partial"] == pytest.approx(0.5)
    # Deprecated aliases (one release) carry the same partial areas.
    assert out["pr_auc"] == out["pr_auc_partial"]
    assert out["roc_auc"] == out["roc_auc_partial"]


def test_threshold_sweep_filters_only_gliner_findings():
    # Finding [11]: production's dlp.gliner.threshold gates only gliner, so
    # the sweep must not drop regex/keyword findings — every point stays a
    # config the production knob can express.
    docs = [_labeled("A", [SSN]), _labeled("B", [EMAIL]), _labeled("C", [])]
    findings = {
        "A": [_finding(SSN, 0.9, scanner="regex")],    # keyword-style conf 0.9
        "B": [_finding(EMAIL, 0.4, scanner="gliner")],
        "C": [],
    }
    out = metrics.threshold_sweep(docs, findings, [0.05, 0.5, 0.95],
                                  _stub_matcher)
    recalls = [pt["span_recall"] for pt in out["points"]]
    # gliner EMAIL (0.4) drops at t=0.5; regex SSN survives even t=0.95>0.9.
    assert recalls == [1.0, 0.5, 0.5]


def test_threshold_sweep_no_valid_points():
    docs = [_labeled("A", [SSN])]
    out = metrics.threshold_sweep(docs, {"A": []}, [0.5], _stub_matcher)
    assert out["pr_auc_partial"] is None
    assert out["roc_auc_partial"] is None
    assert out["pr_auc"] is None       # deprecated alias
    assert out["roc_auc"] is None      # deprecated alias
    assert out["best_f1"] == {"threshold": None, "f1": None}


# ---------------------------------------------------------------------------
# percentile / summarize_latencies
# ---------------------------------------------------------------------------

def test_percentile_linear_interpolation():
    assert metrics.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert metrics.percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)  # unsorted ok
    assert metrics.percentile([1, 2, 3, 4], 25) == pytest.approx(1.75)
    assert metrics.percentile([1, 2, 3, 4], 0) == 1.0
    assert metrics.percentile([1, 2, 3, 4], 100) == 4.0
    assert metrics.percentile([5], 99) == 5.0
    assert metrics.percentile([], 50) is None


def test_percentile_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        metrics.percentile([1, 2], 101)


def test_summarize_latencies():
    out = metrics.summarize_latencies([10, 20, 30, 40])
    assert out["n"] == 4
    assert out["mean"] == pytest.approx(25.0)
    assert out["p50"] == pytest.approx(25.0)
    assert out["p90"] == pytest.approx(37.0)
    assert out["max"] == 40


def test_summarize_latencies_empty():
    out = metrics.summarize_latencies([])
    assert out == {"n": 0, "mean": None, "p50": None, "p90": None,
                   "p95": None, "p99": None, "max": None}
