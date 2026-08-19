############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/metrics.py: Statistics engine for the DLP
# evaluation harness — span/document confusion, grouped
# recall, scope splits, bootstrap CIs, threshold sweeps,
# and latency summaries over DocEval lists.
#
# stdlib-only; every ratio guards its denominator and
# returns None (never NaN, never a crash) when undefined.
# Docs with scan_error are excluded from accuracy math and
# surfaced separately, mirroring SCANNER_ERROR_CATEGORY.
#
############################################################

"""Statistics engine for the DLP evaluation harness."""

import math
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from .constants import CANONICAL_CATEGORIES, SEVERITY_ORDER
from .schemas import DocEval, Finding, LabeledDocument

UNMAPPED = "unmapped"    # FP attribution bucket for findings with no canonical category


# ---------------------------------------------------------------------------
# Guarded arithmetic
# ---------------------------------------------------------------------------

def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2.0 * p * r / (p + r)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _scanned(doc_evals: Iterable[DocEval]) -> Iterable[DocEval]:
    return (d for d in doc_evals if not d.scan_error)


# ---------------------------------------------------------------------------
# Span-level (entity) confusion
# ---------------------------------------------------------------------------

def span_confusion(doc_evals: List[DocEval], strict: bool = False) -> dict:
    """Entity-level {tp, fn, fp, precision, recall, f1} per category + overall.

    strict=False counts any detected ground-truth entity as TP; strict=True
    requires the detecting finding to also carry the right category, so a
    detected-but-mislabeled entity counts FN AND the mislabeling finding
    (DocEval.mislabeled_findings) counts FP under its claimed category —
    keeping strict precision and recall over consistent populations. False
    positives are attributed to the finding's canonical category (None ->
    "unmapped"). The overall row is the micro-average of the counts and
    additionally carries macro_* means over every category with at least one
    ground-truth entity or FP.
    """
    per: Dict[str, Dict[str, int]] = {}

    def cell(cat: str) -> Dict[str, int]:
        return per.setdefault(cat, {"tp": 0, "fn": 0, "fp": 0})

    for d in _scanned(doc_evals):
        for em in d.entity_matches:
            hit = em.matched and (em.category_correct if strict else True)
            cell(em.entity.category)["tp" if hit else "fn"] += 1
        for f in d.false_positives:
            cell(f.category if f.category is not None else UNMAPPED)["fp"] += 1
        if strict:
            # A finding that overlapped entities but category-matched none of
            # them is an FP against its claimed category in strict mode.
            for f in d.mislabeled_findings:
                cell(f.category if f.category is not None else UNMAPPED)["fp"] += 1

    # Stable output order: canonical taxonomy first, then extras alphabetically.
    extras = sorted(k for k in per if k not in CANONICAL_CATEGORIES)
    per_category = {}
    for cat in list(CANONICAL_CATEGORIES) + extras:
        if cat not in per:
            continue
        c = per[cat]
        p = _ratio(c["tp"], c["tp"] + c["fp"])
        r = _ratio(c["tp"], c["tp"] + c["fn"])
        per_category[cat] = {"tp": c["tp"], "fn": c["fn"], "fp": c["fp"],
                             "precision": p, "recall": r, "f1": _f1(p, r)}

    tp = sum(c["tp"] for c in per.values())
    fn = sum(c["fn"] for c in per.values())
    fp = sum(c["fp"] for c in per.values())
    p = _ratio(tp, tp + fp)
    r = _ratio(tp, tp + fn)
    # Every row in per exists because it counted >=1 GT entity or FP, so all
    # rows are macro-eligible; undefined (None) ratios are skipped by _mean.
    overall = {
        "tp": tp, "fn": fn, "fp": fp,
        "precision": p, "recall": r, "f1": _f1(p, r),
        "macro_precision": _mean(row["precision"] for row in per_category.values()),
        "macro_recall": _mean(row["recall"] for row in per_category.values()),
        "macro_f1": _mean(row["f1"] for row in per_category.values()),
    }
    return {"per_category": per_category, "overall": overall}


# ---------------------------------------------------------------------------
# Document-level confusion
# ---------------------------------------------------------------------------

def doc_confusion(doc_evals: List[DocEval]) -> dict:
    """Document-level 2x2 (doc_flagged vs is_clean) with derived rates.

    Positive class = dirty document. Docs with scan_error are excluded from
    the matrix and reported in scan_errors.
    """
    tp = fp = tn = fn = scan_errors = 0
    for d in doc_evals:
        if d.scan_error:
            scan_errors += 1
        elif not d.is_clean:
            tp, fn = (tp + 1, fn) if d.doc_flagged else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if d.doc_flagged else (fp, tn + 1)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    mcc_den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": (None if recall is None or specificity is None
                              else (recall + specificity) / 2.0),
        "f1": _f1(precision, recall),
        "mcc": (tp * tn - fp * fn) / math.sqrt(mcc_den) if mcc_den else None,
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "scan_errors": scan_errors,
    }


# ---------------------------------------------------------------------------
# Severity accuracy
# ---------------------------------------------------------------------------

def severity_accuracy(doc_evals: List[DocEval]) -> dict:
    """Predicted-vs-expected severity confusion over dirty flagged docs.

    matrix[expected][predicted] = count. Only dirty, flagged, error-free docs
    participate (severity is meaningless where no alert would fire).
    """
    levels = list(SEVERITY_ORDER)
    matrix: Dict[str, Dict[str, int]] = {e: {p: 0 for p in levels} for e in levels}
    n = exact = 0
    for d in _scanned(doc_evals):
        if d.is_clean or not d.doc_flagged:
            continue
        row = matrix.setdefault(d.severity_expected, {})   # tolerate off-model labels
        row[d.severity_predicted] = row.get(d.severity_predicted, 0) + 1
        n += 1
        exact += d.severity_predicted == d.severity_expected
    return {"matrix": matrix, "n": n, "exact_match_rate": _ratio(exact, n)}


# ---------------------------------------------------------------------------
# Grouped recall
# ---------------------------------------------------------------------------

_RECALL_BY_KEYS = {"difficulty", "obfuscation", "generator", "carrier", "category"}


def recall_by(doc_evals: List[DocEval], key: str) -> dict:
    """Recall grouped by a ground-truth attribute -> {group: {tp, fn, recall, n}}.

    key is one of difficulty | obfuscation | generator | category (read from
    each GroundTruthEntity) or carrier (read from the DocEval).
    """
    if key not in _RECALL_BY_KEYS:
        raise ValueError(f"recall_by key must be one of {sorted(_RECALL_BY_KEYS)}, got {key!r}")
    groups: Dict[str, Dict[str, int]] = {}
    for d in _scanned(doc_evals):
        for em in d.entity_matches:
            g = d.carrier if key == "carrier" else getattr(em.entity, key)
            row = groups.setdefault(g, {"tp": 0, "fn": 0})
            row["tp" if em.matched else "fn"] += 1
    return {g: {"tp": row["tp"], "fn": row["fn"],
                "recall": _ratio(row["tp"], row["tp"] + row["fn"]),
                "n": row["tp"] + row["fn"]}
            for g, row in sorted(groups.items())}


def scope_split(doc_evals: List[DocEval], scope: Set[str]) -> dict:
    """Recall split by whether the entity's category is in the scanner's scope.

    in_scope recall is the fair per-scanner number (it could have seen the
    category); out_of_scope recall exposes the honest system-level gap.
    """
    buckets = {"in_scope": {"tp": 0, "fn": 0}, "out_of_scope": {"tp": 0, "fn": 0}}
    for d in _scanned(doc_evals):
        for em in d.entity_matches:
            row = buckets["in_scope" if em.entity.category in scope else "out_of_scope"]
            row["tp" if em.matched else "fn"] += 1
    return {name: {"tp": row["tp"], "fn": row["fn"],
                   "recall": _ratio(row["tp"], row["tp"] + row["fn"]),
                   "n": row["tp"] + row["fn"]}
            for name, row in buckets.items()}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(doc_evals: List[DocEval],
                 metric_fn: Callable[[List[DocEval]], Optional[float]],
                 n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap over DOCUMENTS -> {lo, hi, point, degenerate}.

    Resamples whole documents with replacement (entities within a doc are
    correlated, so the doc is the exchangeable unit). Resamples where
    metric_fn returns None are skipped; if more than 10% are None the
    interval is meaningless and lo=hi=None with degenerate=True.
    """
    point = metric_fn(doc_evals) if doc_evals else None
    if not doc_evals:
        return {"lo": None, "hi": None, "point": point, "degenerate": True}
    rng = random.Random(seed)
    n = len(doc_evals)
    values: List[float] = []
    skipped = 0
    for _ in range(n_boot):
        sample = [doc_evals[rng.randrange(n)] for _ in range(n)]
        v = metric_fn(sample)
        if v is None:
            skipped += 1
        else:
            values.append(v)
    if skipped > 0.10 * n_boot or not values:
        return {"lo": None, "hi": None, "point": point, "degenerate": True}
    return {"lo": percentile(values, 100.0 * (alpha / 2.0)),
            "hi": percentile(values, 100.0 * (1.0 - alpha / 2.0)),
            "point": point, "degenerate": False}


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(docs: List[LabeledDocument],
                    findings_by_doc_id: Dict[str, List[Finding]],
                    thresholds: Sequence[float],
                    matcher: Callable[[List[LabeledDocument], Dict[str, List[Finding]]],
                                      List[DocEval]]) -> dict:
    """Sweep a confidence cutoff over findings and report the tradeoff curves.

    matcher(docs, filtered_findings_by_doc_id) -> List[DocEval] is injected so
    this module never imports the matching module (no circular import). Each
    threshold drops only GLINER findings below it — production's
    dlp.gliner.threshold is the only confidence knob, so regex/keyword
    findings always survive and every sweep point is a config expressible in
    production. Returns {points, pr_auc_partial, roc_auc_partial, best_f1}
    where points carry span- and doc-level rates per threshold, the
    *_partial values are trapezoid areas over ONLY the observed curve points
    (not anchored at (0,0)/(1,1), so NOT comparable to a conventional AUC or
    a 0.5 chance line; None with fewer than two defined points), and best_f1
    selects the threshold maximizing span-level F1. Deprecated aliases
    pr_auc/roc_auc carry the same partial areas for one release.
    """
    points = []
    for t in thresholds:
        filtered = {doc_id: [f for f in fs
                             if f.scanner != "gliner" or f.confidence >= t]
                    for doc_id, fs in findings_by_doc_id.items()}
        evals = matcher(docs, filtered)
        span = span_confusion(evals)["overall"]
        doc = doc_confusion(evals)
        points.append({
            "threshold": t,
            "span_precision": span["precision"],
            "span_recall": span["recall"],
            "span_f1": span["f1"],
            "doc_precision": doc["precision"],
            "doc_recall": doc["recall"],
            "doc_specificity": doc["specificity"],
            "doc_fpr": doc["fpr"],
        })

    best = max((pt for pt in points if pt["span_f1"] is not None),
               key=lambda pt: pt["span_f1"], default=None)
    pr_auc = _trapezoid_auc([(pt["span_recall"], pt["span_precision"]) for pt in points])
    roc_auc = _trapezoid_auc([(pt["doc_fpr"], pt["doc_recall"]) for pt in points])
    return {
        "points": points,
        "pr_auc_partial": pr_auc,
        "roc_auc_partial": roc_auc,
        # Deprecated aliases (kept one release) for pre-rename consumers.
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_f1": ({"threshold": best["threshold"], "f1": best["span_f1"]}
                    if best is not None else {"threshold": None, "f1": None}),
    }


def _trapezoid_auc(pairs: List[tuple]) -> Optional[float]:
    """Trapezoid area under (x, y) points; None-coordinate points are dropped."""
    pts = sorted((x, y) for x, y in pairs if x is not None and y is not None)
    if len(pts) < 2:
        return None
    return sum((x2 - x1) * (y1 + y2) / 2.0
               for (x1, y1), (x2, y2) in zip(pts, pts[1:]))


# ---------------------------------------------------------------------------
# Percentiles / latency summaries
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy default method); None if empty."""
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    # DB-sourced values arrive as decimal.Decimal (TIMESTAMPDIFF math); mixed
    # Decimal/float arithmetic raises, so coerce up front.
    vs = sorted(float(v) for v in values)
    rank = (len(vs) - 1) * (p / 100.0)
    lo_i = math.floor(rank)
    hi_i = math.ceil(rank)
    if lo_i == hi_i:
        return float(vs[lo_i])
    return vs[lo_i] + (rank - lo_i) * (vs[hi_i] - vs[lo_i])


def summarize_latencies(values_ms: Sequence[float]) -> dict:
    """{n, mean, p50, p90, p95, p99, max} over a latency sample (ms)."""
    vs = list(values_ms)
    if not vs:
        return {"n": 0, "mean": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None}
    return {"n": len(vs), "mean": sum(vs) / len(vs),
            "p50": percentile(vs, 50), "p90": percentile(vs, 90),
            "p95": percentile(vs, 95), "p99": percentile(vs, 99),
            "max": max(vs)}
