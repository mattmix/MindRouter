############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/matching.py: Finding normalization and
# ground-truth span matching — turns raw scanner output plus
# a labeled document into a DocEval.
#
############################################################

"""Normalize scanner findings and match them against ground-truth spans.

``normalize_findings`` adapts the two raw shapes the harness encounters —
JSONL dicts (offline evaluator / container runner) and ScanFinding-style
objects (dlp_scanner.py) — into the canonical ``Finding`` dataclass.

``match_document`` implements the harness matching semantics:

* A finding OVERLAPS an entity when their char ranges share at least one
  character; when ``iou_threshold`` > 0 the intersection/union of the two
  ranges must also reach the threshold. The same predicate governs entity
  matching, false-positive determination, and trap attribution.
* LLM findings carry no spans (start == end == 0): they match an entity by
  verbatim text containment in either direction, or by canonical-category
  equality when the doc has exactly one entity of that category.
* ``EntityMatch.matched`` is LENIENT (any overlapping finding, category
  ignored); ``category_correct`` is STRICT (an overlapping finding whose
  canonical category equals the entity's).
* A finding that overlaps >=1 entity but category-matches NONE of the
  entities it overlaps lands in ``DocEval.mislabeled_findings`` (counted as
  an FP by strict span confusion; disjoint from ``false_positives``).
* Findings that match no entity are false positives; those overlapping a
  negative trap span are attributed to the trap's generator.
* ``severity_predicted`` mirrors classify_severity() in dlp_scanner.py with
  the production default rules, keyed on canonical categories.
"""

from collections import Counter
from typing import Any, Iterable, List, Mapping, Optional

from . import constants
from .schemas import DocEval, EntityMatch, Finding, GroundTruthEntity, LabeledDocument


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _as_finding(raw: Any) -> Finding:
    if isinstance(raw, Finding):
        return raw
    if isinstance(raw, dict):
        def get(key, default=None):
            return raw.get(key, default)
    else:
        def get(key, default=None):
            return getattr(raw, key, default)
    # Serialized Finding dicts carry "category_raw"; scanner shapes carry
    # only "category" (the raw label).
    category_raw = str(get("category_raw") or get("category") or "")
    start = int(get("start") or 0)
    end = int(get("end") or 0)
    if end < start:
        end = start  # defensive: malformed span collapses to zero-width
    return Finding(
        scanner=str(get("scanner") or ""),
        category_raw=category_raw,
        category=constants.canonicalize(category_raw),
        text=str(get("text") or ""),
        confidence=float(get("confidence") or 0.0),
        start=start,
        end=end,
    )


def normalize_findings(raw_findings: Optional[Iterable[Any]]) -> List[Finding]:
    """Adapt raw findings (dicts or ScanFinding-style objects) to Finding.

    Already-normalized ``Finding`` instances pass through unchanged (their
    ``category_raw`` is not re-canonicalizable, so re-normalizing would be
    lossy).
    """
    return [_as_finding(f) for f in (raw_findings or [])]


# ---------------------------------------------------------------------------
# Matching predicates
# ---------------------------------------------------------------------------

def _overlaps(f: Finding, start: int, end: int, iou_threshold: float) -> bool:
    inter = min(f.end, end) - max(f.start, start)
    if inter <= 0:
        return False
    if iou_threshold > 0.0:
        union = (f.end - f.start) + (end - start) - inter
        if union <= 0 or inter / union < iou_threshold:
            return False
    return True


def _is_spanless(f: Finding) -> bool:
    return f.scanner == "llm" and f.start == 0 and f.end == 0


def _spanless_matches(f: Finding, entity: GroundTruthEntity,
                      category_counts: Counter) -> bool:
    if f.text and entity.text and (entity.text in f.text or f.text in entity.text):
        return True
    return (f.category is not None
            and f.category == entity.category
            and category_counts.get(entity.category, 0) == 1)


# ---------------------------------------------------------------------------
# Document / corpus evaluation
# ---------------------------------------------------------------------------

def match_document(doc: LabeledDocument, findings: List[Finding],
                   scan_latency_ms: float = 0.0,
                   scan_error: Optional[str] = None,
                   iou_threshold: float = 0.0) -> DocEval:
    """Match findings against one document's ground truth into a DocEval."""
    findings = normalize_findings(findings)
    category_counts = Counter(e.category for e in doc.entities)

    matched_finding_idx = set()
    correct_finding_idx = set()   # consumed findings that category-matched an entity
    entity_matches: List[EntityMatch] = []
    for entity in doc.entities:
        scanners = set()
        best_confidence = 0.0
        category_correct = False
        for i, f in enumerate(findings):
            if _is_spanless(f):
                hit = _spanless_matches(f, entity, category_counts)
            else:
                hit = _overlaps(f, entity.start, entity.end, iou_threshold)
            if not hit:
                continue
            matched_finding_idx.add(i)
            scanners.add(f.scanner)
            best_confidence = max(best_confidence, f.confidence)
            if f.category is not None and f.category == entity.category:
                category_correct = True
                correct_finding_idx.add(i)
        entity_matches.append(EntityMatch(
            entity=entity,
            matched=bool(scanners),
            matched_by=sorted(scanners),
            best_confidence=best_confidence,
            category_correct=category_correct,
        ))

    mislabeled = [findings[i]
                  for i in sorted(matched_finding_idx - correct_finding_idx)]

    false_positives: List[Finding] = []
    trap_hits = set()
    for i, f in enumerate(findings):
        if i in matched_finding_idx:
            continue
        false_positives.append(f)
        if _is_spanless(f):
            continue  # no span to attribute to a trap
        for neg in doc.negatives:
            if _overlaps(f, neg.start, neg.end, iou_threshold):
                trap_hits.add(neg.generator)

    return DocEval(
        doc_id=doc.doc_id,
        profile=doc.profile,
        carrier=doc.carrier,
        is_clean=doc.is_clean,
        scan_error=scan_error,
        entity_matches=entity_matches,
        false_positives=false_positives,
        mislabeled_findings=mislabeled,
        fp_trap_hits=sorted(trap_hits),
        doc_flagged=bool(findings),
        severity_predicted=_predict_severity(findings),
        severity_expected=constants.expected_severity(
            e.category for e in doc.entities),
        scan_latency_ms=float(scan_latency_ms),
        text_chars=len(doc.text),
        findings_count=len(findings),
    )


def _predict_severity(findings: List[Finding]) -> str:
    # Mirrors classify_severity() (dlp_scanner.py) with the production
    # default rules, keyed on canonical categories; None (unmapped) takes
    # the fallback like any unknown category.
    if not findings:
        return "minor"
    highest = "minor"
    for f in findings:
        sev = constants.DEFAULT_SEVERITY_BY_CATEGORY.get(
            f.category, constants.SEVERITY_FALLBACK)
        if constants.SEVERITY_ORDER[sev] > constants.SEVERITY_ORDER[highest]:
            highest = sev
    return highest


def match_corpus(docs: Iterable[LabeledDocument],
                 findings_by_doc_id: Mapping[str, Iterable[Any]],
                 latencies_by_doc_id: Optional[Mapping[str, float]] = None,
                 errors_by_doc_id: Optional[Mapping[str, str]] = None,
                 iou_threshold: float = 0.0) -> List[DocEval]:
    """match_document over a corpus; doc ids absent from the maps default
    to no findings / 0.0 ms / no error."""
    latencies_by_doc_id = latencies_by_doc_id or {}
    errors_by_doc_id = errors_by_doc_id or {}
    return [
        match_document(
            doc,
            normalize_findings(findings_by_doc_id.get(doc.doc_id)),
            scan_latency_ms=latencies_by_doc_id.get(doc.doc_id, 0.0),
            scan_error=errors_by_doc_id.get(doc.doc_id),
            iou_threshold=iou_threshold,
        )
        for doc in docs
    ]
