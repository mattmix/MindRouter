############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/schemas.py: Typed artifacts every harness
# module reads and writes — labeled documents, findings,
# match results, run manifests — plus JSONL persistence.
#
# stdlib-only on purpose: corpus generation, offline eval,
# and in-container execution all import this file, and the
# app container has no harness dependencies installed.
#
############################################################

"""Data contracts for the DLP evaluation harness."""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthEntity:
    """One planted sensitive item with its exact span in the document text."""
    category: str            # canonical key from constants.CANONICAL_CATEGORIES
    text: str                # the exact planted string
    start: int               # char offset in LabeledDocument.text
    end: int                 # exclusive char offset
    generator: str = ""      # which generator produced it (e.g. "ssn.dashed")
    difficulty: str = "plain"   # "plain" | "obfuscated" | "boundary"
    obfuscation: str = ""    # e.g. "spaced_digits", "unicode_homoglyph"
    attrs: Dict[str, Any] = field(default_factory=dict)   # e.g. {"luhn_valid": true}


@dataclass
class LabeledDocument:
    """A synthetic document with span-exact ground truth.

    ``is_clean`` documents contain zero sensitive entities (they may contain
    hard negatives — PII lookalikes — recorded in ``negatives`` so false
    positives can be attributed to the trap that caused them).
    """
    doc_id: str
    text: str
    entities: List[GroundTruthEntity] = field(default_factory=list)
    negatives: List[GroundTruthEntity] = field(default_factory=list)  # traps; category = what they mimic
    profile: str = "accuracy"    # corpus profile that produced this doc
    carrier: str = ""            # template family (e.g. "support_ticket")
    seed: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not self.entities


# ---------------------------------------------------------------------------
# Scanner output (normalized)
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A scanner finding, normalized for matching."""
    scanner: str             # "regex" | "gliner" | "llm"
    category_raw: str        # label as emitted by the scanner
    category: Optional[str]  # canonical key (None if unmapped)
    text: str
    confidence: float
    start: int = 0
    end: int = 0             # llm findings carry no spans (0,0)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

@dataclass
class EntityMatch:
    """Outcome for one ground-truth entity after matching against findings."""
    entity: GroundTruthEntity
    matched: bool
    matched_by: List[str] = field(default_factory=list)     # scanners that hit it
    best_confidence: float = 0.0
    category_correct: bool = False   # a matching finding also had the right category


@dataclass
class DocEval:
    """Full evaluation of one document against one scanner configuration."""
    doc_id: str
    profile: str
    carrier: str
    is_clean: bool
    scan_error: Optional[str] = None      # scanner raised; excluded from accuracy, counted separately
    entity_matches: List[EntityMatch] = field(default_factory=list)
    false_positives: List[Finding] = field(default_factory=list)
    # Findings that overlapped >=1 entity but category-matched none of the
    # entities they overlap (strict-mode FPs; disjoint from false_positives).
    mislabeled_findings: List[Finding] = field(default_factory=list)
    fp_trap_hits: List[str] = field(default_factory=list)   # generator names of triggered negatives
    doc_flagged: bool = False             # any finding at all (mirrors "alert would fire")
    severity_predicted: str = "minor"
    severity_expected: str = "minor"
    scan_latency_ms: float = 0.0
    text_chars: int = 0
    findings_count: int = 0


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """Identity + provenance of one harness run directory."""
    run_id: str
    kind: str                  # "corpus" | "offline" | "e2e" | "load" | "full"
    created_at: str
    argv: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    corpus_path: Optional[str] = None
    base_url: Optional[str] = None
    scanner_mode: Optional[str] = None
    git_rev: Optional[str] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------

_TYPES = {
    "GroundTruthEntity": GroundTruthEntity,
    "LabeledDocument": LabeledDocument,
    "Finding": Finding,
    "EntityMatch": EntityMatch,
    "DocEval": DocEval,
    "RunManifest": RunManifest,
}

# Nested dataclass fields that need recursive rehydration.
_NESTED = {
    LabeledDocument: {"entities": GroundTruthEntity, "negatives": GroundTruthEntity},
    EntityMatch: {"entity": GroundTruthEntity},
    DocEval: {"entity_matches": EntityMatch, "false_positives": Finding,
              "mislabeled_findings": Finding},
}


def to_dict(obj) -> dict:
    return dataclasses.asdict(obj)


def from_dict(cls, data: dict):
    """Rehydrate a dataclass (recursively for known nested fields)."""
    kwargs = {}
    nested = _NESTED.get(cls, {})
    names = {f.name for f in dataclasses.fields(cls)}
    for k, v in data.items():
        if k not in names:
            continue  # forward-compatible: ignore unknown keys
        sub = nested.get(k)
        if sub is not None and v is not None:
            if isinstance(v, list):
                kwargs[k] = [from_dict(sub, item) for item in v]
            else:
                kwargs[k] = from_dict(sub, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def write_jsonl(path: str, objs) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(to_dict(o) if dataclasses.is_dataclass(o) else o,
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str, cls=None):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            out.append(from_dict(cls, data) if cls is not None else data)
    return out


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "dlp_harness_runs")


def new_run_dir(kind: str, run_id: Optional[str] = None) -> str:
    """Create dlp_harness_runs/<utc-ts>-<kind>/ and return its path.

    Auto-generated ids are uniquified on collision — the timestamp has
    1-second granularity and back-to-back CLI invocations otherwise land in
    (and silently overwrite) the same directory.
    """
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = f"{ts}-{kind}"
        n = 1
        while os.path.exists(os.path.join(RUNS_ROOT, candidate)):
            n += 1
            candidate = f"{ts}-{kind}-{n}"
        run_id = candidate
    path = os.path.join(RUNS_ROOT, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_manifest(run_dir: str, manifest: RunManifest) -> None:
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(to_dict(manifest), f, indent=2)


def load_manifest(run_dir: str) -> RunManifest:
    with open(os.path.join(run_dir, "run.json"), "r", encoding="utf-8") as f:
        return from_dict(RunManifest, json.load(f))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
