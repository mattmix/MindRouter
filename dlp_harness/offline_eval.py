############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/offline_eval.py: Offline orchestration — run
# the production DLP scanners over a labeled corpus (locally
# via scanner_bridge or inside the app container via
# container_runner) and reduce the findings to the
# offline_metrics.json artifact.
#
############################################################

"""Offline evaluation orchestration for the DLP harness.

No gateway, no HarnessDB: this module exercises the scanner code itself
(dlp_scanner.py) against span-exact ground truth. The in-container path
shells out to ``docker compose`` following the recipe documented in
container_runner.py, so GLiNER runs with the app's own weights and deps.

Artifacts written to ``out_dir``:

* ``offline_findings.jsonl`` — one line per doc:
  ``{"doc_id", "findings": [...], "latency_ms": {...}, "errors": {...}}``
* ``offline_metrics.json`` — the full metrics artifact (shape below)
* ``run.json`` — RunManifest
"""

import json
import os
import subprocess
import sys
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from dlp_harness import constants, matching, metrics, scanner_bridge
from dlp_harness.schemas import (
    LabeledDocument, RunManifest, read_jsonl, save_manifest, utc_now_iso,
    write_jsonl,
)

# Progress cadence for local scans (docs per scan_documents call).
_CHUNK = 100

# GLiNER threshold used for the single scan a sweep is derived from: scan
# once at the floor, then metrics.threshold_sweep re-filters by confidence.
SWEEP_SCAN_THRESHOLD = 0.05

DEFAULT_SWEEP_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))

# Doc-length buckets for latency_by_length: [lo, hi) pairs over
# (0, 500, 2000, 10000, 50000, 200001), plus an overflow bucket.
_LEN_EDGES = (0, 500, 2000, 10000, 50000, 200001)

# Fixed in-container paths (see container_runner.py's documented recipe).
_C_HARNESS = "/tmp/dlp_harness"
_C_CORPUS = "/tmp/dlp_harness_corpus.jsonl"
_C_FINDINGS = "/tmp/dlp_harness_findings.jsonl"

_SUPPORTED = ("regex", "gliner")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _resolve_corpus(docs: Optional[Iterable[LabeledDocument]],
                    corpus_path: Optional[str]) -> List[LabeledDocument]:
    if docs is None and corpus_path is None:
        raise ValueError("run_offline requires docs or corpus_path")
    if docs is not None:
        return list(docs)
    path = corpus_path
    if os.path.isdir(path):
        path = os.path.join(path, "corpus.jsonl")
    return read_jsonl(path, LabeledDocument)


# ---------------------------------------------------------------------------
# Scanning: local (scanner_bridge) and in-container (subprocess)
# ---------------------------------------------------------------------------

def _scan_local(docs: List[LabeledDocument], scanners: Sequence[str],
                gliner_threshold: float, gliner_categories: Optional[List[str]],
                gliner_max_chars: int,
                progress: Optional[Callable[[str], None]]) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    total = len(docs)
    for i in range(0, total, _CHUNK):
        chunk = docs[i:i + _CHUNK]
        results.update(scanner_bridge.scan_documents(
            {d.doc_id: d.text for d in chunk},
            scanners=scanners,
            gliner_threshold=gliner_threshold,
            gliner_categories=gliner_categories,
            gliner_max_chars=gliner_max_chars,
        ))
        if progress:
            progress(f"[offline] scanned {min(i + _CHUNK, total)}/{total} docs")
    return results


def _compose(cmd: List[str], compose_dir: str,
             timeout_s: float = 3600.0) -> subprocess.CompletedProcess:
    full = ["docker", "compose"] + cmd
    try:
        proc = subprocess.run(full, cwd=compose_dir, capture_output=True,
                              text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"in-container scan failed: {' '.join(full)}: {e}") from e
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-2000:]
        raise RuntimeError(
            f"in-container scan failed: {' '.join(full)} "
            f"(exit {proc.returncode}): {tail}")
    return proc


def _scan_in_container(docs: List[LabeledDocument], corpus_path: Optional[str],
                       out_dir: str, scanners: Sequence[str],
                       gliner_threshold: float, gliner_max_chars: int,
                       compose_dir: str,
                       progress: Optional[Callable[[str], None]]) -> Dict[str, dict]:
    corpus_file = corpus_path
    if corpus_file and os.path.isdir(corpus_file):
        corpus_file = os.path.join(corpus_file, "corpus.jsonl")
    if not corpus_file or not os.path.isfile(corpus_file):
        corpus_file = os.path.join(out_dir, "container_corpus.jsonl")
        write_jsonl(corpus_file, docs)

    findings_local = os.path.join(out_dir, "container_findings.jsonl")
    harness_dir = os.path.dirname(os.path.abspath(__file__))

    if progress:
        progress(f"[offline] copying harness + corpus into app container "
                 f"(compose dir {compose_dir})")
    # rm first: `docker compose cp DIR app:EXISTING_DIR` nests instead of replacing.
    _compose(["exec", "-T", "app", "rm", "-rf", _C_HARNESS], compose_dir)
    _compose(["cp", harness_dir, f"app:{_C_HARNESS}"], compose_dir)
    _compose(["cp", corpus_file, f"app:{_C_CORPUS}"], compose_dir)
    if progress:
        progress(f"[offline] scanning {len(docs)} docs in container "
                 f"({','.join(scanners)}) — gliner model load can take minutes")
    _compose(["exec", "-T", "app", "python", f"{_C_HARNESS}/container_runner.py",
              "--corpus", _C_CORPUS, "--out", _C_FINDINGS,
              "--scanners", ",".join(scanners),
              "--gliner-threshold", str(gliner_threshold),
              "--gliner-max-chars", str(gliner_max_chars)], compose_dir)
    _compose(["cp", f"app:{_C_FINDINGS}", findings_local], compose_dir)

    results: Dict[str, dict] = {}
    for row in read_jsonl(findings_local):
        results[str(row["doc_id"])] = {
            "findings": row.get("findings") or [],
            "latency_ms": row.get("latency_ms") or {},
            "errors": row.get("errors") or {},
        }
    return results


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------

def _scope_for(scanners: Sequence[str],
               gliner_categories: Optional[List[str]]) -> set:
    scope: set = set()
    if "regex" in scanners:
        scope |= constants.REGEX_BUILTIN_SCOPE
    if "gliner" in scanners:
        if gliner_categories:
            scope |= {c for c in (constants.canonicalize(l) for l in gliner_categories)
                      if c is not None}
        else:
            scope |= constants.GLINER_DEFAULT_SCOPE
    return scope


def _len_bucket(n: int) -> str:
    for lo, hi in zip(_LEN_EDGES, _LEN_EDGES[1:]):
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return f">={_LEN_EDGES[-1]}"


def _latency_by_length(docs: List[LabeledDocument],
                       results: Dict[str, dict],
                       scanners: Sequence[str]) -> Dict[str, dict]:
    buckets: Dict[str, Dict[str, list]] = {}
    counts: Counter = Counter()
    for d in docs:
        lat = (results.get(d.doc_id) or {}).get("latency_ms") or {}
        label = _len_bucket(len(d.text))
        counts[label] += 1
        rows = buckets.setdefault(label, {})
        for scanner, ms in lat.items():
            rows.setdefault(scanner, []).append(float(ms))
    out: Dict[str, dict] = {}
    ordered = [f"{lo}-{hi}" for lo, hi in zip(_LEN_EDGES, _LEN_EDGES[1:])]
    ordered.append(f">={_LEN_EDGES[-1]}")
    for label in ordered:
        if label not in counts:
            continue
        row: Dict[str, Any] = {"n": counts[label]}
        for scanner in scanners:
            vals = buckets.get(label, {}).get(scanner)
            row[scanner] = (sum(vals) / len(vals)) if vals else None
        out[label] = row
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_offline(
    docs: Optional[Iterable[LabeledDocument]] = None,
    corpus_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    scanners: Sequence[str] = ("regex",),
    gliner_threshold: float = 0.5,
    gliner_categories: Optional[List[str]] = None,
    gliner_max_chars: int = 10_000,
    sweep: bool = False,
    sweep_thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    in_container: bool = False,
    compose_dir: Optional[str] = None,
    seed: int = 42,
    n_boot: int = 500,
    progress: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """Evaluate the production scanners over a labeled corpus, offline.

    Local path drives scanner_bridge.scan_documents; ``in_container=True``
    shells out to docker compose per container_runner.py's recipe (needed
    for GLiNER when the host lacks the model/deps). When a sweep is
    requested, GLiNER is scanned ONCE at SWEEP_SCAN_THRESHOLD and the main
    metrics are re-filtered at ``gliner_threshold`` (regex findings carry
    confidence 1.0, so they pass every cut).

    Returns the offline_metrics dict, which is also written to
    ``out_dir/offline_metrics.json``.
    """
    if out_dir is None:
        raise ValueError("run_offline requires out_dir")
    requested = tuple(scanners)
    bad = [s for s in requested if s not in _SUPPORTED]
    if not requested or bad:
        raise ValueError(f"unsupported scanners {bad or requested}; supported: {_SUPPORTED}")

    docs = _resolve_corpus(docs, corpus_path)
    os.makedirs(out_dir, exist_ok=True)
    if compose_dir is None:
        compose_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    notes: List[str] = []
    effective = list(requested)
    if not in_container and "gliner" in effective and not scanner_bridge.gliner_available():
        effective = [s for s in effective if s != "gliner"]
        notes.append(
            "GLINER REQUESTED BUT NOT IMPORTABLE IN THIS PROCESS — results are "
            + ("REGEX-ONLY" if effective else "EMPTY")
            + "; install gliner or rerun with in_container=True")

    want_sweep_scan = sweep and ("gliner" in effective or (in_container and "gliner" in requested))
    scan_threshold = min(SWEEP_SCAN_THRESHOLD, gliner_threshold) if want_sweep_scan \
        else gliner_threshold
    if want_sweep_scan and scan_threshold < gliner_threshold:
        notes.append(f"sweep: gliner scanned once at threshold {scan_threshold}; "
                     f"main metrics re-filtered at {gliner_threshold}")
    if in_container and gliner_categories:
        notes.append("in_container ignores gliner_categories "
                     "(container_runner has no flag for it); default label set used")

    if in_container:
        results = _scan_in_container(docs, corpus_path, out_dir, requested,
                                     scan_threshold, gliner_max_chars,
                                     compose_dir, progress)
        scope_scanners: Sequence[str] = requested
    elif effective:
        results = _scan_local(docs, effective, scan_threshold, gliner_categories,
                              gliner_max_chars, progress)
        scope_scanners = effective
    else:  # gliner-only requested and unavailable: nothing scannable
        results = {d.doc_id: {"findings": [], "latency_ms": {},
                              "errors": {"gliner": "gliner not importable"}}
                   for d in docs}
        scope_scanners = requested

    # -- persist raw findings ------------------------------------------------
    findings_rows = []
    for d in docs:
        r = results.get(d.doc_id) or {"findings": [], "latency_ms": {},
                                      "errors": {"harness": "no scan result for doc"}}
        findings_rows.append({"doc_id": d.doc_id, "findings": r["findings"],
                              "latency_ms": r.get("latency_ms") or {},
                              "errors": r.get("errors") or {}})
    write_jsonl(os.path.join(out_dir, "offline_findings.jsonl"), findings_rows)

    # -- normalize + match ---------------------------------------------------
    all_findings = {row["doc_id"]: matching.normalize_findings(row["findings"])
                    for row in findings_rows}
    errors_map = {row["doc_id"]: "; ".join(f"{k}: {v}" for k, v in row["errors"].items())
                  for row in findings_rows if row["errors"]}
    totals = {row["doc_id"]: sum(row["latency_ms"].values())
              for row in findings_rows}

    if scan_threshold < gliner_threshold:
        main_findings = {k: [f for f in v if f.confidence >= gliner_threshold]
                         for k, v in all_findings.items()}
    else:
        main_findings = all_findings

    evals = matching.match_corpus(docs, main_findings,
                                  latencies_by_doc_id=totals,
                                  errors_by_doc_id=errors_map)

    # -- reduce --------------------------------------------------------------
    per_scanner_lat = {
        s: metrics.summarize_latencies(
            [row["latency_ms"][s] for row in findings_rows if s in row["latency_ms"]])
        for s in scope_scanners
    }
    fp_traps = Counter(hit for e in evals for hit in e.fp_trap_hits)

    sweep_result = None
    if sweep:
        sweep_result = metrics.threshold_sweep(
            docs, all_findings, list(sweep_thresholds),
            lambda ds, fmap: matching.match_corpus(ds, fmap,
                                                   errors_by_doc_id=errors_map))

    out: Dict[str, Any] = {
        "run": {
            "kind": "offline",
            "created_at": utc_now_iso(),
            "corpus_path": os.path.abspath(corpus_path) if corpus_path else None,
            "n_docs": len(docs),
            "n_dirty": sum(1 for d in docs if d.entities),
            "n_clean": sum(1 for d in docs if not d.entities),
            "scanners_requested": list(requested),
            "scanners_effective": list(scope_scanners),
            "gliner_threshold": gliner_threshold,
            "gliner_scan_threshold": scan_threshold,
            "gliner_categories": gliner_categories,
            "gliner_max_chars": gliner_max_chars,
            "in_container": in_container,
            "seed": seed,
            "n_boot": n_boot,
            "notes": notes,
        },
        "doc_confusion": metrics.doc_confusion(evals),
        "span_confusion": metrics.span_confusion(evals, strict=False),
        "span_confusion_strict": metrics.span_confusion(evals, strict=True),
        "scope_split": metrics.scope_split(
            evals, _scope_for(scope_scanners, gliner_categories)),
        "recall_by": {
            "difficulty": metrics.recall_by(evals, "difficulty"),
            "generator": metrics.recall_by(evals, "generator"),
            "carrier": metrics.recall_by(evals, "carrier"),
        },
        "fp_traps": dict(sorted(fp_traps.items())),
        "latency_ms": {
            "per_scanner": per_scanner_lat,
            "per_doc_total": metrics.summarize_latencies(list(totals.values())),
        },
        "latency_by_length": _latency_by_length(docs, results, scope_scanners),
        "severity_accuracy": metrics.severity_accuracy(evals),
        "bootstrap": {
            "doc_recall": metrics.bootstrap_ci(
                evals, lambda e: metrics.doc_confusion(e)["recall"],
                n_boot=n_boot, seed=seed),
            "doc_precision": metrics.bootstrap_ci(
                evals, lambda e: metrics.doc_confusion(e)["precision"],
                n_boot=n_boot, seed=seed),
            "span_recall": metrics.bootstrap_ci(
                evals, lambda e: metrics.span_confusion(e)["overall"]["recall"],
                n_boot=n_boot, seed=seed),
        },
        "sweep": sweep_result,
        "scan_errors": sum(1 for e in evals if e.scan_error),
    }

    with open(os.path.join(out_dir, "offline_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)

    save_manifest(out_dir, RunManifest(
        run_id=os.path.basename(os.path.normpath(out_dir)),
        kind="offline",
        created_at=out["run"]["created_at"],
        argv=list(sys.argv[1:]),
        seed=seed,
        corpus_path=out["run"]["corpus_path"],
        scanner_mode=",".join(scope_scanners),
        notes="; ".join(notes),
    ))
    if progress:
        progress(f"[offline] wrote offline_metrics.json + offline_findings.jsonl "
                 f"to {out_dir}")
    return out
