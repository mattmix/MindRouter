############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/container_runner.py: Run the production DLP
# scanners over a corpus JSONL inside the app container,
# writing one findings JSON line per document.
#
############################################################

"""Self-contained in-container corpus scanner.

The app container has the app source and the scanner deps (gliner, torch) but
no harness package; copy the whole dlp_harness directory in and run this file
directly.  From the repo checkout on the docker host:

    docker compose cp dlp_harness app:/tmp/dlp_harness
    docker compose cp /path/to/corpus.jsonl app:/tmp/corpus.jsonl
    docker compose exec -T app python /tmp/dlp_harness/container_runner.py \\
        --corpus /tmp/corpus.jsonl --out /tmp/findings.jsonl \\
        --scanners regex,gliner --gliner-threshold 0.05
    docker compose cp app:/tmp/findings.jsonl ./findings.jsonl

(`python` in the container resolves to /opt/venv/bin/python 3.12; the scanner
source is auto-detected at /app/backend/app/services/dlp_scanner.py.)  Also
runs locally from the repo checkout for smoke tests — regex only unless
gliner is installed.

Input : corpus JSONL; only the doc_id and text fields of each line are read.
Output: one JSON line per doc: {"doc_id", "findings", "latency_ms", "errors"}.
Exit  : 1 if the scanner module cannot load, 2 on bad arguments.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

# The dlp_harness dir may live anywhere (e.g. /tmp/dlp_harness); make its
# parent importable so `dlp_harness.*` resolves, with a loose-files fallback.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dlp_harness import scanner_bridge
except ImportError:
    import scanner_bridge

# Chunk size == progress cadence.  Chunking costs nothing on the gliner side:
# the model is cached module-globally inside dlp_scanner, so it loads once per
# process no matter how many scan_documents calls (event loops) run.
PROGRESS_EVERY = 50


def _locate_scanner() -> Optional[str]:
    """Path to dlp_scanner.py: env override, repo-checkout default, container path."""
    env = os.environ.get("DLP_SCANNER_PATH")
    if env:
        return env
    default = scanner_bridge.resolve_scanner_path()
    if os.path.isfile(default):
        return default
    if os.path.isfile(scanner_bridge.CONTAINER_SCANNER_PATH):
        return scanner_bridge.CONTAINER_SCANNER_PATH
    return None


def _read_corpus(path: str, limit: int) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append((str(d["doc_id"]), d.get("text") or ""))
            if limit and len(docs) >= limit:
                break
    return docs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan a corpus JSONL with the production DLP scanners.")
    ap.add_argument("--corpus", required=True, help="input corpus.jsonl (doc_id + text per line)")
    ap.add_argument("--out", required=True, help="output findings.jsonl")
    ap.add_argument("--scanners", default="regex", help="comma-separated: regex,gliner")
    ap.add_argument("--gliner-threshold", type=float, default=0.5)
    ap.add_argument("--gliner-max-chars", type=int, default=10_000,
                    help="GLiNER prefix cap in chars; 0 disables it")
    ap.add_argument("--no-global-cap", action="store_true",
                    help="skip the 200k global truncation run_dlp_scan applies")
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N docs (smoke runs)")
    args = ap.parse_args(argv)

    scanners = tuple(s.strip() for s in args.scanners.split(",") if s.strip())
    bad = [s for s in scanners if s not in ("regex", "gliner")]
    if not scanners or bad:
        print(f"error: bad --scanners value {args.scanners!r} (supported: regex,gliner)",
              file=sys.stderr)
        return 2

    scanner_path = _locate_scanner()
    if scanner_path is None:
        print("error: cannot locate dlp_scanner.py (set DLP_SCANNER_PATH)", file=sys.stderr)
        return 1
    # scan_documents resolves the path itself; publishing the env var makes
    # every resolution in this process agree with the probe above.
    os.environ["DLP_SCANNER_PATH"] = scanner_path
    try:
        scanner_bridge.load_scanner_module()
    except Exception as e:
        print(f"error: scanner module failed to load from {scanner_path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if "gliner" in scanners and not scanner_bridge.gliner_available():
        print("warning: gliner not importable here; gliner scans will report errors",
              file=sys.stderr)

    docs = _read_corpus(args.corpus, args.limit)
    total = len(docs)
    print(f"scanning {total} docs with {','.join(scanners)} (scanner={scanner_path})",
          file=sys.stderr)

    t0 = time.perf_counter()
    n_done = 0
    n_findings = 0
    n_errors = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(0, total, PROGRESS_EVERY):
            chunk = docs[i:i + PROGRESS_EVERY]
            results = scanner_bridge.scan_documents(
                dict(chunk),
                scanners=scanners,
                gliner_threshold=args.gliner_threshold,
                gliner_max_chars=args.gliner_max_chars,
                apply_global_cap=not args.no_global_cap,
            )
            for doc_id, _ in chunk:
                r = results[doc_id]
                out.write(json.dumps({"doc_id": doc_id, **r}, ensure_ascii=False) + "\n")
                n_findings += len(r["findings"])
                n_errors += len(r["errors"])
                n_done += 1
            print(f"processed {n_done}/{total} docs", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f"done: {n_done} docs, {n_findings} findings, {n_errors} scanner errors "
          f"in {elapsed:.1f}s -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
