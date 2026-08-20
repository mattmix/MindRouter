############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/service_bench.py: Benchmark an off-host DLP
# GPU service (dlp_service) directly over HTTP — accuracy
# parity against ground truth and throughput/latency vs
# concurrency — for the dedicated-host-vs-CPU comparison.
#
# Talks the /scan wire contract straight to the service
# (bypassing MindRouter), so it measures the service itself:
# GLiNER on the GPU plus the dynamic batcher.
#
############################################################

"""Direct benchmark client for the off-host DLP GPU service."""

import asyncio
import time
from typing import Dict, List, Optional

import httpx

from dlp_harness import schemas, matching, metrics
from dlp_harness.constants import canonicalize


async def _scan_one(client, endpoint, key, text, threshold, verify_note):
    t0 = time.monotonic()
    try:
        r = await client.post(
            endpoint.rstrip("/") + "/scan",
            headers={"X-Worker-Key": key},
            json={"text": text, "threshold": threshold},
        )
        dt = (time.monotonic() - t0) * 1000.0
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "latency_ms": dt,
                    "findings": [], "server_ms": None}
        body = r.json()
        return {"ok": True, "status": 200, "latency_ms": dt,
                "server_ms": body.get("latency_ms"),
                "batch_size": body.get("batch_size"),
                "findings": body.get("findings", [])}
    except httpx.HTTPError as e:
        return {"ok": False, "status": None, "latency_ms": (time.monotonic() - t0) * 1000.0,
                "error": type(e).__name__, "findings": [], "server_ms": None}


async def _run_phase(endpoint, key, docs, concurrency, threshold, verify_tls, timeout):
    sem = asyncio.Semaphore(concurrency)
    results: Dict[str, dict] = {}
    limits = httpx.Limits(max_connections=concurrency + 4)
    t_start = time.monotonic()

    async with httpx.AsyncClient(verify=verify_tls, timeout=timeout, limits=limits) as client:
        async def one(doc):
            async with sem:
                res = await _scan_one(client, endpoint, key, doc.text, threshold, verify_tls)
                results[doc.doc_id] = res
        await asyncio.gather(*(one(d) for d in docs))

    wall = time.monotonic() - t_start
    return results, wall


def bench_service(
    endpoint: str,
    key: str,
    docs: List[schemas.LabeledDocument],
    concurrencies=(1, 2, 4, 8, 16, 32),
    threshold: float = 0.5,
    verify_tls: bool = True,
    timeout: float = 60.0,
    score_accuracy: bool = True,
    progress=print,
) -> dict:
    """Drive the service with a corpus at each concurrency; measure + score.

    Returns {"phases": [...per concurrency...], "accuracy": {...}, "endpoint": ...}.
    Accuracy is scored once (findings are concurrency-independent) from the
    highest-concurrency pass that completed cleanly.
    """
    phases = []
    scored_results = None
    for c in concurrencies:
        results, wall = asyncio.run(
            _run_phase(endpoint, key, docs, c, threshold, verify_tls, timeout))
        oks = [r for r in results.values() if r["ok"]]
        lat = [r["latency_ms"] for r in oks]
        server = [r["server_ms"] for r in oks if r.get("server_ms") is not None]
        n_err = len(results) - len(oks)
        rps = len(oks) / wall if wall > 0 else 0.0
        phase = {
            "concurrency": c,
            "n": len(results), "n_ok": len(oks), "n_err": n_err,
            "wall_s": round(wall, 3),
            "throughput_rps": round(rps, 2),
            "client_latency_ms": metrics.summarize_latencies(lat),
            "server_latency_ms": metrics.summarize_latencies(server) if server else None,
        }
        phases.append(phase)
        progress(f"[bench] c={c:3d}  rps={rps:6.1f}  "
                 f"client p50={phase['client_latency_ms']['p50']:.0f}/"
                 f"p95={phase['client_latency_ms']['p95']:.0f}ms  errs={n_err}")
        if len(oks) == len(results):
            scored_results = results

    accuracy = None
    if score_accuracy and scored_results is not None:
        findings_by_doc = {}
        for doc_id, res in scored_results.items():
            findings_by_doc[doc_id] = [
                {"scanner": "gliner", "category": f.get("category"),
                 "text": f.get("text", ""), "confidence": f.get("confidence", 0.0),
                 "start": f.get("start", 0), "end": f.get("end", 0)}
                for f in res["findings"]
            ]
        evals = matching.match_corpus(docs, findings_by_doc)
        accuracy = {
            "doc_confusion": metrics.doc_confusion(evals),
            "span_confusion": metrics.span_confusion(evals)["overall"],
            "recall_by_category": {
                cat: row for cat, row in metrics.span_confusion(evals)["per_category"].items()
            },
        }

    return {"endpoint": endpoint, "threshold": threshold,
            "n_docs": len(docs), "phases": phases, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Aggregate multi-endpoint benchmark (total fleet throughput)
# ---------------------------------------------------------------------------

async def _run_aggregate_phase(endpoint_keys, docs, concurrency, threshold, verify_tls, timeout):
    """Fire docs round-robin across all endpoints at total concurrency C."""
    sem = asyncio.Semaphore(concurrency)
    # One client per endpoint (its own TLS/verify + connection pool).
    clients = {}
    for url, key in endpoint_keys:
        clients[url] = httpx.AsyncClient(
            verify=verify_tls, timeout=timeout,
            limits=httpx.Limits(max_connections=concurrency + 4))
    per_ep = {url: {"ok": 0, "err": 0, "lat": []} for url, _ in endpoint_keys}
    n_ep = len(endpoint_keys)
    t0 = time.monotonic()

    async def one(i, doc):
        url, key = endpoint_keys[i % n_ep]
        async with sem:
            res = await _scan_one(clients[url], url, key, doc.text, threshold, verify_tls)
            if res["ok"]:
                per_ep[url]["ok"] += 1
                per_ep[url]["lat"].append(res["latency_ms"])
            else:
                per_ep[url]["err"] += 1

    try:
        await asyncio.gather(*(one(i, d) for i, d in enumerate(docs)))
    finally:
        for cl in clients.values():
            await cl.aclose()
    wall = time.monotonic() - t0
    total_ok = sum(e["ok"] for e in per_ep.values())
    return per_ep, wall, total_ok


def bench_aggregate(
    endpoint_keys,        # list of (url, key)
    docs,
    concurrencies=(8, 16, 32, 48, 64, 96, 128),
    threshold=0.5,
    verify_tls=True,
    timeout=60.0,
    progress=print,
) -> dict:
    """Total fleet throughput: load spread round-robin across all endpoints."""
    phases = []
    for c in concurrencies:
        per_ep, wall, total_ok = asyncio.run(
            _run_aggregate_phase(endpoint_keys, docs, c, threshold, verify_tls, timeout))
        rps = total_ok / wall if wall > 0 else 0.0
        ep_rps = {url: round(e["ok"] / wall, 1) if wall > 0 else 0.0 for url, e in per_ep.items()}
        phases.append({
            "concurrency": c, "wall_s": round(wall, 3),
            "total_throughput_rps": round(rps, 2),
            "per_endpoint_rps": ep_rps,
            "per_endpoint_p50_ms": {url: metrics.percentile(e["lat"], 50) for url, e in per_ep.items()},
            "errors": {url: e["err"] for url, e in per_ep.items()},
        })
        progress(f"[agg] c={c:3d}  TOTAL={rps:6.1f} rps  per-ep={ep_rps}")
    peak = max((p["total_throughput_rps"] for p in phases), default=0.0)
    return {"endpoints": [u for u, _ in endpoint_keys], "n_docs": len(docs),
            "phases": phases, "peak_total_rps": peak}
