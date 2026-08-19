############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/load.py: Load/overhead matrix driver — runs a
# scanner-mode x concurrency grid of timed traffic phases
# against a LOCAL gateway and measures what DLP costs
# (latency, throughput, CPU) and what it delivers under
# pressure (alert coverage, scan lag, drain time, queue
# drops).
#
# The sender is self-contained (no gateway-module import)
# so the hot path carries zero harness overhead. Request
# correlation: non-stream response body "id" IS
# requests.request_uuid; for streams the gateway stamps
# request_uuid into every SSE chunk "id" (the mock backend
# omits chunk ids on purpose). DLP is post-hoc, so each
# phase ends with a drain measurement — poll the alert
# count until it stops moving — before coverage/lag are
# read from the DB and the phase's alerts are purged.
#
# Measurement pattern mirrors chat_bench.py: staggered
# start, warmup boundary flagging, stop-event then
# drain-then-cancel with a bounded grace.
#
############################################################

"""Load/overhead matrix driver for the DLP evaluation harness."""

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import httpx

from dlp_harness.constants import (
    SAFE_RUN_OVERRIDES,
    SCANNER_ERROR_CATEGORY,
    SCANNER_MODES,
)
from dlp_harness.metrics import summarize_latencies
from dlp_harness.schemas import (
    LabeledDocument,
    RunManifest,
    save_manifest,
    utc_now_iso,
)

if TYPE_CHECKING:  # typing only: keeps load.py importable without pymysql
    from dlp_harness.db import HarnessDB

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

_REQUEST_TIMEOUT_S = 120.0     # per-request ceiling (chat_bench turn_timeout analogue)
_WORKER_GRACE_S = 10.0         # drain-then-cancel grace after the stop event
_SAMPLE_INTERVAL_S = 2.0       # cpu + gateway-queue sampler period
_DRAIN_POLL_S = 2.0            # alert-count poll period during drain
_DOCKER_STATS_TIMEOUT_S = 15.0  # per-sample ceiling for `docker stats --no-stream`


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

def _assert_local_url(base_url: str, allow_prod: bool) -> None:
    host = urllib.parse.urlsplit(base_url).hostname
    if allow_prod:
        return
    if host not in _LOCAL_HOSTS:
        raise RuntimeError(
            f"run_load_matrix refuses non-local base_url {base_url!r} without "
            "allow_prod=True (this driver mutates DLP config and purges alert rows)"
        )


# ---------------------------------------------------------------------------
# Pure parsers (unit-testable, no I/O)
# ---------------------------------------------------------------------------

_MEM_MULT = {
    "b": 1, "kb": 1000, "kib": 1024, "mb": 1000 ** 2, "mib": 1024 ** 2,
    "gb": 1000 ** 3, "gib": 1024 ** 3, "tb": 1000 ** 4, "tib": 1024 ** 4,
}
_MEM_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)$")


def _parse_mem_mb(mem_usage: str) -> Optional[float]:
    used = str(mem_usage).split("/")[0].strip()
    if not used or used == "--":
        return None
    m = _MEM_RE.match(used)
    if not m:
        return None
    mult = _MEM_MULT.get(m.group(2).lower())
    if mult is None:
        return None
    return float(m.group(1)) * mult / (1024 * 1024)


def parse_docker_stats_line(line: str) -> Optional[dict]:
    """One `docker stats --format {{json .}}` line -> {cpu_pct, mem_mb} or None."""
    try:
        data = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    cpu_raw = str(data.get("CPUPerc", "")).strip()
    if not cpu_raw.endswith("%"):
        return None            # "--" = container gone; treat as sampler failure
    try:
        cpu_pct = float(cpu_raw[:-1].replace(",", "."))
    except ValueError:
        return None
    return {"cpu_pct": cpu_pct, "mem_mb": _parse_mem_mb(data.get("MemUsage", ""))}


def parse_gateway_queue(text: str) -> Optional[float]:
    """Extract the mindrouter_queue_size gauge from Prometheus exposition text.

    Tolerates labels ({...}), trailing timestamps, and comment lines; rejects
    metrics that merely share the prefix (e.g. mindrouter_queue_size_bytes).
    """
    name = "mindrouter_queue_size"
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith(name):
            continue
        rest = line[len(name):]
        if rest[:1] not in (" ", "\t", "{"):
            continue
        if rest.startswith("{"):
            close = rest.find("}")
            if close < 0:
                continue
            rest = rest[close + 1:]
        parts = rest.split()
        if not parts:
            continue
        try:
            return float(parts[0])
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Document cycling
# ---------------------------------------------------------------------------

def doc_expected_alert(doc: LabeledDocument) -> bool:
    """Load-profile docs carry meta["expected_alert"]; others fall back to labels."""
    return bool(doc.meta.get("expected_alert", not doc.is_clean))


class DocCycler:
    """Shared round-robin document source for one phase's workers.

    Single-threaded asyncio makes the shared index safe as long as next()
    never awaits (it doesn't). dirty_rate=None preserves the corpus mix in
    corpus order; a float re-mixes dirty/clean draws deterministically via
    random.Random(seed) while cycling each pool round-robin.
    """

    def __init__(self, docs: Sequence[LabeledDocument],
                 dirty_rate: Optional[float] = None, seed: int = 42):
        if not docs:
            raise ValueError("DocCycler needs at least one document")
        self._all = list(docs)
        self._dirty = [d for d in docs if doc_expected_alert(d)]
        self._clean = [d for d in docs if not doc_expected_alert(d)]
        self._rate = dirty_rate
        if dirty_rate is not None:
            if not 0.0 <= dirty_rate <= 1.0:
                raise ValueError(f"dirty_rate must be in [0, 1], got {dirty_rate}")
            if dirty_rate > 0.0 and not self._dirty:
                raise ValueError("dirty_rate > 0 but corpus has no dirty documents")
            if dirty_rate < 1.0 and not self._clean:
                raise ValueError("dirty_rate < 1 but corpus has no clean documents")
        self._rng = random.Random(seed)
        self._i = self._di = self._ci = 0

    def next(self) -> LabeledDocument:
        if self._rate is None:
            doc = self._all[self._i % len(self._all)]
            self._i += 1
            return doc
        if self._rng.random() < self._rate:
            doc = self._dirty[self._di % len(self._dirty)]
            self._di += 1
        else:
            doc = self._clean[self._ci % len(self._clean)]
            self._ci += 1
        return doc


# ---------------------------------------------------------------------------
# Phase summarization (pure)
# ---------------------------------------------------------------------------

def _is_ok(rec: dict) -> bool:
    return (rec.get("error") is None and rec.get("status") is not None
            and 200 <= rec["status"] < 300)


def _summary_or_none(values: List[float]) -> Optional[dict]:
    return summarize_latencies(values) if values else None


def summarize_phase(records: List[dict], *, phase_id: str, scanner_mode: str,
                    concurrency: int, duration_s: float, warmup_s: float,
                    stream: bool) -> dict:
    """Offered-load + latency summary over one phase's request records.

    The measurement window is post-warmup only; rps uses the fixed window
    length (duration_s - warmup_s) rather than observed span so phases are
    comparable. dlp/cpu/gateway_queue are returned as null-shaped defaults
    for the driver to fill in.
    """
    measured = [r for r in records if not r.get("in_warmup")]
    ok = [r for r in measured if _is_ok(r)]
    window_s = duration_s - warmup_s
    rps = (len(ok) / window_s) if window_s > 0 else 0.0

    if stream:
        ttfb = summarize_latencies(
            [r["ttfb_ms"] for r in ok if r.get("ttfb_ms") is not None])
        ttft = summarize_latencies(
            [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None])
    else:
        ttfb = ttft = None

    return {
        "phase_id": phase_id,
        "scanner_mode": scanner_mode,
        "concurrency": concurrency,
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "offered": {
            "n_requests": len(measured),
            "n_ok": len(ok),
            "n_err": len(measured) - len(ok),
            "rps": rps,
            "dirty_sent": sum(1 for r in measured if r.get("expected_alert")),
        },
        "latency_ms": {
            "ttfb": ttfb,
            "ttft": ttft,
            "e2e": summarize_latencies([r["e2e_ms"] for r in ok]),
        },
        "dlp": {
            "coverage_rate": None, "alerts": 0, "dirty_unscannable": None,
            "scan_lag_ms": None, "scan_latency_ms": None,
            "drain_seconds": None, "drain_settled": None,
            "queue_drops_logged": None, "scanner_error_alerts": 0,
        },
        "cpu": {"app_mean_pct": None, "app_max_pct": None},
        "gateway_queue": {"mean": None, "max": None},
    }


def compute_baseline_comparison(phases: List[dict]) -> List[dict]:
    """Pair each non-"off" phase with the "off" phase at the same concurrency."""
    off_by_conc = {p["concurrency"]: p for p in phases
                   if p["scanner_mode"] == "off"}

    def e2e_pct(phase: dict, key: str) -> Optional[float]:
        e2e = phase["latency_ms"]["e2e"] or {}
        return e2e.get(key)

    def delta(phase: dict, off: dict, key: str) -> Optional[float]:
        a, b = e2e_pct(phase, key), e2e_pct(off, key)
        return (a - b) if a is not None and b is not None else None

    out = []
    for p in phases:
        if p["scanner_mode"] == "off":
            continue
        off = off_by_conc.get(p["concurrency"])
        if off is None:
            continue
        rps_off = off["offered"]["rps"]
        rps_mode = p["offered"]["rps"]
        out.append({
            "concurrency": p["concurrency"],
            "mode": p["scanner_mode"],
            "e2e_p50_delta_ms": delta(p, off, "p50"),
            "e2e_p95_delta_ms": delta(p, off, "p95"),
            "throughput_delta_pct": ((rps_mode - rps_off) / rps_off * 100.0
                                     if rps_off else None),
            "coverage_rate": p["dlp"]["coverage_rate"],
        })
    return out


# ---------------------------------------------------------------------------
# Drain measurement (post-hoc DLP settles AFTER traffic stops)
# ---------------------------------------------------------------------------

def measure_drain(count_fn: Callable[[], int], settle_s: float, timeout_s: float,
                  t_start: Optional[float] = None,
                  poll_interval_s: float = _DRAIN_POLL_S,
                  sleep: Callable[[float], None] = time.sleep,
                  now: Callable[[], float] = time.monotonic):
    """Poll count_fn until unchanged for settle_s (or timeout).

    Returns (drain_seconds, settled, last_count). drain_seconds is measured
    from t_start (the phase-end monotonic instant) to the LAST observed count
    change — the settle-confirmation window is not billed to the drain.
    """
    if t_start is None:
        t_start = now()
    last = count_fn()
    last_change = now()
    while True:
        t = now()
        if t - last_change >= settle_s:
            return (max(0.0, last_change - t_start), True, last)
        if t - t_start >= timeout_s:
            return (t - t_start, False, last)
        sleep(poll_interval_s)
        current = count_fn()
        if current != last:
            last = current
            last_change = now()


# ---------------------------------------------------------------------------
# Docker helpers (any failure -> None, never a crash)
# ---------------------------------------------------------------------------

def _resolve_container_id(compose_dir: str) -> Optional[str]:
    try:
        out = subprocess.run(["docker", "compose", "ps", "-q", "app"],
                             cwd=compose_dir, capture_output=True, text=True,
                             timeout=15)
        if out.returncode != 0:
            return None
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


def _count_queue_drops(compose_dir: str, since_iso: str) -> Optional[int]:
    try:
        out = subprocess.run(["docker", "compose", "logs", "app",
                              "--since", since_iso],
                             cwd=compose_dir, capture_output=True, text=True,
                             timeout=60)
        if out.returncode != 0:
            return None
        return out.stdout.count("dlp_queue_full")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Request sender (self-contained; mirrors chat_bench run_turn's shape)
# ---------------------------------------------------------------------------

async def _send(client: httpx.AsyncClient, base_url: str, api_key: str,
                doc: LabeledDocument, *, model: str, max_tokens: int,
                stream: bool, request_timeout_s: float = _REQUEST_TIMEOUT_S,
                rec: Optional[dict] = None) -> dict:
    """One chat completion; returns a load_requests.jsonl record (sans phase fields).

    A caller-supplied rec is filled in place, so a cancellation mid-request
    (worker grace cancel) still leaves the partial record — including any
    request_uuid already captured from a stream chunk — visible to the caller
    for accounting and the final alert purge.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "user", "content": doc.text}]}
    if rec is None:
        rec = {}
    rec.update({"doc_id": doc.doc_id, "request_uuid": None, "stream": stream,
                "status": None, "error": None, "ttfb_ms": None, "ttft_ms": None,
                "e2e_ms": 0.0, "expected_alert": doc_expected_alert(doc)})
    t0 = time.monotonic()
    try:
        async with asyncio.timeout(request_timeout_s):
            if not stream:
                resp = await client.post(url, json=body, headers=headers)
                rec["status"] = resp.status_code
                if resp.status_code == 200:
                    try:
                        rec["request_uuid"] = resp.json().get("id")
                    except ValueError:
                        rec["error"] = "non-JSON 200 body"
                else:
                    rec["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                async with client.stream("POST", url, json=body,
                                         headers=headers) as resp:
                    rec["status"] = resp.status_code
                    if resp.status_code != 200:
                        text = await resp.aread()
                        rec["error"] = (f"HTTP {resp.status_code}: "
                                        f"{text[:200].decode(errors='replace')}")
                    else:
                        buf = b""
                        async for raw in resp.aiter_bytes():
                            now = time.monotonic()
                            if rec["ttfb_ms"] is None:
                                rec["ttfb_ms"] = (now - t0) * 1000.0
                            buf += raw
                            while b"\n" in buf:
                                line, buf = buf.split(b"\n", 1)
                                line = line.strip()
                                if not line.startswith(b"data:"):
                                    continue
                                payload = line[5:].strip()
                                if payload == b"[DONE]":
                                    continue
                                try:
                                    chunk = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue
                                if "error" in chunk:  # gateway emits SSE errors
                                    rec["error"] = json.dumps(chunk["error"])[:200]
                                    continue
                                if rec["request_uuid"] is None and chunk.get("id"):
                                    rec["request_uuid"] = chunk["id"]
                                choices = chunk.get("choices") or []
                                if (rec["ttft_ms"] is None and choices
                                        and (choices[0].get("delta") or {}).get("content")):
                                    rec["ttft_ms"] = (now - t0) * 1000.0
    except (TimeoutError, asyncio.TimeoutError):
        rec["error"] = f"timeout after {request_timeout_s}s"
    except httpx.HTTPError as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    except asyncio.CancelledError:
        if rec.get("error") is None:
            rec["error"] = "cancelled at phase grace"
        raise
    finally:
        rec["e2e_ms"] = (time.monotonic() - t0) * 1000.0
    return rec


# ---------------------------------------------------------------------------
# Background samplers
# ---------------------------------------------------------------------------

async def _interruptible_sleep(event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def _cpu_sampler(container_id: Optional[str], phase_id: str, t0: float,
                       samples: List[dict], stop: asyncio.Event) -> None:
    if not container_id:
        return
    while not stop.is_set():
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stats", "--no-stream", "--format", "{{json .}}",
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(),
                                                timeout=_DOCKER_STATS_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.kill()        # wait_for cancels communicate() only —
                await proc.wait()  # reap so the transport closes in-loop
                return
            lines = [ln for ln in out.decode(errors="replace").splitlines()
                     if ln.strip()]
            parsed = parse_docker_stats_line(lines[-1]) if lines else None
            if parsed is None:
                return                        # failure -> stop sampling silently
            samples.append({"phase_id": phase_id,
                            "ts": time.monotonic() - t0, **parsed})
        except Exception:
            return                            # failure -> stop sampling silently
        await _interruptible_sleep(stop, _SAMPLE_INTERVAL_S)


async def _queue_sampler(base_url: str, samples: List[float],
                         stop: asyncio.Event) -> None:
    url = base_url.rstrip("/") + "/metrics"
    async with httpx.AsyncClient(timeout=10.0) as client:
        while not stop.is_set():
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    value = parse_gateway_queue(resp.text)
                    if value is not None:
                        samples.append(value)
            except httpx.HTTPError:
                pass                          # transient; keep sampling
            await _interruptible_sleep(stop, _SAMPLE_INTERVAL_S)


# ---------------------------------------------------------------------------
# Phase driver
# ---------------------------------------------------------------------------

async def _run_phase_traffic(base_url: str, api_keys: List[str],
                             cycler: DocCycler, *, phase_id: str,
                             concurrency: int, duration_s: float,
                             warmup_s: float, stream: bool, model: str,
                             max_tokens: int, container_id: Optional[str],
                             req_file,
                             records: Optional[List[dict]] = None) -> dict:
    """Run one phase's timed traffic; returns records/samples/t_stop.

    A caller-supplied records list is appended to in place so partial results
    survive even when the surrounding event loop is torn down mid-phase.
    """
    if records is None:
        records = []
    cpu_samples: List[dict] = []
    queue_samples: List[float] = []
    stop = asyncio.Event()          # workers: stop starting new requests
    sampler_stop = asyncio.Event()  # samplers: cover the worker-drain grace too
    t0 = time.monotonic()
    stagger_window = min(warmup_s * 0.8, 5.0)

    async def worker(idx: int) -> None:
        await asyncio.sleep(stagger_window * idx / max(1, concurrency))
        api_key = api_keys[idx % len(api_keys)]
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_S, connect=10.0)) as client:
            while not stop.is_set():
                doc = cycler.next()
                ts = time.monotonic() - t0
                rec: dict = {}

                def _record() -> None:
                    rec["phase_id"] = phase_id
                    rec["ts"] = ts
                    rec["in_warmup"] = ts < warmup_s
                    records.append(rec)
                    req_file.write(json.dumps(rec) + "\n")

                try:
                    await _send(client, base_url, api_key, doc, model=model,
                                max_tokens=max_tokens, stream=stream, rec=rec)
                except asyncio.CancelledError:
                    # grace cancel: keep the partial record so any captured
                    # request_uuid still reaches accounting and the purge
                    if rec:
                        _record()
                    raise
                _record()

    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    samplers = [
        asyncio.create_task(_cpu_sampler(container_id, phase_id, t0,
                                         cpu_samples, sampler_stop)),
        asyncio.create_task(_queue_sampler(base_url, queue_samples,
                                           sampler_stop)),
    ]

    await asyncio.sleep(duration_s)
    stop.set()
    t_stop = time.monotonic()
    try:
        await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True),
                               timeout=_WORKER_GRACE_S)
    except asyncio.TimeoutError:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    sampler_stop.set()
    await asyncio.gather(*samplers, return_exceptions=True)

    return {"records": records, "cpu_samples": cpu_samples,
            "queue_samples": queue_samples, "t_stop": t_stop}


def _is_error_alert(alert: dict) -> bool:
    cats = alert.get("categories")
    if isinstance(cats, list):
        return SCANNER_ERROR_CATEGORY in cats
    return SCANNER_ERROR_CATEGORY in str(cats or "")


def _measure_phase_dlp(db: "HarnessDB", records: List[dict], t_stop: float,
                       settle_s: float, drain_timeout_s: float, *,
                       scanner_mode: str = "regex") -> dict:
    """Drain, then read coverage/lag from the DB. Returns fields + request ids."""
    uuids = [r["request_uuid"] for r in records if r.get("request_uuid")]
    uuid_to_id: Dict[str, int] = {}
    if uuids:
        for row in db.fetch_requests_by_uuids(uuids):
            uuid_to_id[row["request_uuid"]] = row["id"]
    request_ids = list(uuid_to_id.values())

    drain_seconds, settled, _ = measure_drain(
        lambda: db.count_alerts_for_request_ids(request_ids,
                                                exclude_scanner_errors=False),
        settle_s=settle_s, timeout_s=drain_timeout_s, t_start=t_stop)

    alerts = db.fetch_alerts_by_request_ids(request_ids) if request_ids else []
    real_alerts = [a for a in alerts if not _is_error_alert(a)]
    alerted_ids = {a["request_id"] for a in real_alerts}

    measured_dirty = [r for r in records
                      if not r.get("in_warmup") and r.get("expected_alert")]
    # denominator mirrors e2e._score: only gateway-completed requests can be
    # scanned, so failed/timed-out ones must not read as DLP scan drops
    dirty_ids = {uuid_to_id[r["request_uuid"]] for r in measured_dirty
                 if _is_ok(r) and r.get("request_uuid") in uuid_to_id}
    dirty_unscannable = sum(1 for r in measured_dirty if not _is_ok(r))
    if scanner_mode == "off":
        coverage = None            # scanner disabled: coverage is meaningless
    else:
        coverage = (len(dirty_ids & alerted_ids) / len(dirty_ids)
                    if dirty_ids else None)

    lag_rows = db.fetch_scan_lags_ms(request_ids) if request_ids else []
    lags = [row["lag_ms"] for row in lag_rows if row.get("lag_ms") is not None]
    latencies = [a["scan_latency_ms"] for a in real_alerts
                 if a.get("scan_latency_ms") is not None]

    return {
        "fields": {
            "coverage_rate": coverage,
            "alerts": len(real_alerts),
            "dirty_unscannable": dirty_unscannable,
            "scan_lag_ms": _summary_or_none(lags),
            "scan_latency_ms": _summary_or_none(latencies),
            "drain_seconds": drain_seconds,
            "drain_settled": settled,
            "queue_drops_logged": None,     # driver fills from docker logs
            "scanner_error_alerts": len(alerts) - len(real_alerts),
        },
        "request_ids": request_ids,
    }


def _persist_unpurged(out_dir: str, ids: List[int],
                      progress: Callable[[str], None], *, reason: str) -> None:
    path = os.path.join(out_dir, "unpurged_request_ids.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"request_ids": ids}, f, indent=2)
        where = f"re-purge candidates saved to {path}"
    except Exception:
        where = f"purge these request ids manually: {ids}"
    progress(f"[load] WARNING: {reason}; synthetic alerts may remain — {where}")


def _purge_residuals(db: "HarnessDB", purge_ids: Sequence[int],
                     pending_records: Sequence[dict], settle_s: float,
                     timeout_s: float, out_dir: str,
                     progress: Callable[[str], None]) -> None:
    """Final cleanup: purge the whole run's synthetic alerts before restore.

    Late post-hoc scans can write alerts after a phase's own purge (or a phase
    aborted before it ever purged), so this waits for the scan backlog to
    settle under the still-safe (email-off) overrides and then re-purges every
    resolved request id. Never raises — the caller's config restore must
    always run afterwards.
    """
    ids: List[int] = list(purge_ids)
    try:
        uuids = [r.get("request_uuid") for r in pending_records
                 if r.get("request_uuid")]
        if uuids:
            ids.extend(int(row["id"])
                       for row in db.fetch_requests_by_uuids(uuids))
        ids = sorted(set(ids))
        if not ids:
            return
        _, settled, _ = measure_drain(
            lambda: db.count_alerts_for_request_ids(
                ids, exclude_scanner_errors=False),
            settle_s=settle_s, timeout_s=timeout_s)
        db.purge_alerts_for_request_ids(ids)
        if not settled:
            _persist_unpurged(out_dir, ids, progress,
                              reason="alert count still moving at final purge")
    except Exception as e:
        _persist_unpurged(
            out_dir, ids, progress,
            reason=f"final purge failed ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Matrix driver
# ---------------------------------------------------------------------------

def run_load_matrix(
    base_url: str,
    api_keys: List[str],
    admin_key: str,
    db: "HarnessDB",
    out_dir: str,
    docs: List[LabeledDocument],
    modes: Sequence[str] = ("off", "regex"),
    concurrencies: Sequence[int] = (1, 4, 16),
    duration_s: float = 60.0,
    warmup_s: float = 10.0,
    stream: bool = True,
    model: str = "dlp-mock",
    max_tokens: int = 64,
    dirty_rate: Optional[float] = None,
    inter_phase_drain_timeout_s: float = 240.0,
    settle_s: float = 8.0,
    allow_prod: bool = False,
    seed: int = 42,
    compose_dir: Optional[str] = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Run the scanner-mode x concurrency load matrix; return load_phases.json content.

    Writes load_requests.jsonl (incremental, flushed per phase),
    cpu_samples.jsonl, config_snapshot.json, load_phases.json, and run.json
    into out_dir. DLP config is snapshotted before the first phase (and
    persisted to config_snapshot.json for recovery after a hard kill); every
    mode application layers SAFE_RUN_OVERRIDES on top (email off, dedup off).
    Cleanup runs in a finally, ordered so the safe overrides are still in
    force while the scan backlog settles and synthetic alerts are purged, and
    only then is the real config restored. compose_dir defaults to the repo
    root containing this file. admin_key is accepted for interface parity
    with the other drivers (backend/provisioning management happens outside
    this module).
    """
    _assert_local_url(base_url, allow_prod)
    if not api_keys:
        raise ValueError("run_load_matrix needs at least one api key")
    if not docs:
        raise ValueError("run_load_matrix needs a non-empty corpus")
    for mode in modes:
        if mode not in SCANNER_MODES:
            raise ValueError(f"unknown scanner mode {mode!r}; "
                             f"choose from {sorted(SCANNER_MODES)}")
    if warmup_s >= duration_s:
        raise ValueError(f"warmup_s ({warmup_s}) must be < duration_s ({duration_s})")

    if compose_dir is None:
        # repo root derived from this file (mirrors offline_eval.py); a
        # hardcoded path would silently null cpu/queue-drop metrics elsewhere
        compose_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(out_dir, exist_ok=True)
    container_id = _resolve_container_id(compose_dir)
    if container_id is None:
        progress("[load] docker compose app container not found; "
                 "cpu + queue-drop metrics will be null")

    phases: List[dict] = []
    purge_ids: List[int] = []          # every request id resolved so far
    pending_records: List[dict] = []   # in-flight phase records (unresolved)
    snapshot = db.snapshot_dlp_config()
    # persist immediately (mirrors e2e.py): a hard kill mid-run must leave an
    # on-disk record of the pre-run config for manual restore
    with open(os.path.join(out_dir, "config_snapshot.json"), "w",
              encoding="utf-8") as f:
        json.dump(db.snapshot_to_json(snapshot), f, indent=2)
    try:
        with open(os.path.join(out_dir, "load_requests.jsonl"), "w",
                  encoding="utf-8") as req_file, \
             open(os.path.join(out_dir, "cpu_samples.jsonl"), "w",
                  encoding="utf-8") as cpu_file:
            for mode in modes:
                db.apply_overrides({**SCANNER_MODES[mode], **SAFE_RUN_OVERRIDES})
                for concurrency in concurrencies:
                    phase_id = f"{mode}-c{concurrency}"
                    progress(f"[load] phase {phase_id}: {duration_s:.0f}s "
                             f"({warmup_s:.0f}s warmup), stream={stream}")
                    phase_start_iso = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")
                    cycler = DocCycler(docs, dirty_rate=dirty_rate, seed=seed)
                    phase_records: List[dict] = []
                    pending_records = phase_records

                    traffic = asyncio.run(_run_phase_traffic(
                        base_url, list(api_keys), cycler, phase_id=phase_id,
                        concurrency=concurrency, duration_s=duration_s,
                        warmup_s=warmup_s, stream=stream, model=model,
                        max_tokens=max_tokens, container_id=container_id,
                        req_file=req_file, records=phase_records))
                    req_file.flush()
                    for sample in traffic["cpu_samples"]:
                        cpu_file.write(json.dumps(sample) + "\n")
                    cpu_file.flush()

                    summary = summarize_phase(
                        traffic["records"], phase_id=phase_id,
                        scanner_mode=mode, concurrency=concurrency,
                        duration_s=duration_s, warmup_s=warmup_s, stream=stream)

                    dlp = _measure_phase_dlp(db, traffic["records"],
                                             traffic["t_stop"], settle_s,
                                             inter_phase_drain_timeout_s,
                                             scanner_mode=mode)
                    purge_ids.extend(dlp["request_ids"])
                    pending_records = []   # resolved into purge_ids above
                    summary["dlp"].update(dlp["fields"])
                    summary["dlp"]["queue_drops_logged"] = _count_queue_drops(
                        compose_dir, phase_start_iso)

                    cpu_vals = [s["cpu_pct"] for s in traffic["cpu_samples"]]
                    if cpu_vals:
                        summary["cpu"] = {
                            "app_mean_pct": sum(cpu_vals) / len(cpu_vals),
                            "app_max_pct": max(cpu_vals),
                        }
                    queue_vals = traffic["queue_samples"]
                    if queue_vals:
                        summary["gateway_queue"] = {
                            "mean": sum(queue_vals) / len(queue_vals),
                            "max": max(queue_vals),
                        }

                    # keep the alert table lean across a long matrix run
                    # (the finally re-purges everything for late scans)
                    if dlp["request_ids"]:
                        db.purge_alerts_for_request_ids(dlp["request_ids"])

                    phases.append(summary)
                    offered = summary["offered"]
                    progress(f"[load] phase {phase_id}: ok={offered['n_ok']} "
                             f"err={offered['n_err']} rps={offered['rps']:.2f} "
                             f"coverage={summary['dlp']['coverage_rate']} "
                             f"drain={summary['dlp']['drain_seconds']}")
    finally:
        # order matters: wait out late post-hoc scans and purge synthetic
        # alerts while the safe (email-off) overrides are still applied,
        # THEN restore the real config
        try:
            _purge_residuals(db, purge_ids, pending_records, settle_s,
                             inter_phase_drain_timeout_s, out_dir, progress)
        finally:
            db.restore_dlp_config(snapshot)

    result = {"phases": phases,
              "baseline_comparison": compute_baseline_comparison(phases)}
    with open(os.path.join(out_dir, "load_phases.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    save_manifest(out_dir, RunManifest(
        run_id=os.path.basename(os.path.normpath(out_dir)),
        kind="load",
        created_at=utc_now_iso(),
        argv=list(sys.argv),
        seed=seed,
        base_url=base_url,
        scanner_mode=",".join(modes),
        extra={
            "concurrencies": list(concurrencies),
            "duration_s": duration_s,
            "warmup_s": warmup_s,
            "stream": stream,
            "model": model,
            "max_tokens": max_tokens,
            "dirty_rate": dirty_rate,
            "n_docs": len(docs),
            "n_api_keys": len(api_keys),
            "artifacts": ["load_requests.jsonl", "load_phases.json",
                          "cpu_samples.jsonl"],
        }))
    return result
