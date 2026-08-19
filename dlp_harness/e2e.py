############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/e2e.py: End-to-end DLP evaluation through the
# live gateway — snapshot/override config, preflight the
# whole pipeline (health, audit capture, a canary SSN that
# must alert), send a labeled corpus, drain the post-hoc
# worker, join dlp_alerts back to documents, and score.
#
# The DLP worker is post-hoc and per-uvicorn-worker: a clean
# scan writes NO row and an alert arrives seconds after the
# response, so absence-of-alert is only meaningful after the
# drain loop observes a stable alert count. Scanner-failure
# alerts (categories == ["dlp_scanner_error"]) are counted
# separately and excluded from every accuracy number —
# including the coverage denominators, since their docs were
# never actually scanned.
#
# Cleanup is failure-proof: purge ids are hoisted to
# function scope and the finally block (1) drains straggler
# scans while the SAFE overrides (email off) still apply,
# (2) purges every synthetic alert, and (3) only then
# restores the production config snapshot.
#
# The db argument is duck-typed against HarnessDB so this
# module never imports pymysql (tests inject a FakeDB).
#
############################################################

"""End-to-end DLP evaluation orchestrator."""

import asyncio
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from dlp_harness.constants import (
    GLINER_DEFAULT_SCOPE,
    REGEX_BUILTIN_SCOPE,
    SAFE_RUN_OVERRIDES,
    SCANNER_ERROR_CATEGORY,
    SCANNER_MODES,
    SEVERITY_ORDER,
    SEVERITY_RULES_OVERRIDE,
    canonicalize,
    expected_severity,
    regex_can_see,
)
from dlp_harness.gateway import SendResult, require_local, send_chat
from dlp_harness.metrics import summarize_latencies
from dlp_harness.mock_backend import DEFAULT_MODEL
from dlp_harness.schemas import (
    LabeledDocument,
    RunManifest,
    new_run_dir,
    save_manifest,
    utc_now_iso,
    write_jsonl,
)

PROBE_TEXT = ("Quick connectivity check. Summarize the benefits of regular "
              "exercise in one sentence.")
CANARY_TEXT = "For the record, my SSN is 123-45-6789. Please keep it private."

_SCOPE_BY_MODE: Dict[str, Set[str]] = {
    "off": set(),
    "regex": set(REGEX_BUILTIN_SCOPE),
    "gliner": set(GLINER_DEFAULT_SCOPE),
    "regex+gliner": set(REGEX_BUILTIN_SCOPE) | set(GLINER_DEFAULT_SCOPE),
}


def active_scope(scanner_mode: str) -> Set[str]:
    """Canonical categories the selected scanner matrix can possibly detect."""
    if scanner_mode not in _SCOPE_BY_MODE:
        raise ValueError(f"unknown scanner_mode {scanner_mode!r} "
                         f"(one of {sorted(SCANNER_MODES)})")
    return _SCOPE_BY_MODE[scanner_mode]


def entity_in_scope(category: str, generator: str, scanner_mode: str) -> bool:
    """Whether the active scanner matrix could possibly detect this entity.

    GLiNER scope is category-level, but regex scope is variant-aware
    (constants.regex_can_see): the builtin patterns only match specific
    formats — e.g. only a dashed SSN is regex-visible — so a doc whose only
    entity is a spaced SSN must NOT count as in-scope for --mode regex.
    ``generator`` is the ground-truth generator string ("ssn.dashed").
    """
    active_scope(scanner_mode)   # validates the mode name
    parts = set(scanner_mode.split("+"))
    if "gliner" in parts and category in GLINER_DEFAULT_SCOPE:
        return True
    if "regex" in parts and regex_can_see(category, generator):
        return True
    return False


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _f(v) -> Optional[float]:
    """float-coerce DB numerics (Decimal) for JSON; None passes through."""
    return None if v is None else float(v)


def _is_error_alert(alert: dict) -> bool:
    cats = alert.get("categories") or []
    return SCANNER_ERROR_CATEGORY in cats


def _git_rev() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

async def _await_request_row(db, request_uuid: str, timeout_s: float = 10.0,
                             poll_s: float = 0.5) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        rows = db.fetch_requests_by_uuids([request_uuid])
        if rows:
            return rows[0]
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"no requests row appeared for uuid {request_uuid!r} within {timeout_s}s")
        await asyncio.sleep(poll_s)


async def _preflight_async(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    admin_key: str,
    db,
    model: str,
    plant_side: str,
    scanner_mode: str,
    progress=print,
    poll_interval_s: float = 2.0,
    canary_timeout_s: float = 90.0,
    stream_pct: float = 0.0,
    row_timeout_s: float = 10.0,
    aux_ids_out: Optional[List[int]] = None,
) -> dict:
    """Verify every hidden dependency of the pipeline before a run.

    Caller must have applied require_local already. Returns
    {"probe_request_id", "canary_request_id", "aux_request_ids"} so the run
    can purge the alerts these synthetic requests may create. Ids are ALSO
    appended to ``aux_ids_out`` as they resolve, so a caller's finally block
    can purge them even when a later preflight step raises.
    """
    # Probes and canaries always plant prompt-side; "mixed" only affects the
    # per-doc split in the main send (response CAPTURE is still checked below).
    # "echo" passes through: the payload rides in the prompt either way.
    probe_side = "prompt" if plant_side == "mixed" else plant_side
    aux_ids: List[int] = []

    def _track(rid: int) -> None:
        aux_ids.append(rid)
        if aux_ids_out is not None:
            aux_ids_out.append(rid)

    r = await client.get(base_url.rstrip("/") + "/healthz")
    if r.status_code != 200:
        raise RuntimeError(f"preflight: GET /healthz returned {r.status_code} — gateway down?")
    r = await client.get(base_url.rstrip("/") + "/api/admin/backends",
                         headers={"Authorization": f"Bearer {admin_key}"})
    if r.status_code != 200:
        raise RuntimeError(
            f"preflight: GET /api/admin/backends returned {r.status_code} — admin key rejected?")

    probe = await send_chat(client, base_url, api_key, model, PROBE_TEXT,
                            stream=False, plant_side=probe_side, allow_prod=True)
    if not probe.ok:
        raise RuntimeError(
            f"preflight: clean probe chat failed ({probe.error}) — is the mock backend "
            "registered and routable (wait_until_routable)?")
    if not probe.request_uuid:
        raise RuntimeError("preflight: probe response carried no 'id' to correlate on")
    probe_row = await _await_request_row(db, probe.request_uuid,
                                         timeout_s=row_timeout_s)
    probe_rid = int(probe_row["id"])
    _track(probe_rid)

    # Audit capture is the hidden hard dependency: without stored text the
    # worker scans nothing and every alert count is silently zero.
    msg_rows = db.query("SELECT messages FROM requests WHERE id=%s", (probe_rid,))
    if not msg_rows or msg_rows[0].get("messages") in (None, ""):
        raise RuntimeError(
            "preflight: requests.messages is NULL — audit prompt capture is OFF; "
            "DLP cannot see prompts (set AUDIT_LOG_ENABLED + AUDIT_LOG_PROMPTS)")
    if plant_side in ("response", "mixed", "echo"):
        resp_rows = db.query("SELECT content FROM responses WHERE request_id=%s", (probe_rid,))
        if not resp_rows or resp_rows[0].get("content") in (None, ""):
            raise RuntimeError(
                "preflight: responses.content is NULL — audit response capture is OFF; "
                "DLP cannot see responses (set AUDIT_LOG_RESPONSES)")

    if stream_pct > 0:
        # Stream correlation is a MOCK-ONLY contract: the gateway stamps
        # request_uuid into SSE chunk ids only when the backend omits its
        # own. Verify the actual contract here so a real backend fails loudly
        # instead of silently breaking coverage joins and the alert purge.
        sprobe = await send_chat(client, base_url, api_key, model, PROBE_TEXT,
                                 stream=True, plant_side=probe_side,
                                 allow_prod=True)
        if not sprobe.ok:
            raise RuntimeError(
                f"preflight: streamed probe chat failed ({sprobe.error})")
        if not sprobe.request_uuid:
            raise RuntimeError(
                "preflight: streamed probe carried no chunk 'id' to correlate on")
        try:
            srow = await _await_request_row(db, sprobe.request_uuid,
                                            timeout_s=row_timeout_s)
        except RuntimeError as exc:
            raise RuntimeError(
                f"preflight: streamed chunk id {sprobe.request_uuid!r} does not "
                "resolve to requests.request_uuid — stream correlation requires "
                "the mock backend (use --stream-pct 0 or --model dlp-mock)"
            ) from exc
        _track(int(srow["id"]))
        progress("preflight: streamed chunk id resolves to a requests row")

    canary_rid: Optional[int] = None
    if scanner_mode != "off":
        canary = await send_chat(client, base_url, api_key, model, CANARY_TEXT,
                                 stream=False, plant_side=probe_side, allow_prod=True)
        if not canary.ok or not canary.request_uuid:
            raise RuntimeError(f"preflight: canary chat failed ({canary.error})")
        canary_row = await _await_request_row(db, canary.request_uuid,
                                              timeout_s=row_timeout_s)
        canary_rid = int(canary_row["id"])
        _track(canary_rid)
        deadline = time.monotonic() + canary_timeout_s
        while db.count_alerts_for_request_ids([canary_rid]) == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"preflight: DLP worker produced no alert for a canary SSN within "
                    f"{canary_timeout_s:.0f}s — worker dead, scanners disabled, or queue full?")
            await asyncio.sleep(poll_interval_s)
        progress("preflight: canary SSN alerted")

    return {"probe_request_id": probe_rid, "canary_request_id": canary_rid,
            "aux_request_ids": aux_ids}


def preflight(
    base_url: str,
    api_key: str,
    admin_key: str,
    db,
    model: str = DEFAULT_MODEL,
    plant_side: str = "prompt",
    scanner_mode: str = "regex",
    allow_prod: bool = False,
    progress=print,
    timeout: float = 120.0,
    stream_pct: float = 0.0,
) -> dict:
    """Standalone sync preflight (same checks run_e2e performs)."""
    require_local(base_url, allow_prod)
    active_scope(scanner_mode)   # validates the mode name

    async def _go():
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            return await _preflight_async(client, base_url, api_key, admin_key, db,
                                          model, plant_side, scanner_mode, progress,
                                          stream_pct=stream_pct)
    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Send phase
# ---------------------------------------------------------------------------

@dataclass
class _Planned:
    doc: LabeledDocument
    stream: bool
    jitter_ms: float
    plant_side: str = "prompt"
    result: Optional[SendResult] = None


async def _send_all_async(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    docs: List[LabeledDocument],
    plant_side: str,
    stream_pct: float,
    concurrency: int,
    seed: int,
    progress=print,
    plans_out: Optional[List[_Planned]] = None,
) -> Tuple[List[_Planned], float]:
    """Send every doc; returns (order-preserving plans, last-send monotonic).

    ``plans_out`` (when given) is extended with the plans BEFORE any send
    fires, so an interrupted caller can still resolve the uuids of whatever
    partially completed for cleanup.
    """
    # All randomness drawn up front, in doc order, so the plan is a pure
    # function of the seed regardless of task scheduling.
    rng = random.Random(seed)
    plans = []
    for d in docs:
        stream = rng.random() < stream_pct
        jitter = rng.uniform(0.0, 50.0)
        side = plant_side
        if plant_side == "mixed":
            side = "response" if rng.random() < 0.5 else "prompt"
        plans.append(_Planned(doc=d, stream=stream, jitter_ms=jitter,
                              plant_side=side))
    if plans_out is not None:
        plans_out.extend(plans)
    sem = asyncio.Semaphore(max(1, concurrency))
    last_send = {"t": time.monotonic()}
    done = {"n": 0}

    async def one(plan: _Planned) -> None:
        async with sem:
            await asyncio.sleep(plan.jitter_ms / 1000.0)
            plan.result = await send_chat(client, base_url, api_key, model,
                                          plan.doc.text, stream=plan.stream,
                                          plant_side=plan.plant_side,
                                          allow_prod=True)
            last_send["t"] = time.monotonic()
            done["n"] += 1
            if done["n"] % 25 == 0:
                progress(f"sent {done['n']}/{len(plans)}")

    await asyncio.gather(*(one(p) for p in plans))
    return plans, last_send["t"]


# ---------------------------------------------------------------------------
# Drain
# ---------------------------------------------------------------------------

async def _drain_async(
    db,
    request_ids: List[int],
    drain_timeout_s: float,
    settle_s: float,
    started_monotonic: Optional[float] = None,
    poll_interval_s: float = 2.0,
    _clock=time.monotonic,
    _sleep=asyncio.sleep,
) -> Tuple[float, bool]:
    """Wait until the post-hoc alert count stops moving.

    Settled = the count was unchanged for settle_s; gives up at
    drain_timeout_s (measured from started_monotonic, i.e. the last send).
    Returns (seconds since started_monotonic, settled).
    """
    start = started_monotonic if started_monotonic is not None else _clock()
    if not request_ids:
        return (_clock() - start, True)
    last_count = db.count_alerts_for_request_ids(request_ids)
    stable_since = _clock()
    settled = False
    while True:
        now = _clock()
        if now - stable_since >= settle_s:
            settled = True
            break
        if now - start >= drain_timeout_s:
            break
        await _sleep(poll_interval_s)
        count = db.count_alerts_for_request_ids(request_ids)
        if count != last_count:
            last_count = count
            stable_since = _clock()
    return (_clock() - start, settled)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(
    plans: List[_Planned],
    uuid_to_reqid: Dict[str, int],
    alerts: List[dict],
    lags: List[dict],
    scanner_mode: str,
    plant_side: str,
) -> Tuple[List[dict], dict]:
    """Join alerts to docs and compute (result rows, partial metrics).

    Docs whose request carries a scanner-error alert are "degraded": they
    were never (fully) scanned, so they are excluded from every coverage
    denominator and counted in coverage["degraded_docs"] instead. In-scope
    membership is variant-aware for regex (see entity_in_scope).
    """
    active_scope(scanner_mode)   # validates the mode name
    error_alerts = [a for a in alerts if _is_error_alert(a)]
    real_alerts = [a for a in alerts if not _is_error_alert(a)]
    degraded_reqids = {a["request_id"] for a in error_alerts}

    # One kept alert per request (lowest alert id = first written); extras
    # are surfaced as a note, never silently merged.
    kept_by_reqid: Dict[int, dict] = {}
    multi_alert_extras = 0
    for alert in sorted(real_alerts, key=lambda a: a["id"]):
        rid = alert["request_id"]
        if rid in kept_by_reqid:
            multi_alert_extras += 1
        else:
            kept_by_reqid[rid] = alert
    lag_by_alert_id = {row["alert_id"]: _f(row.get("lag_ms")) for row in lags}

    rows: List[dict] = []
    for plan in plans:
        res = plan.result
        rid = uuid_to_reqid.get(res.request_uuid) if res.request_uuid else None
        alert = kept_by_reqid.get(rid) if rid is not None else None
        alert_out = None
        if alert is not None:
            raw_cats = list(alert.get("categories") or [])
            canonical: List[str] = []
            for c in raw_cats:
                canon = canonicalize(c)
                if canon and canon not in canonical:
                    canonical.append(canon)
            alert_out = {
                "alert_id": int(alert["id"]),
                "severity": alert.get("severity"),
                "scanner": alert.get("scanner"),
                "categories": raw_cats,
                "canonical_categories": canonical,
                "confidence": _f(alert.get("confidence")),
                "scan_latency_ms": _f(alert.get("scan_latency_ms")),
                "lag_ms": lag_by_alert_id.get(alert["id"]),
                "entities_n": len(alert.get("entities") or []),
            }
        gt_categories = sorted({e.category for e in plan.doc.entities})
        rows.append({
            "doc_id": plan.doc.doc_id,
            "request_uuid": res.request_uuid,
            "request_id": rid,
            "plant_side": getattr(plan, "plant_side", plant_side),
            "stream": plan.stream,
            "status_code": res.status_code,
            "error": res.error,
            "client_latency_ms": res.e2e_ms,
            "expected_alert": not plan.doc.is_clean,
            "gt_categories": gt_categories,
            "in_scope": any(entity_in_scope(e.category, e.generator, scanner_mode)
                            for e in plan.doc.entities),
            "scan_degraded": rid is not None and rid in degraded_reqids,
            "alert": alert_out,
        })

    ok_rows = [r for r in rows if r["error"] is None and r["status_code"] == 200]
    send = {
        "n_sent": len(rows),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "client_latency_ms": summarize_latencies([r["client_latency_ms"] for r in ok_rows]),
    }

    # Scanner-error-degraded docs drop out of every accuracy denominator:
    # they were never scanned, so counting them as misses (or clean FP
    # non-events) would misattribute scanner outages as detection quality.
    scanned_rows = [r for r in ok_rows if not r["scan_degraded"]]
    degraded_rows = [r for r in ok_rows if r["scan_degraded"]]
    dirty = [r for r in scanned_rows if r["expected_alert"]]
    clean = [r for r in scanned_rows if not r["expected_alert"]]
    dirty_alerted = [r for r in dirty if r["alert"] is not None]
    in_scope_dirty = [r for r in dirty if r["in_scope"]]
    in_scope_alerted = [r for r in in_scope_dirty if r["alert"] is not None]
    clean_alerted = [r for r in clean if r["alert"] is not None]
    coverage = {
        "dirty_sent": len(dirty),
        "dirty_alerted": len(dirty_alerted),
        "rate": _ratio(len(dirty_alerted), len(dirty)),
        "in_scope_dirty_sent": len(in_scope_dirty),
        "in_scope_dirty_alerted": len(in_scope_alerted),
        "in_scope_rate": _ratio(len(in_scope_alerted), len(in_scope_dirty)),
        "degraded_docs": len(degraded_rows),
    }
    clean_fp = {
        "clean_sent": len(clean),
        "clean_alerted": len(clean_alerted),
        "rate": _ratio(len(clean_alerted), len(clean)),
    }

    per_category: Dict[str, dict] = {}
    for r in dirty:
        detected_cats = set(r["alert"]["canonical_categories"]) if r["alert"] else set()
        for cat in r["gt_categories"]:
            cell = per_category.setdefault(cat, {"expected": 0, "detected": 0})
            cell["expected"] += 1
            if cat in detected_cats:
                cell["detected"] += 1
    per_category_detection = {
        cat: {"expected": c["expected"], "detected": c["detected"],
              "recall": _ratio(c["detected"], c["expected"])}
        for cat, c in sorted(per_category.items())
    }

    levels = list(SEVERITY_ORDER)
    matrix: Dict[str, Dict[str, int]] = {e: {p: 0 for p in levels} for e in levels}
    sev_n = sev_exact = 0
    for r in dirty_alerted:
        exp = expected_severity(r["gt_categories"])
        pred = str(r["alert"]["severity"] or "").lower()
        row = matrix.setdefault(exp, {})           # tolerate off-model labels
        row[pred] = row.get(pred, 0) + 1
        sev_n += 1
        sev_exact += pred == exp
    severity = {"matrix": matrix, "exact_match_rate": _ratio(sev_exact, sev_n)}

    joined_alerts = [r["alert"] for r in rows if r["alert"] is not None]
    scanner_counts: Dict[str, int] = {}
    for a in joined_alerts:
        scanner_counts[a["scanner"]] = scanner_counts.get(a["scanner"], 0) + 1

    notes: Dict[str, Any] = {"multi_alert_extras": multi_alert_extras}
    if error_alerts:
        # dlp_worker rate-limits scanner-error alert rows (one per scanner
        # per window), so the observable degraded set is a LOWER BOUND.
        notes["scanner_error_note"] = (
            "degraded_docs is a lower bound: the worker rate-limits "
            "scanner-error alerts, so other docs may also have been skipped "
            "without an error row; coverage may still be contaminated")

    metrics = {
        "send": send,
        "coverage": coverage,
        "clean_fp": clean_fp,
        "per_category_detection": per_category_detection,
        "severity": severity,
        "scan_latency_ms": summarize_latencies(
            [a["scan_latency_ms"] for a in joined_alerts
             if a["scan_latency_ms"] is not None]),
        "scan_lag_ms": summarize_latencies(
            [a["lag_ms"] for a in joined_alerts if a["lag_ms"] is not None]),
        "scanner_counts": scanner_counts,
        "scanner_error_alerts": len(error_alerts),
        "notes": notes,
    }
    return rows, metrics


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_e2e_async(
    docs: List[LabeledDocument],
    base_url: str,
    api_key: str,
    admin_key: str,
    db,
    out_dir: Optional[str],
    scanner_mode: str = "regex",
    plant_side: str = "prompt",
    stream_pct: float = 0.3,
    concurrency: int = 8,
    model: str = DEFAULT_MODEL,
    drain_timeout_s: float = 300,
    settle_s: float = 10,
    keep_alerts: bool = False,
    allow_prod: bool = False,
    seed: int = 42,
    skip_config: bool = False,
    progress=print,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    poll_interval_s: float = 2.0,
    canary_timeout_s: float = 90.0,
) -> dict:
    require_local(base_url, allow_prod)
    active_scope(scanner_mode)   # validates the mode name
    if plant_side not in ("prompt", "response", "mixed", "echo"):
        raise ValueError(
            f"plant_side must be 'prompt', 'response', 'mixed' or 'echo', "
            f"got {plant_side!r}")

    if out_dir is None:
        out_dir = new_run_dir("e2e")
    else:
        os.makedirs(out_dir, exist_ok=True)
    run_id = os.path.basename(os.path.normpath(out_dir))
    created_at = utc_now_iso()
    metrics_path = os.path.join(out_dir, "e2e_metrics.json")

    # Hoisted so the finally block can resolve and purge no matter where a
    # failure or interrupt lands.
    snap = None
    metrics: Optional[dict] = None
    uuid_to_reqid: Dict[str, int] = {}
    aux_request_ids: List[int] = []
    sent_plans: List[_Planned] = []
    try:
        if not skip_config:
            snap = db.snapshot_dlp_config()
            with open(os.path.join(out_dir, "config_snapshot.json"), "w",
                      encoding="utf-8") as f:
                json.dump(db.snapshot_to_json(snap), f, indent=2)
            db.apply_overrides(SCANNER_MODES[scanner_mode])
            db.apply_overrides(SAFE_RUN_OVERRIDES)   # never email, never dedup
            # Pin the severity model: production classify_severity reads the
            # ambient dlp.severity_rules (falling back to "moderate" for any
            # unlisted label), so an unpinned run would measure config drift,
            # not the scanner. The snapshot covers the key (DLP_CONFIG_KEYS),
            # so the finally-path restore puts the real value back.
            db.set_config("dlp.severity_rules", SEVERITY_RULES_OVERRIDE)

        async with httpx.AsyncClient(transport=transport,
                                     timeout=httpx.Timeout(120.0)) as client:
            # aux ids (probe/stream-probe/canary) land in aux_request_ids AS
            # THEY RESOLVE via aux_ids_out, so the finally block can purge
            # them even when preflight itself raises partway through.
            await _preflight_async(client, base_url, api_key, admin_key, db,
                                   model, plant_side, scanner_mode, progress,
                                   poll_interval_s=poll_interval_s,
                                   canary_timeout_s=canary_timeout_s,
                                   stream_pct=stream_pct,
                                   aux_ids_out=aux_request_ids)
            progress(f"preflight OK; sending {len(docs)} docs "
                     f"(mode={scanner_mode}, side={plant_side})")
            plans, last_send = await _send_all_async(
                client, base_url, api_key, model, docs, plant_side,
                stream_pct, concurrency, seed, progress, plans_out=sent_plans)

        uuids = [p.result.request_uuid for p in plans
                 if p.result is not None and p.result.request_uuid]
        req_rows = db.fetch_requests_by_uuids(uuids) if uuids else []
        uuid_to_reqid.update({row["request_uuid"]: int(row["id"])
                              for row in req_rows})
        request_ids = sorted(uuid_to_reqid.values())

        drain_seconds, settled = await _drain_async(
            db, request_ids, drain_timeout_s, settle_s,
            started_monotonic=last_send, poll_interval_s=poll_interval_s)
        if not settled:
            progress(f"WARNING: alert count still moving at drain_timeout_s="
                     f"{drain_timeout_s}s — coverage numbers are a lower bound")

        alerts = db.fetch_alerts_by_request_ids(request_ids) if request_ids else []
        lags = db.fetch_scan_lags_ms(request_ids) if request_ids else []
        rows, metrics = _score(plans, uuid_to_reqid, alerts, lags,
                               scanner_mode, plant_side)

        # The purge lives in the finally block (it must survive mid-run
        # failures); cleanup numbers are backfilled there and this file is
        # re-written with them.
        metrics["cleanup"] = {"alerts_purged": None,
                              "residual_drain_settled": None}
        metrics["drain"] = {"seconds": drain_seconds, "settled": settled}
        metrics["run"] = {
            "run_id": run_id,
            "kind": "e2e",
            "created_at": created_at,
            "scanner_mode": scanner_mode,
            "plant_side": plant_side,
            "stream_pct": stream_pct,
            "concurrency": concurrency,
            "base_url": base_url,
            "model": model,
            "seed": seed,
            "n_docs": len(docs),
        }

        write_jsonl(os.path.join(out_dir, "e2e_results.jsonl"), rows)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        save_manifest(out_dir, RunManifest(
            run_id=run_id, kind="e2e", created_at=created_at,
            argv=list(sys.argv), seed=seed, base_url=base_url,
            scanner_mode=scanner_mode, git_rev=_git_rev(),
            extra={"plant_side": plant_side, "stream_pct": stream_pct,
                   "concurrency": concurrency, "model": model,
                   "n_docs": len(docs), "out_dir": os.path.abspath(out_dir)}))
        progress(f"e2e run complete -> {out_dir}")
        return metrics
    finally:
        # Cleanup runs on EVERY exit path, strictly ordered: (1) residual
        # drain while the SAFE overrides (email off) are still applied, so
        # straggler scans land before prod email config comes back; (2)
        # purge every synthetic alert those requests created; (3) only then
        # restore the config snapshot. Restoring first would let late scans
        # of planted PII email real recipients and survive the purge.
        purged: Optional[int] = None
        residual_settled = True
        try:
            unresolved = [p.result.request_uuid for p in sent_plans
                          if p.result is not None and p.result.request_uuid
                          and p.result.request_uuid not in uuid_to_reqid]
            if unresolved:   # interrupted before the post-send resolve ran
                for row in db.fetch_requests_by_uuids(unresolved):
                    uuid_to_reqid[row["request_uuid"]] = int(row["id"])
            purge_ids = sorted(set(uuid_to_reqid.values()) | set(aux_request_ids))
            if purge_ids:
                _, residual_settled = await _drain_async(
                    db, purge_ids, drain_timeout_s, settle_s,
                    poll_interval_s=poll_interval_s)
                if not residual_settled:
                    progress(f"WARNING: alerts still arriving after the residual "
                             f"drain window — re-purge dlp_alerts for request "
                             f"ids {purge_ids} manually")
                if not keep_alerts:
                    purged = db.purge_alerts_for_request_ids(purge_ids)
        except BaseException as exc:
            ids = sorted(set(uuid_to_reqid.values()) | set(aux_request_ids))
            progress(f"WARNING: alert cleanup failed ({type(exc).__name__}: {exc})"
                     f" — manually purge dlp_alerts for request ids {ids}")
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                raise
        finally:
            if snap is not None:
                db.restore_dlp_config(snap)
        if metrics is not None:
            metrics["cleanup"] = {"alerts_purged": purged,
                                  "residual_drain_settled": residual_settled}
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            except OSError:
                pass


def run_e2e(
    docs: List[LabeledDocument],
    base_url: str,
    api_key: str,
    admin_key: str,
    db,
    out_dir: Optional[str],
    scanner_mode: str = "regex",
    plant_side: str = "prompt",
    stream_pct: float = 0.3,
    concurrency: int = 8,
    model: str = DEFAULT_MODEL,
    drain_timeout_s: float = 300,
    settle_s: float = 10,
    keep_alerts: bool = False,
    allow_prod: bool = False,
    seed: int = 42,
    skip_config: bool = False,
    progress=print,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    poll_interval_s: float = 2.0,
    canary_timeout_s: float = 90.0,
) -> dict:
    """Sync entry point: run the full e2e evaluation, return the metrics dict."""
    return asyncio.run(run_e2e_async(
        docs, base_url, api_key, admin_key, db, out_dir,
        scanner_mode=scanner_mode, plant_side=plant_side, stream_pct=stream_pct,
        concurrency=concurrency, model=model, drain_timeout_s=drain_timeout_s,
        settle_s=settle_s, keep_alerts=keep_alerts, allow_prod=allow_prod,
        seed=seed, skip_config=skip_config, progress=progress,
        transport=transport, poll_interval_s=poll_interval_s,
        canary_timeout_s=canary_timeout_s))
