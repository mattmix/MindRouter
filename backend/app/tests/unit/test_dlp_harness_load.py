############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_harness_load.py: Unit tests for the DLP harness
# load/overhead matrix driver (dlp_harness/load.py).
#
# Pure-function coverage only: phase summarization, baseline
# pairing, docker-stats / Prometheus parsers, drain-settle
# logic (fake clock + FakeDB), the prod guard, doc cycling
# determinism, and the driver's snapshot-persist / residual-
# purge / restore cleanup contract. No network (MockTransport
# only), no real DB, no docker.
#
############################################################

import asyncio
import json
import os
import sys
import time

import httpx
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dlp_harness import load as dlp_load                      # noqa: E402
from dlp_harness.schemas import (                             # noqa: E402
    GroundTruthEntity,
    LabeledDocument,
)


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------

def _rec(e2e=100.0, ttfb=None, ttft=None, status=200, error=None,
         warm=False, dirty=False, uuid="uuid-x", stream=True):
    return {"phase_id": "p", "ts": 0.0, "doc_id": "d", "request_uuid": uuid,
            "stream": stream, "status": status, "error": error,
            "ttfb_ms": ttfb, "ttft_ms": ttft, "e2e_ms": e2e,
            "expected_alert": dirty, "in_warmup": warm}


def _doc(i, dirty=False, with_entity=False, meta_flag=True):
    entities = ([GroundTruthEntity(category="ssn", text="123-45-6789",
                                   start=0, end=11)] if with_entity else [])
    meta = {"expected_alert": dirty} if meta_flag else {}
    return LabeledDocument(doc_id=f"doc-{i}", text=f"document text {i}",
                           entities=entities, meta=meta)


def _phase(mode, conc, p50=None, p95=None, rps=1.0, coverage=None):
    e2e = (None if p50 is None and p95 is None
           else {"n": 10, "mean": p50, "p50": p50, "p90": p95, "p95": p95,
                 "p99": p95, "max": p95})
    return {"phase_id": f"{mode}-c{conc}", "scanner_mode": mode,
            "concurrency": conc, "duration_s": 60.0, "warmup_s": 10.0,
            "offered": {"n_requests": 10, "n_ok": 10, "n_err": 0,
                        "rps": rps, "dirty_sent": 2},
            "latency_ms": {"ttfb": None, "ttft": None, "e2e": e2e},
            "dlp": {"coverage_rate": coverage, "alerts": 0,
                    "dirty_unscannable": 0,
                    "scan_lag_ms": None, "scan_latency_ms": None,
                    "drain_seconds": None, "drain_settled": None,
                    "queue_drops_logged": None, "scanner_error_alerts": 0},
            "cpu": {"app_mean_pct": None, "app_max_pct": None},
            "gateway_queue": {"mean": None, "max": None}}


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


# ---------------------------------------------------------------------------
# summarize_phase
# ---------------------------------------------------------------------------

def test_summarize_phase_warmup_exclusion_and_rps():
    records = [
        _rec(e2e=999.0, ttfb=900.0, ttft=950.0, warm=True),        # excluded
        _rec(e2e=888.0, ttfb=800.0, ttft=850.0, warm=True, dirty=True),
        _rec(e2e=100.0, ttfb=10.0, ttft=20.0, dirty=True),
        _rec(e2e=200.0, ttfb=20.0, ttft=30.0),
        _rec(e2e=300.0, ttfb=30.0, ttft=40.0, dirty=True),
        _rec(e2e=400.0, ttfb=40.0, ttft=50.0),
        _rec(e2e=50.0, status=500, error="HTTP 500: boom"),        # err
        _rec(e2e=60.0, status=None, error="ConnectError: x"),      # err
    ]
    s = dlp_load.summarize_phase(records, phase_id="regex-c4",
                                 scanner_mode="regex", concurrency=4,
                                 duration_s=60.0, warmup_s=10.0, stream=True)
    assert s["phase_id"] == "regex-c4"
    assert s["offered"]["n_requests"] == 6         # warmup rows excluded
    assert s["offered"]["n_ok"] == 4
    assert s["offered"]["n_err"] == 2
    assert s["offered"]["dirty_sent"] == 2         # post-warmup dirty only
    assert s["offered"]["rps"] == pytest.approx(4 / 50.0)

    e2e = s["latency_ms"]["e2e"]
    assert e2e["n"] == 4                           # errors + warmup excluded
    assert e2e["mean"] == pytest.approx(250.0)
    assert e2e["p50"] == pytest.approx(250.0)
    assert e2e["max"] == pytest.approx(400.0)
    assert s["latency_ms"]["ttfb"]["n"] == 4
    assert s["latency_ms"]["ttfb"]["max"] == pytest.approx(40.0)
    assert s["latency_ms"]["ttft"]["max"] == pytest.approx(50.0)
    # null-shaped defaults for the driver to fill
    assert s["dlp"]["coverage_rate"] is None
    assert s["cpu"] == {"app_mean_pct": None, "app_max_pct": None}
    assert s["gateway_queue"] == {"mean": None, "max": None}


def test_summarize_phase_nonstream_nulls_ttfb_ttft():
    records = [_rec(e2e=120.0, ttfb=None, ttft=None, stream=False)]
    s = dlp_load.summarize_phase(records, phase_id="off-c1",
                                 scanner_mode="off", concurrency=1,
                                 duration_s=30.0, warmup_s=5.0, stream=False)
    assert s["latency_ms"]["ttfb"] is None
    assert s["latency_ms"]["ttft"] is None
    assert s["latency_ms"]["e2e"]["n"] == 1
    assert s["offered"]["rps"] == pytest.approx(1 / 25.0)


# ---------------------------------------------------------------------------
# baseline comparison
# ---------------------------------------------------------------------------

def test_compute_baseline_comparison_pairing_and_math():
    phases = [
        _phase("off", 1, p50=100.0, p95=150.0, rps=10.0),
        _phase("off", 4, p50=110.0, p95=160.0, rps=0.0),      # zero-rps baseline
        _phase("regex", 1, p50=120.0, p95=180.0, rps=9.0, coverage=0.97),
        _phase("regex", 16, p50=500.0, p95=900.0, rps=5.0),   # no off-c16
        _phase("gliner", 4, p50=200.0, p95=400.0, rps=4.0, coverage=0.9),
    ]
    out = dlp_load.compute_baseline_comparison(phases)
    assert [(row["mode"], row["concurrency"]) for row in out] == \
        [("regex", 1), ("gliner", 4)]              # regex-c16 has no baseline

    regex = out[0]
    assert regex["e2e_p50_delta_ms"] == pytest.approx(20.0)
    assert regex["e2e_p95_delta_ms"] == pytest.approx(30.0)
    assert regex["throughput_delta_pct"] == pytest.approx(-10.0)
    assert regex["coverage_rate"] == pytest.approx(0.97)

    gliner = out[1]
    assert gliner["throughput_delta_pct"] is None  # off rps == 0 -> undefined
    assert gliner["e2e_p50_delta_ms"] == pytest.approx(90.0)


def test_compute_baseline_comparison_handles_null_latency():
    phases = [
        _phase("off", 1, p50=None, p95=None, rps=10.0),
        _phase("regex", 1, p50=120.0, p95=180.0, rps=9.0),
    ]
    out = dlp_load.compute_baseline_comparison(phases)
    assert len(out) == 1
    assert out[0]["e2e_p50_delta_ms"] is None
    assert out[0]["e2e_p95_delta_ms"] is None
    assert out[0]["throughput_delta_pct"] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# docker stats parser
# ---------------------------------------------------------------------------

def test_parse_docker_stats_line_real_shape():
    line = ('{"BlockIO":"4.84MB / 36.9kB","CPUPerc":"1.87%","ID":"d84ecc2ee9dc",'
            '"MemPerc":"3.73%","MemUsage":"292MiB / 7.653GiB",'
            '"Name":"mindrouter2-app-1","NetIO":"3.56kB / 1.77kB","PIDs":"8"}')
    parsed = dlp_load.parse_docker_stats_line(line)
    assert parsed == {"cpu_pct": pytest.approx(1.87),
                      "mem_mb": pytest.approx(292.0)}


def test_parse_docker_stats_line_units_and_dashes():
    gib = dlp_load.parse_docker_stats_line(
        '{"CPUPerc":"250.00%","MemUsage":"1.5GiB / 7.6GiB"}')
    assert gib["cpu_pct"] == pytest.approx(250.0)
    assert gib["mem_mb"] == pytest.approx(1536.0)

    no_mem = dlp_load.parse_docker_stats_line(
        '{"CPUPerc":"0.50%","MemUsage":"-- / --"}')
    assert no_mem == {"cpu_pct": pytest.approx(0.5), "mem_mb": None}

    assert dlp_load.parse_docker_stats_line(
        '{"CPUPerc":"--","MemUsage":"-- / --"}') is None    # container gone
    assert dlp_load.parse_docker_stats_line("not json at all") is None
    assert dlp_load.parse_docker_stats_line("[1, 2, 3]") is None
    assert dlp_load.parse_docker_stats_line(
        '{"CPUPerc":"1.0%","MemUsage":"9zz / 1GiB"}')["mem_mb"] is None


# ---------------------------------------------------------------------------
# gateway /metrics parser
# ---------------------------------------------------------------------------

def test_parse_gateway_queue_unlabeled_and_comments():
    text = ("# HELP mindrouter_queue_size Current queue size\n"
            "# TYPE mindrouter_queue_size gauge\n"
            "mindrouter_queue_size 3.0\n"
            "mindrouter_active_backends 12\n")
    assert dlp_load.parse_gateway_queue(text) == pytest.approx(3.0)


def test_parse_gateway_queue_labeled_with_timestamp():
    text = 'mindrouter_queue_size{worker="1",pid="7"} 7 1699999999999\n'
    assert dlp_load.parse_gateway_queue(text) == pytest.approx(7.0)


def test_parse_gateway_queue_rejects_prefix_collisions_and_absence():
    assert dlp_load.parse_gateway_queue("mindrouter_queue_size_bytes 99\n") is None
    assert dlp_load.parse_gateway_queue("some_other_metric 1\n") is None
    assert dlp_load.parse_gateway_queue("") is None
    assert dlp_load.parse_gateway_queue(None) is None


# ---------------------------------------------------------------------------
# drain settle logic
# ---------------------------------------------------------------------------

class FakeCountDB:
    """FakeDB surface for drain: count_alerts_for_request_ids over a script."""

    def __init__(self, counts):
        self._counts = list(counts)
        self.calls = 0

    def count_alerts_for_request_ids(self, request_ids,
                                     exclude_scanner_errors=True):
        i = min(self.calls, len(self._counts) - 1)
        self.calls += 1
        return self._counts[i]


def test_measure_drain_settles_after_counts_stabilize():
    clock = FakeClock(t=100.0)
    fake = FakeCountDB([0, 3, 5, 5, 5, 5, 5])
    drain_s, settled, last = dlp_load.measure_drain(
        lambda: fake.count_alerts_for_request_ids([1, 2, 3]),
        settle_s=4.0, timeout_s=60.0, t_start=98.0,
        poll_interval_s=2.0, sleep=clock.sleep, now=clock.now)
    assert settled is True
    assert last == 5
    # count last changed at t=104 (third poll); phase ended at t_start=98
    assert drain_s == pytest.approx(6.0)


def test_measure_drain_times_out_when_counts_keep_moving():
    clock = FakeClock(t=100.0)
    fake = FakeCountDB(list(range(100)))          # strictly increasing forever
    drain_s, settled, last = dlp_load.measure_drain(
        lambda: fake.count_alerts_for_request_ids([1]),
        settle_s=4.0, timeout_s=10.0,
        poll_interval_s=2.0, sleep=clock.sleep, now=clock.now)
    assert settled is False
    assert drain_s == pytest.approx(10.0)
    assert last == fake.calls - 1                 # saw the latest count


def test_measure_drain_immediate_settle_with_zero_settle_window():
    clock = FakeClock(t=50.0)
    fake = FakeCountDB([0])
    drain_s, settled, last = dlp_load.measure_drain(
        lambda: fake.count_alerts_for_request_ids([]),
        settle_s=0.0, timeout_s=60.0, t_start=50.0,
        sleep=clock.sleep, now=clock.now)
    assert settled is True
    assert last == 0
    assert drain_s == pytest.approx(0.0)
    assert fake.calls == 1                        # settled without re-polling


# ---------------------------------------------------------------------------
# phase DLP measurement (FakeDB, zero settle window -> no sleeping)
# ---------------------------------------------------------------------------

class FakeHarnessDB:
    def __init__(self):
        self.requests = [{"request_uuid": "u1", "id": 1},
                         {"request_uuid": "u2", "id": 2},
                         {"request_uuid": "u3", "id": 3}]
        self.alerts = [
            {"id": 10, "request_id": 1, "categories": ["ssn"],
             "scan_latency_ms": 12.5},
            {"id": 11, "request_id": 2, "categories": ["dlp_scanner_error"],
             "scan_latency_ms": 400.0},
        ]
        self.purged = []

    def fetch_requests_by_uuids(self, uuids):
        return [r for r in self.requests if r["request_uuid"] in set(uuids)]

    def count_alerts_for_request_ids(self, ids, exclude_scanner_errors=True):
        rows = [a for a in self.alerts if a["request_id"] in set(ids)]
        if exclude_scanner_errors:
            rows = [a for a in rows
                    if "dlp_scanner_error" not in a["categories"]]
        return len(rows)

    def fetch_alerts_by_request_ids(self, ids):
        return [a for a in self.alerts if a["request_id"] in set(ids)]

    def fetch_scan_lags_ms(self, ids):
        return [{"request_id": a["request_id"], "alert_id": a["id"],
                 "scan_latency_ms": a["scan_latency_ms"], "lag_ms": 33.0}
                for a in self.alerts if a["request_id"] in set(ids)]


def test_measure_phase_dlp_coverage_excludes_errors_and_warmup():
    records = [
        _rec(uuid="u1", dirty=True),               # dirty, alerted -> covered
        _rec(uuid="u2", dirty=True),               # dirty, only an error alert
        _rec(uuid="u3", dirty=True, warm=True),    # dirty but warmup -> excluded
    ]
    out = dlp_load._measure_phase_dlp(FakeHarnessDB(), records, t_stop=0.0,
                                      settle_s=0.0, drain_timeout_s=5.0)
    fields = out["fields"]
    assert sorted(out["request_ids"]) == [1, 2, 3]
    assert fields["coverage_rate"] == pytest.approx(0.5)   # u1 of {u1, u2}
    assert fields["alerts"] == 1                           # error alert excluded
    assert fields["scanner_error_alerts"] == 1
    assert fields["scan_latency_ms"]["n"] == 1             # non-error only
    assert fields["scan_latency_ms"]["max"] == pytest.approx(12.5)
    assert fields["scan_lag_ms"]["n"] == 2
    assert fields["drain_settled"] is True
    assert fields["dirty_unscannable"] == 0        # every record completed


def test_measure_phase_dlp_off_mode_nulls_coverage():
    # finding: an "off" phase used to report coverage 0.0 (no alerts can be
    # written with the scanner disabled), tripping a false CRITICAL in the
    # report; disabled-scanner coverage must be None
    records = [_rec(uuid="u1", dirty=True), _rec(uuid="u2", dirty=True)]
    out = dlp_load._measure_phase_dlp(FakeHarnessDB(), records, t_stop=0.0,
                                      settle_s=0.0, drain_timeout_s=5.0,
                                      scanner_mode="off")
    assert out["fields"]["coverage_rate"] is None
    assert sorted(out["request_ids"]) == [1, 2]    # ids still purgeable


def test_measure_phase_dlp_denominator_excludes_failed_requests():
    # finding: requests the gateway never completed (timeout/transport error)
    # can never be scanned, so they must not deflate coverage; they surface
    # in dirty_unscannable instead
    records = [
        _rec(uuid="u1", dirty=True),                                # ok, alerted
        _rec(uuid="u2", dirty=True),                                # ok, missed
        _rec(uuid="u3", dirty=True, error="timeout after 120.0s"),  # failed
        _rec(uuid=None, dirty=True, status=None,
             error="ConnectError: x"),                              # failed, no uuid
        _rec(uuid=None, dirty=True, warm=True, status=None,
             error="ConnectError: x"),                              # warmup: ignored
    ]
    out = dlp_load._measure_phase_dlp(FakeHarnessDB(), records, t_stop=0.0,
                                      settle_s=0.0, drain_timeout_s=5.0,
                                      scanner_mode="regex")
    fields = out["fields"]
    assert fields["coverage_rate"] == pytest.approx(0.5)   # u1 of {u1, u2}
    assert fields["dirty_unscannable"] == 2                # u3 + connect error
    assert sorted(out["request_ids"]) == [1, 2, 3]         # u3 still purged


def test_measure_phase_dlp_no_uuids_yields_nulls():
    records = [_rec(uuid=None, dirty=True)]
    out = dlp_load._measure_phase_dlp(FakeHarnessDB(), records, t_stop=0.0,
                                      settle_s=0.0, drain_timeout_s=5.0)
    assert out["request_ids"] == []
    assert out["fields"]["coverage_rate"] is None
    assert out["fields"]["alerts"] == 0
    assert out["fields"]["scan_lag_ms"] is None
    assert out["fields"]["dirty_unscannable"] == 0


# ---------------------------------------------------------------------------
# prod guard + argument validation
# ---------------------------------------------------------------------------

class GuardFakeDB:
    def __init__(self):
        self.snapshot_calls = 0

    def snapshot_dlp_config(self):
        self.snapshot_calls += 1
        return {}

    def restore_dlp_config(self, snap):
        pass

    def apply_overrides(self, overrides):
        pass


def test_run_load_matrix_refuses_non_local_base_url(tmp_path):
    fake = GuardFakeDB()
    with pytest.raises(RuntimeError, match="non-local"):
        dlp_load.run_load_matrix(
            "https://mindrouter.uidaho.edu", ["mr2_key"], "mr2_admin",
            fake, str(tmp_path / "out"), [_doc(0, dirty=True)])
    assert fake.snapshot_calls == 0               # refused before any DB touch
    assert not (tmp_path / "out").exists()        # and before any file I/O


def test_assert_local_url_accepts_local_hosts():
    for url in ("http://127.0.0.1:8000", "http://localhost:8000",
                "http://[::1]:8000"):
        dlp_load._assert_local_url(url, allow_prod=False)
    dlp_load._assert_local_url("https://mindrouter.uidaho.edu", allow_prod=True)


def test_run_load_matrix_validates_arguments(tmp_path):
    fake = GuardFakeDB()
    docs = [_doc(0, dirty=True)]
    with pytest.raises(ValueError, match="api key"):
        dlp_load.run_load_matrix("http://localhost:8000", [], "a", fake,
                                 str(tmp_path), docs)
    with pytest.raises(ValueError, match="unknown scanner mode"):
        dlp_load.run_load_matrix("http://localhost:8000", ["k"], "a", fake,
                                 str(tmp_path), docs, modes=("off", "llm"))
    with pytest.raises(ValueError, match="warmup_s"):
        dlp_load.run_load_matrix("http://localhost:8000", ["k"], "a", fake,
                                 str(tmp_path), docs, duration_s=10.0,
                                 warmup_s=10.0)
    with pytest.raises(ValueError, match="corpus"):
        dlp_load.run_load_matrix("http://localhost:8000", ["k"], "a", fake,
                                 str(tmp_path), [])
    assert fake.snapshot_calls == 0


# ---------------------------------------------------------------------------
# doc cycling
# ---------------------------------------------------------------------------

def test_doc_cycler_round_robin_preserves_corpus_mix():
    docs = [_doc(i, dirty=(i % 3 == 0)) for i in range(5)]
    cycler = dlp_load.DocCycler(docs, dirty_rate=None, seed=1)
    ids = [cycler.next().doc_id for _ in range(12)]
    assert ids == [f"doc-{i % 5}" for i in range(12)]


def test_doc_cycler_deterministic_across_instances():
    docs = [_doc(i, dirty=(i % 2 == 0)) for i in range(10)]
    a = dlp_load.DocCycler(docs, dirty_rate=0.5, seed=7)
    b = dlp_load.DocCycler(docs, dirty_rate=0.5, seed=7)
    seq_a = [a.next().doc_id for _ in range(40)]
    seq_b = [b.next().doc_id for _ in range(40)]
    assert seq_a == seq_b
    assert any(d != seq_a[0] for d in seq_a)      # actually mixes documents


def test_doc_cycler_dirty_rate_extremes():
    docs = [_doc(i, dirty=(i < 3)) for i in range(6)]
    all_dirty = dlp_load.DocCycler(docs, dirty_rate=1.0, seed=3)
    assert all(dlp_load.doc_expected_alert(all_dirty.next()) for _ in range(20))
    all_clean = dlp_load.DocCycler(docs, dirty_rate=0.0, seed=3)
    assert not any(dlp_load.doc_expected_alert(all_clean.next())
                   for _ in range(20))


def test_doc_cycler_validation():
    with pytest.raises(ValueError):
        dlp_load.DocCycler([], dirty_rate=None)
    clean_only = [_doc(i, dirty=False) for i in range(3)]
    with pytest.raises(ValueError, match="no dirty"):
        dlp_load.DocCycler(clean_only, dirty_rate=0.5)
    dirty_only = [_doc(i, dirty=True) for i in range(3)]
    with pytest.raises(ValueError, match="no clean"):
        dlp_load.DocCycler(dirty_only, dirty_rate=0.5)
    with pytest.raises(ValueError, match="dirty_rate"):
        dlp_load.DocCycler(clean_only, dirty_rate=1.5)


def test_doc_expected_alert_meta_overrides_labels():
    # load-profile doc: meta flag wins even with no entity spans
    assert dlp_load.doc_expected_alert(_doc(0, dirty=True)) is True
    assert dlp_load.doc_expected_alert(_doc(1, dirty=False)) is False
    # no meta: fall back to ground-truth labels
    assert dlp_load.doc_expected_alert(
        _doc(2, with_entity=True, meta_flag=False)) is True
    assert dlp_load.doc_expected_alert(_doc(3, meta_flag=False)) is False


# ---------------------------------------------------------------------------
# _cpu_sampler subprocess hygiene
# ---------------------------------------------------------------------------

class HungProc:
    """Fake asyncio subprocess whose communicate() never returns."""

    def __init__(self):
        self.killed = False
        self.reaped = False
        self.returncode = None

    async def communicate(self):
        await asyncio.sleep(3600)

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.reaped = True
        return self.returncode


def test_cpu_sampler_kills_hung_docker_stats(monkeypatch):
    # finding: asyncio.wait_for cancels communicate() only — without an
    # explicit kill+wait the docker stats child leaks (one hung process
    # per phase) and its transport GCs after the loop closes
    proc = HungProc()

    async def fake_exec(*args, **kwargs):
        return proc

    monkeypatch.setattr(dlp_load.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(dlp_load, "_DOCKER_STATS_TIMEOUT_S", 0.05)
    samples = []

    async def run():
        await dlp_load._cpu_sampler("cid", "p1", 0.0, samples, asyncio.Event())

    asyncio.run(run())
    assert proc.killed is True
    assert proc.reaped is True                    # reaped inside the live loop
    assert samples == []


# ---------------------------------------------------------------------------
# matrix driver cleanup contract (snapshot persist / hoisted purge / restore)
# ---------------------------------------------------------------------------

_MISSING = object()


class MatrixFakeDB:
    """Event-recording HarnessDB fake: uuid "uN" resolves to request id N."""

    def __init__(self, snapshot=None, purge_raises=False):
        self.snapshot = ({"dlp.enabled": "false"} if snapshot is None
                         else snapshot)
        self.events = []
        self.purge_raises = purge_raises

    def snapshot_dlp_config(self):
        self.events.append(("snapshot",))
        return dict(self.snapshot)

    def snapshot_to_json(self, snap):
        return {k: (None if v is _MISSING else v) for k, v in snap.items()}

    def apply_overrides(self, overrides):
        self.events.append(("apply", dict(overrides)))

    def restore_dlp_config(self, snap):
        self.events.append(("restore", dict(snap)))

    def fetch_requests_by_uuids(self, uuids):
        return [{"request_uuid": u, "id": int(u[1:])} for u in uuids]

    def count_alerts_for_request_ids(self, ids, exclude_scanner_errors=True):
        return len(list(ids))

    def fetch_alerts_by_request_ids(self, ids):
        return [{"id": 100 + i, "request_id": i, "categories": ["ssn"],
                 "scan_latency_ms": 5.0} for i in ids]

    def fetch_scan_lags_ms(self, ids):
        return [{"request_id": i, "lag_ms": 10.0, "scan_latency_ms": 5.0}
                for i in ids]

    def purge_alerts_for_request_ids(self, ids):
        if self.purge_raises:
            raise RuntimeError("purge exploded")
        self.events.append(("purge", sorted(ids)))
        return len(list(ids))


def _make_fake_traffic(counter, fail_on_phase=None):
    """_run_phase_traffic stand-in: two ok dirty records per phase, appended
    in place (the real contract) so a phase that dies mid-run still leaves
    its records visible to the caller's pending-record accounting."""

    async def fake(base_url, api_keys, cycler, *, phase_id, concurrency,
                   duration_s, warmup_s, stream, model, max_tokens,
                   container_id, req_file, records=None):
        if records is None:
            records = []
        for _ in range(2):
            counter[0] += 1
            records.append({
                "phase_id": phase_id, "ts": warmup_s + 0.5, "doc_id": "d",
                "request_uuid": f"u{counter[0]}", "stream": stream,
                "status": 200, "error": None, "ttfb_ms": None,
                "ttft_ms": None, "e2e_ms": 50.0, "expected_alert": True,
                "in_warmup": False,
            })
        if fail_on_phase is not None and phase_id == fail_on_phase:
            raise RuntimeError(f"traffic died in {phase_id}")
        return {"records": records, "cpu_samples": [], "queue_samples": [],
                "t_stop": time.monotonic()}

    return fake


def _run_matrix(db, out_dir, monkeypatch, *, fake_traffic,
                modes=("off", "regex"), concurrencies=(1,), **kwargs):
    monkeypatch.setattr(dlp_load, "_resolve_container_id", lambda cd: None)
    monkeypatch.setattr(dlp_load, "_count_queue_drops", lambda cd, since: None)
    monkeypatch.setattr(dlp_load, "_run_phase_traffic", fake_traffic)
    return dlp_load.run_load_matrix(
        "http://127.0.0.1:8000", ["mr2_k"], "mr2_admin", db, out_dir,
        [_doc(0, dirty=True), _doc(1, dirty=False)],
        modes=modes, concurrencies=concurrencies,
        duration_s=2.0, warmup_s=1.0, settle_s=0.0,
        inter_phase_drain_timeout_s=1.0, progress=lambda msg: None, **kwargs)


def test_run_load_matrix_persists_snapshot_before_first_phase(tmp_path,
                                                              monkeypatch):
    # finding: the pre-run DLP config must be on disk (e2e.py shape) BEFORE
    # any mutation, so a hard kill leaves a restorable record
    class ApplyBoom(MatrixFakeDB):
        def apply_overrides(self, overrides):
            raise RuntimeError("db died before the first phase")

    db = ApplyBoom(snapshot={"dlp.enabled": "true",
                             "dlp.gliner.threshold": _MISSING})
    out = tmp_path / "run"
    monkeypatch.setattr(dlp_load, "_resolve_container_id", lambda cd: None)
    with pytest.raises(RuntimeError, match="before the first phase"):
        dlp_load.run_load_matrix(
            "http://127.0.0.1:8000", ["k"], "a", db, str(out),
            [_doc(0, dirty=True)], modes=("regex",), concurrencies=(1,),
            duration_s=2.0, warmup_s=1.0, settle_s=0.0,
            progress=lambda m: None)
    snap = json.loads((out / "config_snapshot.json").read_text())
    assert snap == {"dlp.enabled": "true", "dlp.gliner.threshold": None}
    # config restore still ran despite the failure
    assert db.events[-1] == ("restore", {"dlp.enabled": "true",
                                         "dlp.gliner.threshold": _MISSING})


def test_run_load_matrix_success_purges_residuals_then_restores(tmp_path,
                                                                monkeypatch):
    # finding-[12] ordering: late post-hoc scans must settle and be purged
    # under the safe (email-off) overrides, and only then may the real
    # config come back
    db = MatrixFakeDB()
    out = tmp_path / "run"
    result = _run_matrix(db, str(out), monkeypatch,
                         fake_traffic=_make_fake_traffic([0]))
    phases = result["phases"]
    assert [p["phase_id"] for p in phases] == ["off-c1", "regex-c1"]
    assert phases[0]["dlp"]["coverage_rate"] is None       # scanner off
    assert phases[1]["dlp"]["coverage_rate"] == pytest.approx(1.0)
    kinds = [e[0] for e in db.events]
    assert kinds[-1] == "restore"                          # restore is LAST
    assert kinds[-2] == "purge"
    assert db.events[-2][1] == [1, 2, 3, 4]                # whole-run residual
    assert (out / "config_snapshot.json").exists()
    assert (out / "load_phases.json").exists()


def test_run_load_matrix_failure_purges_hoisted_and_pending_ids(tmp_path,
                                                                monkeypatch):
    # finding: a mid-matrix crash must still resolve the in-flight phase's
    # uuids, purge every id seen so far, and only then restore
    db = MatrixFakeDB()
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="regex-c1"):
        _run_matrix(db, str(out), monkeypatch,
                    fake_traffic=_make_fake_traffic([0],
                                                    fail_on_phase="regex-c1"))
    kinds = [e[0] for e in db.events]
    assert kinds[-1] == "restore"
    assert kinds[-2] == "purge"
    # off-c1 resolved [1, 2]; the dying regex-c1 phase left u3/u4 pending
    assert db.events[-2][1] == [1, 2, 3, 4]
    assert (out / "config_snapshot.json").exists()


def test_purge_residuals_persists_ids_when_purge_fails(tmp_path):
    # cleanup must never raise (the caller's restore has to run) but a
    # failed purge must leave an operator-actionable id list on disk
    db = MatrixFakeDB(purge_raises=True)
    msgs = []
    dlp_load._purge_residuals(db, [1, 2], [{"request_uuid": "u3"}],
                              0.0, 1.0, str(tmp_path), msgs.append)
    saved = json.loads((tmp_path / "unpurged_request_ids.json").read_text())
    assert saved == {"request_ids": [1, 2, 3]}
    assert any("WARNING" in m for m in msgs)


def test_purge_residuals_noop_without_ids(tmp_path):
    db = MatrixFakeDB()
    dlp_load._purge_residuals(db, [], [], 0.0, 1.0, str(tmp_path),
                              lambda m: None)
    assert db.events == []
    assert not (tmp_path / "unpurged_request_ids.json").exists()


def test_run_load_matrix_compose_dir_defaults_to_repo_root(tmp_path,
                                                           monkeypatch):
    # finding: a hardcoded dev-laptop compose_dir silently nulled CPU and
    # queue-drop metrics on any other host; None must derive the repo root
    seen = []

    def capture(compose_dir):
        seen.append(compose_dir)
        return None

    class ApplyBoom(MatrixFakeDB):
        def apply_overrides(self, overrides):
            raise RuntimeError("stop early")

    monkeypatch.setattr(dlp_load, "_resolve_container_id", capture)
    common = dict(modes=("regex",), concurrencies=(1,), duration_s=2.0,
                  warmup_s=1.0, settle_s=0.0, progress=lambda m: None)
    with pytest.raises(RuntimeError, match="stop early"):
        dlp_load.run_load_matrix(
            "http://127.0.0.1:8000", ["k"], "a", ApplyBoom(),
            str(tmp_path / "r1"), [_doc(0, dirty=True)], **common)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.abspath(dlp_load.__file__)))
    assert seen == [repo_root]
    assert os.path.isdir(os.path.join(seen[0], "dlp_harness"))
    with pytest.raises(RuntimeError, match="stop early"):
        dlp_load.run_load_matrix(
            "http://127.0.0.1:8000", ["k"], "a", ApplyBoom(),
            str(tmp_path / "r2"), [_doc(0, dirty=True)],
            compose_dir="/opt/mindrouter", **common)
    assert seen[1] == "/opt/mindrouter"           # explicit dir passes through


# ---------------------------------------------------------------------------
# run_in_container.sh: stale-tree purge must precede the cp
# ---------------------------------------------------------------------------

def test_run_in_container_script_purges_stale_tree_before_cp():
    # finding: `docker compose cp DIR app:EXISTING_DIR` NESTS instead of
    # replacing, leaving the previous run's harness on PYTHONPATH — the
    # script must rm the container-side tree first
    script = os.path.join(os.path.dirname(os.path.abspath(dlp_load.__file__)),
                          "run_in_container.sh")
    with open(script, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    rm_idx = next(i for i, ln in enumerate(lines)
                  if ln.startswith("docker compose exec")
                  and "rm -rf /tmp/dlp_harness" in ln)
    cp_idx = next(i for i, ln in enumerate(lines)
                  if ln.startswith("docker compose cp dlp_harness "))
    assert rm_idx < cp_idx
