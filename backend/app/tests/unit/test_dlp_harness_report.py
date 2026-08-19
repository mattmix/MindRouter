############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# backend/app/tests/unit/test_dlp_harness_report.py:
# Unit tests for the DLP harness report generator
# (dlp_harness/report.py) — fixture run directories, the
# recommendations rule engine, missing/malformed artifact
# robustness, and a real run_offline round-trip that pins
# the producer/consumer artifact schema. No live HTTP, no
# real DB.
#
############################################################

"""Tests for dlp_harness.report: report generation + recommendations."""

import json
import os
import sys

import pytest

# Import dlp_harness from the repo root, not the app package (import-chain rule).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from dlp_harness import report


# ---------------------------------------------------------------------------
# Fixture builders (dicts per the artifact contracts)
# ---------------------------------------------------------------------------

_LAT = {"n": 10, "mean": 5.0, "p50": 4.0, "p90": 8.0, "p95": 9.0,
        "p99": 9.8, "max": 10.0}

_SWEEP_POINTS = [
    {"threshold": 0.3, "span_precision": 0.85, "span_recall": 0.97,
     "span_f1": 0.906, "doc_precision": 0.9, "doc_recall": 0.98,
     "doc_specificity": 0.9, "doc_fpr": 0.1},
    {"threshold": 0.5, "span_precision": 0.94, "span_recall": 0.93,
     "span_f1": 0.935, "doc_precision": 0.97, "doc_recall": 0.95,
     "doc_specificity": 0.97, "doc_fpr": 0.03},
    {"threshold": 0.7, "span_precision": 0.98, "span_recall": 0.80,
     "span_f1": 0.881, "doc_precision": 0.99, "doc_recall": 0.85,
     "doc_specificity": 0.99, "doc_fpr": 0.01},
]

_SEV_MATRIX = {"minor": {"minor": 20, "moderate": 1, "major": 0},
               "moderate": {"minor": 0, "moderate": 30, "major": 2},
               "major": {"minor": 0, "moderate": 3, "major": 39}}


def _offline():
    """offline_metrics.json in run_offline's REAL output shape.

    Mirrors dlp_harness/offline_eval.py: nested recall_by, severity_accuracy,
    latency_ms.{per_scanner,per_doc_total}, latency_by_length as a bucket dict,
    bootstrap.{doc_recall,doc_precision,span_recall}, sweep with the
    *_partial area names (plus transitional pr_auc/roc_auc duplicates).
    """
    return {
        "run": {"kind": "offline", "created_at": "2026-08-19T00:00:00+00:00",
                "corpus_path": None, "n_docs": 200, "n_dirty": 100, "n_clean": 100,
                "scanners_requested": ["regex", "gliner"],
                "scanners_effective": ["regex", "gliner"],
                "gliner_threshold": 0.5, "gliner_scan_threshold": 0.05,
                "gliner_categories": None, "gliner_max_chars": 10000,
                "in_container": False, "seed": 42, "n_boot": 500, "notes": []},
        "doc_confusion": {"tp": 95, "fp": 2, "tn": 98, "fn": 5,
                          "precision": 0.9794, "recall": 0.95, "specificity": 0.98,
                          "accuracy": 0.965, "balanced_accuracy": 0.965,
                          "f1": 0.9645, "mcc": 0.93, "fpr": 0.02, "fnr": 0.05},
        "span_confusion": {
            "per_category": {
                "ssn": {"tp": 40, "fn": 2, "fp": 1, "precision": 0.9756,
                        "recall": 0.9524, "f1": 0.9639},
                "credit_card": {"tp": 30, "fn": 3, "fp": 2, "precision": 0.9375,
                                "recall": 0.9091, "f1": 0.9231},
            },
            "overall": {"tp": 70, "fn": 5, "fp": 3, "precision": 0.9589,
                        "recall": 0.9333, "f1": 0.9459, "macro_precision": 0.956,
                        "macro_recall": 0.931, "macro_f1": 0.943},
        },
        "span_confusion_strict": {
            "per_category": {
                "ssn": {"tp": 38, "fn": 4, "fp": 1, "precision": 0.974,
                        "recall": 0.9048, "f1": 0.938},
                "credit_card": {"tp": 29, "fn": 4, "fp": 2, "precision": 0.935,
                                "recall": 0.8788, "f1": 0.906},
            },
            "overall": {"tp": 67, "fn": 8, "fp": 3, "precision": 0.957,
                        "recall": 0.893, "f1": 0.924},
        },
        "scope_split": {"in_scope": {"tp": 70, "fn": 3, "recall": 0.9589, "n": 73},
                        "out_of_scope": {"tp": 0, "fn": 2, "recall": 0.0, "n": 2}},
        "recall_by": {
            "difficulty": {"plain": {"tp": 60, "fn": 2, "recall": 0.9677, "n": 62}},
            "generator": {"ssn.dashed": {"tp": 40, "fn": 2, "recall": 0.9524, "n": 42}},
            "carrier": {"support_ticket": {"tp": 30, "fn": 1, "recall": 0.9677, "n": 31}},
        },
        "fp_traps": {"order_number": 2, "uuid": 1},
        "latency_ms": {"per_scanner": {"regex": dict(_LAT), "gliner": dict(_LAT)},
                       "per_doc_total": dict(_LAT)},
        "latency_by_length": {
            "0-500": {"n": 60, "regex": 0.4, "gliner": 11.0},
            "500-2000": {"n": 80, "regex": 1.1, "gliner": 45.0},
            "10000-50000": {"n": 20, "regex": 6.0, "gliner": 130.0},
            ">=200001": {"n": 2, "regex": 9.0, "gliner": None},
        },
        "severity_accuracy": {"matrix": _SEV_MATRIX, "n": 95,
                              "exact_match_rate": 0.9368},
        "bootstrap": {
            "doc_recall": {"lo": 0.90, "hi": 0.98, "point": 0.95, "degenerate": False},
            "doc_precision": {"lo": 0.93, "hi": 0.99, "point": 0.979, "degenerate": False},
            "span_recall": {"lo": 0.88, "hi": 0.96, "point": 0.933, "degenerate": False},
        },
        "sweep": {
            "points": [dict(p) for p in _SWEEP_POINTS],
            "pr_auc_partial": 0.93, "roc_auc_partial": 0.95,
            "pr_auc": 0.93, "roc_auc": 0.95,     # transitional duplicates
            "best_f1": {"threshold": 0.5, "f1": 0.935},
        },
        "scan_errors": 0,
    }


def _offline_legacy():
    """offline_metrics.json in the pre-alignment flat/legacy shape.

    Older artifacts on disk carry these keys (scanner_config, flat
    recall_by_*, severity, latency, *_ci bootstrap keys, latency_by_length as
    a list, bare pr_auc/roc_auc); the report must keep rendering them.
    """
    return {
        "scanner_config": {"mode": "regex+gliner", "gliner_threshold": 0.5},
        "doc_confusion": {"tp": 95, "fp": 2, "tn": 98, "fn": 5,
                          "precision": 0.9794, "recall": 0.95, "specificity": 0.98,
                          "f1": 0.9645, "mcc": 0.93, "fpr": 0.02, "fnr": 0.05},
        "span_confusion": {
            "per_category": {"ssn": {"tp": 40, "fn": 2, "fp": 1, "precision": 0.9756,
                                     "recall": 0.9524, "f1": 0.9639}},
            "overall": {"tp": 70, "fn": 5, "fp": 3, "precision": 0.9589,
                        "recall": 0.9333, "f1": 0.9459},
        },
        "recall_by_difficulty": {"plain": {"tp": 60, "fn": 2, "recall": 0.9677, "n": 62}},
        "recall_by_generator": {"ssn.dashed": {"tp": 40, "fn": 2, "recall": 0.9524, "n": 42}},
        "recall_by_carrier": {"support_ticket": {"tp": 30, "fn": 1, "recall": 0.9677, "n": 31}},
        "fp_traps": {"order_number": 2, "uuid": 1},
        "latency": {"regex": dict(_LAT), "gliner": None},
        "latency_by_length": [
            {"chars": 200, "regex_ms": 0.4, "gliner_ms": 11.0},
            {"chars": 2000, "regex_ms": 1.1, "gliner_ms": 45.0},
            {"chars": 20000, "regex_ms": 6.0, "gliner_ms": 130.0},
        ],
        "severity": {"matrix": _SEV_MATRIX, "n": 95, "exact_match_rate": 0.9368},
        "bootstrap": {"doc_recall_ci": {"lo": 0.90, "hi": 0.98, "point": 0.95},
                      "doc_precision_ci": {"lo": 0.93, "hi": 0.99, "point": 0.979},
                      "span_recall_ci": {"lo": 0.88, "hi": 0.96, "point": 0.933}},
        "sweep": {
            "points": [dict(p) for p in _SWEEP_POINTS],
            "pr_auc": 0.93, "roc_auc": 0.95,
            "best_f1": {"threshold": 0.5, "f1": 0.935},
        },
        "scan_errors": 0,
    }


def _e2e(coverage=1.0, drain_s=3.0, err_alerts=0):
    return {
        "run": {"scanner_mode": "regex+gliner", "plant_side": "prompt",
                "stream_pct": 0.5, "concurrency": 4,
                "base_url": "http://localhost:8000", "model": "dlp-mock"},
        "send": {"n_sent": 100, "n_ok": 100, "n_failed": 0,
                 "client_latency_ms": dict(_LAT)},
        "coverage": {"dirty_sent": 50, "dirty_alerted": int(50 * coverage),
                     "rate": coverage, "in_scope_dirty_sent": 48,
                     "in_scope_dirty_alerted": int(48 * coverage),
                     "in_scope_rate": coverage},
        "clean_fp": {"clean_sent": 50, "clean_alerted": 0, "rate": 0.0},
        "per_category_detection": {
            "ssn": {"expected": 25, "detected": 25, "recall": 1.0},
            "email": {"expected": 25, "detected": 24, "recall": 0.96}},
        "severity": {"matrix": {"major": {"major": 25, "moderate": 0, "minor": 0}},
                     "exact_match_rate": 1.0},
        "scan_latency_ms": dict(_LAT), "scan_lag_ms": dict(_LAT),
        "scanner_counts": {"regex": 40, "gliner": 10},
        "drain": {"seconds": drain_s, "settled": True},
        "scanner_error_alerts": err_alerts,
        "cleanup": {"alerts_purged": 50},
    }


def _phase(pid, mode, conc, coverage=1.0, drain=5.0, drops=0, errs=0):
    return {
        "phase_id": pid, "scanner_mode": mode, "concurrency": conc,
        "duration_s": 60, "warmup_s": 5,
        "offered": {"n_requests": 600, "n_ok": 600, "n_err": 0, "rps": 10.0,
                    "dirty_sent": 120},
        "latency_ms": {"ttfb": dict(_LAT), "ttft": None, "e2e": dict(_LAT)},
        "dlp": {"coverage_rate": coverage,
                "alerts": int(120 * coverage) if coverage is not None else 0,
                "scan_lag_ms": dict(_LAT), "scan_latency_ms": dict(_LAT),
                "drain_seconds": drain, "drain_settled": True,
                "queue_drops_logged": drops, "scanner_error_alerts": errs},
        "cpu": {"app_mean_pct": 35.0, "app_max_pct": 60.0},
        "gateway_queue": {"mean": 0.5, "max": 2.0},
    }


def _load_phases():
    # The off phase carries coverage 0.0 — what the load driver historically
    # wrote for scanner-off phases (dirty docs sent, DLP disabled, no alerts).
    # The report must not read that as scan loss.
    return {
        "phases": [_phase("p1", "off", 4, coverage=0.0),
                   _phase("p2", "regex", 4),
                   _phase("p3", "regex+gliner", 4),
                   _phase("p4", "regex+gliner", 16)],
        "baseline_comparison": [
            {"concurrency": 4, "mode": "regex", "e2e_p50_delta_ms": 2.0,
             "e2e_p95_delta_ms": 10.0, "throughput_delta_pct": -1.0,
             "coverage_rate": 1.0},
            {"concurrency": 4, "mode": "regex+gliner", "e2e_p50_delta_ms": 5.0,
             "e2e_p95_delta_ms": 40.0, "throughput_delta_pct": -2.0,
             "coverage_rate": 1.0},
        ],
    }


def _corpus_manifest():
    return {"docs": 100, "dirty_docs": 50, "clean_docs": 50,
            "entities": 75, "negatives": 20,
            "by_profile": {"accuracy": 100},
            "by_carrier": {"support_ticket": 60, "hr_email": 40},
            "by_category": {"ssn": 42, "credit_card": 33},
            "by_difficulty": {"plain": 62, "obfuscated": 13},
            "by_generator": {"ssn.dashed": 42, "cc.plain": 33},
            "char_histogram": {"<=500": 60, "<=1000": 40}}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def full_run_dir(tmp_path):
    rd = tmp_path / "run-full"
    rd.mkdir()
    _write_json(rd / "run.json",
                {"run_id": "t1", "kind": "full",
                 "created_at": "2026-08-19T00:00:00+00:00", "argv": [], "seed": 42,
                 "scanner_mode": "regex+gliner", "git_rev": "abc123"})
    _write_json(rd / "manifest.json", _corpus_manifest())
    _write_json(rd / "offline_metrics.json", _offline())
    _write_json(rd / "e2e_metrics.json", _e2e())
    _write_json(rd / "load_phases.json", _load_phases())
    _write_json(rd / "config_snapshot.json",
                {"dlp.enabled": True, "dlp.gliner.threshold": 0.5,
                 "dlp.dedup.enabled": False})
    _write_jsonl(rd / "offline_findings.jsonl", [{"doc_id": "d1"}])
    _write_jsonl(rd / "e2e_results.jsonl",
                 [{"doc_id": "d1", "request_uuid": "u1"},
                  {"doc_id": "d2", "request_uuid": "u2"}])
    _write_jsonl(rd / "load_requests.jsonl",
                 [{"phase_id": "p1", "doc_id": "d1"},
                  {"phase_id": "p2", "doc_id": "d2"},
                  {"phase_id": "p3", "doc_id": "d3"}])
    _write_jsonl(rd / "cpu_samples.jsonl",
                 [{"phase_id": "p1", "ts": 0.0, "cpu_pct": 30.0, "mem_mb": 512.0},
                  {"phase_id": "p1", "ts": 1.0, "cpu_pct": 32.0, "mem_mb": 514.0}])
    return rd


_OFFLINE_SECTION_HEADINGS = (
    "### Recall by difficulty", "### Recall by generator", "### Recall by carrier",
    "### Severity confusion", "### Scan latency (ms)",
    "### Bootstrap 95% confidence intervals",
)


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

def test_generate_report_full(full_run_dir, tmp_path):
    out_dir = tmp_path / "out"
    result = report.generate_report([str(full_run_dir)], str(out_dir))

    assert os.path.isfile(result["md_path"])
    assert os.path.isfile(result["html_path"])

    with open(result["html_path"], encoding="utf-8") as f:
        html_text = f.read()
    for marker in ("Executive summary", "Corpus", "Offline accuracy",
                   "Threshold sweep", "E2E detection", "Load &amp; overhead",
                   "Scan pipeline health", "Configuration appendix",
                   "Methodology"):
        assert marker in html_text, f"missing section marker: {marker}"
    assert "data:image/png;base64," in html_text

    assert result["charts"], "expected at least one chart"
    for p in result["charts"]:
        assert os.path.isfile(p)
    chart_names = {os.path.basename(p) for p in result["charts"]}
    assert "per_category_recall.png" in chart_names
    assert "load_throughput.png" in chart_names
    assert "threshold_sweep.png" in chart_names
    assert "scan_latency_vs_length.png" in chart_names

    with open(result["md_path"], encoding="utf-8") as f:
        md_text = f.read()
    assert "## Offline accuracy" in md_text
    assert "## Load & overhead" in md_text
    # producer-shape offline sections all render (regression: findings 5/21)
    for heading in _OFFLINE_SECTION_HEADINGS:
        assert heading in md_text, f"missing offline section: {heading}"
    assert "| doc recall |" in md_text            # bootstrap table has rows
    assert "per-doc total" in md_text             # latency_ms.per_doc_total row
    assert "![" in md_text                       # chart references
    assert "| category |" in md_text or "| matching |" in md_text
    # markdown chart links resolve relative to out_dir
    assert (out_dir / "charts" / "per_category_recall.png").is_file()


def test_generate_report_healthy_has_no_findings(full_run_dir, tmp_path):
    result = report.generate_report([str(full_run_dir)], str(tmp_path / "out2"))
    with open(result["html_path"], encoding="utf-8") as f:
        html_text = f.read()
    assert "No findings" in html_text


def test_legacy_offline_shape_still_renders(tmp_path):
    """Old on-disk artifacts (flat keys) keep rendering via the fallbacks."""
    rd = tmp_path / "run-legacy"
    rd.mkdir()
    _write_json(rd / "offline_metrics.json", _offline_legacy())
    result = report.generate_report([str(rd)], str(tmp_path / "out"))
    with open(result["md_path"], encoding="utf-8") as f:
        md_text = f.read()
    for heading in _OFFLINE_SECTION_HEADINGS:
        assert heading in md_text, f"missing offline section: {heading}"
    assert "| doc recall |" in md_text
    chart_names = {os.path.basename(p) for p in result["charts"]}
    assert "scan_latency_vs_length.png" in chart_names


def test_report_renders_real_run_offline_artifact(tmp_path):
    """Round-trip schema contract: run_offline's real artifact must render.

    Generates a tiny corpus, runs the REAL producer (run_offline, regex-only,
    local scanner bridge) into a run dir, then renders the report over it and
    asserts every offline section carries data — so any producer/consumer
    key drift fails the suite instead of silently emptying the report
    (regression: findings 5/21).
    """
    from dlp_harness import corpus, offline_eval

    docs = corpus.generate("accuracy", 20, seed=7)
    run_dir = tmp_path / "offline-run"
    metrics_out = offline_eval.run_offline(
        docs=docs, out_dir=str(run_dir), scanners=("regex",),
        n_boot=30, seed=7, progress=None)
    # sanity: the producer wrote the artifact the report will read
    assert os.path.isfile(run_dir / "offline_metrics.json")
    assert isinstance(metrics_out.get("recall_by"), dict)

    out_dir = tmp_path / "out"
    result = report.generate_report([str(run_dir)], str(out_dir))
    with open(result["md_path"], encoding="utf-8") as f:
        md_text = f.read()
    for heading in _OFFLINE_SECTION_HEADINGS:
        assert heading in md_text, f"missing offline section: {heading}"
    assert "| doc recall |" in md_text            # bootstrap rows present
    assert "per-doc total" in md_text             # per-scanner latency table
    chart_names = {os.path.basename(p) for p in result["charts"]}
    assert "scan_latency_vs_length.png" in chart_names


# ---------------------------------------------------------------------------
# Recommendations rule engine
# ---------------------------------------------------------------------------

def _healthy_data():
    return {"offline": _offline(), "e2e_metrics": _e2e(),
            "load_phases": _load_phases()["phases"],
            "baseline_comparison": _load_phases()["baseline_comparison"]}


def test_recs_silent_on_healthy():
    assert report.build_recommendations(_healthy_data()) == []


def test_rec_luhn_fires_on_low_cc_precision():
    off = _offline()
    off["span_confusion"]["per_category"]["credit_card"] = {
        "tp": 4, "fn": 3, "fp": 6, "precision": 0.4, "recall": 0.5714, "f1": 0.47}
    recs = report.build_recommendations({"offline": off})
    luhn = [r for r in recs if "Luhn" in r["recommendation"]]
    assert luhn, f"no Luhn recommendation in {recs}"
    assert luhn[0]["severity"] == "warn"
    assert "0.40" in luhn[0]["finding"]          # cites the number


def test_rec_specificity_names_top_traps():
    off = _offline()
    off["doc_confusion"]["specificity"] = 0.90
    recs = report.build_recommendations({"offline": off})
    hit = [r for r in recs if "specificity" in r["finding"].lower()]
    assert hit
    assert "order_number" in hit[0]["finding"]
    assert "0.900" in hit[0]["finding"]


def test_rec_threshold_change():
    off = _offline()
    off["sweep"]["best_f1"] = {"threshold": 0.7, "f1": 0.88}
    recs = report.build_recommendations({"offline": off})
    hit = [r for r in recs if "dlp.gliner.threshold" in r["recommendation"]]
    assert hit
    assert hit[0]["severity"] == "info"
    assert "0.70" in hit[0]["recommendation"]


def test_rec_threshold_silent_near_default():
    off = _offline()
    off["sweep"]["best_f1"] = {"threshold": 0.53, "f1": 0.93}   # within 0.05
    recs = report.build_recommendations({"offline": off})
    assert not [r for r in recs if "dlp.gliner.threshold" in r["recommendation"]]


def test_rec_coverage_drop_is_critical_and_names_phase():
    data = {"load_phases": [_phase("p9", "regex+gliner", 32,
                                   coverage=0.9, drops=12)]}
    recs = report.build_recommendations(data)
    assert recs
    assert recs[0]["severity"] == "critical"
    assert "p9" in recs[0]["finding"]
    assert "0.900" in recs[0]["finding"]
    assert "12" in recs[0]["finding"]            # queue_drops_logged cited
    assert "queue" in recs[0]["recommendation"].lower()


def test_rec_coverage_ignores_off_phases():
    """Scanner-off baseline phases (coverage 0.0 by construction — DLP is
    disabled, no alerts can exist) must not fire the scan-loss critical rule
    (regression: finding 6)."""
    data = {"load_phases": [_phase("off-c1", "off", 1, coverage=0.0),
                            _phase("off-c16", "off", 16, coverage=0.0)]}
    assert report.build_recommendations(data) == []
    # ... and an off phase never masks a real drop in a scanning phase
    data["load_phases"].append(_phase("rx-c16", "regex", 16, coverage=0.9))
    recs = report.build_recommendations(data)
    assert recs and recs[0]["severity"] == "critical"
    assert "rx-c16" in recs[0]["finding"]
    assert "off-c1" not in recs[0]["finding"]


def test_exec_worst_coverage_tile_skips_off_phases(tmp_path):
    """The 'Worst load coverage' executive tile must ignore scanner-off
    phases, not report 0.0% on every default (off,regex) run
    (regression: finding 6)."""
    rd = tmp_path / "run-load"
    rd.mkdir()
    _write_json(rd / "load_phases.json",
                {"phases": [_phase("off-c4", "off", 4, coverage=0.0),
                            _phase("rx-c4", "regex", 4, coverage=1.0)],
                 "baseline_comparison": []})
    result = report.generate_report([str(rd)], str(tmp_path / "out"))
    with open(result["html_path"], encoding="utf-8") as f:
        html_text = f.read()
    assert "Worst load coverage" in html_text
    assert '<div class="v">100.0%</div>' in html_text
    assert '<div class="v">0.0%</div>' not in html_text
    assert "Scans were dropped" not in html_text


def test_rec_drain_backlog():
    data = {"load_phases": [_phase("p2", "regex", 8, drain=120.0)]}
    recs = report.build_recommendations(data)
    hit = [r for r in recs if "backlog" in r["finding"].lower()]
    assert hit
    assert "120" in hit[0]["finding"]
    # e2e drain also triggers the same rule
    recs2 = report.build_recommendations({"e2e_metrics": _e2e(drain_s=90.0)})
    assert [r for r in recs2 if "backlog" in r["finding"].lower()]


def test_rec_overhead():
    data = {"baseline_comparison": [
        {"concurrency": 16, "mode": "regex+gliner", "e2e_p50_delta_ms": 100.0,
         "e2e_p95_delta_ms": 400.0, "throughput_delta_pct": -2.0,
         "coverage_rate": 1.0}]}
    recs = report.build_recommendations(data)
    hit = [r for r in recs if "overhead" in r["finding"].lower()]
    assert hit
    assert "400" in hit[0]["finding"]
    # throughput arm of the rule
    data2 = {"baseline_comparison": [
        {"concurrency": 16, "mode": "gliner", "e2e_p50_delta_ms": 1.0,
         "e2e_p95_delta_ms": 5.0, "throughput_delta_pct": -20.0,
         "coverage_rate": 1.0}]}
    assert [r for r in report.build_recommendations(data2)
            if "overhead" in r["finding"].lower()]


def test_rec_scan_errors():
    off = _offline()
    off["scan_errors"] = 3
    recs = report.build_recommendations({"offline": off})
    hit = [r for r in recs if "degraded" in r["finding"].lower()]
    assert hit
    assert "3" in hit[0]["finding"]


def test_rec_truncation_blindness_flat_legacy_key():
    off = _offline_legacy()
    off["recall_by_depth"] = {
        "0": {"tp": 10, "fn": 0, "recall": 1.0, "n": 10},
        "15000": {"tp": 0, "fn": 10, "recall": 0.0, "n": 10}}
    recs = report.build_recommendations({"offline": off})
    hit = [r for r in recs if "truncation" in r["finding"].lower()]
    assert hit
    assert hit[0]["severity"] == "info"
    assert "max_scan_chars" in hit[0]["recommendation"]


def test_rec_truncation_blindness_nested_recall_by():
    # producer shape: depth lives inside the nested recall_by dict
    off = _offline()
    off["recall_by"]["depth"] = {
        "0": {"tp": 10, "fn": 0, "recall": 1.0, "n": 10},
        "15000": {"tp": 0, "fn": 10, "recall": 0.0, "n": 10}}
    recs = report.build_recommendations({"offline": off})
    hit = [r for r in recs if "truncation" in r["finding"].lower()]
    assert hit
    assert hit[0]["severity"] == "info"
    assert "max_scan_chars" in hit[0]["recommendation"]


def test_recs_sorted_most_severe_first():
    off = _offline()
    off["span_confusion"]["per_category"]["credit_card"]["precision"] = 0.4
    data = {"offline": off,
            "load_phases": [_phase("p9", "regex", 8, coverage=0.5, drops=100)]}
    recs = report.build_recommendations(data)
    assert len(recs) >= 2
    assert recs[0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# Sweep partial areas (regression: finding 11)
# ---------------------------------------------------------------------------

def test_sweep_partial_areas_prefers_explicit_names():
    pr, roc = report._sweep_partial_areas(
        {"pr_auc_partial": 0.1, "roc_auc_partial": 0.2,
         "pr_auc": 0.9, "roc_auc": 0.8})
    assert (pr, roc) == (0.1, 0.2)


def test_sweep_partial_areas_falls_back_to_legacy_names():
    assert report._sweep_partial_areas({"pr_auc": 0.9, "roc_auc": 0.8}) == (0.9, 0.8)
    assert report._sweep_partial_areas({}) == (None, None)


def _section_text(sec):
    return "\n".join(str(b[1]) for b in sec["blocks"]
                     if b[0] in ("h3", "p", "note", "pre"))


def test_sweep_section_captions_legacy_areas_as_partial():
    # Even artifacts carrying only the legacy pr_auc/roc_auc names must be
    # captioned as partial/observed-range areas, never as full-curve AUCs.
    sec = report._sec_sweep({"offline": _offline_legacy()}, {})
    text = _section_text(sec)
    assert "Partial PR area 0.930" in text
    assert "partial ROC area 0.950" in text
    assert "observed recall [0.800, 0.970]" in text
    assert "observed FPR [0.010, 0.100]" in text
    assert "not full-curve AUCs" in text
    assert "ROC-AUC" not in text


def test_sweep_chart_caption_marks_partial_area():
    result = report._chart_sweep({"offline": _offline_legacy()})
    assert result is not None
    fig, caption = result
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert "Partial PR area 0.930" in caption
    assert "partial ROC area 0.950" in caption
    assert "not full-curve AUCs" in caption


def test_bucket_chars_parses_length_bucket_labels():
    assert report._bucket_chars("0-500") == 250.0
    assert report._bucket_chars("500-2000") == 1250.0
    assert report._bucket_chars(">=200001") == 200001.0
    assert report._bucket_chars("junk") is None


# ---------------------------------------------------------------------------
# Offline section shape details
# ---------------------------------------------------------------------------

def test_bootstrap_heading_absent_without_rows():
    # a bootstrap dict with no renderable CI rows must not leave a bare heading
    off = _offline()
    off["bootstrap"] = {"doc_recall": None, "unrelated": 3}
    sec = report._sec_offline({"offline": off}, {})
    assert "Bootstrap 95% confidence intervals" not in _section_text(sec)


def test_offline_latency_chart_accepts_bucket_dict():
    fig_caption = report._chart_latency_by_length({"offline": _offline()})
    assert fig_caption is not None
    fig, _caption = fig_caption
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Robustness: missing and malformed artifacts
# ---------------------------------------------------------------------------

def test_empty_run_dir_still_renders(tmp_path):
    empty = tmp_path / "empty-run"
    empty.mkdir()
    result = report.generate_report([str(empty)], str(tmp_path / "out"))
    assert os.path.isfile(result["md_path"])
    assert os.path.isfile(result["html_path"])
    assert result["charts"] == []
    with open(result["html_path"], encoding="utf-8") as f:
        html_text = f.read()
    assert "No data" in html_text
    assert "Methodology" in html_text            # static sections still render


def test_malformed_json_is_skipped_with_warning(tmp_path):
    rd = tmp_path / "bad-run"
    rd.mkdir()
    (rd / "offline_metrics.json").write_text("{not valid json", encoding="utf-8")
    _write_json(rd / "e2e_metrics.json", _e2e())
    result = report.generate_report([str(rd)], str(tmp_path / "out"))
    with open(result["html_path"], encoding="utf-8") as f:
        html_text = f.read()
    assert "malformed JSON skipped" in html_text
    assert "offline_metrics.json" in html_text
    assert "E2E detection" in html_text          # good artifact still rendered
    assert "100" in html_text                    # e2e n_sent made it through


def test_missing_run_dir_warned_not_fatal(tmp_path):
    result = report.generate_report([str(tmp_path / "nope")], str(tmp_path / "out"))
    with open(result["html_path"], encoding="utf-8") as f:
        html_text = f.read()
    assert "run directory not found" in html_text


def test_load_run_data_merges_load_phases_across_dirs(tmp_path):
    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    d1.mkdir(), d2.mkdir()
    _write_json(d1 / "load_phases.json",
                {"phases": [_phase("a1", "regex", 4)], "baseline_comparison": []})
    _write_json(d2 / "load_phases.json",
                {"phases": [_phase("b1", "gliner", 4)], "baseline_comparison": []})
    data = report.load_run_data([str(d1), str(d2)])
    assert {p["phase_id"] for p in data["load_phases"]} == {"a1", "b1"}
