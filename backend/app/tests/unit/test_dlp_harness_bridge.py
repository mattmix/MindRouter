############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_harness_bridge.py: Unit tests for the DLP harness
# scanner bridge — standalone load hygiene and batch scan
# semantics. Must pass without gliner installed.
#
############################################################

"""Unit tests for dlp_harness.scanner_bridge."""

import sys
import time
from pathlib import Path

import pytest

# Established harness-suite pattern: make the repo root importable so
# `dlp_harness` resolves regardless of pytest rootdir.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dlp_harness import constants, scanner_bridge

SSN = "123-45-6789"
EMAIL = "user@example.com"


def _fresh_cache(monkeypatch):
    """Force the next load_scanner_module call to actually load."""
    monkeypatch.setattr(scanner_bridge, "_module_cache", {})


# ===================================================================
# load_scanner_module
# ===================================================================

class TestLoadScannerModule:

    def test_load_and_sys_modules_restored(self, monkeypatch):
        _fresh_cache(monkeypatch)
        keys = ("backend", "backend.app", "backend.app.logging_config")
        before = {k: sys.modules.get(k) for k in keys}
        present_before = {k for k in keys if k in sys.modules}

        mod = scanner_bridge.load_scanner_module()

        assert mod.MAX_SCAN_CHARS == 200_000
        assert mod.GLINER_DEFAULT_MAX_CHARS == 10_000
        assert callable(mod.scan_regex)
        assert callable(mod.run_dlp_scan)
        for k in keys:
            if k in present_before:
                assert sys.modules[k] is before[k], f"{k} not restored"
            else:
                assert k not in sys.modules, f"{k} leaked into sys.modules"

    def test_repeat_calls_return_cached_module(self, monkeypatch):
        _fresh_cache(monkeypatch)
        assert scanner_bridge.load_scanner_module() is scanner_bridge.load_scanner_module()

    def test_env_var_path_override_honored(self, monkeypatch, tmp_path):
        _fresh_cache(monkeypatch)
        monkeypatch.setenv("DLP_SCANNER_PATH", str(tmp_path / "missing.py"))
        with pytest.raises((FileNotFoundError, OSError, ImportError)):
            scanner_bridge.load_scanner_module()

    def test_loaded_logger_is_inert(self):
        # The stub logger must swallow structlog-style kwargs silently.
        mod = scanner_bridge.load_scanner_module()
        mod.logger.warning("event_name", key="value")
        mod.logger.info("event_name")
        mod.logger.error("event_name", a=1, b=2)
        mod.logger.exception("event_name")


# ===================================================================
# scan_documents — regex path
# ===================================================================

class TestScanDocumentsRegex:

    def test_regex_findings_spans_slice_equal(self):
        text = f"My SSN is {SSN}, reach me at {EMAIL} thanks."
        res = scanner_bridge.scan_documents({"d1": text})
        r = res["d1"]

        assert r["errors"] == {}
        assert "regex" in r["latency_ms"] and r["latency_ms"]["regex"] >= 0.0

        by_cat = {}
        for f in r["findings"]:
            assert f["scanner"] == "regex"
            assert text[f["start"]:f["end"]] == f["text"]
            by_cat.setdefault(f["category"], []).append(f)

        assert [f["text"] for f in by_cat["social security number"]] == [SSN]
        assert [f["text"] for f in by_cat["email"]] == [EMAIL]

    def test_global_cap_truncates_before_scanning(self):
        mod = scanner_bridge.load_scanner_module()
        # Space padding fails every builtin pattern in O(1) per position;
        # alphanumeric padding would make the email regex quadratic.
        text = " " * mod.MAX_SCAN_CHARS + SSN

        capped = scanner_bridge.scan_documents({"d1": text})["d1"]
        assert capped["findings"] == []
        assert capped["errors"] == {}

        # Negative control: without the cap the same entity IS found.
        uncapped = scanner_bridge.scan_documents(
            {"d1": text}, apply_global_cap=False)["d1"]
        ssn = [f for f in uncapped["findings"]
               if f["category"] == "social security number"]
        assert len(ssn) == 1
        assert ssn[0]["start"] == mod.MAX_SCAN_CHARS
        assert text[ssn[0]["start"]:ssn[0]["end"]] == SSN

    def test_unknown_scanner_rejected(self):
        with pytest.raises(ValueError):
            scanner_bridge.scan_documents({"d1": "x"}, scanners=("regex", "llm"))


# ===================================================================
# scan_documents — gliner path (gliner absent / untouched)
# ===================================================================

class TestScanDocumentsGliner:

    def test_gliner_absent_reports_error_and_keeps_regex(self, monkeypatch):
        mod = scanner_bridge.load_scanner_module()
        # None in sys.modules forces ImportError even where gliner IS installed,
        # so this test is deterministic in-container too.
        monkeypatch.setitem(sys.modules, "gliner", None)
        monkeypatch.setattr(mod, "_gliner_model", None)

        text = f"SSN {SSN} here"
        r = scanner_bridge.scan_documents(
            {"d1": text}, scanners=("regex", "gliner"))["d1"]

        assert "gliner" in r["errors"]
        assert "unavailable" in r["errors"]["gliner"]
        assert set(r["latency_ms"]) == {"regex", "gliner"}
        cats = [f["category"] for f in r["findings"]]
        assert "social security number" in cats

    def test_regex_only_never_touches_gliner(self, monkeypatch):
        mod = scanner_bridge.load_scanner_module()

        def _boom(*args, **kwargs):
            raise AssertionError("gliner path touched")

        monkeypatch.setattr(mod, "scan_gliner", _boom)
        monkeypatch.setattr(mod, "_load_gliner", _boom)

        r = scanner_bridge.scan_documents(
            {"d1": f"SSN {SSN}"}, scanners=("regex",))["d1"]
        assert r["errors"] == {}
        assert "gliner" not in r["latency_ms"]
        assert any(f["category"] == "social security number" for f in r["findings"])

    def test_gliner_available_false_when_blocked(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gliner", None)
        assert scanner_bridge.gliner_available() is False

    def test_gliner_available_returns_bool(self):
        assert isinstance(scanner_bridge.gliner_available(), bool)


# ===================================================================
# scan_documents — gliner warmup (finding [10])
# ===================================================================

class TestGlinerWarmup:

    @staticmethod
    def _recording_fake(calls, warmup_delay_s=0.0):
        async def fake_scan_gliner(text, categories=None, threshold=0.5,
                                   max_chars=10_000):
            calls.append(text)
            if len(calls) == 1 and warmup_delay_s:
                time.sleep(warmup_delay_s)  # simulate one-time model load
            return []
        return fake_scan_gliner

    def test_warmup_call_precedes_timed_loop(self, monkeypatch):
        mod = scanner_bridge.load_scanner_module()
        calls = []
        monkeypatch.setattr(mod, "scan_gliner", self._recording_fake(calls))
        res = scanner_bridge.scan_documents(
            {"d1": "one", "d2": "two"}, scanners=("gliner",))
        assert calls == ["warmup", "one", "two"]
        assert set(res) == {"d1", "d2"}          # warmup adds no result row
        for r in res.values():
            assert r["errors"] == {}
            assert r["latency_ms"]["gliner"] >= 0.0

    def test_warmup_latency_excluded_from_samples(self, monkeypatch):
        mod = scanner_bridge.load_scanner_module()
        calls = []
        monkeypatch.setattr(mod, "scan_gliner",
                            self._recording_fake(calls, warmup_delay_s=0.3))
        res = scanner_bridge.scan_documents({"d1": "x"}, scanners=("gliner",))
        # The 300ms "model load" happened in the untimed warmup call, so the
        # first per-doc sample must not absorb it.
        assert res["d1"]["latency_ms"]["gliner"] < 250.0

    def test_warmup_failure_swallowed_real_error_attributed(self, monkeypatch):
        mod = scanner_bridge.load_scanner_module()
        calls = []

        async def fake_scan_gliner(text, categories=None, threshold=0.5,
                                   max_chars=10_000):
            calls.append(text)
            raise mod.DlpScannerError("gliner unavailable")

        monkeypatch.setattr(mod, "scan_gliner", fake_scan_gliner)
        res = scanner_bridge.scan_documents({"d1": "x"}, scanners=("gliner",))
        assert calls[0] == "warmup"              # warmup failure swallowed
        assert "unavailable" in res["d1"]["errors"]["gliner"]

    def test_warmup_receives_run_config(self, monkeypatch):
        mod = scanner_bridge.load_scanner_module()
        seen = []

        async def fake_scan_gliner(text, categories=None, threshold=0.5,
                                   max_chars=10_000):
            seen.append((text, categories, threshold, max_chars))
            return []

        monkeypatch.setattr(mod, "scan_gliner", fake_scan_gliner)
        scanner_bridge.scan_documents({"d1": "x"}, scanners=("gliner",),
                                      gliner_threshold=0.7,
                                      gliner_categories=["person"],
                                      gliner_max_chars=123)
        assert seen[0] == ("warmup", ["person"], 0.7, 123)


# ===================================================================
# prod_default_config
# ===================================================================

class TestProdDefaultConfig:

    def test_shape_and_defaults(self):
        cfg = scanner_bridge.prod_default_config()
        assert cfg["regex.enabled"] is True
        assert cfg["gliner.enabled"] is True
        assert cfg["gliner.threshold"] == 0.5
        assert cfg["gliner.max_scan_chars"] == 10_000
        assert cfg["llm.enabled"] is False
        # Severity rules keyed by RAW labels, per classify_severity lookup.
        assert cfg["severity_rules"]["social security number"] == "major"
        assert cfg["severity_rules"]["credit card number"] == "major"
        assert cfg["severity_rules"]["email"] == "minor"
        assert cfg["severity_rules"]["phone number"] == "minor"
        assert cfg["severity_rules"]["date of birth"] == "moderate"

    def test_severity_rules_shared_with_constants(self):
        # Deduplicated: the bridge sources SEVERITY_RULES_OVERRIDE from
        # constants, but hands out a mutation-safe copy.
        cfg = scanner_bridge.prod_default_config()
        assert cfg["severity_rules"] == constants.SEVERITY_RULES_OVERRIDE
        assert cfg["severity_rules"] is not constants.SEVERITY_RULES_OVERRIDE
