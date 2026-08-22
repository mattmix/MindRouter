############################################################
# test_websearch_dlp_gate.py: DLP screening of web-search queries
############################################################
"""Scan -> redact -> re-scan -> block, before a query reaches a provider.

The gate exists so user text cannot reach a third party unscreened, so the
tests care most about the paths where it must REFUSE: redaction that does not
clear, redaction that removes everything, and a scanner outage.
"""
import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
GATE_SRC = (_APP_DIR / "services" / "search" / "dlp_gate.py").read_text()
AUDIT_SRC = (_APP_DIR / "services" / "search" / "audit.py").read_text()
DLP_ROUTES_SRC = (_APP_DIR / "dashboard" / "dlp_routes.py").read_text()
DLP_HTML = (_APP_DIR / "dashboard" / "templates" / "admin" / "dlp.html").read_text()
WS_HTML = (_APP_DIR / "dashboard" / "templates" / "admin" / "_audit_web_search.html").read_text()
MIGRATIONS = _APP_DIR / "db" / "migrations" / "versions"

from backend.app.services.dlp_scanner import ScanFinding, ScanResult  # noqa: E402
from backend.app.services.search import dlp_gate as G  # noqa: E402

RULES = {"social security number": "major", "email": "minor", "person": "ignore"}


def _gate(enabled=True, min_severity="moderate", on_error="block"):
    async def _load(db):
        return {"enabled": enabled, "min_severity": min_severity,
                "on_scanner_error": on_error}
    return _load


def _scanner(sequence):
    """Feed run_dlp_scan a scripted sequence of results, one per pass."""
    calls = {"n": 0, "texts": []}

    async def _scan(text, cfg):
        calls["texts"].append(text)
        i = calls["n"]
        calls["n"] += 1
        item = sequence[i] if i < len(sequence) else sequence[-1]
        return item(text) if callable(item) else item

    return _scan, calls


def _patch(monkeypatch, gate_loader, scan):
    monkeypatch.setattr(G, "load_gate_config", gate_loader)
    import backend.app.services.dlp_scanner as DS
    monkeypatch.setattr(DS, "run_dlp_scan", scan)
    worker = types.ModuleType("backend.app.services.dlp_worker")

    async def _load_dlp_config(db):
        return {"severity_rules": RULES}

    worker._load_dlp_config = _load_dlp_config
    monkeypatch.setitem(sys.modules, "backend.app.services.dlp_worker", worker)


def _ssn_finding(value="123-45-6789"):
    return ScanFinding("regex", "social security number", value, 1.0, 0, len(value))


class TestDisabledAndClean:
    def test_disabled_does_not_scan(self, monkeypatch):
        scan, calls = _scanner([None])
        _patch(monkeypatch, _gate(enabled=False), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "my ssn is 123-45-6789"))
        assert out.allowed and out.scanned is False
        assert out.query == "my ssn is 123-45-6789"
        assert calls["n"] == 0, "an off toggle must cost nothing"
        assert out.audit_detail() is None

    def test_clean_query_passes_untouched(self, monkeypatch):
        scan, calls = _scanner([None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "weather in moscow idaho"))
        assert out.allowed and out.scanned and out.action == G.ACTION_NONE
        assert out.query == "weather in moscow idaho"
        assert calls["n"] == 1, "clean means one pass, not two"

    def test_finding_below_the_threshold_is_left_alone(self, monkeypatch):
        """An email is minor; at threshold=moderate the query goes as-is."""
        res = ScanResult(findings=[ScanFinding("regex", "email", "a@b.edu", 1.0, 0, 7)])
        scan, calls = _scanner([res])
        _patch(monkeypatch, _gate(min_severity="moderate"), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "email a@b.edu"))
        assert out.allowed and out.action == G.ACTION_NONE
        assert out.query == "email a@b.edu"
        assert out.severity == "minor"
        assert calls["n"] == 1

    def test_lowering_the_bar_catches_the_same_finding(self, monkeypatch):
        res = ScanResult(findings=[ScanFinding("regex", "email", "a@b.edu", 1.0, 0, 7)])
        scan, _ = _scanner([res, None])
        _patch(monkeypatch, _gate(min_severity="minor"), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "email a@b.edu"))
        assert out.allowed and out.action == G.ACTION_REDACTED
        assert "a@b.edu" not in out.query


class TestRedaction:
    def test_redacts_and_clears_on_the_second_pass(self, monkeypatch):
        scan, calls = _scanner([ScanResult(findings=[_ssn_finding()]), None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "my ssn is 123-45-6789 refund?"))
        assert out.allowed and out.action == G.ACTION_REDACTED
        assert "123-45-6789" not in out.query
        assert "[REDACTED: social security number]" in out.query
        assert calls["n"] == 2, "the redaction must be re-verified, not assumed"
        assert "123-45-6789" not in calls["texts"][1], "pass 2 scans the REDACTED text"
        assert out.severity == "major" and out.second_severity is None

    def test_masked_evidence_never_holds_the_raw_value(self, monkeypatch):
        scan, _ = _scanner([ScanResult(findings=[_ssn_finding()]), None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789"))
        detail = out.audit_detail()
        assert detail["action"] == "redacted"
        blob = str(detail)
        assert "123-45-6789" not in blob, "the audit row must not copy what it refused"
        assert detail["masked"][0]["text"].startswith("12") and "*" in detail["masked"][0]["text"]

    def test_every_non_ignored_finding_goes_once_triggered(self, monkeypatch):
        """A minor email rides along when a major SSN trips the threshold."""
        res = ScanResult(findings=[
            _ssn_finding(),
            ScanFinding("regex", "email", "a@b.edu", 1.0, 0, 7),
            ScanFinding("gliner", "person", "Ada", 0.9, 0, 3),   # Ignore rule
        ])
        scan, _ = _scanner([res, None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789 mail a@b.edu for Ada"))
        assert "123-45-6789" not in out.query
        assert "a@b.edu" not in out.query
        assert "Ada" in out.query, "an Ignored category is excluded from screening too"

    def test_longest_value_redacted_first(self, monkeypatch):
        """A short value that is a substring of a longer one must not corrupt it."""
        res = ScanResult(findings=[
            ScanFinding("regex", "email", "a@b.edu", 1.0, 0, 7),
            ScanFinding("regex", "social security number", "a@b.edu.long-secret", 1.0, 0, 19),
        ])
        scan, _ = _scanner([res, None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "x a@b.edu.long-secret y"))
        assert "long-secret" not in out.query


class TestBlocking:
    def test_second_pass_still_dirty_blocks(self, monkeypatch):
        dirty = ScanResult(findings=[_ssn_finding()])
        scan, calls = _scanner([dirty, dirty])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert out.second_severity == "major"
        assert "still contains sensitive data" in out.reason
        assert calls["n"] == 2

    def test_unredactable_phrase_blocks(self, monkeypatch):
        """GLiNER flagging a phrase rather than a value leaves nothing to remove."""
        res = ScanResult(findings=[
            ScanFinding("gliner", "social security number", "", 0.9, 0, 0)
        ])
        scan, calls = _scanner([res])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "how do I collect SSNs"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert "could not be redacted" in out.reason
        assert calls["n"] == 1, "no point re-scanning an unchanged query"

    def test_redacting_everything_blocks(self, monkeypatch):
        """Nothing searchable left is not a search worth sending."""
        res = ScanResult(findings=[_ssn_finding("123-45-6789")])
        scan, _ = _scanner([res, None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "123-45-6789"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert "Nothing searchable" in out.reason

    def test_blocked_screen_reports_itself_for_the_audit_row(self, monkeypatch):
        dirty = ScanResult(findings=[_ssn_finding()])
        scan, _ = _scanner([dirty, dirty])
        _patch(monkeypatch, _gate(), scan)
        detail = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789")).audit_detail()
        assert detail["action"] == "blocked"
        assert detail["severity"] == "major" and detail["second_severity"] == "major"
        assert detail["categories"] == ["social security number"]


class TestScannerOutage:
    _DEGRADED = ScanResult(findings=[], scanner_errors=["gliner: model load failed"])

    def test_fails_closed_by_default(self, monkeypatch):
        scan, calls = _scanner([self._DEGRADED])
        _patch(monkeypatch, _gate(on_error="block"), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "anything"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert out.degraded is True
        assert "unavailable" in out.reason
        assert calls["n"] == 1

    def test_allow_policy_sends_but_records_the_gap(self, monkeypatch):
        scan, _ = _scanner([self._DEGRADED])
        _patch(monkeypatch, _gate(on_error="allow"), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "anything"))
        assert out.allowed is True
        assert out.degraded is True, "the coverage gap must still be visible"

    def test_degradation_during_the_second_pass_blocks(self, monkeypatch):
        scan, _ = _scanner([ScanResult(findings=[_ssn_finding()]), self._DEGRADED])
        _patch(monkeypatch, _gate(on_error="block"), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert "while verifying the redaction" in out.reason

    def test_screening_bug_honours_the_posture(self, monkeypatch):
        """If the gate itself throws, block (default) rather than send blind."""
        async def _boom(text, cfg):
            raise RuntimeError("scanner import exploded")

        _patch(monkeypatch, _gate(on_error="block"), _boom)
        out = asyncio.run(G.screen_query(MagicMock(), "q"))
        assert out.allowed is False and out.degraded is True

    def test_screening_bug_with_allow_posture_sends(self, monkeypatch):
        async def _boom(text, cfg):
            raise RuntimeError("scanner import exploded")

        _patch(monkeypatch, _gate(on_error="allow"), _boom)
        out = asyncio.run(G.screen_query(MagicMock(), "q"))
        assert out.allowed is True


class TestGateConfig:
    def test_invalid_values_fall_back(self, monkeypatch):
        from backend.app.db import crud as real_crud

        vals = {"dlp.websearch.enabled": True,
                "dlp.websearch.min_severity": "catastrophic",
                "dlp.websearch.on_scanner_error": "shrug"}

        async def cfg(db, key, default=None):
            return vals.get(key, default)

        monkeypatch.setattr(real_crud, "get_config_json", cfg)
        out = asyncio.run(G.load_gate_config(MagicMock()))
        assert out["min_severity"] == "moderate" and out["on_scanner_error"] == "block"

    def test_unreadable_config_disables_rather_than_guesses(self, monkeypatch):
        from backend.app.db import crud as real_crud

        async def boom(db, key, default=None):
            raise RuntimeError("db down")

        monkeypatch.setattr(real_crud, "get_config_json", boom)
        out = asyncio.run(G.load_gate_config(MagicMock()))
        assert out["enabled"] is False

    def test_threshold_ordering(self):
        assert G._meets_threshold("major", "moderate") is True
        assert G._meets_threshold("moderate", "moderate") is True
        assert G._meets_threshold("minor", "moderate") is False
        assert G._meets_threshold(None, "minor") is False

    def test_blocked_error_is_a_valueerror(self):
        """So every existing call site's except ValueError already handles it."""
        assert issubclass(G.WebSearchBlockedError, ValueError)


class TestWiringAndSurfaces:
    def test_gate_runs_before_any_provider_is_contacted(self):
        fn = AUDIT_SRC[AUDIT_SRC.index("async def run_logged_search"):]
        assert fn.index("screen_query(db, query)") < fn.index("provider.search_exchange("), \
            "screening must precede the provider call, not follow it"
        assert "blocked=True" in fn
        assert "query = screen.query or query" in fn, "the redacted query is what gets sent"

    def test_blocked_search_is_still_audited(self):
        fn = AUDIT_SRC[AUDIT_SRC.index("async def run_logged_search"):]
        blocked = fn[fn.index("if not screen.allowed:"):fn.index("raise WebSearchBlockedError")]
        assert "record_search(" in blocked, "a blocked search is the most important row"
        assert '"websearch_blocked_by_dlp"' in blocked

    def test_row_status_covers_blocked(self):
        from backend.app.services.search.audit import _row_status

        assert _row_status(None, True) == "blocked"
        assert _row_status(ValueError(), False) == "error"
        assert _row_status(None, False) == "success"

    def test_call_sites_report_a_block_accurately(self):
        api = (_APP_DIR / "api" / "search_api.py").read_text()
        mcp = (_APP_DIR / "api" / "mcp_server.py").read_text()
        rsp = (_APP_DIR / "services" / "responses_websearch.py").read_text()
        assert "except WebSearchBlockedError" in api and "422" in api
        assert "except WebSearchBlockedError" in mcp
        assert "except WebSearchBlockedError" in rsp
        # the tool result must not invite a retry loop
        assert "Do not retry" in rsp

    def test_admin_controls_exist_and_are_validated(self):
        for name in ("websearch_dlp_enabled", "websearch_dlp_min_severity",
                     "websearch_dlp_on_error"):
            assert f'name="{name}"' in DLP_HTML, name
            assert name in DLP_ROUTES_SRC, name
        assert 'set_config(db, "dlp.websearch.enabled"' in DLP_ROUTES_SRC
        assert 'set_config(db, "dlp.websearch.min_severity"' in DLP_ROUTES_SRC
        assert 'set_config(db, "dlp.websearch.on_scanner_error"' in DLP_ROUTES_SRC
        # validated against the gate's own tuples, so the two cannot drift
        assert "VALID_MIN_SEVERITIES as WS_SEVERITIES" in DLP_ROUTES_SRC

    def test_audit_viewer_surfaces_the_outcome(self):
        assert 'name="dlp_action_filter"' in WS_HTML
        assert "shield-x" in WS_HTML and "shield-check" in WS_HTML
        assert "value=\"blocked\"" in WS_HTML
        audit = (_APP_DIR / "dashboard" / "templates" / "admin" / "audit.html").read_text()
        assert "DLP screening: " in audit, "the expand view must show the screening detail"

    def test_migration_082(self):
        mig = (MIGRATIONS / "20260822_000001_082_websearch_dlp_screening.py").read_text()
        assert 'revision = "082"' in mig and 'down_revision = "081"' in mig
        assert "dlp_action" in mig and "dlp_detail" in mig
        assert '"dlp.websearch.enabled": False' in mig, "opt-in, not on by deploy"
        assert '"dlp.websearch.on_scanner_error": "block"' in mig
        assert "op.drop_column" in mig
