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
        assert "***********" in out.query, "the value is masked with asterisks"
        assert calls["n"] == 2, "the redaction must be re-verified, not assumed"
        assert "123-45-6789" not in calls["texts"][1], "pass 2 scans the REDACTED text"
        assert out.severity == "major" and out.second_severity is None

    def test_mask_is_asterisks_not_a_word_shaped_placeholder(self, monkeypatch):
        """A placeholder TOKEN re-primes the scanner; asterisks cannot.

        Measured against the production scanners: "[REDACTED: social security
        number]" is itself flagged as a social security number, and even a bare
        "[REDACTED]" leaves a word for GLiNER to bind the user's label word to.
        Asterisks carry no semantics, so nothing survives to re-flag.
        """
        scan, calls = _scanner([ScanResult(findings=[_ssn_finding()]), None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789 financial aid"))
        assert "REDACTED" not in out.query
        assert "social security number" not in out.query
        # and the text handed to the verifying pass is the masked one
        assert "REDACTED" not in calls["texts"][1]
        assert "123-45-6789" not in calls["texts"][1]

    def test_mask_span_preserves_whitespace_shape(self):
        assert G.mask_span("ssn 456-78-9012") == "*** ***********"
        assert G.mask_span("") == ""

    def test_has_content_ignores_masks(self):
        assert G._has_content("*** ***") is False
        assert G._has_content("*** *** for records") is True

    def test_mask_uses_offsets_for_multi_word_spans(self):
        """Masking a span like "ssn 456-78-9012" needs offsets, not a value swap."""
        text = "contact 208-885-6111 or ssn 456-78-9012 for records"
        start = text.index("ssn 456-78-9012")
        f = ScanFinding("gliner", "social security number", "ssn 456-78-9012",
                        0.9, start, start + len("ssn 456-78-9012"))
        assert G._mask_findings(text, [f], {}) == (
            "contact 208-885-6111 or *** *********** for records"
        )

    def test_mask_falls_back_to_value_when_offsets_are_unusable(self):
        """The LLM scanner reports no offsets; the finding must still be masked."""
        text = "ssn 456-78-9012 here"
        f = ScanFinding("llm", "social security number", "456-78-9012", 0.9, 0, 0)
        assert G._mask_findings(text, [f], {}) == "ssn *********** here"

    def test_evidence_is_masked_even_though_the_trail_keeps_the_original(self, monkeypatch):
        """Two different things, deliberately.

        The provenance trail keeps the caller's ORIGINAL query (that is the
        point of the before/after audit, and it is governed by
        search.audit.store_original_query). The EVIDENCE — the per-finding
        snippets — is always masked, so a reader scanning the findings never
        sees a raw value they did not ask to see.
        """
        scan, _ = _scanner([ScanResult(findings=[_ssn_finding()]), None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789 aid"))
        detail = out.audit_detail()
        assert detail["action"] == "redacted"

        # evidence: masked
        assert "123-45-6789" not in str(detail["masked"])
        assert detail["masked"][0]["text"].startswith("12") and "*" in detail["masked"][0]["text"]
        assert "123-45-6789" not in str([p["findings"] for p in detail["passes"]])

        # provenance: the original is kept, by design, in pass 1's text only
        assert detail["passes"][0]["text"] == "ssn 123-45-6789 aid"
        assert "123-45-6789" not in (detail["passes"][1]["text"] or "")
        assert "123-45-6789" not in out.query, "the outbound query is masked"

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


class TestMaskRounds:
    """Masking a value can expose the user's own label word as a NEW span, so
    the gate re-masks and re-verifies within a bounded number of rounds."""

    def test_second_round_clears_a_newly_exposed_span(self, monkeypatch):
        text = "contact 208-885-6111 or ssn 456-78-9012 for records"
        ssn_at = text.index("456-78-9012")

        def pass1(t):
            return ScanResult(findings=[ScanFinding(
                "regex", "social security number", "456-78-9012", 1.0,
                ssn_at, ssn_at + 11)])

        def pass2(t):
            # GLiNER now binds the label word to the mask beside it.
            span = "ssn " + "*" * 11
            i = t.index(span)
            return ScanResult(findings=[ScanFinding(
                "gliner", "social security number", span, 0.9, i, i + len(span))])

        scan, calls = _scanner([pass1, pass2, None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), text))
        assert out.allowed and out.action == G.ACTION_REDACTED
        assert out.query == "contact 208-885-6111 or *** *********** for records"
        assert out.rounds == 2
        assert calls["n"] == 3, "one scan, then a scan after each masking round"

    def test_no_progress_blocks_instead_of_looping(self, monkeypatch):
        """A scanner flagging an already-masked run cannot be masked further."""
        def flag_mask(t):
            run = "*" * 11
            if run not in t:
                i = t.index("456-78-9012")
                return ScanResult(findings=[ScanFinding(
                    "regex", "social security number", "456-78-9012", 1.0, i, i + 11)])
            i = t.index(run)
            return ScanResult(findings=[ScanFinding(
                "gliner", "social security number", run, 0.9, i, i + 11)])

        scan, calls = _scanner([flag_mask])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "my ssn is 456-78-9012 please"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert "cannot be masked further" in out.reason
        assert calls["n"] <= G.MAX_REDACTION_ROUNDS + 1, "the loop must terminate"

    def test_rounds_are_bounded(self, monkeypatch):
        """A scanner that flags something new every round still terminates."""
        state = {"i": 0}

        def always_new(t):
            state["i"] += 1
            # Flag a different real word each round so masking always progresses.
            words = ["alpha", "bravo", "charlie", "delta", "echo"]
            w = words[min(state["i"] - 1, len(words) - 1)]
            if w not in t:
                return ScanResult(findings=[])
            i = t.index(w)
            return ScanResult(findings=[ScanFinding(
                "gliner", "social security number", w, 0.9, i, i + len(w))])

        scan, calls = _scanner([always_new])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "alpha bravo charlie delta echo"))
        assert out.allowed is False and out.action == G.ACTION_BLOCKED
        assert out.rounds == G.MAX_REDACTION_ROUNDS
        assert calls["n"] == G.MAX_REDACTION_ROUNDS + 1

    def test_rounds_recorded_on_the_audit_detail(self, monkeypatch):
        scan, _ = _scanner([ScanResult(findings=[_ssn_finding()]), None])
        _patch(monkeypatch, _gate(), scan)
        detail = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789 aid")).audit_detail()
        assert detail["rounds"] == 1


class TestProvenance:
    """The audit trail must answer: what was submitted, what was sent, and
    what every DLP pass decided."""

    def test_each_pass_is_recorded_with_its_text_and_verdict(self, monkeypatch):
        text = "contact 208-885-6111 or ssn 456-78-9012 for records"
        ssn_at = text.index("456-78-9012")

        def p1(t):
            return ScanResult(findings=[ScanFinding(
                "regex", "social security number", "456-78-9012", 1.0, ssn_at, ssn_at + 11)])

        def p2(t):
            span = "ssn " + "*" * 11
            i = t.index(span)
            return ScanResult(findings=[ScanFinding(
                "gliner", "social security number", span, 0.9, i, i + len(span))])

        scan, _ = _scanner([p1, p2, None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), text))

        passes = out.audit_detail()["passes"]
        assert [p["pass"] for p in passes] == [1, 2, 3]
        assert [p["verdict"] for p in passes] == ["fail", "fail", "pass"]
        assert passes[0]["text"] == text, "pass 1 records the caller's own query"
        assert passes[1]["text"] == "contact 208-885-6111 or ssn *********** for records"
        assert passes[2]["text"] == out.query, "the last pass records what was sent"
        assert passes[0]["severity"] == "major" and passes[2]["severity"] is None
        # findings are masked evidence, never the raw value
        assert "456-78-9012" not in str(passes[0]["findings"])
        assert out.original_query == text

    def test_blocked_trail_shows_how_far_masking_got(self, monkeypatch):
        def flag(t):
            run = "*" * 11
            if run in t:
                i = t.index(run)
                return ScanResult(findings=[ScanFinding(
                    "gliner", "social security number", run, 0.9, i, i + 11)])
            i = t.index("456-78-9012")
            return ScanResult(findings=[ScanFinding(
                "regex", "social security number", "456-78-9012", 1.0, i, i + 11)])

        scan, _ = _scanner([flag])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "my ssn is 456-78-9012 please"))
        assert out.allowed is False
        passes = out.audit_detail()["passes"]
        assert [p["verdict"] for p in passes] == ["fail", "fail"]
        # outbound_text is the furthest-masked form, NOT the original
        assert out.outbound_text() == "my ssn is *********** please"
        assert out.original_query == "my ssn is 456-78-9012 please"

    def test_clean_query_records_a_single_passing_pass(self, monkeypatch):
        scan, _ = _scanner([None])
        _patch(monkeypatch, _gate(), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "library hours"))
        passes = out.audit_detail()["passes"]
        assert len(passes) == 1 and passes[0]["verdict"] == "pass"
        assert passes[0]["text"] == "library hours"

    def test_original_can_be_withheld_by_policy(self, monkeypatch):
        """store_original off keeps the verdicts but not the caller's text."""
        async def _load(db):
            return {"enabled": True, "min_severity": "moderate",
                    "on_scanner_error": "block", "store_original": False}

        scan, _ = _scanner([ScanResult(findings=[_ssn_finding()]), None])
        _patch(monkeypatch, _load, scan)
        out = asyncio.run(G.screen_query(MagicMock(), "ssn 123-45-6789 aid"))
        passes = out.audit_detail()["passes"]
        assert passes[0]["text"] is None and passes[0]["text_stored"] is False
        assert passes[0]["text_chars"] == len("ssn 123-45-6789 aid")
        assert passes[0]["verdict"] == "fail", "the verdict is still recorded"
        # later passes scan masked text, so they are always kept
        assert passes[1]["text_stored"] is True

    def test_degraded_pass_is_flagged_in_the_trail(self, monkeypatch):
        degraded = ScanResult(findings=[], scanner_errors=["gliner: down"])
        scan, _ = _scanner([degraded])
        _patch(monkeypatch, _gate(on_error="allow"), scan)
        out = asyncio.run(G.screen_query(MagicMock(), "q"))
        p = out.audit_detail()["passes"][0]
        assert p["degraded"] is True and p["scanner_errors"] == ["gliner: down"]

    def test_audit_row_carries_both_texts(self):
        """record_search stores the original only when screening changed it."""
        assert "query_original=original_raw" in AUDIT_SRC
        assert 'candidate != (query or "")' in AUDIT_SRC, "no pointless duplicate"
        assert 'first.get("text_stored", True)' in AUDIT_SRC, "honours the policy"
        # a blocked row records the furthest-masked form, not the original
        assert "query=screen.outbound_text()" in AUDIT_SRC

    def test_viewer_shows_before_after_and_the_pass_table(self):
        audit = (_APP_DIR / "dashboard" / "templates" / "admin" / "audit.html").read_text()
        assert "wsDlpProvenance" in audit
        assert "Query submitted by caller" in audit
        assert "Query sent to provider" in audit
        assert "Furthest-masked form (not sent)" in audit
        assert "nothing was sent to the provider" in audit.lower()
        for col in ("Pass", "Verdict", "Severity", "Categories", "Text scanned"):
            assert f"<th>{col}</th>" in audit, col
        table = (_APP_DIR / "dashboard" / "templates" / "admin" / "_audit_web_search.html").read_text()
        assert "log.query_original" in table, "the row hints at the before-text"

    def test_export_includes_both_texts(self):
        routes = (_APP_DIR / "dashboard" / "routes.py").read_text()
        assert '"query_original": row.query_original' in routes
        assert '"query", "query_original"' in routes

    def test_migration_083(self):
        mig = (MIGRATIONS / "20260822_000002_083_websearch_query_provenance.py").read_text()
        assert 'revision = "083"' in mig and 'down_revision = "082"' in mig
        assert "query_original" in mig
        assert "search.audit.store_original_query" in mig
        assert "op.drop_column" in mig


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
        assert "wsDlpProvenance(data)" in audit, "the expand view must show the screening detail"

    def test_migration_082(self):
        mig = (MIGRATIONS / "20260822_000001_082_websearch_dlp_screening.py").read_text()
        assert 'revision = "082"' in mig and 'down_revision = "081"' in mig
        assert "dlp_action" in mig and "dlp_detail" in mig
        assert '"dlp.websearch.enabled": False' in mig, "opt-in, not on by deploy"
        assert '"dlp.websearch.on_scanner_error": "block"' in mig
        assert "op.drop_column" in mig
