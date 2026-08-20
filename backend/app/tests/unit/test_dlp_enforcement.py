############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_enforcement.py: Unit tests for inline DLP actions
#     (dlp_enforcement.py) — the inline gate, per-severity
#     block/redact decision, block-over-redact precedence,
#     redaction text substitution, fail-open behavior, and the
#     non-leaking 422 message.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for inline DLP actions (dlp_enforcement)."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ----------------------------------------------------------------
# Direct-load dlp_enforcement.py to avoid the DB / telemetry import chain.
# Its only hard dependency is structlog; crud / scanner / worker are imported
# lazily inside functions and are faked per-test via patch.dict(sys.modules).
# ----------------------------------------------------------------

_svc_dir = Path(__file__).resolve().parents[2] / "services"
_spec = importlib.util.spec_from_file_location(
    "dlp_enforcement_under_test", _svc_dir / "dlp_enforcement.py",
    submodule_search_locations=[],
)
enforce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enforce)


# ---- Fakes ------------------------------------------------------

class _Finding:
    def __init__(self, category, text="X"):
        self.category = category
        self.text = text


class _ScanResult:
    def __init__(self, findings, scanner_errors=None):
        self.findings = findings
        self.scanner_errors = scanner_errors or []


def _fake_modules(config, scan_result=None, scan_raises=False):
    """Build sys.modules fakes for the lazily-imported dependencies."""
    crud = types.ModuleType("backend.app.db.crud")

    async def get_config_json(db, key, default=None):
        return config.get(key, default)

    crud.get_config_json = get_config_json

    db_pkg = types.ModuleType("backend.app.db")
    db_pkg.crud = crud

    scanner = types.ModuleType("backend.app.services.dlp_scanner")
    scanner._SEVERITY_ORDER = {"minor": 0, "moderate": 1, "major": 2}

    async def run_dlp_scan(text, cfg):
        if scan_raises:
            raise RuntimeError("pool down")
        return scan_result

    scanner.run_dlp_scan = run_dlp_scan

    worker = types.ModuleType("backend.app.services.dlp_worker")

    async def _load_dlp_config(db):
        return {"severity_rules": config.get("__severity_rules__", {})}

    worker._load_dlp_config = _load_dlp_config

    return {
        "backend.app.db": db_pkg,
        "backend.app.db.crud": crud,
        "backend.app.services.dlp_scanner": scanner,
        "backend.app.services.dlp_worker": worker,
    }


def _gate(enabled=True, scope="prompt", block=(), redact=()):
    return {
        "enabled": enabled, "scope": scope,
        "block_severities": set(block), "redact_severities": set(redact),
    }


def _reset_gate():
    enforce.invalidate_gate_cache()


# ===================================================================
# Pure gate logic
# ===================================================================

class TestActiveGate:
    def test_inactive_when_master_off(self):
        assert enforce._active(_gate(enabled=False, block={"major"}), "prompt") is False

    def test_inactive_when_no_actions(self):
        assert enforce._active(_gate(scope="both"), "prompt") is False
        assert enforce._active(_gate(scope="both"), "response") is False

    def test_active_for_block_only(self):
        assert enforce._active(_gate(scope="prompt", block={"major"}), "prompt") is True

    def test_active_for_redact_only(self):
        # Redaction alone activates the inline path.
        assert enforce._active(_gate(scope="prompt", redact={"minor"}), "prompt") is True

    def test_prompt_scope_gates_sides(self):
        g = _gate(scope="prompt", block={"major"})
        assert enforce._active(g, "prompt") is True
        assert enforce._active(g, "response") is False

    def test_response_scope_gates_sides(self):
        g = _gate(scope="response", redact={"major"})
        assert enforce._active(g, "prompt") is False
        assert enforce._active(g, "response") is True

    def test_both_scope(self):
        g = _gate(scope="both", redact={"minor"})
        assert enforce._active(g, "prompt") is True
        assert enforce._active(g, "response") is True


# ===================================================================
# redact_text
# ===================================================================

class TestRedactText:
    def test_replaces_value_with_labeled_placeholder(self):
        out = enforce.redact_text("My SSN is 123-45-6789.", [("123-45-6789", "social security number")])
        assert "123-45-6789" not in out
        assert "[REDACTED: social security number]" in out

    def test_multiple_values(self):
        out = enforce.redact_text(
            "SSN 123-45-6789 card 4111111111111111",
            [("123-45-6789", "social security number"), ("4111111111111111", "credit card number")],
        )
        assert "123-45-6789" not in out and "4111111111111111" not in out

    def test_noops(self):
        assert enforce.redact_text(None, [("x", "y")]) is None
        assert enforce.redact_text("clean", []) == "clean"


# ===================================================================
# Client message never leaks the matched values
# ===================================================================

class TestClientMessage:
    def test_message_names_categories_not_values(self):
        err = enforce.DlpBlockedError(
            categories=["social security number"], severity="major", side="prompt"
        )
        msg = err.client_message()
        assert "social security number" in msg and "prompt" in msg
        assert "123-45-6789" not in msg


# ===================================================================
# evaluate_prompt_inline / evaluate_response_inline
# ===================================================================

@pytest.mark.asyncio
class TestEvaluateInline:
    async def test_returns_none_when_inline_off(self):
        _reset_gate()
        cfg = {"dlp.enabled": False}
        with patch.dict(sys.modules, _fake_modules(cfg, _ScanResult([_Finding("x", "v")]))):
            assert await enforce.evaluate_prompt_inline(None, "SSN 123-45-6789") is None

    async def test_blocks_when_severity_configured(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "__severity_rules__": {"social security number": "major"},
        }
        result = _ScanResult([_Finding("social security number", "123-45-6789")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            action = await enforce.evaluate_prompt_inline(None, "SSN 123-45-6789")
        assert action is not None
        assert isinstance(action.block, enforce.DlpBlockedError)
        assert action.block.severity == "major"
        assert action.block.categories == ["social security number"]
        assert action.redactions == []

    async def test_redacts_when_severity_configured(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.minor.redact": True,
            "__severity_rules__": {"email": "minor"},
        }
        result = _ScanResult([_Finding("email", "a@b.com")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            action = await enforce.evaluate_prompt_inline(None, "mail a@b.com")
        assert action is not None
        assert action.block is None
        assert action.redactions == [("a@b.com", "email")]

    async def test_block_takes_precedence_over_redact(self):
        _reset_gate()
        # SSN -> major (block); email -> minor (redact). Block must win.
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "dlp.action.minor.redact": True,
            "__severity_rules__": {"social security number": "major", "email": "minor"},
        }
        result = _ScanResult([
            _Finding("social security number", "123-45-6789"),
            _Finding("email", "a@b.com"),
        ])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            action = await enforce.evaluate_prompt_inline(None, "SSN 123-45-6789 a@b.com")
        assert isinstance(action.block, enforce.DlpBlockedError)
        assert action.redactions == []

    async def test_no_action_when_severity_unconfigured(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,  # only major blocks; finding is minor
            "__severity_rules__": {"email": "minor"},
        }
        result = _ScanResult([_Finding("email", "a@b.com")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            assert await enforce.evaluate_prompt_inline(None, "a@b.com") is None

    async def test_fail_open_on_scanner_exception(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "__severity_rules__": {"social security number": "major"},
        }
        with patch.dict(sys.modules, _fake_modules(cfg, None, scan_raises=True)):
            assert await enforce.evaluate_prompt_inline(None, "SSN 123-45-6789") is None

    async def test_fail_open_on_degraded_empty_result(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.major.redact": True,
            "__severity_rules__": {"social security number": "major"},
        }
        degraded = _ScanResult([], scanner_errors=["gliner: pool unreachable"])
        with patch.dict(sys.modules, _fake_modules(cfg, degraded)):
            assert await enforce.evaluate_prompt_inline(None, "SSN 123-45-6789") is None

    async def test_empty_text_never_acts(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
        }
        with patch.dict(sys.modules, _fake_modules(cfg, _ScanResult([_Finding("x", "v")]))):
            assert await enforce.evaluate_prompt_inline(None, "   ") is None

    async def test_response_scope_leaves_prompt_inactive(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True, "dlp.block.scope": "response",
            "dlp.action.major.redact": True,
            "__severity_rules__": {"social security number": "major"},
        }
        result = _ScanResult([_Finding("social security number", "123-45-6789")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            assert await enforce.evaluate_prompt_inline(None, "SSN 123-45-6789") is None
            action = await enforce.evaluate_response_inline(None, "SSN 123-45-6789")
        assert action is not None and action.redactions == [("123-45-6789", "social security number")]
