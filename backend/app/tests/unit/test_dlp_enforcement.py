############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_enforcement.py: Unit tests for inline DLP blocking
#     (dlp_enforcement.py) — the block gate, per-severity block
#     decision, fail-open behavior, and the non-leaking 422 message.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for inline DLP blocking (dlp_enforcement)."""

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
    def __init__(self, category):
        self.category = category


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


def _reset_gate():
    enforce.invalidate_gate_cache()


# ===================================================================
# Pure gate logic
# ===================================================================

class TestActiveGate:
    def test_inactive_when_master_off(self):
        gate = {"enabled": False, "scope": "prompt", "block_severities": {"major"}}
        assert enforce._active(gate, "prompt") is False

    def test_inactive_when_no_block_severities(self):
        gate = {"enabled": True, "scope": "both", "block_severities": set()}
        assert enforce._active(gate, "prompt") is False
        assert enforce._active(gate, "response") is False

    def test_prompt_scope_gates_sides(self):
        gate = {"enabled": True, "scope": "prompt", "block_severities": {"major"}}
        assert enforce._active(gate, "prompt") is True
        assert enforce._active(gate, "response") is False

    def test_response_scope_gates_sides(self):
        gate = {"enabled": True, "scope": "response", "block_severities": {"major"}}
        assert enforce._active(gate, "prompt") is False
        assert enforce._active(gate, "response") is True

    def test_both_scope(self):
        gate = {"enabled": True, "scope": "both", "block_severities": {"minor"}}
        assert enforce._active(gate, "prompt") is True
        assert enforce._active(gate, "response") is True


# ===================================================================
# Client message never leaks the matched values
# ===================================================================

class TestClientMessage:
    def test_message_names_categories_not_values(self):
        err = enforce.DlpBlockedError(
            categories=["social security number"], severity="major", side="prompt"
        )
        msg = err.client_message()
        assert "social security number" in msg
        assert "prompt" in msg
        # The raw value must never appear.
        assert "123-45-6789" not in msg


# ===================================================================
# evaluate_prompt_block — gate, decision, fail-open
# ===================================================================

@pytest.mark.asyncio
class TestEvaluatePromptBlock:
    async def test_returns_none_when_blocking_off(self):
        _reset_gate()
        cfg = {"dlp.enabled": False}
        with patch.dict(sys.modules, _fake_modules(cfg, _ScanResult([_Finding("x")]))):
            assert await enforce.evaluate_prompt_block(None, "SSN 123-45-6789") is None

    async def test_blocks_when_severity_configured(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True,
            "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "dlp.action.moderate.block": False,
            "dlp.action.minor.block": False,
            "__severity_rules__": {"social security number": "major"},
        }
        result = _ScanResult([_Finding("social security number")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            err = await enforce.evaluate_prompt_block(None, "SSN 123-45-6789")
        assert isinstance(err, enforce.DlpBlockedError)
        assert err.severity == "major"
        assert err.categories == ["social security number"]

    async def test_no_block_when_severity_not_configured(self):
        _reset_gate()
        # 'email' maps to minor, but only major blocks -> allow.
        cfg = {
            "dlp.enabled": True,
            "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "dlp.action.moderate.block": False,
            "dlp.action.minor.block": False,
            "__severity_rules__": {"email": "minor"},
        }
        result = _ScanResult([_Finding("email")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            assert await enforce.evaluate_prompt_block(None, "a@b.com") is None

    async def test_fail_open_on_scanner_exception(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True,
            "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "__severity_rules__": {"social security number": "major"},
        }
        with patch.dict(sys.modules, _fake_modules(cfg, None, scan_raises=True)):
            # Scanner outage -> fail open (allow), not raise.
            assert await enforce.evaluate_prompt_block(None, "SSN 123-45-6789") is None

    async def test_fail_open_on_degraded_empty_result(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True,
            "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
            "__severity_rules__": {"social security number": "major"},
        }
        degraded = _ScanResult([], scanner_errors=["gliner: pool unreachable"])
        with patch.dict(sys.modules, _fake_modules(cfg, degraded)):
            assert await enforce.evaluate_prompt_block(None, "SSN 123-45-6789") is None

    async def test_empty_text_never_blocks(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True,
            "dlp.block.scope": "prompt",
            "dlp.action.major.block": True,
        }
        with patch.dict(sys.modules, _fake_modules(cfg, _ScanResult([_Finding("x")]))):
            assert await enforce.evaluate_prompt_block(None, "   ") is None

    async def test_response_scope_does_not_block_prompt(self):
        _reset_gate()
        cfg = {
            "dlp.enabled": True,
            "dlp.block.scope": "response",
            "dlp.action.major.block": True,
            "__severity_rules__": {"social security number": "major"},
        }
        result = _ScanResult([_Finding("social security number")])
        with patch.dict(sys.modules, _fake_modules(cfg, result)):
            # scope=response -> prompt path is inactive.
            assert await enforce.evaluate_prompt_block(None, "SSN 123-45-6789") is None
            # ...but the response path blocks.
            err = await enforce.evaluate_response_block(None, "SSN 123-45-6789")
        assert isinstance(err, enforce.DlpBlockedError)
