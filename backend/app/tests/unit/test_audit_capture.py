"""Tests for the audit content-capture toggles (2.9.5).

audit_log_enabled / audit_log_prompts / audit_log_responses were dead
settings; they now gate prompt/response content persistence in the
inference audit path.  Source-contract tests (inference.py cannot be
imported standalone — see MEMORY.md import-chain gotcha).
"""

import re
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[2]
INFERENCE_SRC = (_APP_DIR / "services" / "inference.py").read_text()
SETTINGS_SRC = (_APP_DIR / "settings.py").read_text()
MAIN_SRC = (_APP_DIR / "main.py").read_text()


class TestPromptCapture:
    def test_prompt_capture_gated(self):
        """messages/prompt extraction (and image offload) must be inside
        the capture_prompts gate."""
        assert "capture_prompts = (" in INFERENCE_SRC
        gate = INFERENCE_SRC.split("capture_prompts = (", 1)[1]
        head = gate[:2000]
        assert "audit_log_enabled" in head
        assert "audit_log_prompts" in head
        assert "if capture_prompts:" in head
        # image offload must be inside the gate (no files written when off)
        block = head.split("if capture_prompts:", 1)[1].split("# Extract parameters")[0]
        assert "_extract_images_from_messages" in block

    def test_ungated_extraction_removed(self):
        """The old unconditional extraction must be gone."""
        assert not re.search(
            r"^        if hasattr\(request, \"messages\"\):",
            INFERENCE_SRC, re.M,
        ), "messages extraction is no longer allowed outside the capture gate"


class TestResponseCapture:
    def test_both_create_response_sites_gated(self):
        """Streaming and non-streaming completion writers must null
        content when response capture is off."""
        sites = INFERENCE_SRC.split("await crud.create_response(")
        assert len(sites) == 3, "expected exactly two create_response call sites"
        # Check the text immediately preceding each call site
        for i, before in enumerate(sites[:-1], start=1):
            tail = before[-900:]
            assert "audit_log_responses" in tail, (
                f"create_response call site {i} lacks the capture gate"
            )
            assert "content = None" in tail


class TestSettingsAndStartup:
    def test_settings_document_semantics(self):
        assert "audit_log_enabled: bool = True" in SETTINGS_SRC
        # The comment must state the DLP interaction
        block = SETTINGS_SRC.split("# Audit Logging")[1][:800]
        assert "DLP" in block

    def test_dead_settings_removed(self):
        for dead in (
            "conversation_retention_days",
            "conversation_cleanup_interval",
            "artifact_retention_days",
            "artifact_max_size_mb",
        ):
            assert dead not in SETTINGS_SRC, f"dead setting {dead} still present"

    def test_startup_warns_when_capture_disabled(self):
        assert "audit_capture_disabled" in MAIN_SRC

    def test_compose_passes_capture_toggles_through(self):
        """Env vars the app reads MUST be listed in docker-compose.yml —
        pydantic reads env_file only inside the container (MEMORY.md)."""
        compose = (
            Path(__file__).resolve().parents[4] / "docker-compose.yml"
        ).read_text()
        for var in ("AUDIT_LOG_ENABLED", "AUDIT_LOG_PROMPTS", "AUDIT_LOG_RESPONSES"):
            assert f"{var}=${{{var}:-true}}" in compose, f"{var} not passed through"


class TestRetentionDefaultsAligned:
    def test_code_fallbacks_match_migration_seeds(self):
        """_DEFAULTS must equal the migration-029 seeded values (the
        three-way docs/seed/code disagreement was a confirmed doc bug)."""
        retention_src = (_APP_DIR / "services" / "retention.py").read_text()
        migration_src = next(
            (_APP_DIR / "db" / "migrations" / "versions").glob("*029_seed_retention*")
        ).read_text()
        seeds = dict(
            re.findall(r'\("(retention\.[a-z_.0-9]+)", (\d+),', migration_src)
        )
        defaults_block = retention_src.split("_DEFAULTS: dict[str, Any] = {")[1].split("}")[0]
        code = dict(re.findall(r'"(retention\.[a-z_.0-9]+)": (\d+),', defaults_block))
        for key, seeded in seeds.items():
            assert code.get(key) == seeded, (
                f"{key}: code fallback {code.get(key)} != migration seed {seeded}"
            )


class TestDocsTruthful:
    def test_no_dead_retention_vars_in_docs(self):
        for doc in (
            Path(__file__).resolve().parents[4] / "docs" / "index.md",
            _APP_DIR / "dashboard" / "templates" / "public" / "documentation.html",
        ):
            text = doc.read_text()
            hits = [
                line for line in text.splitlines()
                if "CONVERSATION_RETENTION_DAYS" in line
                or "CONVERSATION_CLEANUP_INTERVAL" in line
            ]
            assert not hits, f"dead retention vars still documented in {doc.name}: {hits[:2]}"

    def test_removed_artifact_vars_not_presented_as_live(self):
        """ARTIFACT_MAX_SIZE_MB / ARTIFACT_RETENTION_DAYS may only appear
        in prose explaining that they were REMOVED — never as live config."""
        for doc in (
            Path(__file__).resolve().parents[4] / "docs" / "index.md",
            _APP_DIR / "dashboard" / "templates" / "public" / "documentation.html",
        ):
            for line in doc.read_text().splitlines():
                if "ARTIFACT_MAX_SIZE_MB" in line or "ARTIFACT_RETENTION_DAYS" in line:
                    assert "removed" in line.lower(), (
                        f"{doc.name} still documents a removed artifact var as live: {line.strip()[:120]}"
                    )

    def test_new_admin_endpoints_documented(self):
        for doc in (
            Path(__file__).resolve().parents[4] / "docs" / "index.md",
            _APP_DIR / "dashboard" / "templates" / "public" / "documentation.html",
        ):
            text = doc.read_text()
            assert "reset-password" in text, f"{doc.name} missing reset-password"
            assert "api-keys/{id}/revoke" in text, f"{doc.name} missing admin key revoke"

    def test_capture_toggles_documented_with_dlp_note(self):
        for doc in (
            Path(__file__).resolve().parents[4] / "docs" / "index.md",
            _APP_DIR / "dashboard" / "templates" / "public" / "documentation.html",
        ):
            text = doc.read_text()
            assert "AUDIT_LOG_PROMPTS" in text, f"{doc.name} missing capture toggles"
            idx = text.index("AUDIT_LOG_ENABLED")
            assert "DLP" in text[idx - 1500:idx + 1500], (
                f"{doc.name}: DLP interaction not documented near the toggles"
            )
