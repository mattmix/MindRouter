"""Contract tests for the voice-backend foundation (migration 072).

Source inspection only — no db/telemetry imports at module level (see MEMORY
"Import Chain Gotcha"). Mirrors test_video_schema_contract.py, which pins the
same properties for the video engine added by 065.

Guards the schema half of routing TTS/STT through the registry: the enum
widening, the ORM members, and the discovery mapping. Without these, a
registered Kokoro backend's models fall through to the CHAT default and a
downgrade silently drops enum values.
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MIGRATION_072 = (
    "backend/app/db/migrations/versions/"
    "20260808_000000_072_add_voice_backend_engines.py"
)


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


class TestMigration072:
    def test_revision_chain(self):
        src = _read(MIGRATION_072)
        assert re.search(r'^revision = "072"', src, re.M)
        assert re.search(r'^down_revision = "071"', src, re.M)

    def test_is_the_only_072(self):
        """A duplicate revision id makes alembic pick one arbitrarily."""
        d = os.path.join(ROOT, "backend/app/db/migrations/versions")
        hits = [
            f for f in os.listdir(d)
            if f.endswith(".py") and re.search(r'^revision = "072"', _read(
                f"backend/app/db/migrations/versions/{f}"), re.M)
        ]
        assert len(hits) == 1, f"multiple migrations claim revision 072: {hits}"

    def test_enum_lists_spelled_out_and_append_only(self):
        """The ALTER replaces the whole value list, so the OLD list must match
        what 065 left behind exactly — a missing value would be DROPPED, and
        MariaDB maps ENUM by position, silently rewriting existing rows."""
        src = _read(MIGRATION_072)
        old = re.search(r'OLD_ENGINE = "([^"]+)"', src).group(1)
        new = re.search(r'NEW_ENGINE = "([^"]+)"', src).group(1)

        assert old == "'ollama','vllm','diffusion','video'"
        assert new == "'ollama','vllm','diffusion','video','tts','stt'"
        # append-only: OLD must be a prefix of NEW so existing rows keep their
        # ordinal positions
        assert new.startswith(old), "enum change must be append-only"

    def test_old_list_matches_what_065_left(self):
        """Cross-check against the previous migration rather than trusting the
        literal in isolation."""
        prev = _read(
            "backend/app/db/migrations/versions/"
            "20260722_000000_065_add_video_modality_and_engine.py"
        )
        prev_new = re.search(r'NEW_ENGINE = "([^"]+)"', prev).group(1)
        cur_old = re.search(r'OLD_ENGINE = "([^"]+)"', _read(MIGRATION_072)).group(1)
        assert cur_old == prev_new, (
            "072's OLD_ENGINE must equal 065's NEW_ENGINE, or the ALTER drops values"
        )

    def test_only_backends_table_is_altered(self):
        """Modality already had TTS/STT, so requests/models must NOT be touched
        — requests is the largest table in prod."""
        src = _read(MIGRATION_072)
        assert "ALTER TABLE backends" in src
        assert "ALTER TABLE requests" not in src
        assert "ALTER TABLE models" not in src

    def test_downgrade_documents_the_narrowing_hazard(self):
        src = _read(MIGRATION_072)
        i = src.index("def downgrade")
        assert "engine IN ('tts','stt')" in src[i:], (
            "downgrade must tell the operator how to find rows that block the narrowing"
        )


class TestBackendEngineMembers:
    def test_orm_has_tts_and_stt(self):
        src = _read("backend/app/db/models.py")
        i = src.index("class BackendEngine")
        block = src[i:i + 800]
        assert 'TTS = "tts"' in block
        assert 'STT = "stt"' in block

    def test_modality_already_had_tts_stt(self):
        """If these ever move, migration 072's 'no modality change' premise breaks."""
        src = _read("backend/app/db/models.py")
        i = src.index("class Modality")
        block = src[i:i + 800]
        assert 'TTS = "tts"' in block
        assert 'STT = "stt"' in block


class TestDiscoveryModalityMapping:
    def test_registry_maps_voice_engines(self):
        """Without this branch a Kokoro backend's models (kokoro, tts-1,
        tts-1-hd) are stored as Modality.CHAT and become selectable chat
        models."""
        src = _read("backend/app/core/telemetry/registry.py")
        assert "BackendEngine.TTS" in src and "Modality.TTS" in src
        assert "BackendEngine.STT" in src and "Modality.STT" in src

    def test_mapping_precedes_the_name_heuristics(self):
        """Engine-based mapping must win over the 'embed'/'rerank' substring
        checks — a whisper model id containing neither must still be STT."""
        src = _read("backend/app/core/telemetry/registry.py")
        i = src.index("BackendEngine.STT")
        j = src.index('"embed" in model_info.name.lower()')
        assert i < j, "engine checks must come before the name heuristics"
