############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 072_add_voice_backend_engines.py: Add the 'tts' and 'stt'
#     backend engine types so voice services can be registered
#     as first-class backends and load-balanced by the scheduler
#     instead of being a single hardcoded URL in app_config.
#
# Mirrors 057 (diffusion engine) and 065 (video engine).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add tts and stt backend engine types.

Revision ID: 072
Revises: 071

Until now TTS and STT were reached through `app_config.voice.tts_url` /
`voice.stt_url` — a single URL each, with no health checking, no circuit
breaker, no failover and no per-container attribution, unlike every other
modality. Widening `backends.engine` lets a Kokoro or speaches container be
registered as an ordinary backend.

NOTE: `Modality` already contains 'tts' and 'stt' (present since the initial
schema), so neither `requests.modality` nor `models.modality` changes here.
This migration touches ONE small table.

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL):

  1. `backends` has tens of rows; a plain ALTER is fine and fast. We are
     APPENDING two values to the end of a 4-value ENUM, so the column stays
     1 byte and no row rewrite is needed.
  2. The downgrade NARROWS the enum, which is not an instant operation and
     will FAIL outright if any backend row still uses 'tts' or 'stt'. Delete
     or re-engine those backends first — see downgrade() for the query.
"""

from alembic import op

revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None

# Spelled out in full — never rely on the ORM enum at migration time, since
# the code may have moved on by the time an old database is upgraded.
OLD_ENGINE = "'ollama','vllm','diffusion','video'"
NEW_ENGINE = "'ollama','vllm','diffusion','video','tts','stt'"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({NEW_ENGINE}) NOT NULL"
    )


def downgrade() -> None:
    # Narrowing fails while any row still uses a removed value. Check first:
    #   SELECT id, name, engine FROM backends WHERE engine IN ('tts','stt');
    # and delete or re-engine those rows before running this.
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({OLD_ENGINE}) NOT NULL"
    )
