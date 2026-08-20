############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 077_add_dlp_backend_engine.py: Add the 'dlp' backend engine
#     type so a DLP (GLiNER) scan service can be registered as a
#     first-class backend and be a fleet member for status +
#     GPU/power telemetry (via the per-node sidecar) — while
#     serving NO models, so it is never eligible for inference
#     routing and never appears in the model catalog.
#
# Mirrors 065 (video engine) and 072 (tts/stt engines).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add dlp backend engine type.

Revision ID: 077
Revises: 076

A node can now host a vLLM backend on one GPU and a DLP (GLiNER) backend on
another, each with its own engine / gpu_indices / health / telemetry. The DLP
backend's adapter probes GET /healthz (no auth) and discovers ZERO models, so
crud.get_backends_with_model (which requires a `models` row) never returns it
and it is absent from /v1/models — it exists purely so the fleet can report its
health and, through the per-node sidecar, its GPU/power draw.

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL — a failure mid-migration
leaves partial state that must be cleaned up by hand):

  1. `backends` has tens of rows; we are APPENDING one value to the end of a
     6-value ENUM, so the column stays 1 byte and no row rewrite is needed. We
     request ALGORITHM=INSTANT, LOCK=NONE explicitly rather than letting the
     server choose. FALLBACK: if the server rejects INSTANT on this version,
     re-run the single failing statement without the ALGORITHM/LOCK clause (it
     will pick INPLACE).
  2. The downgrade NARROWS the enum, which is not an instant operation. It will
     also silently corrupt any surviving row, so downgrade() REFUSES to run
     while any backend is still engine='dlp' — delete or re-engine those rows
     first.
"""

from alembic import op

revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None

# Spelled out in full — never rely on the ORM enum at migration time, since
# the code may have moved on by the time an old database is upgraded.
OLD_ENGINE = "'ollama','vllm','diffusion','video','tts','stt'"
NEW_ENGINE = "'ollama','vllm','diffusion','video','tts','stt','dlp'"


def upgrade() -> None:
    # backends.engine — append 'dlp' (small table; instant 1-byte append)
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({NEW_ENGINE}) NOT NULL, ALGORITHM=INSTANT, LOCK=NONE"
    )


def downgrade() -> None:
    # Refuse to narrow the enum while live data still uses 'dlp'; MariaDB would
    # otherwise coerce those rows to '' and corrupt the fleet registry.
    bind = op.get_bind()
    remaining = bind.exec_driver_sql(
        "SELECT COUNT(*) FROM backends WHERE engine = 'dlp'"
    ).scalar()
    if remaining:
        raise RuntimeError(
            f"Cannot downgrade: {remaining} backend row(s) still use "
            f"engine='dlp'. Delete or re-engine them first "
            f"(SELECT id, name FROM backends WHERE engine='dlp';)."
        )

    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({OLD_ENGINE}) NOT NULL"
    )
