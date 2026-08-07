############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 071_dlp_key_removal_and_index.py: Retire the DLP internal
#     API key (plaintext at rest) and index dlp_alerts by
#     scan time for the new retention sweep
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Remove DLP internal API key config; add dlp_alerts.scanned_at index

Revision ID: 071
Revises: 070
Create Date: 2026-08-07 00:00:00.000000

Before 2.9.9 the DLP LLM scanner authenticated to MindRouter's own /v1
endpoint with an auto-minted API key whose RAW value was stored in
app_config under ``dlp.internal_api_key_raw`` — the only unhashed key in
a system that otherwise persists Argon2 + SHA-256 only.  The key was
owned by whichever row ``SELECT id FROM users LIMIT 1`` returned (in
practice the bootstrap admin), so it carried admin rights, and it never
expired.

The scanner now dispatches straight to a backend and holds no
credential.  This migration revokes any key that was minted and drops
both config rows.  Revoke before delete: dropping the id row first would
orphan a live, never-expiring, admin-capable key with no pointer left to
find it by.

Installs that never enabled the LLM scanner have nothing to clean up —
``dlp.internal_api_key_raw`` is only written at runtime.

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "071"
down_revision = "070"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Revoke the minted key, if any.  Matched by the recorded id first; the
    # name is the fallback for an install whose config row was lost.
    conn.execute(
        sa.text(
            "UPDATE api_keys SET status = 'revoked' WHERE id IN ("
            "  SELECT * FROM ("
            "    SELECT CAST(JSON_UNQUOTE(value) AS UNSIGNED) FROM app_config"
            "     WHERE `key` = 'dlp.internal_api_key_id'"
            "       AND value IS NOT NULL AND value <> 'null'"
            "  ) AS k"
            ")"
        )
    )
    conn.execute(
        sa.text(
            "UPDATE api_keys SET status = 'revoked'"
            " WHERE name = 'DLP Internal Scanner' AND status = 'active'"
        )
    )

    conn.execute(
        sa.text(
            "DELETE FROM app_config WHERE `key` IN"
            " ('dlp.internal_api_key_raw', 'dlp.internal_api_key_id')"
        )
    )

    # Retention sweeps dlp_alerts by scanned_at alone.  The existing
    # composite (user_id, scanned_at) can't serve that predicate without a
    # user_id prefix, so the sweep would full-scan a table that until now
    # had no delete path at all.
    op.create_index("ix_dlp_alerts_scanned_at", "dlp_alerts", ["scanned_at"])


def downgrade() -> None:
    op.drop_index("ix_dlp_alerts_scanned_at", table_name="dlp_alerts")
    # The config rows are deliberately NOT restored: a revoked credential
    # cannot be un-burned, and re-creating the keys would reintroduce the
    # plaintext-at-rest defect this migration exists to remove.
