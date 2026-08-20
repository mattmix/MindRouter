############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 079_dlp_redact_action.py: Seed the per-severity DLP "Redact"
#     action to false, so the third inline action (alongside
#     Block and Alert) has an explicit, off-by-default value.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Seed DLP redact action.

Revision ID: 079
Revises: 078

Adds `dlp.action.{severity}.redact` (bool) alongside the block/alert keys from
078. Redaction replaces the offending spans in the outbound prompt/response with
a placeholder while leaving the stored audit content intact. Seeded to false so
behavior is unchanged on deploy; INSERT IGNORE never clobbers a configured value.
"""

import json

from alembic import op

revision = "079"
down_revision = "078"
branch_labels = None
depends_on = None

SEVERITIES = ("major", "moderate", "minor")


def upgrade() -> None:
    bind = op.get_bind()
    for sev in SEVERITIES:
        bind.exec_driver_sql(
            "INSERT IGNORE INTO app_config (`key`, value, description) "
            "VALUES (%s, %s, %s)",
            (f"dlp.action.{sev}.redact", json.dumps(False),
             "DLP redact action (added in 079)"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    for sev in SEVERITIES:
        bind.exec_driver_sql(
            "DELETE FROM app_config WHERE `key` = %s",
            (f"dlp.action.{sev}.redact",),
        )
