############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 080_apikey_expiry_reminders.py: per-key "reminder sent"
#     watermarks for the API key expiry reminder emails, plus
#     the reminder configuration defaults (opt-in, off).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API key expiry reminder watermarks + config.

Revision ID: 080
Revises: 079

Adds two nullable timestamp columns to api_keys so each key is reminded at
most once per tier (early / urgent).  They are cleared on renewal (a renewed
key re-arms both reminders) and are naturally NULL for every existing key, so
no backfill is needed.  Seeds the reminder config in app_config, all OFF by
default — no email goes out until an admin enables reminders.
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "080"
down_revision = "079"
branch_labels = None
depends_on = None

_CONFIG = {
    "apikey.reminders.enabled": False,
    "apikey.reminders.early_days": 10,
    "apikey.reminders.urgent_days": 2,
    "apikey.reminders.test_recipient": "",
}


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("reminder_early_sent_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "api_keys",
        sa.Column("reminder_urgent_sent_at", sa.DateTime(timezone=True), nullable=True),
    )
    bind = op.get_bind()
    for key, value in _CONFIG.items():
        bind.exec_driver_sql(
            "INSERT IGNORE INTO app_config (`key`, value, description) VALUES (%s, %s, %s)",
            (key, json.dumps(value), "API key expiry reminder config (added in 080)"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql(
        "DELETE FROM app_config WHERE `key` IN "
        "('apikey.reminders.enabled','apikey.reminders.early_days',"
        "'apikey.reminders.urgent_days','apikey.reminders.test_recipient')"
    )
    op.drop_column("api_keys", "reminder_urgent_sent_at")
    op.drop_column("api_keys", "reminder_early_sent_at")
