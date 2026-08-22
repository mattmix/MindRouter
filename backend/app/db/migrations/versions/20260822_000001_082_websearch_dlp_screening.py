############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 082_websearch_dlp_screening.py: DLP screening of web-search
#     queries before they reach a third-party provider.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pre-send DLP screening for web search.

Revision ID: 082
Revises: 081

Adds the screening outcome to the web-search audit row (dlp_action +
dlp_detail) and seeds the configuration.  Screening is OFF by default: it adds
a synchronous scan to every search, so an operator opts in.
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "082"
down_revision = "081"
branch_labels = None
depends_on = None

_CONFIG = {
    # Off by default — turning it on adds a blocking scan before every search.
    "dlp.websearch.enabled": False,
    # Redact at this severity and above ("moderate or high risk").
    "dlp.websearch.min_severity": "moderate",
    # Fail CLOSED when the scanners are unavailable: the point of the feature
    # is that unscanned text must not reach a third party.
    "dlp.websearch.on_scanner_error": "block",
}


def upgrade() -> None:
    op.add_column(
        "web_search_logs", sa.Column("dlp_action", sa.String(length=16), nullable=True)
    )
    op.add_column("web_search_logs", sa.Column("dlp_detail", sa.JSON(), nullable=True))
    op.create_index(
        "ix_web_search_logs_dlp_action", "web_search_logs", ["dlp_action", "created_at"]
    )

    bind = op.get_bind()
    for key, value in _CONFIG.items():
        bind.exec_driver_sql(
            "INSERT IGNORE INTO app_config (`key`, value, description) VALUES (%s, %s, %s)",
            (key, json.dumps(value), "Web-search DLP screening (added in 082)"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql(
        "DELETE FROM app_config WHERE `key` IN "
        "('dlp.websearch.enabled','dlp.websearch.min_severity',"
        "'dlp.websearch.on_scanner_error')"
    )
    op.drop_index("ix_web_search_logs_dlp_action", table_name="web_search_logs")
    op.drop_column("web_search_logs", "dlp_detail")
    op.drop_column("web_search_logs", "dlp_action")
