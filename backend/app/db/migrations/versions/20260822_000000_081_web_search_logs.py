############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 081_web_search_logs.py: first-class audit entity for
#     outbound web-search provider calls.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Web search audit log.

Revision ID: 081
Revises: 080

Creates `web_search_logs`: one row per outbound call to a search provider,
recording the provider, the payload sent (credentials redacted), the HTTP
status, latency, the verbatim response body, the normalized results, and the
failure detail when there is one.

Also seeds the audit/retention configuration:
  search.audit.enabled              — master switch (default on)
  search.audit.store_response_body  — keep the verbatim body (default on)
  search.audit.max_body_chars       — cap on the stored body
  retention.web_search_logs_days    — 0 = keep forever (matches dlp_alerts)

Indexes lead with the filter column and end on created_at so the audit
viewer's "newest first, filtered by X" is served by an index rather than a
filesort.
"""

import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "081"
down_revision = "080"
branch_labels = None
depends_on = None

_CONFIG = {
    "search.audit.enabled": True,
    "search.audit.store_response_body": True,
    "search.audit.max_body_chars": 200000,
    "retention.web_search_logs_days": 0,
}


def upgrade() -> None:
    op.create_table(
        "web_search_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("search_uuid", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("api_key_id", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.BigInteger(), nullable=True),
        sa.Column("client_ip", sa.String(length=45), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("max_results", sa.Integer(), nullable=True),
        sa.Column("request_url", sa.String(length=1000), nullable=True),
        sa.Column("request_params", sa.JSON(), nullable=True),
        sa.Column("request_headers", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("response_body", mysql.MEDIUMTEXT(), nullable=True),
        sa.Column("response_truncated", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("response_meta", sa.JSON(), nullable=True),
        sa.Column("error_type", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"]),
        sa.ForeignKeyConstraint(["request_id"], ["requests.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("search_uuid"),
    )
    op.create_index("ix_web_search_logs_created_at", "web_search_logs", ["created_at"])
    op.create_index("ix_web_search_logs_provider_time", "web_search_logs", ["provider", "created_at"])
    op.create_index("ix_web_search_logs_status_time", "web_search_logs", ["status", "created_at"])
    op.create_index("ix_web_search_logs_source_time", "web_search_logs", ["source", "created_at"])
    op.create_index("ix_web_search_logs_user_time", "web_search_logs", ["user_id", "created_at"])
    op.create_index("ix_web_search_logs_http_status", "web_search_logs", ["http_status"])
    op.create_index("ix_web_search_logs_request", "web_search_logs", ["request_id"])

    bind = op.get_bind()
    for key, value in _CONFIG.items():
        bind.exec_driver_sql(
            "INSERT IGNORE INTO app_config (`key`, value, description) VALUES (%s, %s, %s)",
            (key, json.dumps(value), "Web search audit log (added in 081)"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql(
        "DELETE FROM app_config WHERE `key` IN "
        "('search.audit.enabled','search.audit.store_response_body',"
        "'search.audit.max_body_chars','retention.web_search_logs_days')"
    )
    # MariaDB will not drop an index backing a foreign key, so the table goes
    # in one piece rather than index-by-index.
    op.drop_table("web_search_logs")
