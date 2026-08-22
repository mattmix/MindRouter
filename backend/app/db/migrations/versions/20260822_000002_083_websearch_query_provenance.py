############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 083_websearch_query_provenance.py: keep the caller's original
#     query alongside the redacted one that was actually sent.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Before/after provenance for screened web-search queries.

Revision ID: 083
Revises: 082

`web_search_logs.query` holds the OUTBOUND text (post-redaction). This adds
`query_original` for what the caller submitted, so an auditor can see both
halves of the before/after trail, plus the switch that governs keeping it.

The per-pass verdicts live in the existing dlp_detail JSON (`passes`), which
needs no schema change.
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None

_CONFIG = {
    # On: an auditor investigating a block needs to see what was actually
    # typed. Off keeps the masked evidence, the per-pass verdicts and the
    # outbound text, but not the original.
    "search.audit.store_original_query": True,
}


def upgrade() -> None:
    op.add_column("web_search_logs", sa.Column("query_original", sa.Text(), nullable=True))
    bind = op.get_bind()
    for key, value in _CONFIG.items():
        bind.exec_driver_sql(
            "INSERT IGNORE INTO app_config (`key`, value, description) VALUES (%s, %s, %s)",
            (key, json.dumps(value), "Web-search query provenance (added in 083)"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql(
        "DELETE FROM app_config WHERE `key` = 'search.audit.store_original_query'"
    )
    op.drop_column("web_search_logs", "query_original")
