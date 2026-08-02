############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 069: api_keys.key_sha256 for O(1) hot-path key lookup
#
############################################################

"""Add api_keys.key_sha256 (SHA-256 hexdigest of the full key).

Nullable: keys created before this migration are backfilled lazily by
verify_api_key on their first successful Argon2 verification. The Argon2
key_hash column is kept as-is for rollback safety.

Kept deliberately minimal (one column + one index, no FK involvement) —
MariaDB DDL is non-transactional, so a mid-migration failure must leave
as little partial state as possible.

Revision ID: 069
Revises: 068
"""

import sqlalchemy as sa
from alembic import op

revision = "069"
down_revision = "068"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Type and index name deliberately match what Base.metadata renders for
    # the model's String(64) + unique=True + index=True declaration, so a
    # future `alembic revision --autogenerate` (compare_type=True) produces
    # no drop/recreate churn on the uniqueness guarantee the fast path
    # depends on.
    op.add_column("api_keys", sa.Column("key_sha256", sa.String(64), nullable=True))
    op.create_index(
        "ix_api_keys_key_sha256",
        "api_keys",
        ["key_sha256"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_api_keys_key_sha256", table_name="api_keys")
    op.drop_column("api_keys", "key_sha256")
