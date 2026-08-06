############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 070: make historical user references nullable
#
############################################################

"""Make blog_posts.author_id, email_log.sent_by, and
admin_audit_log.user_id nullable.

Hard-deleting a user must not destroy blog posts, the email log, or the
admin audit trail (the audit log is retained permanently by policy).
delete_user now detaches these references (SET NULL) instead, which
requires the columns to be nullable.  MariaDB keeps the existing FK
constraints intact across a MODIFY that only changes nullability.

Revision ID: 070
Revises: 069
"""

import sqlalchemy as sa
from alembic import op

revision = "070"
down_revision = "069"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "blog_posts", "author_id",
        existing_type=sa.Integer(), nullable=True,
    )
    op.alter_column(
        "email_log", "sent_by",
        existing_type=sa.Integer(), nullable=True,
    )
    op.alter_column(
        "admin_audit_log", "user_id",
        existing_type=sa.Integer(), nullable=True,
    )


def downgrade() -> None:
    # Rows detached by a user deletion (NULL refs) must be removed or
    # repointed manually before tightening the columns back down.
    op.alter_column(
        "admin_audit_log", "user_id",
        existing_type=sa.Integer(), nullable=False,
    )
    op.alter_column(
        "email_log", "sent_by",
        existing_type=sa.Integer(), nullable=False,
    )
    op.alter_column(
        "blog_posts", "author_id",
        existing_type=sa.Integer(), nullable=False,
    )
