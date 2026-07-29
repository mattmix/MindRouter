############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 068: generic SSO identity columns (provider, subject)
#
############################################################

"""Add users.sso_provider / users.sso_subject for non-Azure SSO providers.

Google / generic OIDC / SAML logins key on the (provider, subject) pair.
azure_oid is untouched — the Azure AD driver keeps its original column.

Revision ID: 068
Revises: 067
"""

import sqlalchemy as sa
from alembic import op

revision = "068"
down_revision = "067"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("sso_provider", sa.String(40), nullable=True))
    op.add_column("users", sa.Column("sso_subject", sa.String(255), nullable=True))
    op.create_index(
        "uq_users_sso_identity",
        "users",
        ["sso_provider", "sso_subject"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_users_sso_identity", table_name="users")
    op.drop_column("users", "sso_subject")
    op.drop_column("users", "sso_provider")
