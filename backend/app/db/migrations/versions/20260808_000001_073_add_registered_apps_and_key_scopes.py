############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 073_add_registered_apps_and_key_scopes.py: Registered apps
#     and scoped API keys, so a first-party app can provision
#     its users without being a full deployment admin.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add registered apps and API key scopes.

Revision ID: 073
Revises: 072

Until now authorization was a single boolean derived from the owning user's
group (`api/auth.py` require_admin -> user.group.is_admin). There was no way
to grant a credential *some* privilege: to let an app create users and mint
keys you had to put its key's owner in an admin group, which also grants
reading every stored prompt and revoking anyone's key. This migration adds the
two columns that make a bounded credential possible.

`api_keys.scopes` is NULL for every existing key and NULL means "legacy" —
privilege continues to be derived from the owner's group exactly as before.
Only keys created with an explicit scope list are restricted. That keeps this
migration a pure addition with no behaviour change on upgrade.

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL):

  1. `api_keys` is small (hundreds of rows); two nullable column adds are
     metadata-only and fast. `requests` is NOT touched — app attribution is
     derived by joining `api_keys.app_id`, so the largest table in the
     database keeps its current row width.
  2. `apps.created_by` is nullable ON PURPOSE. Migration 070 established the
     rule that user references are detached rather than cascade-deleted, so
     deleting an admin never destroys the apps they registered.
  3. The downgrade drops both columns and the table. Any app-minted keys
     survive as ordinary keys — review them before downgrading, since they
     lose their scope restriction and revert to owner-derived privilege:
       SELECT id, name, user_id FROM api_keys WHERE app_id IS NOT NULL;
"""

import sqlalchemy as sa
from alembic import op

revision = "073"
down_revision = "072"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "apps",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        # Stable identifier used in API paths, e.g. "vandalchat".
        # Uniqueness comes from the explicit ix_apps_slug index below, NOT from
        # unique=True here: both would emit an index over the same column, and
        # MariaDB refuses to drop an index backing a constraint (error 1553),
        # so the spare would be unremovable by a later autogenerate.
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="active"
        ),
        # The Entra application (client) ID whose tokens this app may present.
        # Incoming id_tokens must carry EXACTLY this audience — accepting any
        # token from the tenant would let every app in the tenant provision
        # MindRouter accounts.
        sa.Column("entra_client_id", sa.String(64), nullable=True),
        sa.Column("entra_tenant_id", sa.String(64), nullable=True),
        # Lifetime of the per-user keys this app mints. Rotated silently on
        # each user login, so this bounds how long a leaked key stays useful.
        sa.Column(
            "key_ttl_days", sa.Integer, nullable=False, server_default="30"
        ),
        # Detach-not-delete: see migration 070.
        sa.Column(
            "created_by", sa.Integer, sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_apps_slug", "apps", ["slug"], unique=True)

    # Which app minted this key. NULL for every key that exists today and for
    # any key a human creates by hand. Attribution only — policy continues to
    # come from the owning user's group, because a person who uses both an app
    # and the API directly is one user with one budget and one fair share.
    op.add_column(
        "api_keys",
        sa.Column("app_id", sa.Integer, sa.ForeignKey("apps.id"), nullable=True),
    )
    op.create_index("ix_api_keys_app_id", "api_keys", ["app_id"])

    # Comma-separated scope allowlist. NULL = legacy, privilege derived from
    # the owner's group as before. A non-NULL value RESTRICTS the key and can
    # only ever remove privilege, never add it.
    op.add_column(
        "api_keys", sa.Column("scopes", sa.String(255), nullable=True)
    )

    # Keys an app manages on a user's behalf are not shown in that user's own
    # dashboard: the user never chose them and cannot meaningfully rotate them.
    op.add_column(
        "api_keys",
        sa.Column(
            "hidden", sa.Boolean, nullable=False, server_default=sa.text("0")
        ),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "hidden")
    op.drop_column("api_keys", "scopes")
    op.drop_index("ix_api_keys_app_id", table_name="api_keys")
    op.drop_column("api_keys", "app_id")
    op.drop_index("ix_apps_slug", table_name="apps")
    op.drop_table("apps")
