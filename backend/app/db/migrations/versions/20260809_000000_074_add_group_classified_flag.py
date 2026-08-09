############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 074_add_group_classified_flag.py: Mark users whose group
#     was assigned without the directory attribute that
#     normally decides it, so it can be settled later.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add users.group_classified.

Revision ID: 074
Revises: 073

A user's group is chosen once, at first provisioning, from the Azure
`jobTitle` attribute (`azure_auth._map_job_title_to_group`), and is never
re-evaluated afterwards — the update branch refreshes name/department/college
and last_login only.

That was harmless while every Azure user arrived through MindRouter's own
sign-in, which fetches jobTitle from Microsoft Graph. It stops being harmless
once a registered app can provision users from an id_token, because an
id_token does NOT carry jobTitle. A faculty member who used the app before
ever signing in to MindRouter would land in the default group and stay there
permanently, with the wrong token budget and scheduler weight, and nothing
would flag it.

This column marks such users. Their group is settled on their first direct
MindRouter sign-in, where Graph data is available.

DEFAULT IS TRUE ON PURPOSE: every existing user is treated as already
classified, so no one's group is re-evaluated on upgrade. Only accounts
created without jobTitle are marked false.

Note this does not address the narrower pre-existing case of an Azure user
whose jobTitle is empty or unrecognised — they also land in the default group
permanently. Marking those false here would silently reassign existing users'
groups on their next login, which is a policy change and belongs in its own
deliberate migration.
"""

import sqlalchemy as sa
from alembic import op

revision = "074"
down_revision = "073"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "group_classified",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("1"),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "group_classified")
