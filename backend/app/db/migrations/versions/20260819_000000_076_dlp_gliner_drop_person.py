############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 076_dlp_gliner_drop_person.py: Remove "person" from the
#     stored GLiNER category list (measured precision 0.34)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Remove "person" from the stored dlp.gliner.categories list

Revision ID: 076
Revises: 075
Create Date: 2026-08-19 00:00:00.000000

The DLP evaluation harness measured GLiNER's "person" category at 0.34
precision on a labeled corpus: the model tags section headers and
greetings ("Chief complaint", "CONTACT", "hey") as people, and those
false positives dominated the clean-traffic alert rate.  "person" was
removed from scan_gliner's code default, but migration 044 SEEDED
``dlp.gliner.categories`` in app_config with a list that includes
"person", and the worker passes the stored row to the scanner — so on
any deployment that ran 044 (including prod) the code-default change
alone is inert.  This migration edits the stored list in place.

Admins can still re-enable "person" deliberately via Admin -> DLP; this
only removes it from lists that carry it, preserving any other
customization in the row.

Downgrade is a no-op: resurrecting a category that was removed for
measured noise would be wrong, and an admin who wants it back has the
UI.
"""

import json

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None

_KEY = "dlp.gliner.categories"


def upgrade() -> None:
    bind = op.get_bind()
    row = bind.execute(
        sa.text("SELECT value FROM app_config WHERE `key` = :k"), {"k": _KEY}
    ).fetchone()
    if row is None:
        return
    try:
        categories = json.loads(row[0])
    except (ValueError, TypeError):
        return
    if not isinstance(categories, list) or "person" not in categories:
        return
    categories = [c for c in categories if c != "person"]
    bind.execute(
        sa.text("UPDATE app_config SET value = :v WHERE `key` = :k"),
        {"v": json.dumps(categories), "k": _KEY},
    )


def downgrade() -> None:
    # Deliberate no-op — see module docstring.
    pass
