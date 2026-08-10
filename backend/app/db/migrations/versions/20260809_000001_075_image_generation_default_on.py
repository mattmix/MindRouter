############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 075_image_generation_default_on.py: Make image access a
#     global default with per-user exceptions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Make users.image_generation_enabled tri-state and default it ON.

Revision ID: 075
Revises: 074

Image access was pure opt-in: migration 058 created the column NOT NULL
DEFAULT 0 and an administrator granted it one person at a time. The
expectation is now the opposite — everyone has it unless told otherwise,
including accounts created by SSO and by a registered application, which no
administrator sees before they start using the product.

So the column becomes TRI-STATE:

    NULL   inherit the `img.enabled_by_default` global   (the normal state)
    True   force ON  regardless of the global
    False  force OFF regardless of the global

WHY THE 0s BECOME NULL BUT THE 1s DO NOT. At the time of writing production
had 49 rows set to 1 and 206 set to 0, and the admin audit log held 47
`images_config.toggle_user` entries of which EVERY ONE was false -> true.
Nobody has ever been deliberately denied, so a 0 records only "never got
around to granting it" and carries no intent to preserve. A 1 is the opposite:
it is the surviving record of a deliberate grant, and it is exactly the
allow-list that keeps those people working if an operator ever flips the
global back OFF. The asymmetry is the point.

Without the backfill this migration would be a no-op for 206 of 255 users, and
the admin page's new exception list would open with 206 phantom "denied"
entries.

THE DDL TRAP: `server_default=None` must be passed explicitly. Supplying only
`existing_server_default` emits `BOOL NULL DEFAULT 0`, which keeps the
database-level default — so every INSERT that omits the column still writes a
forced deny, and inheritance never happens. After upgrading, the column must
read `tinyint(1) DEFAULT NULL` in SHOW CREATE TABLE.

MariaDB DDL is non-transactional, so each step is separately re-runnable: the
ALTER is idempotent in effect, the UPDATE matches nothing on a second pass, and
the seed is a SELECT-then-INSERT.

DEPLOY ORDER, AND THE ONE THING TO DO AFTERWARDS. Migrate FIRST, then recreate
the app container — the reverse order breaks provisioning outright, because the
new code writes NULL into a column that is still NOT NULL (error 1048).

The cost of migrating first is a short window in which the OLD image is still
serving, and the OLD model declaration carries a Python-side `default=False`.
A Python-side default is applied by the ORM regardless of what the database
says, so any account created by SSO or by the app-provisioning endpoint during
that window is written with an explicit 0 — a force-OFF that looks exactly like
a deliberate administrative denial and will never be revisited.

So once the new container is healthy, re-run the backfill. It is idempotent and
by then nothing can write a fresh 0:

    UPDATE users SET image_generation_enabled = NULL
    WHERE image_generation_enabled = 0;

Skipping it strands anyone who happened to sign in mid-deploy.
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "075"
down_revision = "074"
branch_labels = None
depends_on = None


_CONFIG_KEY = "img.enabled_by_default"
_CONFIG_VALUE = True
_CONFIG_DESCRIPTION = (
    "Image generation is available to all users unless a per-user override "
    "says otherwise"
)


def upgrade() -> None:
    # 1. Widen to nullable AND drop the server default. Both halves matter —
    #    see THE DDL TRAP above.
    op.alter_column(
        "users",
        "image_generation_enabled",
        existing_type=sa.Boolean(),
        nullable=True,
        existing_nullable=False,
        server_default=None,
        existing_server_default=sa.text("0"),
    )

    # 2. Backfill. Must follow the ALTER: NULL cannot be written into a NOT
    #    NULL column. Explicit grants (1) are deliberately left alone.
    op.execute(
        sa.text(
            "UPDATE users SET image_generation_enabled = NULL "
            "WHERE image_generation_enabled = 0"
        )
    )

    # 3. Seed the global. The `img.*` namespace has never been seeded and its
    #    lazy call-site defaults have already drifted from each other; on an
    #    access flag that is not acceptable, so the row is materialised.
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )
    conn = op.get_bind()
    existing = conn.execute(
        sa.text("SELECT 1 FROM app_config WHERE `key` = :k"), {"k": _CONFIG_KEY}
    ).fetchone()
    if not existing:
        op.execute(
            app_config.insert().values(
                key=_CONFIG_KEY,
                value=json.dumps(_CONFIG_VALUE),
                description=_CONFIG_DESCRIPTION,
            )
        )


def downgrade() -> None:
    """Restore the pre-075 shape.

    LOSSY BY CONSTRUCTION: NULL means "inherit", and the NOT NULL column has no
    way to say that. Inheriting users are materialised to 0 — the exact
    pre-075 state, not the current global default — so downgrading revokes
    access from everyone who was relying on inheritance. That is the honest
    reversal of this migration; anything else would invent grants that never
    existed.

    The NULLs must be materialised BEFORE the column is tightened: MariaDB runs
    STRICT_TRANS_TABLES here and rejects the narrowing rather than coercing.

    ORDER: the config row is deleted FIRST, before any DDL. MariaDB implicitly
    commits around DDL, so a statement that fails after the ALTER leaves the
    schema already changed while alembic rolls back its own (now empty)
    transaction and never advances alembic_version — a half-applied downgrade
    that then refuses to re-run because the column is already NOT NULL. Putting
    the only non-DDL statement first means any failure there leaves nothing
    committed.
    """
    # Bind parameters go through the connection: Operations.execute() takes
    # `execution_options` as its only other argument and it is KEYWORD-ONLY, so
    # passing a params dict positionally raises TypeError before any SQL runs.
    conn = op.get_bind()
    conn.execute(
        sa.text("DELETE FROM app_config WHERE `key` = :k"), {"k": _CONFIG_KEY}
    )
    op.execute(
        sa.text(
            "UPDATE users SET image_generation_enabled = 0 "
            "WHERE image_generation_enabled IS NULL"
        )
    )
    op.alter_column(
        "users",
        "image_generation_enabled",
        existing_type=sa.Boolean(),
        nullable=False,
        existing_nullable=True,
        server_default=sa.text("0"),
    )
