############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 078_dlp_detection_actions.py: Seed the per-severity DLP
#     "Detection Action" configuration (Block and/or Alert),
#     the global block scope, and the per-severity notify-user
#     flag — deriving values from the pre-existing email config
#     so behavior is UNCHANGED on deploy (alert-only, no blocking).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Seed DLP detection-action config.

Revision ID: 078
Revises: 077

Introduces a per-severity action model on top of the existing DLP config:

  dlp.action.{severity}.block   (bool)  — reject the request with a 422
  dlp.action.{severity}.alert   (bool)  — send an email alert (async)
  dlp.email.{severity}.notify_user (bool) — also email the requesting user
  dlp.block.scope               (str)   — "prompt" | "response" | "both"

These are stored as JSON-encoded values in app_config, exactly like every other
dlp.* key (crud.set_config / get_config_json). This migration only SEEDS keys
that are absent; it never overwrites an admin's existing value.

Behavior preservation: `alert` is derived from the current email delivery mode
(`dlp.email.{severity}.mode`): alert = (mode != "off"). A missing mode key means
the old code used its "immediate" default, so alert defaults to true there. Every
`block` seeds to false and scope to "prompt", so the inline blocking path stays
completely inert until an admin turns a Block on — no latency change on deploy.

DDL note: app_config is a plain key/value table; these are row INSERTs guarded by
INSERT IGNORE (the unique PK on `key` makes an existing row a no-op), so the
migration is safe to re-run and never clobbers configured values.
"""

import json

from alembic import op

revision = "078"
down_revision = "077"
branch_labels = None
depends_on = None

SEVERITIES = ("major", "moderate", "minor")


def _seed(bind, key: str, value) -> None:
    """INSERT IGNORE one JSON-encoded config key (no-op if it already exists)."""
    bind.exec_driver_sql(
        "INSERT IGNORE INTO app_config (`key`, value, description) "
        "VALUES (%s, %s, %s)",
        (key, json.dumps(value), "DLP detection action (added in 078)"),
    )


def _existing_mode(bind, severity: str) -> str:
    """Return the stored email delivery mode for a severity, or 'immediate'.

    Values are JSON-encoded strings (e.g. '"off"'); decode defensively.
    """
    row = bind.exec_driver_sql(
        "SELECT value FROM app_config WHERE `key` = %s",
        (f"dlp.email.{severity}.mode",),
    ).scalar()
    if row is None:
        return "immediate"  # matches the old GET default
    try:
        val = json.loads(row)
        return val if isinstance(val, str) else "immediate"
    except (ValueError, TypeError):
        return "immediate"


def upgrade() -> None:
    bind = op.get_bind()

    # Global block scope — default to prompt-only (lowest cost, cleanest 422,
    # does not touch streaming).
    _seed(bind, "dlp.block.scope", "prompt")

    for sev in SEVERITIES:
        # Blocking starts OFF everywhere so the inline scan path is inert.
        _seed(bind, f"dlp.action.{sev}.block", False)
        # Alert mirrors the old "send email?" gate: mode != off.
        _seed(bind, f"dlp.action.{sev}.alert", _existing_mode(bind, sev) != "off")
        # Do not email the end user by default.
        _seed(bind, f"dlp.email.{sev}.notify_user", False)


def downgrade() -> None:
    bind = op.get_bind()
    keys = ["dlp.block.scope"]
    for sev in SEVERITIES:
        keys.extend(
            [
                f"dlp.action.{sev}.block",
                f"dlp.action.{sev}.alert",
                f"dlp.email.{sev}.notify_user",
            ]
        )
    for key in keys:
        bind.exec_driver_sql("DELETE FROM app_config WHERE `key` = %s", (key,))
