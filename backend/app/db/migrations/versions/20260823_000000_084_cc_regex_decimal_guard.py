############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 084_cc_regex_decimal_guard.py: stop the credit-card pattern
#     matching the fractional part of a decimal number.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Credit-card regex: don't start inside a decimal number.

Revision ID: 084
Revises: 083

`.` is not a word character, so the bare \\b in the built-in card pattern let a
match begin immediately AFTER a decimal point and consume the fractional
digits. A game-telemetry line like

    {"x":33,"y":12,"z":33,"dist":2.808820224788294}

yielded the 15-digit run 808820224788294, which passes the Luhn check by
chance (roughly one random run in ten does) and raised a MAJOR "credit card
number" alert. Six such alerts fired against one session before this was
caught.

Since 2.9.46 the stored `dlp.regex.patterns` list is authoritative once an
admin has saved the DLP page (`dlp.regex.builtins_in_list`), so fixing the
code constant alone would not change a running deployment — this migration
rewrites the stored pattern too.

It only replaces an entry whose pattern is still EXACTLY the old built-in, so
a pattern an operator has customised is left alone.
"""

import json

from alembic import op

revision = "084"
down_revision = "083"
branch_labels = None
depends_on = None

_OLD = r"\b(?:\d[ -]*?){13,19}\b"
_NEW = r"(?<!\d)(?<!\d\.)\b(?:\d[ -]*?){13,19}\b"

_KEY = "dlp.regex.patterns"


def _swap(bind, frm: str, to: str) -> None:
    row = bind.exec_driver_sql(
        "SELECT value FROM app_config WHERE `key` = %s", (_KEY,)
    ).scalar()
    if row is None:
        return
    try:
        patterns = json.loads(row)
    except (ValueError, TypeError):
        return
    if not isinstance(patterns, list):
        return

    changed = False
    for entry in patterns:
        # Match on the pattern text, not the name: an operator may have renamed
        # the rule, and we must not rewrite one they have already edited.
        if isinstance(entry, dict) and entry.get("pattern") == frm:
            entry["pattern"] = to
            changed = True

    if changed:
        bind.exec_driver_sql(
            "UPDATE app_config SET value = %s WHERE `key` = %s",
            (json.dumps(patterns), _KEY),
        )


def upgrade() -> None:
    _swap(op.get_bind(), _OLD, _NEW)


def downgrade() -> None:
    _swap(op.get_bind(), _NEW, _OLD)
