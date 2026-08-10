############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# feature_access.py: Per-user feature access with a global
#     default and explicit per-user exceptions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Who may generate images.

Image access used to be pure opt-in: `users.image_generation_enabled` was a
NOT NULL boolean defaulting to 0, and an administrator granted it one person at
a time. The expectation is now the opposite — everyone has it unless told
otherwise, including accounts created by SSO and by a registered application,
which no administrator ever sees before they start using the product.

So the column becomes TRI-STATE and the policy moves to one config value:

    NULL   inherit the global default   (the normal state)
    True   force ON  regardless of the global
    False  force OFF regardless of the global

    effective = override if override is not None else global_default

THE HAZARD THIS MODULE EXISTS TO REMOVE: a nullable boolean is FALSY. Python's
`if not user.image_generation_enabled`, Jinja's `{% if ... %}` and SQL's
`col == True` all read NULL as "no". Every one of those, left alone, silently
denies access to every user who inherits the default — which after migration
075 is almost everyone. Resolution lives here, once, and a drift guard in
test_image_access_tristate.py fails the build if the column is read directly
anywhere else.

TWO PATHS, DELIBERATELY DIFFERENT:

  * `image_generation_allowed()` reads the database and is the ONLY thing an
    enforcement gate may use.
  * `image_access()` reads a module cache and exists so a Jinja template can
    decide whether to draw a nav link without a DB round-trip per render. It is
    NEVER an authorization boundary. If it is briefly stale the worst outcome
    is a link shown or hidden a few seconds early; the route behind it
    re-resolves against the database.

The cache/refresh-loop shape is copied from services/branding.py, which solved
the same "one config value, needed synchronously in every template, must
converge across uvicorn workers" problem.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from sqlalchemy import or_

from backend.app.db import crud
from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Namespace note: every other setting this subsystem owns is `img.*`
# (img.enabled, img.default_model, img.max_n ...). A second prefix for the same
# feature is how a future read site looks up a key that does not exist, gets
# None, and — on an access flag — denies everyone.
IMAGE_DEFAULT_KEY = "img.enabled_by_default"

# Fallback when the row is absent. Must be identical at every call site:
# get_config_json returns the CALLER's default when the key is missing, so a
# site passing something else is a silent policy fork.
IMAGE_DEFAULT_FALLBACK = True

# Distinct from IMAGE_DEFAULT_KEY: `img.enabled` is the subsystem kill switch
# (off => 503 for everyone). This module does not read it; the gates check it
# separately so a globally-down service still reports "unavailable" rather than
# "forbidden".

_REFRESH_INTERVAL = 15  # seconds; how quickly a save propagates across workers
_DEFAULT_CACHE: bool = IMAGE_DEFAULT_FALLBACK


# --------------------------------------------------------------------------
# Pure logic — no imports, no I/O, directly unit-testable
# --------------------------------------------------------------------------

def resolve_feature_access(user_value: Optional[bool], global_default: bool) -> bool:
    """Resolve a tri-state override against the global default.

    `None` means inherit. Anything else is an explicit decision that outranks
    the global in BOTH directions.
    """
    if user_value is None:
        return bool(global_default)
    return bool(user_value)


def access_filter(column: Any, global_default: bool):
    """SQL predicate matching users who EFFECTIVELY have access.

    Takes the column object rather than importing the model, so this module's
    only backend dependency stays `crud` and tests can load it in isolation.
    """
    if global_default:
        return or_(column.is_(True), column.is_(None))
    return column.is_(True)


def exception_kind(global_default: bool) -> str:
    """Which overrides are EXCEPTIONS, as a `crud.get_users(image_override=...)`
    selector.

    An override equal to the global is not an exception — it resolves to the
    same answer and would be noise. When the default is ON the exceptions are
    the denied; when it is OFF they are the allowed.

    Returns a selector rather than a SQL predicate on purpose: `crud.get_users`
    already owns the string->predicate mapping, and `crud` cannot import this
    module (this module imports `crud`). Expressing the policy here and the SQL
    there keeps ONE implementation of each — an earlier revision had this
    function build its own predicate that nothing called, while the admin page
    hand-derived the same rule, so the tests guarded code that never ran.
    """
    return "off" if global_default else "on"


def normalize_legacy_image_access(data: dict) -> int:
    """Convert pre-075 `image_generation_enabled: false` rows to inherit.

    Before migration 075 that column was NOT NULL, so EVERY export taken before
    this release records `false` for every user who had not been individually
    granted access — which was almost all of them. Restoring such a backup onto
    a populated database is harmless (existing usernames are skipped), but
    restoring into a FRESH one — the actual disaster-recovery scenario —
    would insert an explicit force-OFF for the entire user base and silently
    deny everyone, with no error and nothing in the logs.

    `false` in a legacy export means "was never granted", not "was denied": at
    the time of the change the audit log held 47 access toggles and every one
    was an enable. Coercing it to NULL restores the intended meaning, which is
    "inherit whatever the global default is". An explicit `true` is a real
    grant and is left alone.

    HOW A PRE-075 EXPORT IS IDENTIFIED. On a FACT, not on the shape of the
    data: migration 075 seeds `img.enabled_by_default`, so any export taken
    after it carries that app_config row and any export taken before it cannot.

    The obvious shortcut — "no user row is NULL, therefore pre-075" — is wrong
    in the dangerous direction. A small deployment where an administrator has
    explicitly classified every user contains no NULLs while being perfectly
    post-075, and the shortcut would rewrite each deliberate DENIAL to inherit;
    on restore those users would silently REGAIN access that was deliberately
    revoked. That is a fail-open in the disaster-recovery path, which is the
    one path that must fail closed.

    Returns the number of rows normalized.
    """
    users = data.get("users")
    if not isinstance(users, list) or not users:
        return 0

    config_rows = data.get("app_config")
    if isinstance(config_rows, list) and any(
        isinstance(r, dict) and r.get("key") == "img.enabled_by_default"
        for r in config_rows
    ):
        # Post-075 export: every value in it is deliberate. Leave it alone.
        return 0

    normalized = 0
    for row in users:
        # `0` as well as `False`: JSON round-trips and older exporters have both.
        if isinstance(row, dict) and row.get("image_generation_enabled") in (False, 0):
            row["image_generation_enabled"] = None
            normalized += 1
    if normalized:
        logger.warning(
            "legacy_image_access_normalized",
            rows=normalized,
            detail=(
                "pre-075 backup (no img.enabled_by_default row): force-OFF "
                "image access coerced to inherit the global default"
            ),
        )
    return normalized

# --------------------------------------------------------------------------
# Authoritative path — every enforcement gate uses this
# --------------------------------------------------------------------------

async def image_default_enabled(db) -> bool:
    """The global default, read from the database."""
    return bool(await crud.get_config_json(db, IMAGE_DEFAULT_KEY, IMAGE_DEFAULT_FALLBACK))


async def image_generation_allowed(db, user) -> bool:
    """May this user generate or edit images?

    The single answer every gate must ask for. Returns False for a missing
    user so a caller that forgot its own None check still fails closed.
    """
    if user is None:
        return False
    return resolve_feature_access(
        getattr(user, "image_generation_enabled", None),
        await image_default_enabled(db),
    )


# --------------------------------------------------------------------------
# Cached path — navigation rendering only, never authorization
# --------------------------------------------------------------------------

async def refresh_feature_access_cache(db=None) -> bool:
    """Reload the global default into the module cache. Never raises."""
    global _DEFAULT_CACHE
    try:
        if db is not None:
            value = await image_default_enabled(db)
        else:
            async with get_async_db_context() as own_db:
                value = await image_default_enabled(own_db)
        _DEFAULT_CACHE = value
        return value
    except Exception:  # pragma: no cover - defensive; keep serving last-known
        logger.warning("feature_access_cache_refresh_failed", exc_info=True)
        return _DEFAULT_CACHE


def image_access(user) -> bool:
    """Synchronous accessor for Jinja. NAV VISIBILITY ONLY — not a gate."""
    if user is None:
        return False
    return resolve_feature_access(
        getattr(user, "image_generation_enabled", None), _DEFAULT_CACHE
    )


async def feature_access_refresh_loop() -> None:
    """Background task: converge the cache across workers after an admin save."""
    while True:
        try:
            await asyncio.sleep(_REFRESH_INTERVAL)
            await refresh_feature_access_cache()
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.warning("feature_access_refresh_loop_error", exc_info=True)
