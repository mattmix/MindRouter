############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# scopes.py: Bounded privilege for API keys
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API key scopes.

Before this existed, authorization was a single boolean derived from the
owning user's group: a credential could either do everything an admin can —
including reading every stored prompt and revoking anyone's key — or nothing
administrative at all. There was no way to say "this credential may provision
users for one app and nothing else", which is what a registered application
needs.

TWO INVARIANTS, both load-bearing:

1. **NULL scopes means legacy.** Every key that existed before migration 073
   has no scope list, and for those the group-derived checks apply exactly as
   they always did. Adding scopes is therefore a pure addition — no existing
   credential changes behaviour.

2. **Scopes only ever REMOVE privilege.** A scope list is an allowlist
   intersected with what the owner could already do; it can never grant
   something the owner's group does not permit. So an app key belonging to an
   administrator is still not an admin credential — which is the point, since
   administrators use first-party apps too, and a key minted by an app should
   never carry its owner's admin rights. (This is the same class of mistake as
   the DLP internal key, which inherited admin because privilege was derived
   solely from whoever happened to own it.)
"""

from typing import Iterable, Optional, Set

# Call inference endpoints (/v1/chat/completions, embeddings, images, ...).
SCOPE_INFERENCE = "inference"

# Reach administrative endpoints. Still ALSO requires the owner's group to be
# admin — this scope permits, it does not grant.
SCOPE_ADMIN = "admin"

# Provision users and mint per-user keys, limited to the app the credential
# belongs to. This is what a registered application's own credential carries.
SCOPE_APP_PROVISION = "apps:provision"

ALL_SCOPES = frozenset({SCOPE_INFERENCE, SCOPE_ADMIN, SCOPE_APP_PROVISION})

# What an app mints for one of its end users: inference and nothing else,
# whoever owns it.
APP_USER_KEY_SCOPES = (SCOPE_INFERENCE,)

# What a registered app's own credential carries.
APP_CREDENTIAL_SCOPES = (SCOPE_APP_PROVISION,)


def parse_scopes(raw: Optional[str]) -> Optional[Set[str]]:
    """Parse the stored scope string.

    Returns None for a legacy key (no restriction), otherwise the set of
    granted scopes. An explicitly empty string is NOT legacy — it is a key
    that may do nothing, which is a meaningful state for a disabled
    credential.
    """
    if raw is None:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}


def format_scopes(scopes: Optional[Iterable[str]]) -> Optional[str]:
    """Serialize a scope collection for storage. None stays None (legacy)."""
    if scopes is None:
        return None
    return ",".join(sorted({s.strip() for s in scopes if s and s.strip()}))


def key_has_scope(api_key, scope: str) -> bool:
    """Does this key permit ``scope``?

    A legacy key (NULL scopes) permits everything at this layer — the caller's
    group-derived checks remain the gate, unchanged. A scoped key permits only
    what it lists.
    """
    granted = parse_scopes(getattr(api_key, "scopes", None))
    if granted is None:
        return True
    return scope in granted


def is_scoped(api_key) -> bool:
    """True when this key carries an explicit scope list."""
    return getattr(api_key, "scopes", None) is not None
