############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# api_keys.py: API key generation, hashing, and verification
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API key generation and verification."""

import asyncio
import hashlib
import hmac
import secrets
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.models import ApiKey

# API key format: mr2_<random_string>
API_KEY_PREFIX = "mr2_"
API_KEY_LENGTH = 48  # Total length including prefix

# Use Argon2 for hashing
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
)

# Each Argon2 verify allocates memory_cost (64 MiB) and runs off-thread
# (argon2-cffi releases the GIL); the cap keeps a flood of unknown keys
# from blowing up worker RSS.
_argon2_verify_semaphore = asyncio.Semaphore(4)


def generate_api_key() -> Tuple[str, str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, key_prefix, key_sha256)
        - full_key: The complete API key to give to the user (store nowhere!)
        - key_hash: Argon2 hash to store in database
        - key_prefix: First 8 chars for identification
        - key_sha256: SHA-256 hexdigest for hot-path lookup
    """
    # SECURITY INVARIANT: token_urlsafe(32) = 256 bits of entropy. Storing a
    # plain SHA-256 of the key is safe ONLY because the key is high-entropy
    # and unguessable — never lower this, and never apply this scheme to
    # low-entropy secrets like passwords.
    random_part = secrets.token_urlsafe(32)

    # Full key with prefix
    full_key = f"{API_KEY_PREFIX}{random_part}"

    # Argon2 hash kept alongside key_sha256 for rollback safety
    key_hash = hash_api_key(full_key)

    # SHA-256 digest for O(1) verification
    key_sha256 = hashlib.sha256(full_key.encode()).hexdigest()

    # Prefix for identification (first 8 chars of random part)
    key_prefix = f"{API_KEY_PREFIX}{random_part[:8]}"

    return full_key, key_hash, key_prefix, key_sha256


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using Argon2.

    For API keys, we use a faster hash since we need to look them up frequently.
    We use SHA-256 first to normalize, then Argon2 for the actual hash.

    Args:
        api_key: The raw API key

    Returns:
        Argon2 hash of the key
    """
    # First normalize with SHA-256 (fast, deterministic)
    normalized = hashlib.sha256(api_key.encode()).hexdigest()

    # Then hash with Argon2
    return _hasher.hash(normalized)


def _verify_key_hash(api_key: str, key_hash: str) -> bool:
    """
    Verify an API key against a stored hash.

    Args:
        api_key: The raw API key to verify
        key_hash: The stored Argon2 hash

    Returns:
        True if the key matches
    """
    try:
        # Normalize with SHA-256
        normalized = hashlib.sha256(api_key.encode()).hexdigest()

        # Verify against Argon2 hash
        _hasher.verify(key_hash, normalized)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False


async def verify_api_key(db: AsyncSession, api_key: str) -> Optional[ApiKey]:
    """
    Verify an API key and return the ApiKey record if valid.

    Fast path: unique lookup on the SHA-256 digest — no Argon2 work.
    Fallback (keys created before migration 069): prefix lookup + Argon2
    verify, then backfill key_sha256 so the next request takes the fast path.

    Status/expiry/user-active checks live in the callers (auth.py et al.)
    against the returned row — this function only proves key possession.

    Args:
        db: Database session
        api_key: The raw API key to verify

    Returns:
        ApiKey record if valid, None otherwise
    """
    if not api_key.startswith(API_KEY_PREFIX):
        return None

    digest = hashlib.sha256(api_key.encode()).hexdigest()

    # Fast path: key_sha256 is unique, so a hit identifies the key
    db_key = await crud.get_api_key_by_sha256(db, digest)
    if db_key is not None:
        # Belt-and-braces constant-time recheck of the stored digest
        if db_key.key_sha256 and hmac.compare_digest(db_key.key_sha256, digest):
            return db_key
        return None

    # Fallback: prefix lookup (first 8 chars after mr2_) + Argon2 verify
    random_part = api_key[len(API_KEY_PREFIX):]
    key_prefix = f"{API_KEY_PREFIX}{random_part[:8]}"

    db_key = await crud.get_api_key_by_prefix(db, key_prefix)

    if not db_key:
        return None

    async with _argon2_verify_semaphore:
        verified = await asyncio.to_thread(_verify_key_hash, api_key, db_key.key_hash)

    if verified:
        # Verify-and-upgrade: the request session commits at teardown,
        # persisting the backfill without an extra commit here
        db_key.key_sha256 = digest
        return db_key

    return None


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)
