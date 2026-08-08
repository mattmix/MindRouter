############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# voice_router.py: Pick a TTS/STT service for one request
#
# Prefers a registered backend (health-checked, circuit-broken
# and load-balanced like every other modality) and falls back to
# the legacy single-URL app_config entry when none is registered.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Resolve which voice service should serve a request.

Before 2.9.10 TTS and STT were a single hardcoded URL each in app_config
(``voice.tts_url`` / ``voice.stt_url``): no health check, no failover, no
circuit breaker, and ``requests.backend_id`` always NULL, unlike chat,
embeddings, rerank, image and video which are all registry-routed.

This module is the seam. Callers ask for a target; whether that target came
from the registry or the legacy config key is an implementation detail, so
voice services can be migrated one at a time with no flag day.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# app_config keys holding the legacy single URL / credential per kind.
_LEGACY_KEYS = {
    "tts": ("voice.tts_url", "voice.tts_api_key"),
    "stt": ("voice.stt_url", "voice.stt_api_key"),
}


@dataclass
class VoiceTarget:
    """One resolved voice service."""

    url: str                      # base URL, no trailing slash
    api_key: Optional[str] = None
    backend_id: Optional[int] = None   # None when resolved from legacy config
    backend_name: Optional[str] = None

    @property
    def source(self) -> str:
        return "registry" if self.backend_id is not None else "config_fallback"


async def resolve_voice_backend(db, kind: str) -> Optional[VoiceTarget]:
    """Return a service to handle one ``kind`` ("tts" or "stt") request.

    Resolution order:
      1. A healthy registered backend of the matching engine whose circuit is
         not open. Chosen at random to spread load — the scheduler's
         latency-aware scoring is built around token-bearing jobs, and a
         voice request carries no tokens, so uniform choice is both simpler
         and honest about what we know.
      2. The legacy ``voice.<kind>_url`` app_config value.

    Returns None when neither is available, which the caller should surface
    as a configuration error rather than a backend failure.
    """
    if kind not in _LEGACY_KEYS:
        raise ValueError(f"unknown voice kind: {kind!r}")

    target = await _from_registry(kind)
    if target is not None:
        # The Backend model has no credential column, so an operator-set
        # voice.<kind>_api_key still applies to registered backends. Without
        # this, registering a backend would silently stop sending a key that
        # was previously being sent — an auth failure with no obvious cause.
        target.api_key = await _config_api_key(db, kind)
        return target

    return await _from_config(db, kind)


async def _config_api_key(db, kind: str) -> Optional[str]:
    """Read the configured upstream credential for this voice kind."""
    from backend.app.db import crud

    try:
        return await crud.get_config_json(db, _LEGACY_KEYS[kind][1], None)
    except Exception:
        logger.exception("voice_api_key_lookup_failed", kind=kind)
        return None


async def _from_registry(kind: str) -> Optional[VoiceTarget]:
    """Pick a healthy, circuit-closed registered backend, or None."""
    try:
        from backend.app.core.telemetry.registry import get_registry
        from backend.app.db.models import BackendEngine

        engine = BackendEngine.TTS if kind == "tts" else BackendEngine.STT
        registry = get_registry()
        backends = await registry.get_healthy_backends(engine=engine)
    except Exception:
        # A registry problem must not take voice down while the legacy
        # config path still works.
        logger.exception("voice_registry_lookup_failed", kind=kind)
        return None

    available: List = []
    for b in backends:
        try:
            if await registry.is_backend_available(b.id):
                available.append(b)
        except Exception:
            continue

    if not available:
        return None

    backend = random.choice(available)
    return VoiceTarget(
        url=str(backend.url).rstrip("/"),
        api_key=None,          # filled in by the caller from app_config
        backend_id=backend.id,
        backend_name=backend.name,
    )


async def _from_config(db, kind: str) -> Optional[VoiceTarget]:
    """Legacy single-URL path, kept until every deployment has registered
    its voice services."""
    from backend.app.db import crud

    url_key, key_key = _LEGACY_KEYS[kind]
    url = await crud.get_config_json(db, url_key, None)
    if not url:
        return None

    api_key = await crud.get_config_json(db, key_key, None)
    return VoiceTarget(url=str(url).rstrip("/"), api_key=api_key)
