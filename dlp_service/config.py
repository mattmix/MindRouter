############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_service/config.py: Env-driven configuration for the
# standalone GPU DLP microservice.
#
# Fully standalone — no backend.app.* imports. The service
# runs on a dedicated GPU node but must also import + run on
# a CPU-only Mac (device auto-detect) so its unit tests never
# require gliner, torch, or a GPU.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Environment-driven configuration for the DLP GPU microservice."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

# The production default label set for GLiNER PII detection. Mirrors the
# default categories in backend/app/services/dlp_scanner.py scan_gliner().
# "person" is deliberately absent (measured precision 0.34 — fires on section
# headers and greetings); admins opt in per-request via the categories field.
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "phone number",
    "email",
    "credit card number",
    "social security number",
    "date of birth",
    "driver license number",
    "passport number",
    "bank account number",
)


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    val = env.get(key)
    return val if val is not None and val != "" else default


def _env_opt_str(env: Mapping[str, str], key: str) -> Optional[str]:
    val = env.get(key)
    return val if val else None


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_categories(env: Mapping[str, str], key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = env.get(key)
    if not raw:
        return default
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or default


@dataclass(frozen=True)
class ServiceConfig:
    """Immutable service configuration, built from the process environment.

    Every field has a safe default so the module imports (and its tests run)
    with an empty environment.  ``key`` empty means the auth check FAILS CLOSED:
    with no shared secret configured, no request can authenticate.
    """

    key: str = ""                                  # DLP_SERVICE_KEY (shared secret)
    host: str = "0.0.0.0"                           # DLP_SERVICE_HOST
    port: int = 8710                                # DLP_SERVICE_PORT
    model: str = "urchade/gliner_multi_pii-v1"      # DLP_MODEL
    device: str = "auto"                            # DLP_DEVICE (auto|cuda:0|cpu)
    replicas: int = 2                               # DLP_REPLICAS
    max_batch: int = 32                             # DLP_MAX_BATCH
    batch_window_ms: float = 8.0                    # DLP_BATCH_WINDOW_MS
    max_queue: int = 512                            # DLP_MAX_QUEUE (total, across replicas)
    default_categories: Tuple[str, ...] = DEFAULT_CATEGORIES  # DLP_DEFAULT_CATEGORIES
    max_chars_cap: int = 10_000                     # DLP_MAX_CHARS_CAP (hard ceiling + default)
    hf_home: Optional[str] = None                   # DLP_HF_HOME (HuggingFace cache dir)
    torch_threads: int = 4                          # DLP_TORCH_THREADS

    @property
    def batch_window_s(self) -> float:
        """Batch-collection window in seconds (for asyncio.wait_for deadlines)."""
        return max(0.0, self.batch_window_ms) / 1000.0

    @property
    def replica_count(self) -> int:
        """Number of model replicas, never below 1."""
        return max(1, self.replicas)

    @property
    def per_replica_queue(self) -> int:
        """Per-replica queue maxsize.

        The global MAX_QUEUE is split evenly across replicas; each replica's
        own asyncio.Queue enforces its slice so an oversubscribed replica
        rejects with 503 while the others keep serving.
        """
        return max(1, self.max_queue // self.replica_count)

    @property
    def max_queue_total(self) -> int:
        """Effective total queue capacity = per-replica maxsize x replicas."""
        return self.per_replica_queue * self.replica_count

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ServiceConfig":
        """Build a config from ``env`` (defaults to os.environ)."""
        e = os.environ if env is None else env
        return cls(
            key=_env_str(e, "DLP_SERVICE_KEY", ""),
            host=_env_str(e, "DLP_SERVICE_HOST", "0.0.0.0"),
            port=_env_int(e, "DLP_SERVICE_PORT", 8710),
            model=_env_str(e, "DLP_MODEL", "urchade/gliner_multi_pii-v1"),
            device=_env_str(e, "DLP_DEVICE", "auto"),
            replicas=_env_int(e, "DLP_REPLICAS", 2),
            max_batch=_env_int(e, "DLP_MAX_BATCH", 32),
            batch_window_ms=_env_float(e, "DLP_BATCH_WINDOW_MS", 8.0),
            max_queue=_env_int(e, "DLP_MAX_QUEUE", 512),
            default_categories=_env_categories(e, "DLP_DEFAULT_CATEGORIES", DEFAULT_CATEGORIES),
            max_chars_cap=_env_int(e, "DLP_MAX_CHARS_CAP", 10_000),
            hf_home=_env_opt_str(e, "DLP_HF_HOME"),
            torch_threads=_env_int(e, "DLP_TORCH_THREADS", 4),
        )
