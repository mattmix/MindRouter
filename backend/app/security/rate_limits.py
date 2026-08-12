############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# rate_limits.py: Rate limiting
#
# Primary RPM enforcement is handled via Redis in
# backend/app/core/redis_client.py (check_rpm). This module
# provides a bounded, in-process fixed-window fallback that
# check_rpm consults ONLY when Redis is unavailable, so a
# Redis outage no longer fails fully open.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Rate limiting — Redis-backed RPM with a bounded in-memory fallback.

The authoritative, cross-worker limiter lives in
:func:`backend.app.core.redis_client.check_rpm`. When Redis is
unavailable that function previously admitted every request (fail-open).
This module supplies a small per-process fixed-window limiter so each
worker still enforces a local bound during an outage. It is deliberately
per-worker: with N workers the effective cluster limit degrades to at
most N × rpm_limit, which is a strict improvement over unbounded.

The limiter is memory-bounded: at most ``max_keys`` identities are
tracked, evicting the least-recently-used key when full, so a flood of
distinct identities cannot grow it without limit.
"""

import threading
import time
from collections import OrderedDict

# Mirror the Redis limiter's fixed window so behaviour matches during an
# outage and when Redis recovers.
_WINDOW_SECONDS = 60
# Hard cap on tracked identities. Each entry is tiny (a str key plus two
# numbers); 50k keys bounds worst-case memory to a few MB per worker.
_MAX_KEYS = 50_000


class InMemoryRateLimiter:
    """Bounded per-process fixed-window RPM limiter.

    Keyed by an opaque identity string (e.g. ``"user:123"``). Each key
    maps to ``(count, window_start)``. A key whose window has elapsed is
    reset on next access; an LRU eviction policy caps the total number of
    tracked keys.
    """

    def __init__(
        self, window_seconds: int = _WINDOW_SECONDS, max_keys: int = _MAX_KEYS
    ) -> None:
        self._window = window_seconds
        self._max_keys = max_keys
        # OrderedDict as an LRU: most-recently-touched key at the end,
        # so the front is the eviction candidate.
        self._buckets: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Drop expired windows, then LRU-evict down to the size cap.

        Caller must hold ``self._lock``.
        """
        # Expired windows are dead weight; reclaim them opportunistically.
        expired = [
            k
            for k, (_, start) in self._buckets.items()
            if now - start >= self._window
        ]
        for k in expired:
            del self._buckets[k]
        # Enforce the hard cap by evicting least-recently-used keys.
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)

    def check(self, key: str, limit: int, now: float | None = None) -> tuple[bool, int]:
        """Register a request and report whether it is allowed.

        Mirrors :func:`check_rpm`'s contract: returns
        ``(allowed, current_count)`` where *current_count* includes this
        request when allowed, and excludes it (the pre-existing count in
        the window) when denied.
        """
        if limit <= 0:
            return True, 0
        if now is None:
            now = time.monotonic()
        with self._lock:
            entry = self._buckets.get(key)
            if entry is None or now - entry[1] >= self._window:
                # New key or expired window: start a fresh window at 1.
                self._buckets[key] = (1, now)
                self._buckets.move_to_end(key)
                self._prune(now)
                return True, 1
            count, start = entry
            if count >= limit:
                # Over the limit — do not count this request (matches the
                # Redis path, which decrements back after detecting the
                # overflow and returns the pre-request count).
                self._buckets.move_to_end(key)
                return False, count
            count += 1
            self._buckets[key] = (count, start)
            self._buckets.move_to_end(key)
            return True, count


# Process-wide singleton used by the Redis client's fallback path.
_limiter = InMemoryRateLimiter()


def check_rpm_local(key: str, rpm_limit: int) -> tuple[bool, int]:
    """In-process fallback for :func:`check_rpm` when Redis is down.

    Same ``(allowed, current_count)`` contract as the Redis limiter.
    """
    return _limiter.check(key, rpm_limit)


def reset_local_limiter() -> None:
    """Clear all tracked windows (test helper)."""
    with _limiter._lock:
        _limiter._buckets.clear()
