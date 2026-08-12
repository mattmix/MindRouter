############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_rate_limit_fallback.py: bounded in-memory RPM fallback
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the F38 fix: check_rpm must not fail fully open when
Redis is unavailable.

Covers:
- InMemoryRateLimiter fixed-window semantics: Nth request denied within
  the window, count contract mirrors check_rpm (includes-on-allow,
  excludes-on-deny), window resets after it elapses.
- LRU memory bound: distinct identities never exceed max_keys.
- check_rpm healthy path unchanged: when Redis answers, the in-memory
  limiter is never consulted.
- check_rpm degraded path: Redis unavailable + fallback enabled routes to
  the in-memory limiter and enforces the limit.
- gating: fallback disabled preserves the historical fully-open behaviour.
- rpm_limit <= 0 is always allowed.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_CORE_DIR = Path(__file__).resolve().parents[2] / "core"

# rate_limits.py is self-contained (stdlib only) so a plain import is safe
# and gives us the SAME module instance check_rpm's fallback imports.
from backend.app.security.rate_limits import (  # noqa: E402
    InMemoryRateLimiter,
    reset_local_limiter,
)


def _load_rc():
    """Resolve the real redis_client module, dodging sibling sys.modules
    stubs (see MEMORY.md "Import Chain Gotcha")."""
    import backend.app.core.redis_client as pkg_rc

    if hasattr(pkg_rc, "check_rpm"):
        return pkg_rc
    spec = importlib.util.spec_from_file_location(
        "mr2_redis_client_rpm_under_test", _CORE_DIR / "redis_client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_rc()


@pytest.fixture(autouse=True)
def _clean_limiter():
    reset_local_limiter()
    yield
    reset_local_limiter()


# ── InMemoryRateLimiter unit semantics ────────────────────────────────


def test_window_allows_up_to_limit_then_denies():
    lim = InMemoryRateLimiter(window_seconds=60)
    t = 1000.0
    # limit=3: three allowed, fourth denied.
    assert lim.check("user:1", 3, now=t) == (True, 1)
    assert lim.check("user:1", 3, now=t) == (True, 2)
    assert lim.check("user:1", 3, now=t) == (True, 3)
    # Denied: count excludes this request (mirrors the Redis decr-back).
    assert lim.check("user:1", 3, now=t) == (False, 3)
    assert lim.check("user:1", 3, now=t) == (False, 3)


def test_window_resets_after_it_elapses():
    lim = InMemoryRateLimiter(window_seconds=60)
    assert lim.check("user:1", 1, now=1000.0) == (True, 1)
    assert lim.check("user:1", 1, now=1030.0) == (False, 1)  # same window
    # A full window later the counter resets.
    assert lim.check("user:1", 1, now=1061.0) == (True, 1)


def test_distinct_identities_are_independent():
    lim = InMemoryRateLimiter(window_seconds=60)
    assert lim.check("user:1", 1, now=1000.0) == (True, 1)
    assert lim.check("user:2", 1, now=1000.0) == (True, 1)
    assert lim.check("user:1", 1, now=1000.0) == (False, 1)


def test_zero_or_negative_limit_always_allowed():
    lim = InMemoryRateLimiter()
    assert lim.check("user:1", 0) == (True, 0)
    assert lim.check("user:1", -5) == (True, 0)


def test_memory_bound_evicts_oldest_keys():
    lim = InMemoryRateLimiter(window_seconds=600, max_keys=100)
    t = 5000.0
    for i in range(250):
        # distinct times so LRU order is well-defined and windows stay live
        lim.check(f"user:{i}", 5, now=t + i * 0.001)
    assert len(lim._buckets) <= 100
    # The oldest identity was evicted; a fresh window starts on next touch.
    assert lim.check("user:0", 5, now=t + 1.0) == (True, 1)


def test_expired_windows_are_pruned():
    lim = InMemoryRateLimiter(window_seconds=60, max_keys=100)
    for i in range(50):
        lim.check(f"user:{i}", 5, now=1000.0)
    assert len(lim._buckets) == 50
    # Touching a new key well past the window prunes all the stale ones.
    lim.check("user:new", 5, now=2000.0)
    assert len(lim._buckets) == 1


# ── check_rpm wiring ──────────────────────────────────────────────────


class _FakeRedis:
    """Minimal Redis honoring check_rpm's INCR/TTL pipeline + expire/decr."""

    def __init__(self):
        self.store = {}

    def pipeline(self, transaction=True):
        redis = self

        class _P:
            def __init__(self):
                self.ops = []

            def incr(self, k):
                self.ops.append(("incr", k))

            def ttl(self, k):
                self.ops.append(("ttl", k))

            async def execute(self):
                out = []
                for op in self.ops:
                    if op[0] == "incr":
                        redis.store[op[1]] = int(redis.store.get(op[1], 0)) + 1
                        out.append(redis.store[op[1]])
                    else:  # ttl — pretend already set so no re-expire
                        out.append(30)
                return out

        return _P()

    async def expire(self, k, ttl):
        return True

    async def decr(self, k):
        redis = self.store
        redis[k] = int(redis.get(k, 0)) - 1
        return redis[k]


@pytest.mark.asyncio
async def test_healthy_path_uses_redis_not_local(monkeypatch):
    """When Redis answers, the in-memory limiter stays untouched."""
    monkeypatch.setattr(rc, "_available", True)
    monkeypatch.setattr(rc, "_redis", _FakeRedis())
    # Two requests under a limit of 5 both allowed via Redis.
    assert await rc.check_rpm("user:7", 5) == (True, 1)
    assert await rc.check_rpm("user:7", 5) == (True, 2)
    # The local limiter never saw "user:7".
    assert "user:7" not in rc.__dict__.get("_buckets", {})
    from backend.app.security.rate_limits import _limiter

    assert "user:7" not in _limiter._buckets


@pytest.mark.asyncio
async def test_rpm_limit_zero_always_allowed(monkeypatch):
    monkeypatch.setattr(rc, "_available", False)
    monkeypatch.setattr(rc, "_redis", None)
    assert await rc.check_rpm("user:7", 0) == (True, 0)


@pytest.mark.asyncio
async def test_redis_down_falls_back_and_enforces(monkeypatch):
    """Redis unavailable + fallback on -> local limiter denies the Nth."""
    monkeypatch.setattr(rc, "_available", False)
    monkeypatch.setattr(rc, "_redis", None)
    monkeypatch.setattr(
        rc, "get_settings", lambda: SimpleNamespace(rate_limit_local_fallback=True)
    )
    assert await rc.check_rpm("user:99", 3) == (True, 1)
    assert await rc.check_rpm("user:99", 3) == (True, 2)
    assert await rc.check_rpm("user:99", 3) == (True, 3)
    allowed, _ = await rc.check_rpm("user:99", 3)
    assert allowed is False


@pytest.mark.asyncio
async def test_redis_exception_falls_back(monkeypatch):
    """A raising Redis op also routes to the local limiter."""

    class _BoomRedis:
        def pipeline(self, transaction=True):
            raise RuntimeError("redis exploded")

    monkeypatch.setattr(rc, "_available", True)
    monkeypatch.setattr(rc, "_redis", _BoomRedis())
    monkeypatch.setattr(
        rc, "get_settings", lambda: SimpleNamespace(rate_limit_local_fallback=True)
    )
    assert await rc.check_rpm("user:5", 2) == (True, 1)
    assert await rc.check_rpm("user:5", 2) == (True, 2)
    allowed, _ = await rc.check_rpm("user:5", 2)
    assert allowed is False


@pytest.mark.asyncio
async def test_fallback_disabled_preserves_fail_open(monkeypatch):
    """With the gate off, behaviour is the historical fully-open path."""
    monkeypatch.setattr(rc, "_available", False)
    monkeypatch.setattr(rc, "_redis", None)
    monkeypatch.setattr(
        rc, "get_settings", lambda: SimpleNamespace(rate_limit_local_fallback=False)
    )
    for _ in range(20):
        assert await rc.check_rpm("user:1", 3) == (True, 0)


@pytest.mark.asyncio
async def test_fallback_default_on_when_setting_absent(monkeypatch):
    """If the setting is not yet present, getattr defaults to enabled."""
    monkeypatch.setattr(rc, "_available", False)
    monkeypatch.setattr(rc, "_redis", None)
    monkeypatch.setattr(rc, "get_settings", lambda: SimpleNamespace())
    assert await rc.check_rpm("user:42", 1) == (True, 1)
    allowed, _ = await rc.check_rpm("user:42", 1)
    assert allowed is False
