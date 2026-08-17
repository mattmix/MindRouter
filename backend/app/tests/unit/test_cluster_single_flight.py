"""Tests for the /api/cluster/* single-flight TTL cache (F40).

_SingleFlightTTL lives in backend/app/api/health.py, which imports the db
package chain at module top (Request/RequestStatus, AsyncSessionLocal).
Per the project's import-chain rules we do NOT import that at module level;
the import happens inside a fixture and skips cleanly if the chain's deps
(pymysql, prometheus_client, ...) are unavailable in the test env.
"""

import asyncio

import pytest


@pytest.fixture(scope="module")
def SingleFlightTTL():
    try:
        from backend.app.api.health import _SingleFlightTTL
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"health module import unavailable: {exc}")
    return _SingleFlightTTL


@pytest.mark.asyncio
async def test_concurrent_misses_collapse_to_one_call(SingleFlightTTL):
    calls = {"n": 0}

    async def producer():
        calls["n"] += 1
        await asyncio.sleep(0.02)  # hold the lock so peers pile up behind it
        return {"v": calls["n"]}, True

    cache = SingleFlightTTL(60.0)
    results = await asyncio.gather(*[cache.get(producer) for _ in range(25)])

    # Exactly one producer call served all 25 concurrent misses.
    assert calls["n"] == 1
    assert all(r == {"v": 1} for r in results)


@pytest.mark.asyncio
async def test_cached_value_served_within_ttl(SingleFlightTTL):
    calls = {"n": 0}

    async def producer():
        calls["n"] += 1
        return {"v": calls["n"]}, True

    cache = SingleFlightTTL(60.0)
    first = await cache.get(producer)
    second = await cache.get(producer)
    assert first == second
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_non_cacheable_result_not_stored(SingleFlightTTL):
    calls = {"n": 0}

    async def producer():
        calls["n"] += 1
        return {"v": calls["n"]}, False  # fallback: returned but not cached

    cache = SingleFlightTTL(60.0)
    r1 = await cache.get(producer)
    r2 = await cache.get(producer)
    # Each call re-runs because nothing was cached.
    assert calls["n"] == 2
    assert r1 == {"v": 1}
    assert r2 == {"v": 2}


@pytest.mark.asyncio
async def test_value_refetched_after_ttl_expiry(SingleFlightTTL):
    calls = {"n": 0}

    async def producer():
        calls["n"] += 1
        return {"v": calls["n"]}, True

    cache = SingleFlightTTL(0.01)
    await cache.get(producer)
    await asyncio.sleep(0.03)  # let the entry expire
    await cache.get(producer)
    assert calls["n"] == 2
