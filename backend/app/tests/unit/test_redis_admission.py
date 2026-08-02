############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_redis_admission.py: Redis-shared per-backend admission counters
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for fleet-wide max_concurrent admission counters.

Each worker owns its own Redis subkey mr:adm:{backend_id}:{worker_id};
the fleet-wide in-flight count is the sum over all live subkeys.  A dead
worker stops refreshing its TTLs, so its leaked slots self-heal.

Covers:
- incr/decr symmetry on a worker's subkey
- snapshot summing across workers, zero-fill for idle backends,
  foreign/malformed keys ignored
- negative-count guard (decr never goes below zero)
- TTL refresh call shape on incr, decr, and reconcile-set
- reconcile set: absolute SET with TTL, delete at zero
- fail-open: unavailable or raising Redis returns None / no-ops
- route_job prefers the global snapshot, falls back to local per-worker
  depths when the snapshot is None (routing spec-imported with
  backend.app.db* pre-mocked — see MEMORY.md "Import Chain Gotcha")
- claim/release paths mirror into the shared counter
- source contracts: GC eviction decrements, phantom reset zeroes,
  maintenance loop reconciles
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_CORE_DIR = Path(__file__).resolve().parents[2] / "core"
_ROUTING_PY = _CORE_DIR / "scheduler" / "routing.py"


def _load_rc():
    """Resolve the real redis_client module.

    Other test files in this suite stub backend.app.core.* into
    sys.modules at collection time without restoring; if the package
    import yields such an empty stub, spec-load the .py directly.
    """
    import backend.app.core.redis_client as pkg_rc

    if hasattr(pkg_rc, "_redis"):
        return pkg_rc
    spec = importlib.util.spec_from_file_location(
        "mr2_redis_client_under_test", _CORE_DIR / "redis_client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_rc()


class FakePipeline:
    """Buffers ops like redis-py's pipeline, replays them on execute()."""

    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    def incr(self, key):
        self.ops.append(("incr", key))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    async def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "incr":
                results.append(await self.redis.incr(op[1]))
            else:
                results.append(await self.redis.expire(op[1], op[2]))
        return results


class FakeRedis:
    """Minimal async Redis: INCR/EXPIRE pipeline, DECR-floor Lua, SCAN/MGET."""

    def __init__(self):
        self.store = {}
        self.ttls = {}
        self.expire_calls = []  # (key, ttl) — TTL call-shape assertions

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def incr(self, key):
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        self.expire_calls.append((key, ttl))
        return 1

    async def set(self, key, val, nx=False, ex=None):
        self.store[key] = val
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def delete(self, key):
        self.store.pop(key, None)
        self.ttls.pop(key, None)
        return 1

    async def eval(self, script, numkeys, key, *args):
        # Only the admission DECR-with-floor script is exercised here
        assert "decr" in script
        v = int(self.store.get(key, 0)) - 1
        if v < 0:
            v = 0
        self.store[key] = v
        await self.expire(key, int(args[0]))
        return v

    async def mget(self, keys):
        return [self.store.get(k) for k in keys]

    async def scan_iter(self, match=None, count=None):
        prefix = match.rstrip("*") if match else ""
        for key in list(self.store):
            if key.startswith(prefix):
                yield key


class RaisingRedis:
    """Every operation raises — helpers must fail open, never propagate."""

    def pipeline(self, transaction=False):
        raise RuntimeError("boom")

    async def eval(self, *args, **kwargs):
        raise RuntimeError("boom")

    def scan_iter(self, **kwargs):
        raise RuntimeError("boom")

    async def set(self, *args, **kwargs):
        raise RuntimeError("boom")

    async def delete(self, *args):
        raise RuntimeError("boom")


@pytest.fixture
def fake(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(rc, "_redis", r)
    monkeypatch.setattr(rc, "_available", True)
    return r


# ===================================================================
# Counter helpers
# ===================================================================


@pytest.mark.asyncio
async def test_incr_decr_symmetry(fake):
    assert await rc.incr_backend_inflight(5, "wA") == 1
    assert await rc.incr_backend_inflight(5, "wA") == 2
    assert await rc.decr_backend_inflight(5, "wA") == 1
    assert await rc.decr_backend_inflight(5, "wA") == 0
    assert fake.store["mr:adm:5:wA"] == 0


@pytest.mark.asyncio
async def test_default_worker_id_is_module_uuid(fake):
    await rc.incr_backend_inflight(3)
    assert f"mr:adm:3:{rc.WORKER_ID}" in fake.store


@pytest.mark.asyncio
async def test_negative_count_guard(fake):
    # Decrement on a missing/expired subkey must floor at zero
    assert await rc.decr_backend_inflight(9, "wA") == 0
    assert await rc.decr_backend_inflight(9, "wA") == 0
    assert fake.store["mr:adm:9:wA"] == 0


@pytest.mark.asyncio
async def test_ttl_refreshed_on_every_touch(fake):
    await rc.incr_backend_inflight(5, "wA")
    assert ("mr:adm:5:wA", rc._ADM_TTL_SECONDS) in fake.expire_calls
    fake.expire_calls.clear()
    await rc.decr_backend_inflight(5, "wA")
    assert ("mr:adm:5:wA", rc._ADM_TTL_SECONDS) in fake.expire_calls
    await rc.set_backend_inflight(5, 2, "wA")
    assert fake.ttls["mr:adm:5:wA"] == rc._ADM_TTL_SECONDS


@pytest.mark.asyncio
async def test_reconcile_set_is_absolute_and_deletes_at_zero(fake):
    await rc.incr_backend_inflight(5, "wA")
    await rc.set_backend_inflight(5, 3, "wA")  # SET, not INCR
    assert fake.store["mr:adm:5:wA"] == 3
    await rc.set_backend_inflight(5, 0, "wA")
    assert "mr:adm:5:wA" not in fake.store


# ===================================================================
# Snapshot
# ===================================================================


@pytest.mark.asyncio
async def test_snapshot_sums_across_workers(fake):
    await rc.incr_backend_inflight(1, "wA")
    await rc.incr_backend_inflight(1, "wA")
    await rc.incr_backend_inflight(1, "wB")
    await rc.incr_backend_inflight(2, "wB")
    snap = await rc.get_backend_inflight_snapshot([1, 2, 3])
    assert snap == {1: 3, 2: 1, 3: 0}


@pytest.mark.asyncio
async def test_snapshot_ignores_foreign_and_malformed_keys(fake):
    fake.store["quota:tokens:1"] = 99
    fake.store["mr:adm:notanint:wA"] = 7
    snap = await rc.get_backend_inflight_snapshot([1])
    assert snap == {1: 0}


# ===================================================================
# Fail-open
# ===================================================================


@pytest.mark.asyncio
async def test_fail_open_when_unavailable(monkeypatch):
    monkeypatch.setattr(rc, "_redis", None)
    monkeypatch.setattr(rc, "_available", False)
    assert await rc.incr_backend_inflight(1, "wA") is None
    assert await rc.decr_backend_inflight(1, "wA") is None
    assert await rc.get_backend_inflight_snapshot([1]) is None
    await rc.set_backend_inflight(1, 2, "wA")  # no raise


@pytest.mark.asyncio
async def test_fail_open_when_client_raises(monkeypatch):
    monkeypatch.setattr(rc, "_redis", RaisingRedis())
    monkeypatch.setattr(rc, "_available", True)
    assert await rc.incr_backend_inflight(1, "wA") is None
    assert await rc.decr_backend_inflight(1, "wA") is None
    assert await rc.get_backend_inflight_snapshot([1]) is None
    await rc.set_backend_inflight(1, 2, "wA")  # no raise


# ===================================================================
# Routing integration (spec-imported with backend.app.db* pre-mocked)
# ===================================================================


@pytest.fixture(scope="module")
def routing_mod():
    """Import routing with the db package chain mocked; restore afterwards."""
    added_mocks = []
    for name in [
        "backend.app.db",
        "backend.app.db.session",
        "backend.app.db.models",
        "backend.app.db.crud",
    ]:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
            added_mocks.append(name)
    before = set(sys.modules)
    mod = importlib.import_module("backend.app.core.scheduler.routing")
    yield mod
    # Drop everything this import pulled in under mocked db so later test
    # files re-import the scheduler chain against the real models
    for name in set(sys.modules) - before:
        sys.modules.pop(name, None)
    for name in added_mocks:
        sys.modules.pop(name, None)


class CaptureScorer:
    """Records the queue_depths route_job passes to the scorer."""

    def __init__(self, scores=None):
        self.captured_depths = None
        self.scores = scores or []

    def rank_backends(self, backends, job, backend_models,
                      gpu_utilizations=None, queue_depths=None,
                      latency_emas=None):
        self.captured_depths = queue_depths
        return self.scores


def _job(**kwargs):
    defaults = dict(
        request_id="r1", user_id=1, model="m", priority=0.0,
        assigned_backend_id=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _redis_stub(snapshot):
    return SimpleNamespace(
        get_backend_inflight_snapshot=AsyncMock(return_value=snapshot),
        incr_backend_inflight=AsyncMock(return_value=1),
        decr_backend_inflight=AsyncMock(return_value=0),
        set_backend_inflight=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_route_job_scores_with_global_snapshot(routing_mod, monkeypatch):
    router = routing_mod.BackendRouter()
    router._backend_queue_depths = {7: 3}
    scorer = CaptureScorer()
    router.scorer = scorer
    stub = _redis_stub({7: 5})
    monkeypatch.setattr(routing_mod, "redis_client", stub)

    await router.route_job(_job(), [SimpleNamespace(id=7, name="b7")], {})

    # Fleet-wide count (5) used for the capacity gate, not the local 3
    assert scorer.captured_depths == {7: 5}
    stub.get_backend_inflight_snapshot.assert_awaited_once_with([7])


@pytest.mark.asyncio
async def test_route_job_falls_back_to_local_depths(routing_mod, monkeypatch):
    router = routing_mod.BackendRouter()
    router._backend_queue_depths = {7: 3}
    scorer = CaptureScorer()
    router.scorer = scorer
    monkeypatch.setattr(routing_mod, "redis_client", _redis_stub(None))

    await router.route_job(_job(), [SimpleNamespace(id=7, name="b7")], {})

    # Redis down → today's per-worker semantics
    assert scorer.captured_depths == {7: 3}


@pytest.mark.asyncio
async def test_route_job_claim_mirrors_to_redis(routing_mod, monkeypatch):
    router = routing_mod.BackendRouter()
    score = SimpleNamespace(total_score=1.0, backend_id=7, failed_constraints=[])
    router.scorer = CaptureScorer(scores=[score])
    stub = _redis_stub({7: 0})
    monkeypatch.setattr(routing_mod, "redis_client", stub)

    decision = await router.route_job(
        _job(), [SimpleNamespace(id=7, name="b7")], {}
    )

    assert decision.success
    assert router._backend_queue_depths == {7: 1}  # local always maintained
    stub.incr_backend_inflight.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_completion_and_failure_release_redis_slot(routing_mod, monkeypatch):
    router = routing_mod.BackendRouter()
    router.fair_share = SimpleNamespace(on_job_completed=AsyncMock())
    router.queue = SimpleNamespace(cancel_job=AsyncMock(return_value=False))
    router._try_complete_drain = AsyncMock()
    router._backend_queue_depths = {7: 2}
    stub = _redis_stub({})
    monkeypatch.setattr(routing_mod, "redis_client", stub)

    await router.on_job_completed(_job(), 7, tokens_used=100)
    assert router._backend_queue_depths == {7: 1}
    stub.decr_backend_inflight.assert_awaited_once_with(7)

    await router.on_job_failed(_job(), 7)
    assert router._backend_queue_depths == {7: 0}
    assert stub.decr_backend_inflight.await_count == 2
    router._try_complete_drain.assert_awaited_once_with(7)  # only at depth 0


# ===================================================================
# Source contracts: GC eviction + maintenance-loop reconcile
# ===================================================================


def test_gc_and_reconcile_source_contract():
    src = _ROUTING_PY.read_text()
    gc = src.split("async def _gc_stale_jobs")[1]
    # Stale-job eviction releases the shared slot
    assert "decr_backend_inflight" in gc
    # Phantom-depth reset zeroes this worker's shared counters
    assert "set_backend_inflight" in gc
    loop = src.split("async def _maintenance_loop")[1].split("async def ")[0]
    # 30s reconcile: absolute SET of local in-flight counts (TTL refresh)
    assert "set_backend_inflight" in loop
