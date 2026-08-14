############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_backend_poll_concurrency.py: the backend poll sweep and
#     capability discovery must be concurrency-bounded so N uvicorn
#     workers polling M backends can't exhaust MariaDB's connection
#     pool (each health check / discovery opens a DB session).
#
############################################################

"""Bounded-concurrency guard for the backend registry poll/discovery.

The registry source is checked structurally (importing it pulls the DB
chain — see MEMORY "Import Chain Gotcha"); the actual semaphore behaviour is
exercised functionally via a deferred import inside the test so it never runs
at collection time.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

_REG_SRC = (
    Path(__file__).resolve().parents[2] / "core" / "telemetry" / "registry.py"
).read_text()
_SETTINGS_SRC = (
    Path(__file__).resolve().parents[2] / "settings.py"
).read_text()


class TestPollConcurrencyWiring:
    def test_setting_exists(self):
        assert "backend_poll_concurrency: int = 8" in _SETTINGS_SRC

    def test_bounded_helper_used_in_both_poll_paths(self):
        assert "async def _gather_bounded" in _REG_SRC
        # full sweep + fast-poll both go through the bounded helper, not a
        # raw unbounded gather over every backend.
        assert "await self._gather_bounded(all_tasks)" in _REG_SRC
        assert "await self._gather_bounded(fast_tasks)" in _REG_SRC
        assert _REG_SRC.count("asyncio.gather(*all_tasks") == 0
        assert _REG_SRC.count("asyncio.gather(*fast_tasks") == 0

    def test_discovery_is_bounded(self):
        assert "self._discover_sem = asyncio.Semaphore(" in _REG_SRC
        assert "async with self._discover_sem:" in _REG_SRC
        assert "async def _discover_backend_impl" in _REG_SRC


class TestGatherBoundedBehavior:
    @pytest.mark.asyncio
    async def test_caps_concurrent_execution(self):
        from backend.app.core.telemetry.registry import BackendRegistry

        reg = BackendRegistry.__new__(BackendRegistry)  # skip heavy __init__
        reg._settings = SimpleNamespace(backend_poll_concurrency=3)

        state = {"cur": 0, "max": 0}

        async def _worker():
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.01)
            state["cur"] -= 1

        await reg._gather_bounded([_worker() for _ in range(20)])

        assert state["max"] <= 3, f"concurrency exceeded the limit: {state['max']}"
        assert state["max"] >= 1
        assert state["cur"] == 0  # everything ran and drained

    @pytest.mark.asyncio
    async def test_runs_all_and_swallows_exceptions(self):
        from backend.app.core.telemetry.registry import BackendRegistry

        reg = BackendRegistry.__new__(BackendRegistry)
        reg._settings = SimpleNamespace(backend_poll_concurrency=2)

        ran = []

        async def _ok(i):
            ran.append(i)

        async def _boom():
            raise RuntimeError("one bad check must not abort the sweep")

        # return_exceptions=True inside the helper: a failing check is isolated.
        await reg._gather_bounded([_ok(0), _boom(), _ok(1), _boom(), _ok(2)])
        assert sorted(ran) == [0, 1, 2]
