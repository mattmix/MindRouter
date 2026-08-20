############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_service.py: Unit tests for the standalone GPU DLP
# microservice (dlp_service/). Must pass with NO gpu, NO
# gliner, and NO torch installed — the model manager is mocked
# via the MODEL_FACTORY hook.
#
# Covers: shared-key auth (401), body validation (400),
# oversubscription (503) when a replica queue fills, the
# dynamic batching path (concurrent /scan calls coalesced into
# one GPU batch — batch_size > 1 observed), max_chars prefix
# truncation, /healthz and /stats shapes, and the invariant
# that request text NEVER appears in the service logs.
#
############################################################

"""Unit tests for the standalone GPU DLP microservice."""

import asyncio
import contextlib
import logging
import sys
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

# Established harness/service-suite pattern: make the repo root importable so
# `dlp_service` resolves regardless of pytest rootdir.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dlp_service import server  # noqa: E402
from dlp_service.config import ServiceConfig  # noqa: E402

KEY = "test-worker-key"


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Stand-in for a loaded GLiNER model.

    Records every batch_predict_entities call (so tests can assert batch shape
    / truncation) and flags any text containing the marker "SECRET".
    """

    def __init__(self, blocker: "threading.Event | None" = None):
        self.calls = []
        self._lock = threading.Lock()
        self._blocker = blocker

    def batch_predict_entities(self, texts, labels, threshold, batch_size):
        if self._blocker is not None:
            # Simulate a slow GPU so requests back up behind an in-flight batch.
            self._blocker.wait(timeout=2.0)
        with self._lock:
            self.calls.append(
                {"texts": list(texts), "labels": list(labels),
                 "threshold": threshold, "batch_size": batch_size}
            )
        out = []
        for t in texts:
            ents = []
            idx = t.find("SECRET")
            if idx != -1:
                ents.append({
                    "label": labels[0] if labels else "unknown",
                    "text": "SECRET",
                    "score": 0.99,
                    "start": idx,
                    "end": idx + len("SECRET"),
                })
            out.append(ents)
        return out

    def real_calls(self):
        """Calls excluding the startup warmup predict (["warmup"])."""
        return [c for c in self.calls if c["texts"] != ["warmup"]]


def _install_factory(monkeypatch, blocker=None):
    """Point MODEL_FACTORY at a recording FakeModel; return the models list."""
    models = []

    def factory(model_name, device):
        m = FakeModel(blocker=blocker)
        models.append(m)
        return m

    monkeypatch.setattr(server, "MODEL_FACTORY", factory)
    return models


def _config(**overrides) -> ServiceConfig:
    base = dict(key=KEY, device="cpu", replicas=1, max_batch=32,
                batch_window_ms=25.0, max_queue=512)
    base.update(overrides)
    return ServiceConfig(**base)


@contextlib.asynccontextmanager
async def _async_client(app):
    """Run the app's lifespan and yield an AsyncClient over ASGITransport."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://dlp.test") as ac:
            yield ac


# ===========================================================================
# Auth (401) + body validation (400)
# ===========================================================================

class TestAuth:

    def test_scan_missing_key_is_401(self, monkeypatch):
        _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            r = client.post("/scan", json={"text": "hello"})
        assert r.status_code == 401
        assert r.json() == {"error": "unauthorized"}

    def test_scan_wrong_key_is_401(self, monkeypatch):
        _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            r = client.post("/scan", json={"text": "hello"}, headers={"X-Worker-Key": "nope"})
        assert r.status_code == 401
        assert r.json() == {"error": "unauthorized"}

    def test_stats_requires_key(self, monkeypatch):
        _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            assert client.get("/stats").status_code == 401
            assert client.get("/stats", headers={"X-Worker-Key": "nope"}).status_code == 401
            assert client.get("/stats", headers={"X-Worker-Key": KEY}).status_code == 200

    def test_empty_configured_key_fails_closed(self, monkeypatch):
        """With no shared secret configured, even a matching-looking key fails."""
        _install_factory(monkeypatch)
        app = server.create_app(_config(key=""))
        with TestClient(app) as client:
            r = client.post("/scan", json={"text": "hi"}, headers={"X-Worker-Key": ""})
        assert r.status_code == 401

    def test_bad_body_is_400(self, monkeypatch):
        _install_factory(monkeypatch)
        h = {"X-Worker-Key": KEY}
        with TestClient(_build(monkeypatch)) as client:
            assert client.post("/scan", json={"nope": 1}, headers=h).status_code == 400          # no text
            assert client.post("/scan", json={"text": 5}, headers=h).status_code == 400           # text not str
            assert client.post("/scan", json={"text": "x", "categories": [1]}, headers=h).status_code == 400
            assert client.post("/scan", json={"text": "x", "threshold": "hi"}, headers=h).status_code == 400
            assert client.post("/scan", json={"text": "x", "max_chars": "big"}, headers=h).status_code == 400
            # A raw non-JSON body is a 400, not a 500.
            assert client.post("/scan", content=b"not json", headers=h).status_code == 400


# ===========================================================================
# Happy path + finding shape
# ===========================================================================

class TestScan:

    def test_scan_returns_finding_shape(self, monkeypatch):
        _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            r = client.post(
                "/scan",
                json={"text": "here is a SECRET value", "categories": ["email"], "threshold": 0.5},
                headers={"X-Worker-Key": KEY},
            )
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"findings", "latency_ms", "queued_ms", "batch_size"}
        assert body["batch_size"] == 1
        assert isinstance(body["latency_ms"], float)
        assert isinstance(body["queued_ms"], float)
        assert len(body["findings"]) == 1
        f = body["findings"][0]
        # Same field names/shape as ScanFinding EXCEPT "scanner" (caller stamps it).
        assert set(f) == {"category", "text", "confidence", "start", "end"}
        assert "scanner" not in f
        assert f["category"] == "email"
        assert f["text"] == "SECRET"
        # Offsets are into the (possibly truncated) text: "here is a SECRET..."
        assert f["start"] == "here is a SECRET value".index("SECRET") == 10
        assert f["end"] == 16

    def test_null_categories_uses_service_defaults(self, monkeypatch):
        models = _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            r = client.post("/scan", json={"text": "plain", "categories": None},
                            headers={"X-Worker-Key": KEY})
        assert r.status_code == 200
        # The model was called with the full 8-category production default set.
        call = models[0].real_calls()[0]
        assert tuple(call["labels"]) == ServiceConfig().default_categories

    def test_empty_categories_scans_nothing(self, monkeypatch):
        models = _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            r = client.post("/scan", json={"text": "has a SECRET", "categories": []},
                            headers={"X-Worker-Key": KEY})
        assert r.status_code == 200
        body = r.json()
        assert body["findings"] == []
        assert body["batch_size"] == 0
        # Short-circuited BEFORE the model — no real predict for this request.
        assert models[0].real_calls() == []


# ===========================================================================
# max_chars truncation (prefix cut), mirroring scan_gliner
# ===========================================================================

class TestTruncation:

    def test_max_chars_prefix_truncation_applied_before_model(self, monkeypatch):
        models = _install_factory(monkeypatch)
        # "SECRET" sits AFTER the 100-char cut, so a correct prefix-truncation
        # both shortens the text the model sees AND drops the finding.
        text = ("A" * 200) + "SECRET" + ("B" * 500)
        with TestClient(_build(monkeypatch)) as client:
            r = client.post(
                "/scan",
                json={"text": text, "categories": ["email"], "max_chars": 100},
                headers={"X-Worker-Key": KEY},
            )
        assert r.status_code == 200
        assert r.json()["findings"] == []
        call = models[0].real_calls()[0]
        assert len(call["texts"][0]) == 100

    def test_hard_cap_ceilings_requested_max_chars(self, monkeypatch):
        models = _install_factory(monkeypatch)
        app = server.create_app(_config(max_chars_cap=50))
        text = "C" * 500
        with TestClient(app) as client:
            # Request asks for 400, but the service cap is 50 -> 50 wins.
            r = client.post("/scan", json={"text": text, "categories": ["email"], "max_chars": 400},
                            headers={"X-Worker-Key": KEY})
        assert r.status_code == 200
        call = models[0].real_calls()[0]
        assert len(call["texts"][0]) == 50


# ===========================================================================
# Dynamic batching: concurrent /scan calls get coalesced (batch_size > 1)
# ===========================================================================

class TestBatching:

    async def test_concurrent_scans_are_batched(self, monkeypatch):
        _install_factory(monkeypatch)
        # One replica + a generous window so the burst coalesces into one batch.
        app = server.create_app(_config(replicas=1, batch_window_ms=100.0, max_batch=32))
        payload = {"text": "a SECRET here", "categories": ["email"], "threshold": 0.5}
        headers = {"X-Worker-Key": KEY}
        async with _async_client(app) as ac:
            responses = await asyncio.gather(
                *[ac.post("/scan", json=payload, headers=headers) for _ in range(8)]
            )
        assert all(r.status_code == 200 for r in responses)
        sizes = [r.json()["batch_size"] for r in responses]
        # The whole point: at least one request observed a multi-item GPU batch.
        assert max(sizes) > 1, sizes

        # Stats corroborate the coalescing (same manager, read after the run).
        stats = app.state.manager.stats
        assert stats.scans_total == 8
        assert stats.batches_total < 8            # fewer batches than requests
        assert stats.avg_batch_size > 1.0

    async def test_distinct_signatures_split_into_separate_groups(self, monkeypatch):
        """A mixed batch is split by (labels, threshold) — GLiNER needs uniform."""
        _install_factory(monkeypatch)
        app = server.create_app(_config(replicas=1, batch_window_ms=100.0, max_batch=32))
        headers = {"X-Worker-Key": KEY}
        async with _async_client(app) as ac:
            r_email = [ac.post("/scan", json={"text": "SECRET", "categories": ["email"]}, headers=headers)
                       for _ in range(3)]
            r_phone = [ac.post("/scan", json={"text": "SECRET", "categories": ["phone number"]}, headers=headers)
                       for _ in range(3)]
            responses = await asyncio.gather(*r_email, *r_phone)
        assert all(r.status_code == 200 for r in responses)
        # Each response's finding category reflects its own requested label,
        # proving the groups were not merged.
        cats = {r.json()["findings"][0]["category"] for r in responses}
        assert cats == {"email", "phone number"}


# ===========================================================================
# Oversubscription (503) when the replica queue fills
# ===========================================================================

class TestOversubscription:

    async def test_full_queue_returns_503(self, monkeypatch):
        gate = threading.Event()
        _install_factory(monkeypatch, blocker=gate)
        # 1 replica, batch of 1, queue of 1 -> capacity 2 (1 in-flight + 1 queued).
        app = server.create_app(_config(replicas=1, max_batch=1, batch_window_ms=0.0, max_queue=1))
        headers = {"X-Worker-Key": KEY}
        try:
            async with _async_client(app) as ac:
                async def _release():
                    await asyncio.sleep(0.15)
                    gate.set()
                asyncio.get_event_loop().create_task(_release())
                responses = await asyncio.gather(
                    *[ac.post("/scan", json={"text": "x"}, headers=headers) for _ in range(12)]
                )
        finally:
            gate.set()

        codes = [r.status_code for r in responses]
        assert 503 in codes, codes
        rejected = [r for r in responses if r.status_code == 503]
        body = rejected[0].json()
        assert body["error"] == "oversubscribed"
        assert isinstance(body["queue_depth"], int)
        assert isinstance(body["max_queue"], int)
        assert body["max_queue"] == 1
        # The rejection counter matches the number of 503s served.
        assert app.state.manager.stats.rejected_503 == len(rejected)


# ===========================================================================
# /healthz + /stats shapes
# ===========================================================================

class TestHealthAndStats:

    def test_healthz_shape_no_auth(self, monkeypatch):
        _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            r = client.get("/healthz")  # no key required
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"status", "model", "device", "replicas",
                             "queue_depth", "max_queue", "warm"}
        assert body["status"] == "ok"
        assert body["model"] == "urchade/gliner_multi_pii-v1"
        assert body["device"] == "cpu"          # no torch/gpu in the test env
        assert isinstance(body["replicas"], int) and body["replicas"] >= 1
        assert isinstance(body["queue_depth"], int)
        assert isinstance(body["max_queue"], int)
        assert isinstance(body["warm"], bool)

    def test_stats_shape(self, monkeypatch):
        _install_factory(monkeypatch)
        with TestClient(_build(monkeypatch)) as client:
            client.post("/scan", json={"text": "SECRET", "categories": ["email"]},
                        headers={"X-Worker-Key": KEY})
            r = client.get("/stats", headers={"X-Worker-Key": KEY})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"scans_total", "batches_total", "avg_batch_size",
                             "scan_p50_ms", "scan_p95_ms", "queue_depth",
                             "inflight", "rejected_503"}
        assert body["scans_total"] >= 1
        assert body["batches_total"] >= 1
        assert isinstance(body["avg_batch_size"], float)
        assert isinstance(body["scan_p50_ms"], float)
        assert isinstance(body["scan_p95_ms"], float)
        assert body["inflight"] == 0
        assert body["rejected_503"] == 0


# ===========================================================================
# Content-safety invariant: request text is NEVER logged
# ===========================================================================

class TestNoContentLogging:

    def test_request_text_never_appears_in_logs(self, monkeypatch, caplog):
        _install_factory(monkeypatch)
        secret = "HUNTER2-SSN-999-88-7777-do-not-log"
        with caplog.at_level(logging.DEBUG):  # capture everything, incl. dlp_service
            with TestClient(_build(monkeypatch)) as client:
                r = client.post(
                    "/scan",
                    json={"text": f"my ssn is {secret} SECRET", "categories": ["social security number"]},
                    headers={"X-Worker-Key": KEY},
                )
        assert r.status_code == 200
        # The scan ran (a finding came back) yet the raw text is nowhere in logs.
        assert secret not in caplog.text
        for record in caplog.records:
            assert secret not in record.getMessage()
            assert secret not in str(getattr(record, "args", ""))


# ---------------------------------------------------------------------------
# Small helper used by the TestClient-based cases above.
# ---------------------------------------------------------------------------

def _build(monkeypatch) -> "server.FastAPI":  # type: ignore[name-defined]
    """A default single-replica app (factory already installed by the caller)."""
    return server.create_app(_config())
