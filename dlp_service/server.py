############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_service/server.py: FastAPI app + dynamic GPU batcher +
# GLiNER model manager for the standalone DLP microservice.
#
# The service loads R GLiNER replicas, each with its own
# asyncio batch loop and single-thread executor (one replica
# = one CUDA stream user). /scan requests are load-balanced
# across replicas (least-queue-depth), coalesced into GPU
# batches within a small time window, grouped by uniform
# (labels, threshold) signature, and served through
# batch_predict_entities.
#
# NEVER logs or persists request text — only counts and
# latencies. Auth is a shared X-Worker-Key secret; TLS is
# terminated by a front proxy (see README). gliner + torch
# are imported LAZILY so the module imports (and its unit
# tests run) on a CPU-only Mac with neither installed.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""FastAPI DLP microservice with a dynamic GPU batcher over GLiNER."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import math
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import ServiceConfig

logger = logging.getLogger("dlp_service")

# ---------------------------------------------------------------------------
# Test / injection hook
#
# MODEL_FACTORY, when set, is called as MODEL_FACTORY(model_name, device) and
# must return an object exposing:
#     batch_predict_entities(texts, labels, threshold, batch_size)
#         -> list[ list[ {label, text, score, start, end} ] ]
# one inner list per input text.  When None (production), the manager lazily
# imports gliner and loads GLiNER.from_pretrained.  Tests inject a fake model
# so the suite runs with no gliner, no torch, and no GPU.
# ---------------------------------------------------------------------------
MODEL_FACTORY: Optional[Callable[[str, str], Any]] = None


# ---------------------------------------------------------------------------
# Device + torch thread configuration
# ---------------------------------------------------------------------------

def resolve_device(configured: str) -> str:
    """Resolve DLP_DEVICE to a concrete device string.

    "auto" -> "cuda:0" when torch reports a CUDA device, else "cpu".  torch is
    imported inside a guard so this returns "cpu" on a machine with no torch
    installed (the test/dev Mac).  An explicit "cuda:0"/"cpu" is honored as-is.
    """
    if configured and configured != "auto":
        return configured
    try:  # pragma: no cover - torch not installed in the unit environment
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def configure_torch_threads(n: int) -> None:
    """Bound intra-op torch threads per process.

    WHY: the on-host DLP scanner suffered a measured ~27x slowdown from thread
    oversubscription — many concurrent GLiNER predicts each spun up a full
    torch thread pool and thrashed the CPU.  On a dedicated GPU node the GPU
    does the matmuls, but CPU-side tokenization still benefits from a small,
    bounded thread count instead of one-pool-per-inflight-request.  A no-op
    when torch is absent.
    """
    try:  # pragma: no cover - torch not installed in the unit environment
        import torch

        torch.set_num_threads(max(1, int(n)))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Result / request records
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Resolved result handed back to a single /scan request."""
    findings: List[Dict[str, Any]]
    queued_ms: float
    latency_ms: float
    batch_size: int


@dataclass
class _Request:
    """One in-flight scan request waiting on its replica's batch loop."""
    text: str
    labels: Tuple[str, ...]
    threshold: float
    enqueue_time: float
    future: "asyncio.Future[BatchResult]"
    queued_ms: float = 0.0


class Oversubscribed(Exception):
    """Raised when the picked replica's queue is full (drives the 503)."""

    def __init__(self, queue_depth: int, max_queue: int):
        super().__init__("oversubscribed")
        self.queue_depth = queue_depth
        self.max_queue = max_queue


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    """Process-lifetime counters and a bounded latency window.

    Holds NO request text — only counts and millisecond latencies.
    """
    scans_total: int = 0
    batches_total: int = 0
    batch_size_sum: int = 0
    rejected_503: int = 0
    inflight: int = 0
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=4096))

    def record_batch(self, size: int) -> None:
        self.batches_total += 1
        self.batch_size_sum += size

    def record_scan(self, latency_ms: float) -> None:
        self.scans_total += 1
        self.latencies.append(latency_ms)

    @property
    def avg_batch_size(self) -> float:
        return (self.batch_size_sum / self.batches_total) if self.batches_total else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        data = sorted(self.latencies)
        if len(data) == 1:
            return float(data[0])
        k = (len(data) - 1) * (p / 100.0)
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return float(data[int(k)])
        return float(data[lo] + (data[hi] - data[lo]) * (k - lo))


# ---------------------------------------------------------------------------
# Entity -> finding conversion
# ---------------------------------------------------------------------------

def _entity_to_finding(ent: Any) -> Dict[str, Any]:
    """Convert a GLiNER entity to the wire finding shape.

    Same field names/shape as dlp_scanner.ScanFinding EXCEPT "scanner" (the
    caller stamps "gliner").  category = GLiNER label, confidence = model
    score, start/end = char offsets into the (possibly truncated) text.
    """
    if isinstance(ent, dict):
        get = ent.get
    else:  # object with attributes
        get = lambda k, d=None: getattr(ent, k, d)  # noqa: E731
    return {
        "category": get("label", "unknown"),
        "text": get("text", ""),
        "confidence": float(get("score", 0.0) or 0.0),
        "start": int(get("start", 0) or 0),
        "end": int(get("end", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Replica: one model + one batch loop + one single-thread executor
# ---------------------------------------------------------------------------

class Replica:
    """A single GLiNER replica with its own queue, batch loop, and executor."""

    def __init__(self, index: int, config: ServiceConfig, device: str, stats: Stats):
        self.index = index
        self.config = config
        self.device = device
        self.stats = stats
        self.queue: "asyncio.Queue[_Request]" = asyncio.Queue(maxsize=config.per_replica_queue)
        # One thread => one CUDA stream user for this replica; torch releases
        # the GIL during inference so the event loop keeps serving other work.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"dlp-replica-{index}")
        self.model: Any = None
        self.warm = False
        self._load_lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def max_queue(self) -> int:
        return self.queue.maxsize

    def enqueue(self, req: _Request) -> None:
        """Enqueue or raise asyncio.QueueFull (the oversubscription signal)."""
        self.queue.put_nowait(req)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"dlp-replica-loop-{self.index}")

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.executor.shutdown(wait=False)

    # -- model load (lazy + guarded) ---------------------------------------

    def _build_model(self) -> Any:
        """Construct the model.  Runs inside the replica's thread executor."""
        if MODEL_FACTORY is not None:
            return MODEL_FACTORY(self.config.model, self.device)
        # Lazy import so the module loads without gliner/torch installed.
        if self.config.hf_home:
            os.environ.setdefault("HF_HOME", self.config.hf_home)
        from gliner import GLiNER

        model = GLiNER.from_pretrained(self.config.model)
        try:
            model.to(self.device)
        except Exception:
            logger.warning("dlp_service_to_device_failed replica=%d device=%s", self.index, self.device)
        return model

    async def ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        async with self._load_lock:
            if self.model is not None:
                return self.model
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(self.executor, self._build_model)
            self.model = model
            self.warm = True
            logger.info("dlp_service_replica_loaded replica=%d device=%s", self.index, self.device)
            return model

    async def warmup(self) -> None:
        """Load the model and run one dummy predict to trigger kernel compile."""
        try:
            await self.ensure_model()
            loop = asyncio.get_event_loop()
            labels = list(self.config.default_categories)
            await loop.run_in_executor(self.executor, self._predict, ["warmup"], labels, 0.5)
            logger.info("dlp_service_replica_warm replica=%d", self.index)
        except Exception:
            logger.warning("dlp_service_warmup_failed replica=%d", self.index, exc_info=False)

    def _predict(self, texts: List[str], labels: List[str], threshold: float) -> Any:
        """Blocking model call.  Runs inside the replica's thread executor."""
        return self.model.batch_predict_entities(
            texts, list(labels), threshold=threshold, batch_size=len(texts)
        )

    # -- batch loop ---------------------------------------------------------

    async def _run(self) -> None:
        """Collect a batch within the window, then process it."""
        while not self._stopped:
            try:
                first = await self.queue.get()
            except asyncio.CancelledError:
                raise
            batch: List[_Request] = [first]
            deadline = time.monotonic() + self.config.batch_window_s
            while len(batch) < self.config.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    raise
                batch.append(nxt)
            try:
                await self._process_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A defect in batch handling must never kill the loop.
                logger.warning("dlp_service_batch_loop_error replica=%d", self.index, exc_info=False)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError("batch processing failed"))

    async def _process_batch(self, batch: List[_Request]) -> None:
        loop = asyncio.get_event_loop()
        try:
            await self.ensure_model()
        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return

        # GLiNER's batch_predict_entities needs uniform labels + threshold, so
        # split a mixed batch into per-(labels, threshold) groups; each group
        # is one model call = one "batch" for stats.
        groups: "defaultdict[Tuple[frozenset, float], List[_Request]]" = defaultdict(list)
        for req in batch:
            groups[(frozenset(req.labels), req.threshold)].append(req)

        batch_start = time.monotonic()
        for (_sig, threshold), reqs in groups.items():
            labels = list(reqs[0].labels)
            texts = [r.text for r in reqs]
            for r in reqs:
                r.queued_ms = (batch_start - r.enqueue_time) * 1000.0

            try:
                results = await loop.run_in_executor(self.executor, self._predict, texts, labels, threshold)
            except Exception as exc:
                logger.error(
                    "dlp_service_predict_failed replica=%d error=%s size=%d",
                    self.index, type(exc).__name__, len(reqs),
                )
                for r in reqs:
                    if not r.future.done():
                        r.future.set_exception(exc)
                continue

            self.stats.record_batch(len(reqs))
            logger.debug("dlp_service_batch replica=%d size=%d", self.index, len(reqs))
            now = time.monotonic()
            results = results or []
            for i, r in enumerate(reqs):
                if r.future.done():
                    continue
                if i >= len(results):
                    r.future.set_exception(RuntimeError("model returned fewer results than inputs"))
                    continue
                findings = [_entity_to_finding(e) for e in (results[i] or [])]
                latency_ms = (now - r.enqueue_time) * 1000.0
                self.stats.record_scan(latency_ms)
                r.future.set_result(
                    BatchResult(
                        findings=findings,
                        queued_ms=r.queued_ms,
                        latency_ms=latency_ms,
                        batch_size=len(reqs),
                    )
                )


# ---------------------------------------------------------------------------
# ModelManager: R replicas + load balancing + submission
# ---------------------------------------------------------------------------

class ModelManager:
    """Holds the replicas and load-balances /scan submissions across them."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.stats = Stats()
        self.replicas: List[Replica] = [
            Replica(i, config, self.device, self.stats) for i in range(config.replica_count)
        ]
        self._rr = 0

    @property
    def warm(self) -> bool:
        return all(r.warm for r in self.replicas)

    def queue_depth(self) -> int:
        return sum(r.queue_depth for r in self.replicas)

    @property
    def max_queue_total(self) -> int:
        return sum(r.max_queue for r in self.replicas)

    def pick_replica(self) -> Replica:
        """Least-queue-depth with round-robin tie-break."""
        min_depth = min(r.queue_depth for r in self.replicas)
        candidates = [r for r in self.replicas if r.queue_depth == min_depth]
        self._rr = (self._rr + 1) % len(candidates)
        return candidates[self._rr]

    async def start(self) -> None:
        configure_torch_threads(self.config.torch_threads)
        for r in self.replicas:
            await r.start()

    async def warmup(self) -> None:
        await asyncio.gather(*[r.warmup() for r in self.replicas], return_exceptions=True)

    async def stop(self) -> None:
        for r in self.replicas:
            await r.stop()

    async def submit(self, text: str, labels: List[str], threshold: float) -> BatchResult:
        """Enqueue a scan and await its batched result, or raise Oversubscribed."""
        loop = asyncio.get_event_loop()
        future: "asyncio.Future[BatchResult]" = loop.create_future()
        req = _Request(
            text=text,
            labels=tuple(labels),
            threshold=threshold,
            enqueue_time=time.monotonic(),
            future=future,
        )
        replica = self.pick_replica()
        try:
            replica.enqueue(req)
        except asyncio.QueueFull:
            self.stats.rejected_503 += 1
            raise Oversubscribed(queue_depth=replica.queue_depth, max_queue=replica.max_queue)

        self.stats.inflight += 1
        try:
            return await future
        finally:
            self.stats.inflight -= 1


# ---------------------------------------------------------------------------
# Request validation helpers
# ---------------------------------------------------------------------------

def _authorized(request: Request, config: ServiceConfig) -> bool:
    """Constant-time shared-key check.  Empty configured key => fail closed."""
    if not config.key:
        return False
    provided = request.headers.get("X-Worker-Key")
    if provided is None:
        return False
    return hmac.compare_digest(provided, config.key)


def _truncate(text: str, max_chars: Optional[int], cap: int) -> str:
    """Prefix-truncate text, mirroring scan_gliner and enforcing the hard cap.

    ``cap`` (DLP_MAX_CHARS_CAP) is both the default and the hard ceiling: a
    per-request ``max_chars`` only ever LOWERS the effective limit, never
    raises it above the cap that protects the GPU from a pathological request.
    """
    effective: Optional[int] = cap if cap and cap > 0 else None
    if isinstance(max_chars, int) and not isinstance(max_chars, bool) and max_chars > 0:
        effective = min(effective, max_chars) if effective is not None else max_chars
    if effective is not None and len(text) > effective:
        return text[:effective]
    return text


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Build the FastAPI app with a lifespan-managed ModelManager."""
    config = config or ServiceConfig.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = ModelManager(config)
        app.state.config = config
        app.state.manager = manager
        await manager.start()
        # Warmup runs in the background: /healthz reports warm=false until the
        # replicas finish loading; the batch loop also lazy-loads on first use.
        warm_task = asyncio.create_task(manager.warmup(), name="dlp-warmup")
        app.state.warm_task = warm_task
        try:
            yield
        finally:
            warm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await warm_task
            await manager.stop()

    app = FastAPI(title="MindRouter DLP GPU Service", lifespan=lifespan)

    @app.post("/scan")
    async def scan(request: Request):  # noqa: ANN202
        cfg: ServiceConfig = request.app.state.config
        manager: ModelManager = request.app.state.manager

        if not _authorized(request, cfg):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json body"}, status_code=400)

        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

        text = body.get("text")
        if not isinstance(text, str):
            return JSONResponse({"error": "text must be a string"}, status_code=400)

        categories = body.get("categories", None)
        if categories is not None:
            if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
                return JSONResponse(
                    {"error": "categories must be a list of strings or null"}, status_code=400
                )

        threshold = body.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            return JSONResponse({"error": "threshold must be a number"}, status_code=400)
        threshold = float(threshold)

        max_chars = body.get("max_chars", None)
        if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int)):
            return JSONResponse({"error": "max_chars must be an integer or null"}, status_code=400)

        # Category resolution mirrors scan_gliner: null => defaults; an
        # explicitly-empty list means "scan nothing" (honored, not defaulted).
        if categories is None:
            labels = list(cfg.default_categories)
        elif len(categories) == 0:
            return JSONResponse(
                {"findings": [], "latency_ms": 0.0, "queued_ms": 0.0, "batch_size": 0}
            )
        else:
            labels = list(categories)

        text = _truncate(text, max_chars, cfg.max_chars_cap)

        try:
            result = await manager.submit(text, labels, threshold)
        except Oversubscribed as ov:
            return JSONResponse(
                {"error": "oversubscribed", "queue_depth": ov.queue_depth, "max_queue": ov.max_queue},
                status_code=503,
            )
        except Exception:
            # No request text in the log — only the event name.
            logger.error("dlp_service_scan_failed", exc_info=False)
            return JSONResponse({"error": "scan_failed"}, status_code=500)

        return JSONResponse(
            {
                "findings": result.findings,
                "latency_ms": round(result.latency_ms, 3),
                "queued_ms": round(result.queued_ms, 3),
                "batch_size": result.batch_size,
            }
        )

    @app.get("/healthz")
    async def healthz(request: Request):  # noqa: ANN202
        cfg: ServiceConfig = request.app.state.config
        manager: ModelManager = request.app.state.manager
        return {
            "status": "ok",
            "model": cfg.model,
            "device": manager.device,
            "replicas": len(manager.replicas),
            "queue_depth": manager.queue_depth(),
            "max_queue": manager.max_queue_total,
            "warm": manager.warm,
        }

    @app.get("/stats")
    async def stats(request: Request):  # noqa: ANN202
        cfg: ServiceConfig = request.app.state.config
        manager: ModelManager = request.app.state.manager
        if not _authorized(request, cfg):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        s = manager.stats
        return {
            "scans_total": s.scans_total,
            "batches_total": s.batches_total,
            "avg_batch_size": round(s.avg_batch_size, 3),
            "scan_p50_ms": round(s.percentile(50), 3),
            "scan_p95_ms": round(s.percentile(95), 3),
            "queue_depth": manager.queue_depth(),
            "inflight": s.inflight,
            "rejected_503": s.rejected_503,
        }

    return app
