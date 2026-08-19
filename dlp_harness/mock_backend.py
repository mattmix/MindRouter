############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/mock_backend.py: Deterministic fake vLLM
# backend + admin-API registration client.
#
# Serving a mock backend makes DLP measurable in isolation:
# the harness controls BOTH sides of the scanned text (a
# reply marker lets each request script its own response, so
# PII can be planted in responses as well as prompts), and
# inference latency collapses to a configurable constant so
# DLP overhead dominates the measurement instead of GPU
# noise.
#
# Contract mirrored from core/telemetry/adapters/vllm.py and
# admin_api.py:
#   GET /health            -> must return 200 (health probe)
#   GET /v1/models         -> OpenAI list shape (discovery)
#   POST /v1/chat/completions  (stream + non-stream)
# Streaming chunks deliberately OMIT the "id" field: the
# gateway translator (vllm_out.py) then stamps the DB row's
# request_uuid into every chunk, giving the harness an exact
# client-side correlation key for streams too.
#
############################################################

"""Fake OpenAI-compatible backend for DLP end-to-end runs."""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Optional

import httpx

REPLY_MARKER = "<<<REPLY>>>"
# Base64 variant: the prompt carries only opaque base64 (invisible to every
# scanner), the decoded plaintext comes back in the RESPONSE — this isolates
# response-side DLP detection from prompt-side detection.
REPLY_B64_MARKER = "<<<REPLY_B64>>>"
DEFAULT_MODEL = "dlp-mock"

# Neutral response used when the request does not script one. Deliberately
# free of anything the scanners could flag.
DEFAULT_REPLY = (
    "Here is a summary of the topic you asked about. The main points are "
    "clear and there are no outstanding concerns. Let me know if you would "
    "like more detail on any part of this answer."
)


@dataclass
class MockConfig:
    model: str = DEFAULT_MODEL
    ttft_ms: float = 0.0          # delay before first byte (stream) / body (non-stream)
    latency_ms: float = 0.0       # additional whole-request delay
    jitter_ms: float = 0.0        # uniform [0, jitter] extra
    stream_chunks: int = 6        # content split across this many SSE chunks
    chunk_delay_ms: float = 0.0   # inter-chunk delay
    seed: int = 1234


def _extract_reply(body: dict) -> str:
    """Reply scripted by the caller via REPLY_MARKER, else the neutral text."""
    messages = body.get("messages") or []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict) and p.get("type") == "text")
        if isinstance(content, str) and REPLY_B64_MARKER in content:
            import base64
            payload = content.split(REPLY_B64_MARKER, 1)[1].strip()
            try:
                return base64.b64decode(payload).decode("utf-8") or DEFAULT_REPLY
            except Exception:
                return DEFAULT_REPLY
        if isinstance(content, str) and REPLY_MARKER in content:
            return content.split(REPLY_MARKER, 1)[1].strip() or DEFAULT_REPLY
        break
    return DEFAULT_REPLY


def _usage(body: dict, reply: str) -> dict:
    prompt_chars = len(json.dumps(body.get("messages", [])))
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": max(1, len(reply) // 4),
        "total_tokens": max(2, prompt_chars // 4 + len(reply) // 4),
    }


def build_app(cfg: MockConfig):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    app = FastAPI(title="dlp-harness-mock-backend", docs_url=None, redoc_url=None)
    rng = random.Random(cfg.seed)
    state = {"requests": 0, "streamed": 0}

    async def _delay(extra_ms: float = 0.0) -> None:
        total = cfg.latency_ms + extra_ms
        if cfg.jitter_ms > 0:
            total += rng.uniform(0, cfg.jitter_ms)
        if total > 0:
            await asyncio.sleep(total / 1000.0)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/version")
    async def version():
        return {"version": "0.0.0-dlp-harness-mock"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            f"# mock backend\nmock_requests_total {state['requests']}\n")

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": cfg.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dlp-harness",
                "max_model_len": 32768,
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        state["requests"] += 1
        body = await request.json()
        if body.get("model") not in (cfg.model, None):
            return JSONResponse(status_code=404,
                                content={"error": {"message": "model not found"}})
        reply = _extract_reply(body)
        usage = _usage(body, reply)
        created = int(time.time())

        if body.get("stream"):
            state["streamed"] += 1
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def gen():
                await _delay(cfg.ttft_ms)
                base = {"object": "chat.completion.chunk", "created": created,
                        "model": cfg.model}
                first = dict(base, choices=[{"index": 0,
                                             "delta": {"role": "assistant"},
                                             "finish_reason": None}])
                yield f"data: {json.dumps(first)}\n\n"
                n = max(1, cfg.stream_chunks)
                step = max(1, (len(reply) + n - 1) // n)
                for i in range(0, len(reply), step):
                    if cfg.chunk_delay_ms > 0:
                        await asyncio.sleep(cfg.chunk_delay_ms / 1000.0)
                    chunk = dict(base, choices=[{"index": 0,
                                                 "delta": {"content": reply[i:i + step]},
                                                 "finish_reason": None}])
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = dict(base, choices=[{"index": 0, "delta": {},
                                             "finish_reason": "stop"}])
                yield f"data: {json.dumps(final)}\n\n"
                if include_usage:
                    yield f"data: {json.dumps(dict(base, choices=[], usage=usage))}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        await _delay(cfg.ttft_ms)
        return {
            "id": f"chatcmpl-mock-{state['requests']}",
            "object": "chat.completion",
            "created": created,
            "model": cfg.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    return app


# ---------------------------------------------------------------------------
# Registration client (talks to the MindRouter admin API)
# ---------------------------------------------------------------------------

def _headers(admin_key: str) -> dict:
    return {"Authorization": f"Bearer {admin_key}"}

def register_mock_backend(
    gateway_url: str,
    admin_key: str,
    backend_url: str,
    name: str = "dlp-harness-mock",
    max_concurrent: int = 64,
    timeout: float = 30.0,
) -> dict:
    """Register (upsert) the mock backend, force discovery + health probe.

    Returns the backend record. Registration itself may mark the backend
    HEALTHY before any probe succeeds (discovery quirk), so callers must
    still wait_until_routable() before trusting it.
    """
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            f"{gateway_url}/api/admin/backends/register",
            headers=_headers(admin_key),
            json={"name": name, "url": backend_url, "engine": "vllm",
                  "max_concurrent": max_concurrent, "upsert": True},
        )
        r.raise_for_status()
        backend = r.json()
        bid = backend["id"]
        # refresh = re-run discovery (model rows); enable = immediate health probe
        client.post(f"{gateway_url}/api/admin/backends/{bid}/refresh",
                    headers=_headers(admin_key)).raise_for_status()
        client.post(f"{gateway_url}/api/admin/backends/{bid}/enable",
                    headers=_headers(admin_key)).raise_for_status()
        return backend


def disable_backend(gateway_url: str, admin_key: str, backend_id: int,
                    timeout: float = 30.0) -> None:
    with httpx.Client(timeout=timeout) as client:
        client.post(f"{gateway_url}/api/admin/backends/{backend_id}/disable",
                    headers=_headers(admin_key)).raise_for_status()


def wait_until_routable(
    gateway_url: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 90.0,
    consecutive: int = 4,
) -> bool:
    """Probe with tiny chat requests until the model routes reliably.

    Needs several consecutive successes because each uvicorn worker has its
    own registry and picks up a new backend on its own poll cycle (<=30s).
    """
    deadline = time.monotonic() + timeout
    streak = 0
    with httpx.Client(timeout=15.0) as client:
        while time.monotonic() < deadline:
            try:
                r = client.post(
                    f"{gateway_url}/v1/chat/completions",
                    headers=_headers(api_key),
                    json={"model": model, "max_tokens": 8,
                          "messages": [{"role": "user", "content": "routable probe"}]},
                )
                streak = streak + 1 if r.status_code == 200 else 0
            except httpx.HTTPError:
                streak = 0
            if streak >= consecutive:
                return True
            time.sleep(1.0)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DLP harness mock backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ttft-ms", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--stream-chunks", type=int, default=6)
    parser.add_argument("--chunk-delay-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    import uvicorn
    cfg = MockConfig(model=args.model, ttft_ms=args.ttft_ms,
                     latency_ms=args.latency_ms, jitter_ms=args.jitter_ms,
                     stream_chunks=args.stream_chunks,
                     chunk_delay_ms=args.chunk_delay_ms, seed=args.seed)
    uvicorn.run(build_app(cfg), host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
