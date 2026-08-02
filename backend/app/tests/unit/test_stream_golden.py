"""Golden-stream tests for the vLLM SSE translation + framing pipeline.

The fixture ``fixtures/golden_vllm_stream.json`` was captured from the
pre-orjson pipeline (Pydantic CanonicalStreamChunk +
``model_dump_json(exclude_none=True, by_alias=True)`` framing).  These
tests pin the refactored dict pipeline to that exact wire behavior:
semantically identical JSON per event and identical event partitioning
with coalescing disabled.

Scenarios covered: plain content deltas, <think> block extraction
(including tags split across SSE events and byte chunks), Qwen3.5
reasoning_content passthrough and promotion, tool-call delta fragments,
usage-with-empty-choices final chunk, finish_reason, malformed frame
silent drop, and [DONE].

Also tests the StreamCoalescer flush boundaries (first-event, N-events,
T-ms, force/finish, error-path drain, disabled passthrough).
"""

import json
import os

import orjson
import pytest

from backend.app.core.canonical_schemas import (
    CanonicalStreamChoice,
    CanonicalStreamChunk,
    CanonicalStreamDelta,
    MessageRole,
    UsageInfo,
)
from backend.app.core.stream_coalesce import StreamCoalescer
from backend.app.core.translators.ollama_out import OllamaOutTranslator
from backend.app.core.translators.vllm_out import VLLMOutTranslator

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "golden_vllm_stream.json"
)

with open(FIXTURE_PATH) as f:
    _GOLDEN = json.load(f)

SCENARIOS = {sc["name"]: sc for sc in _GOLDEN["scenarios"]}


async def _aiter(str_chunks):
    for c in str_chunks:
        yield c.encode()


async def _consume(chunk_aiter, include_usage):
    """Mirror stream_chat_completion's account → frame consumer loop.

    Returns (framed_events, full_content, last_finish_reason, real_usage)
    with coalescing disabled — one SSE block per event, exactly as the
    per-chunk write path emits them.
    """
    events = []
    full_content = ""
    last_finish_reason = None
    real_usage = None

    async for chunk in chunk_aiter:
        # Ollama's translator still yields Pydantic chunks; normalize the
        # same way stream_chat_completion does.
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump(exclude_none=True, by_alias=True)

        usage = chunk.get("usage")
        choices = chunk.get("choices") or []

        if usage is not None:
            real_usage = usage
            if not choices and not include_usage:
                continue

        events.append(b"data: " + orjson.dumps(chunk) + b"\n\n")

        for choice in choices:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                full_content += delta["content"]
            if choice.get("finish_reason"):
                last_finish_reason = choice["finish_reason"]

    events.append(b"data: [DONE]\n\n")
    return events, full_content, last_finish_reason, real_usage


def _parse_event(raw: bytes):
    """Parse one framed SSE event back to its payload (dict or '[DONE]')."""
    assert raw.startswith(b"data: ") and raw.endswith(b"\n\n")
    payload = raw[len(b"data: "):-2]
    if payload == b"[DONE]":
        return "[DONE]"
    return json.loads(payload)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.asyncio
async def test_golden_scenario(name):
    """Refactored pipeline emits the exact pre-refactor wire events."""
    scenario = SCENARIOS[name]
    events, full_content, finish_reason, real_usage = await _consume(
        VLLMOutTranslator.translate_chat_stream(
            _aiter(scenario["input_chunks"]),
            "req-golden",
            "test-model",
            thinking_enabled=scenario["thinking_enabled"],
        ),
        include_usage=scenario["include_usage"],
    )

    parsed = [_parse_event(e) for e in events]

    # Identical event partitioning: same number of events, in order,
    # each in its own SSE block (coalescing disabled)
    assert len(parsed) == len(scenario["expected_events"])
    # Semantically identical JSON per event (key order may differ)
    for got, expected in zip(parsed, scenario["expected_events"]):
        assert got == expected

    # Accounting side-effects the quota path derives from the chunks
    assert full_content == scenario["expected_full_content"]
    assert finish_reason == scenario["expected_finish_reason"]
    assert real_usage == scenario["expected_real_usage"]


@pytest.mark.asyncio
async def test_pydantic_chunk_normalization_orjson_safe():
    """The Ollama normalize path (model_dump with a MessageRole enum)
    must serialize through orjson identically to model_dump_json."""
    chunk = CanonicalStreamChunk(
        id="chatcmpl-x",
        created=1700000000,
        model="m",
        choices=[
            CanonicalStreamChoice(
                index=0,
                delta=CanonicalStreamDelta(
                    role=MessageRole.ASSISTANT, content="hi", reasoning="think"
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    dumped = chunk.model_dump(exclude_none=True, by_alias=True)
    via_orjson = json.loads(orjson.dumps(dumped))
    via_pydantic = json.loads(chunk.model_dump_json(exclude_none=True, by_alias=True))
    assert via_orjson == via_pydantic
    assert via_orjson["choices"][0]["delta"]["role"] == "assistant"
    assert via_orjson["choices"][0]["delta"]["reasoning_content"] == "think"


@pytest.mark.asyncio
async def test_ollama_final_chunk_usage_captured_and_forwarded():
    """Ollama's final chunk carries usage AND choices: real usage must be
    captured for quota accounting while the chunk is still forwarded to
    the client (the suppress path only applies to empty-choices chunks)."""
    ndjson = [
        json.dumps({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hi"},
            "done": False,
        }) + "\n",
        json.dumps({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "!"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 2,
        }) + "\n",
    ]
    events, full_content, finish_reason, real_usage = await _consume(
        OllamaOutTranslator.translate_chat_stream(
            _aiter(ndjson), "req-ollama", "llama3.2"
        ),
        include_usage=False,
    )
    parsed = [_parse_event(e) for e in events]
    # Final chunk forwarded (not suppressed) and [DONE] appended
    assert len(parsed) == 3
    assert parsed[-1] == "[DONE]"
    assert parsed[1]["usage"]["prompt_tokens"] == 5
    assert full_content == "Hi!"
    assert finish_reason == "stop"
    # The pre-fix consumer missed this usage (chunk has non-empty choices)
    assert real_usage == {
        "prompt_tokens": 5, "completion_tokens": 2,
        "total_tokens": 7, "is_estimated": False,
    }


# ── StreamCoalescer flush boundaries ────────────────────────


class TestStreamCoalescer:
    def test_first_event_flushes_immediately(self):
        c = StreamCoalescer(8, 50)
        assert c.add(b"e1") == b"e1"  # TTFT never delayed

    def test_n_events_flush(self):
        c = StreamCoalescer(3, 10_000)
        assert c.add(b"e1") == b"e1"
        assert c.add(b"e2") is None
        assert c.add(b"e3") is None
        assert c.add(b"e4") == b"e2e3e4"
        assert c.add(b"e5") is None

    def test_time_flush(self, monkeypatch):
        from backend.app.core import stream_coalesce

        now = [100.0]
        monkeypatch.setattr(stream_coalesce.time, "monotonic", lambda: now[0])
        c = StreamCoalescer(100, 50)
        assert c.add(b"e1") == b"e1"
        assert c.add(b"e2") is None
        now[0] += 0.049
        assert c.add(b"e3") is None
        now[0] += 0.002  # 51ms since last flush
        assert c.add(b"e4") == b"e2e3e4"

    def test_force_flush_drains_buffer_in_order(self):
        c = StreamCoalescer(8, 10_000)
        assert c.add(b"e1") == b"e1"
        assert c.add(b"e2") is None
        # finish_reason/usage/[DONE] events force a flush
        assert c.add(b"data: [DONE]\n\n", force=True) == b"e2data: [DONE]\n\n"

    def test_error_path_flush_drains(self):
        c = StreamCoalescer(8, 10_000)
        assert c.add(b"e1") == b"e1"
        assert c.add(b"e2") is None
        assert c.flush() == b"e2"
        assert c.flush() is None

    def test_disabled_passthrough(self):
        # 0/1 events or 0 ms disables coalescing entirely
        for events, ms in [(0, 50), (1, 50), (8, 0)]:
            c = StreamCoalescer(events, ms)
            assert not c.enabled
            assert c.add(b"e1") == b"e1"
            assert c.add(b"e2") == b"e2"
            assert c.add(b"e3", force=True) == b"e3"
            assert c.flush() is None

    def test_force_on_first_event_single_write(self):
        c = StreamCoalescer(8, 50)
        assert c.add(b"only", force=True) == b"only"
