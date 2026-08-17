"""Unit tests for ws6 DoS / validation hardening.

Covers:
  - CanonicalChatRequest.n clamp (canonical_schemas)
  - CanonicalImageRequest.model format validator (canonical_schemas)
  - count_request_text_chars + _tiktoken_estimate_sync char cap (inference)
  - async _estimate_input_tokens offload + memoization (inference)
  - voice_api._validate_model_format (voice_api)

canonical_schemas, inference and voice_api all import cleanly in the test
environment, so they are imported directly and get_settings is monkeypatched
per-module (each module holds its own reference to get_settings).
"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import backend.app.api.voice_api as voice_api
import backend.app.core.canonical_schemas as cs
import backend.app.services.inference as inf
import backend.app.settings as settings_mod
from fastapi import HTTPException


def _settings_obj(**overrides):
    base = dict(
        max_completions_n=8,
        tokenize_max_input_chars=2_000_000,
        default_tokenizer="cl100k_base",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def patch_settings(monkeypatch):
    """Patch get_settings in every module namespace that reads it directly."""
    def _apply(**overrides):
        obj = _settings_obj(**overrides)
        monkeypatch.setattr(settings_mod, "get_settings", lambda: obj)
        monkeypatch.setattr(inf, "get_settings", lambda: obj)
        return obj
    return _apply


def _chat(**kw):
    kw.setdefault("model", "test-model")
    kw.setdefault("messages", [cs.CanonicalMessage(role=cs.MessageRole.USER, content="hi")])
    return cs.CanonicalChatRequest(**kw)


# ---------------------------------------------------------------------------
# n clamp
# ---------------------------------------------------------------------------

def test_n_clamped_to_max(patch_settings):
    patch_settings(max_completions_n=8)
    assert _chat(n=50).n == 8


def test_n_within_range_unchanged(patch_settings):
    patch_settings(max_completions_n=8)
    assert _chat(n=3).n == 3


def test_n_default_is_one(patch_settings):
    patch_settings(max_completions_n=8)
    assert _chat().n == 1


def test_n_below_one_becomes_one(patch_settings):
    patch_settings(max_completions_n=8)
    assert _chat(n=0).n == 1
    assert _chat(n=-4).n == 1


def test_n_respects_configured_bound(patch_settings):
    patch_settings(max_completions_n=2)
    assert _chat(n=9).n == 2


def test_n_falls_back_to_default_when_setting_absent(monkeypatch):
    # Settings object without max_completions_n (concurrent settings work not
    # yet landed) must not raise — getattr fallback clamps to the contract
    # default of 8.
    monkeypatch.setattr(settings_mod, "get_settings", lambda: SimpleNamespace())
    assert _chat(n=100).n == 8


# ---------------------------------------------------------------------------
# image model format validator
# ---------------------------------------------------------------------------

def test_image_model_hf_style_allowed():
    req = cs.CanonicalImageRequest(model="black-forest-labs/FLUX.2-dev", prompt="a cat")
    assert req.model == "black-forest-labs/FLUX.2-dev"


def test_image_model_simple_name_allowed():
    assert cs.CanonicalImageRequest(model="flux.2-klein", prompt="x").model == "flux.2-klein"


@pytest.mark.parametrize("bad", [
    "<script>alert(1)</script>",
    'model" onload="x',
    "a b",              # space
    "a&b",              # ampersand
    "model'inject",     # quote
    "x" * 129,          # too long
    "",                 # empty
])
def test_image_model_rejects_metacharacters(bad):
    with pytest.raises(ValidationError):
        cs.CanonicalImageRequest(model=bad, prompt="x")


# ---------------------------------------------------------------------------
# count_request_text_chars
# ---------------------------------------------------------------------------

def test_count_request_text_chars_str_content():
    req = _chat(messages=[cs.CanonicalMessage(role=cs.MessageRole.USER, content="hello")])
    assert inf.count_request_text_chars(req) == 5


def test_count_request_text_chars_block_content():
    blocks = [cs.TextContent(text="ab"), cs.TextContent(text="cde")]
    req = _chat(messages=[cs.CanonicalMessage(role=cs.MessageRole.USER, content=blocks)])
    assert inf.count_request_text_chars(req) == 5


# ---------------------------------------------------------------------------
# _tiktoken_estimate_sync char cap
# ---------------------------------------------------------------------------

class _FakeEncoder:
    """One token per character — makes token counts equal char counts."""
    def encode(self, text):
        return list(text)


def test_estimate_sync_caps_encoded_chars():
    big = "x" * 10_000
    req = _chat(messages=[cs.CanonicalMessage(role=cs.MessageRole.USER, content=big)])
    enc = _FakeEncoder()
    # With a 100-char cap the encoded contribution is bounded at 100 tokens,
    # plus the fixed +4 per-message overhead.
    capped = inf._tiktoken_estimate_sync(req, enc, max_chars=100)
    assert capped == 100 + 4


def test_estimate_sync_uncapped_counts_all():
    text = "x" * 50
    req = _chat(messages=[cs.CanonicalMessage(role=cs.MessageRole.USER, content=text)])
    enc = _FakeEncoder()
    assert inf._tiktoken_estimate_sync(req, enc, max_chars=1_000_000) == 50 + 4


def test_estimate_sync_cap_shared_across_messages():
    msgs = [
        cs.CanonicalMessage(role=cs.MessageRole.USER, content="a" * 80),
        cs.CanonicalMessage(role=cs.MessageRole.USER, content="b" * 80),
    ]
    req = _chat(messages=msgs)
    enc = _FakeEncoder()
    # Budget of 100 is consumed across both messages; per-message overhead
    # (+4 * 2) is always added.
    assert inf._tiktoken_estimate_sync(req, enc, max_chars=100) == 100 + 8


# ---------------------------------------------------------------------------
# async _estimate_input_tokens: offload + memoization
# ---------------------------------------------------------------------------

async def test_estimate_input_tokens_async_and_memoized(patch_settings, monkeypatch):
    patch_settings(tokenize_max_input_chars=2_000_000)
    monkeypatch.setattr(inf, "_ensure_encoder", lambda: _FakeEncoder())

    req = _chat(messages=[cs.CanonicalMessage(role=cs.MessageRole.USER, content="abcd")])

    first = await inf._estimate_input_tokens(req)
    # Second call must hit the memo, not recompute.
    second = await inf._estimate_input_tokens(req)
    assert first == 4 + 4  # 4 content chars + per-message overhead
    assert second == first
    assert getattr(req, "_est_input_tokens") == first


async def test_estimate_input_tokens_applies_char_cap(patch_settings, monkeypatch):
    patch_settings(tokenize_max_input_chars=10)
    monkeypatch.setattr(inf, "_ensure_encoder", lambda: _FakeEncoder())

    req = _chat(messages=[cs.CanonicalMessage(role=cs.MessageRole.USER, content="z" * 1000)])
    result = await inf._estimate_input_tokens(req)
    assert result == 10 + 4  # capped at 10 chars + overhead


# ---------------------------------------------------------------------------
# voice_api._validate_model_format
# ---------------------------------------------------------------------------

def test_voice_model_valid_names_pass():
    for name in ("kokoro", "whisper-large-v3-turbo", "gpt-4o-transcribe", "org/model.v2"):
        voice_api._validate_model_format(name)  # must not raise


@pytest.mark.parametrize("bad", [
    "<script>alert(1)</script>",
    'kokoro" onerror="x',
    "has space",
    "a&b",
    "x" * 129,
    "",
])
def test_voice_model_rejects_metacharacters(bad):
    with pytest.raises(HTTPException) as ei:
        voice_api._validate_model_format(bad)
    assert ei.value.status_code == 400
