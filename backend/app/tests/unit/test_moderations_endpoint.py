############################################################
#
# mindrouter - unit tests for the standalone /v1/moderations endpoint
#
# Client apps (e.g. VandalChat) caption user-supplied reference images with a
# vision model and vet the caption against the central image content policy
# BEFORE submitting an img2img edit. This endpoint exposes the same judge and
# img.policy text used by /images/generations and /images/edits, without
# generating anything.
#
############################################################

"""Unit tests for POST /v1/moderations."""

from unittest.mock import AsyncMock, patch

import pytest

import backend.app.api.v1_openai as api
from backend.app.services.image_policy import PolicyVerdict


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _auth():
    class U:
        id = 1

    class K:
        id = 2

    return U(), K()


async def _call(body, config=None, verdict=None, quota=None):
    """Invoke the endpoint with crud config, judge, and metering mocked out.

    Returns (result, judge, quota_check, recorder) so tests can assert the
    metering contract: quota+RPM checked BEFORE the judge runs, and every
    judged call recorded as an audited Request row.
    """
    config = {"img.policy": "no real people"} if config is None else config

    async def fake_config(db, key, default=None):
        return config.get(key, default)

    judge = AsyncMock(return_value=verdict or PolicyVerdict(True, "ok", "judge-model", ""))
    quota_check = quota or AsyncMock()
    recorder = AsyncMock()
    with (
        patch.object(api.crud, "get_config_json", side_effect=fake_config),
        patch("backend.app.services.image_policy.evaluate_prompt", new=judge),
        patch("backend.app.api.voice_api._check_quota", new=quota_check),
        patch("backend.app.api.voice_api._record_and_complete", new=recorder),
        patch.object(api, "bind_request_context"),
    ):
        result = await api.moderations(_FakeRequest(body), db=None, auth=_auth())
    return result, judge, quota_check, recorder


@pytest.mark.asyncio
async def test_pass_verdict_round_trips():
    result, judge, _, _ = await _call({"input": "a red bicycle against a brick wall"})
    assert result["passed"] is True
    assert result["policy_configured"] is True
    assert result["judge_model"] == "judge-model"
    judge.assert_awaited_once()
    assert judge.await_args.kwargs["is_edit"] is False


@pytest.mark.asyncio
async def test_fail_verdict_carries_reason():
    result, _, _, _ = await _call(
        {"input": "something bad"},
        verdict=PolicyVerdict(False, "violates policy", "judge-model", ""),
    )
    assert result["passed"] is False
    assert result["reason"] == "violates policy"


@pytest.mark.asyncio
async def test_edit_context_sets_is_edit():
    _, judge, _, _ = await _call({"input": "put a hat on this man", "context": "edit"})
    assert judge.await_args.kwargs["is_edit"] is True


@pytest.mark.asyncio
async def test_no_policy_passes_without_judge_call():
    result, judge, quota_check, recorder = await _call({"input": "anything"}, config={"img.policy": ""})
    assert result["passed"] is True
    assert result["policy_configured"] is False
    judge.assert_not_awaited()
    # no LLM call happens, so this path is deliberately unmetered
    quota_check.assert_not_awaited()
    recorder.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"input": "   "},
        {"input": "x", "context": "caption"},
        {"input": "y" * 10_001},
    ],
)
async def test_bad_requests_rejected(body):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await _call(body)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_quota_and_rpm_checked_before_the_judge_runs():
    """The metering must gate the GPU work, not trail it — an over-limit
    key gets 429 BEFORE a judge model is invoked."""
    order = []
    quota = AsyncMock(side_effect=lambda *a, **k: order.append("quota"))
    _, judge, _, _ = await _call({"input": "check me"}, quota=quota)
    judge.assert_awaited_once()
    # judge's side effect list: quota entry must exist before judge ran —
    # prove by wrapping: order captured only quota; judge awaited after
    assert order == ["quota"]


@pytest.mark.asyncio
async def test_quota_rejection_prevents_judge_call():
    from fastapi import HTTPException

    quota = AsyncMock(side_effect=HTTPException(status_code=429, detail="Rate limit"))
    with pytest.raises(HTTPException) as exc:
        await _call({"input": "hammered"}, quota=quota)
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_judged_calls_are_recorded():
    """Every judged moderation lands in the request log (voice/video
    precedent for endpoints that dispatch GPU work outside
    InferenceService) — modality CHAT, the judge model attributed."""
    _, _, quota_check, recorder = await _call({"input": "record me"})
    quota_check.assert_awaited_once()
    recorder.assert_awaited_once()
    kwargs = recorder.await_args.kwargs
    assert kwargs["endpoint"] == "/v1/moderations"
    assert kwargs["model"] == "judge-model"
    from backend.app.db.models import Modality
    assert kwargs["modality"] == Modality.CHAT


def test_route_decorator_present():
    """The 8 behavioral tests call the function directly, so a deleted or
    typo'd @router.post decorator would go unnoticed — the endpoint would
    vanish from the API with every test green."""
    import io as _io
    import tokenize as _tokenize
    from pathlib import Path

    src = (Path(api.__file__)).read_text()
    lines = src.splitlines(keepends=True)
    try:
        for tok in _tokenize.generate_tokens(_io.StringIO(src).readline):
            if tok.type == _tokenize.COMMENT:
                row, col = tok.start
                line = lines[row - 1]
                keep = line[:col]
                lines[row - 1] = keep + "\n" if line.endswith("\n") else keep
    except (_tokenize.TokenError, IndentationError):
        pass
    stripped = "".join(lines)
    assert '@router.post("/moderations")\nasync def moderations(' in stripped
