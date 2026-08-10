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


async def _call(body, config=None, verdict=None):
    """Invoke the endpoint with crud config + judge mocked out."""
    config = {"img.policy": "no real people"} if config is None else config

    async def fake_config(db, key, default=None):
        return config.get(key, default)

    judge = AsyncMock(return_value=verdict or PolicyVerdict(True, "ok", "judge-model", ""))
    with (
        patch.object(api.crud, "get_config_json", side_effect=fake_config),
        patch("backend.app.services.image_policy.evaluate_prompt", new=judge),
        patch.object(api, "bind_request_context"),
    ):
        result = await api.moderations(_FakeRequest(body), db=None, auth=_auth())
    return result, judge


@pytest.mark.asyncio
async def test_pass_verdict_round_trips():
    result, judge = await _call({"input": "a red bicycle against a brick wall"})
    assert result["passed"] is True
    assert result["policy_configured"] is True
    assert result["judge_model"] == "judge-model"
    judge.assert_awaited_once()
    assert judge.await_args.kwargs["is_edit"] is False


@pytest.mark.asyncio
async def test_fail_verdict_carries_reason():
    result, _ = await _call(
        {"input": "something bad"},
        verdict=PolicyVerdict(False, "violates policy", "judge-model", ""),
    )
    assert result["passed"] is False
    assert result["reason"] == "violates policy"


@pytest.mark.asyncio
async def test_edit_context_sets_is_edit():
    _, judge = await _call({"input": "put a hat on this man", "context": "edit"})
    assert judge.await_args.kwargs["is_edit"] is True


@pytest.mark.asyncio
async def test_no_policy_passes_without_judge_call():
    result, judge = await _call({"input": "anything"}, config={"img.policy": ""})
    assert result["passed"] is True
    assert result["policy_configured"] is False
    judge.assert_not_awaited()


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
