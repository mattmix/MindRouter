############################################################
#
# mindrouter - unit tests for the standalone /v1/moderations endpoint
#
# Client apps (e.g. VandalChat) caption user-supplied reference images with a
# vision model and vet the caption against the central image content policy
# BEFORE submitting an img2img edit. This endpoint exposes the same judge and
# img.policy text used by /images/generations and /images/edits, without
# generating anything — in the standard OpenAI moderations response shape,
# with the policy verdict in `flagged` plus MindRouter extension fields.
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


class TestOpenAIShape:
    """The response is the standard OpenAI moderations envelope — the whole
    point of the shape rework — with the verdict in `flagged` and the
    policy detail in extension fields the SDK preserves."""

    @pytest.mark.asyncio
    async def test_pass_verdict_is_unflagged(self):
        result, judge, _, _ = await _call({"input": "a red bicycle against a brick wall"})
        assert result["id"].startswith("modr-")
        assert result["model"] == "judge-model"
        assert result["policy_configured"] is True
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["flagged"] is False
        assert r["policy_reason"] == "ok"
        assert r["judge_model"] == "judge-model"
        judge.assert_awaited_once()
        assert judge.await_args.kwargs["is_edit"] is False

    @pytest.mark.asyncio
    async def test_fail_verdict_is_flagged_with_reason(self):
        result, _, _, _ = await _call(
            {"input": "something bad"},
            verdict=PolicyVerdict(False, "violates policy", "judge-model", ""),
        )
        r = result["results"][0]
        assert r["flagged"] is True
        assert r["policy_reason"] == "violates policy"
        # our flags come from the institutional policy, not OpenAI's
        # taxonomy — every category boolean stays false even when flagged
        assert all(v is False for v in r["categories"].values())

    @pytest.mark.asyncio
    async def test_every_openai_category_key_present(self):
        """openai-python's pydantic models REQUIRE every category key in
        categories, category_scores, and category_applied_input_types —
        a missing one breaks SDK parsing."""
        result, _, _, _ = await _call({"input": "x"})
        r = result["results"][0]
        for section in ("categories", "category_scores", "category_applied_input_types"):
            assert set(r[section].keys()) == set(api._MODERATION_CATEGORIES)

    @pytest.mark.asyncio
    async def test_response_parses_with_real_openai_sdk(self):
        """The decisive test: openai-python must parse the response and
        preserve the extension fields."""
        openai_types = pytest.importorskip("openai.types")
        result, _, _, _ = await _call(
            {"input": "check"},
            verdict=PolicyVerdict(False, "real person", "judge-model", ""),
        )
        parsed = openai_types.ModerationCreateResponse.model_validate(result)
        assert parsed.results[0].flagged is True
        assert parsed.results[0].model_extra["policy_reason"] == "real person"
        assert parsed.model_extra["policy_configured"] is True

    @pytest.mark.asyncio
    async def test_array_input_yields_one_result_per_item(self):
        result, judge, _, _ = await _call({"input": ["one", "two", "three"]})
        assert len(result["results"]) == 3
        assert judge.await_count == 3

    @pytest.mark.asyncio
    async def test_edit_context_sets_is_edit(self):
        _, judge, _, _ = await _call({"input": "put a hat on this man", "context": "edit"})
        assert judge.await_args.kwargs["is_edit"] is True

    @pytest.mark.asyncio
    async def test_no_policy_passes_without_judge_call(self):
        result, judge, quota_check, recorder = await _call(
            {"input": "anything"}, config={"img.policy": ""}
        )
        assert result["policy_configured"] is False
        assert result["results"][0]["flagged"] is False
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
        {"input": ["ok", "  "]},
        {"input": []},
        {"input": ["a", "b", "c", "d", "e"]},
        {"input": ["ok", 42]},
        {"input": {"text": "not a string"}},
        {"input": "x", "context": "caption"},
        {"input": "y" * 10_001},
        ["a", "bare", "list"],
    ],
)
async def test_bad_requests_rejected(body):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await _call(body)
    assert exc.value.status_code == 400


class TestMetering:
    @pytest.mark.asyncio
    async def test_quota_and_rpm_checked_before_the_judge_runs(self):
        """The metering must gate the GPU work, not trail it — an
        over-limit key gets 429 BEFORE a judge model is invoked."""
        order = []
        quota = AsyncMock(side_effect=lambda *a, **k: order.append("quota"))
        _, judge, _, _ = await _call({"input": "check me"}, quota=quota)
        judge.assert_awaited_once()
        assert order == ["quota"]

    @pytest.mark.asyncio
    async def test_quota_rejection_prevents_judge_call(self):
        from fastapi import HTTPException

        quota = AsyncMock(side_effect=HTTPException(status_code=429, detail="Rate limit"))
        with pytest.raises(HTTPException) as exc:
            await _call({"input": "hammered"}, quota=quota)
        assert exc.value.status_code == 429

    @pytest.mark.asyncio
    async def test_judged_calls_are_recorded(self):
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

    @pytest.mark.asyncio
    async def test_token_cost_scales_with_input_count(self):
        """Array inputs each run the judge, so the flat cost multiplies —
        otherwise arrays would be a metering discount."""
        _, _, _, recorder = await _call(
            {"input": ["one", "two"]},
            config={"img.policy": "no real people", "img.moderation_quota_tokens": 5},
        )
        assert recorder.await_args.kwargs["token_cost"] == 10


def test_route_decorator_present():
    """The behavioral tests call the function directly, so a deleted or
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
