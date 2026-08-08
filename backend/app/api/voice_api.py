############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# voice_api.py: Public TTS/STT API endpoints (/v1/audio/*)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Public voice API endpoints (TTS and STT) with API key authentication."""

import time
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.db import crud
from backend.app.db.models import ApiKey, Modality, RequestStatus, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/audio", tags=["voice"])


# Upstream TTS call budget.  Covers connect through the last audio chunk;
# httpx applies it per-operation, so it is a gap timeout while streaming.
TTS_TIMEOUT_SECONDS = 30.0


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request body."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _check_quota(db: AsyncSession, user: User, api_key: ApiKey = None):
    """Check if user has sufficient quota."""
    await crud.reset_quota_if_needed(db, user.id)
    quota = await crud.get_user_quota(db, user.id)
    group_budget = user.group.token_budget if user.group else 0
    if quota and group_budget > 0 and quota.tokens_used >= group_budget:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Token quota exceeded",
        )

    # RPM rate limit (Redis-backed, shared across all workers)
    rpm_limit = 0
    if api_key and api_key.rpm_limit:
        rpm_limit = api_key.rpm_limit
    elif quota:
        rpm_limit = quota.rpm_limit
    if rpm_limit > 0:
        from backend.app.core.redis_client import check_rpm
        allowed, current = await check_rpm(f"user:{user.id}", rpm_limit)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {rpm_limit} requests per minute (current: {current})",
            )


async def _record_and_complete(
    db: AsyncSession,
    user: User,
    api_key: ApiKey,
    http_request: Request,
    endpoint: str,
    modality: Modality,
    token_cost: int,
    model: str,
    error_message: Optional[str] = None,
):
    """Create a request record, mark it completed, update quota, and commit."""
    client_ip = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("user-agent")

    db_request = await crud.create_request(
        db=db,
        user_id=user.id,
        api_key_id=api_key.id,
        endpoint=endpoint,
        model=model,
        modality=modality,
        client_ip=client_ip,
        user_agent=user_agent,
    )

    if error_message:
        await crud.update_request_failed(db, db_request.id, error_message)
    else:
        await crud.update_request_completed(
            db, db_request.id,
            prompt_tokens=token_cost,
            completion_tokens=0,
            tokens_estimated=True,
        )
        await crud.update_quota_usage(db, user.id, token_cost)

    await db.commit()

    if not error_message:
        await crud.incr_quota_redis(user.id, token_cost)


# ---------------------------------------------------------------------------
# POST /v1/audio/speech   (TTS)
# ---------------------------------------------------------------------------

@router.post("/speech")
async def tts_speech(
    request: Request,
    body: TTSRequest,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible text-to-speech endpoint.

    Proxies to the configured TTS service (Kokoro).
    Requires API key authentication.
    """
    user, api_key = auth

    await _check_quota(db, user, api_key)

    if not body.input.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    # Read TTS config from DB (same as chat.py)
    tts_enabled = await crud.get_config_json(db, "voice.tts_enabled", False)
    if not tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is not enabled")

    # Prefer a registered, health-checked TTS backend; fall back to the legacy
    # single-URL config when none is registered (see services/voice_router.py).
    from backend.app.services.voice_router import resolve_voice_backend

    target = await resolve_voice_backend(db, "tts")
    if target is None:
        raise HTTPException(status_code=500, detail="TTS service URL not configured")
    tts_url = target.url

    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    payload = {
        "model": body.model,
        "input": body.input,
        "voice": body.voice,
        "speed": body.speed,
        "response_format": body.response_format,
    }

    token_cost = await crud.get_config_json(db, "voice_api.tts_quota_tokens", 100)

    # Send the upstream request and inspect its status BEFORE returning a
    # response body or charging quota.  Previously the request was recorded
    # completed (and the caller billed) before the generator ran at all, so a
    # dead TTS service produced HTTP 200 with zero bytes, billed in full, and
    # left nothing in the audit trail to show it had failed.
    #
    # The client and response are opened here rather than in an `async with`
    # so the connection can outlive this function and be consumed by the
    # streaming generator; the generator's `finally` closes both.
    client = httpx.AsyncClient(timeout=TTS_TIMEOUT_SECONDS)
    upstream_url = f"{tts_url.rstrip('/')}/v1/audio/speech"

    async def _fail(detail: str, error_message: str, status_code: int = 502):
        """Record the failure, charge nothing, and surface a real error.

        The audit write must never mask the upstream failure — if it throws we
        still raise, so the caller gets 502 rather than an unbound-variable 500.
        """
        try:
            await _record_and_complete(
                db, user, api_key, request,
                endpoint="/v1/audio/speech",
                modality=Modality.TTS,
                token_cost=0,
                model=body.model,
                error_message=error_message,
            )
        except Exception:
            logger.exception("tts_failure_record_failed")
        raise HTTPException(status_code=status_code, detail=detail)

    try:
        upstream_req = client.build_request("POST", upstream_url, json=payload, headers=headers)
        resp = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        await client.aclose()
        logger.warning("tts_api_proxy_timeout", timeout_s=TTS_TIMEOUT_SECONDS)
        await _fail("TTS service timed out", "TTS upstream timed out")
    except Exception as e:
        await client.aclose()
        logger.warning("tts_api_proxy_error", error=str(e))
        await _fail("TTS service is unavailable", f"TTS upstream unreachable: {e}")

    if resp.status_code != 200:
        # Deliberately not logging the body: an upstream validation error can
        # echo the caller's input text back, which is user content.
        error_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        logger.warning(
            "tts_api_proxy_error",
            status=resp.status_code,
            body_chars=len(error_body),
        )
        await _fail(
            "TTS service returned an error",
            f"TTS upstream returned HTTP {resp.status_code}",
        )

    # Upstream accepted the request — only now is it correct to bill.
    await _record_and_complete(
        db, user, api_key, request,
        endpoint="/v1/audio/speech",
        modality=Modality.TTS,
        token_cost=token_cost,
        model=body.model,
    )

    async def stream_audio():
        try:
            async for chunk in resp.aiter_bytes(4096):
                yield chunk
        except Exception as e:
            # The upstream accepted and we already billed; a break here means
            # the caller gets truncated audio.  Nothing to un-charge, but it
            # must not be silent.
            logger.warning("tts_stream_interrupted", error=str(e))
        finally:
            await resp.aclose()
            await client.aclose()

    content_type = "audio/mpeg" if body.response_format == "mp3" else f"audio/{body.response_format}"
    return StreamingResponse(stream_audio(), media_type=content_type)


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions   (STT)
# ---------------------------------------------------------------------------

_STT_RESPONSE_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
_STT_TEXT_FORMATS = {"text", "srt", "vtt"}

# OpenAI-dialect STT model names. Clients following OpenAI conventions send
# these; the backing Whisper server only knows its locally-installed model
# name, so forwarding them verbatim 404s upstream (surfaced as an opaque 502
# before 2.8.43). Treat them as aliases for the operator-configured model.
# Other explicit names still pass through, so a second installed model stays
# reachable by its real name.
_STT_MODEL_ALIASES = {
    "whisper", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
    "default", "default-stt",
}


@router.post("/transcriptions")
async def stt_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible speech-to-text endpoint.

    Proxies to the configured STT service (Whisper).
    Requires API key authentication.
    """
    user, api_key = auth

    await _check_quota(db, user, api_key)

    fmt = response_format.lower() if isinstance(response_format, str) else "json"
    if fmt not in _STT_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format '{response_format}'. "
                   f"Supported: {', '.join(sorted(_STT_RESPONSE_FORMATS))}",
        )

    # Read STT config from DB (same as chat.py)
    stt_enabled = await crud.get_config_json(db, "voice.stt_enabled", False)
    if not stt_enabled:
        raise HTTPException(status_code=404, detail="STT is not enabled")

    # Prefer a registered, health-checked STT backend; fall back to the legacy
    # single-URL config when none is registered (see services/voice_router.py).
    from backend.app.services.voice_router import resolve_voice_backend

    target = await resolve_voice_backend(db, "stt")
    if target is None:
        raise HTTPException(status_code=500, detail="STT service URL not configured")
    stt_url = target.url

    stt_default = await crud.get_config_json(db, "voice.stt_model", "whisper-large-v3-turbo")
    requested_model = (model or "").strip()
    if not requested_model or requested_model.lower() in _STT_MODEL_ALIASES:
        stt_model = stt_default
    else:
        stt_model = requested_model

    headers = {}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    audio_data = await file.read()

    token_cost = await crud.get_config_json(db, "voice_api.stt_quota_tokens", 200)

    data = {"model": stt_model, "response_format": fmt}
    if language:
        data["language"] = language

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{stt_url.rstrip('/')}/v1/audio/transcriptions",
                files={"file": (file.filename or "audio.webm", audio_data, file.content_type or "application/octet-stream")},
                data=data,
                headers=headers,
            )
            if resp.status_code != 200:
                logger.warning("stt_api_proxy_error", status=resp.status_code, body=resp.text[:500])
                await _record_and_complete(
                    db, user, api_key, request,
                    endpoint="/v1/audio/transcriptions",
                    modality=Modality.STT,
                    token_cost=0,
                    model=stt_model,
                    error_message=f"STT service error: {resp.status_code}",
                )
                if resp.status_code == 404:
                    # The Whisper server doesn't know this model name — tell the
                    # caller what IS valid instead of an opaque 502.
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unknown STT model '{stt_model}'. Omit 'model' to use the "
                            f"configured default ('{stt_default}'), or pass an installed "
                            f"model name."
                        ),
                    )
                raise HTTPException(status_code=502, detail="STT service error")

    except httpx.TimeoutException as e:
        logger.warning("stt_api_proxy_timeout", error=str(e))
        await _record_and_complete(
            db, user, api_key, request,
            endpoint="/v1/audio/transcriptions",
            modality=Modality.STT,
            token_cost=0,
            model=stt_model,
            error_message=f"STT timeout: {e}",
        )
        raise HTTPException(status_code=502, detail="STT service timed out (model may be loading)")
    except httpx.HTTPError as e:
        logger.warning("stt_api_proxy_error", error=str(e))
        await _record_and_complete(
            db, user, api_key, request,
            endpoint="/v1/audio/transcriptions",
            modality=Modality.STT,
            token_cost=0,
            model=stt_model,
            error_message=f"STT unavailable: {e}",
        )
        raise HTTPException(status_code=502, detail=f"STT service unavailable: {e}")

    # Success — record and deduct quota
    await _record_and_complete(
        db, user, api_key, request,
        endpoint="/v1/audio/transcriptions",
        modality=Modality.STT,
        token_cost=token_cost,
        model=stt_model,
    )

    if fmt in _STT_TEXT_FORMATS:
        content_types = {"text": "text/plain", "srt": "text/plain", "vtt": "text/vtt"}
        return StreamingResponse(
            iter([resp.content]),
            media_type=content_types.get(fmt, "text/plain"),
        )

    return JSONResponse(resp.json())
