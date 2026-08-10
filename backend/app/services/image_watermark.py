############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# image_watermark.py: Invisible provenance watermarking for
#     generated images (Adobe TrustMark).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Invisible provenance watermark for generated images.

Uses TrustMark (Adobe, MIT, https://github.com/adobe/trustmark), the
watermark of the Content Authenticity Initiative. The Q model with BCH_5
error correction carries a 61-bit payload — 8 ASCII characters — embedded
imperceptibly (PSNR ~43) and robust to at least PNG and JPEG-90
re-encoding (verified empirically). Encoding costs ~250 ms of CPU per
1024x1024 image; decode needs no key, so anyone with the open-source
tooling can verify provenance.

The ``trustmark`` package is installed WITHOUT its declared dependencies
(see Dockerfile): its ``numpy<2`` pin conflicts with this image's numpy
2.x and is empirically unnecessary. Its real runtime deps (torchvision,
einops, omegaconf, lightning) are declared in pyproject.toml. When the
package is absent (e.g. local dev without the extra installs),
watermarking degrades to a logged no-op rather than breaking the image
path.

Config (app_config, editable on /admin/images-config, read per request):
  img.watermark_enabled  bool, default True
  img.watermark_text     <=8 printable ASCII chars, default "UIMR-AI"
"""

import asyncio
import base64
import io
import threading
from typing import Optional

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# TrustMark Q + BCH_5 = 61 payload bits at 7 bits per ASCII character.
WATERMARK_MAX_CHARS = 8
WATERMARK_DEFAULT_TEXT = "UIMR-AI"

_tm = None
_tm_unavailable = False
# One model, one encode at a time: serializes both the lazy load and the
# ~250 ms CPU encodes so concurrent image requests can't stack torch work.
_tm_lock = threading.Lock()


def validate_watermark_text(text: str) -> Optional[str]:
    """Return an error message for an invalid payload, or None if valid.

    TrustMark's text mode packs 7-bit ASCII, so the payload must be 1..8
    printable ASCII characters.
    """
    if not isinstance(text, str):
        return "Watermark text must be a string."
    if not text:
        return "Watermark text must not be empty."
    if len(text) > WATERMARK_MAX_CHARS:
        return (
            f"Watermark text must be at most {WATERMARK_MAX_CHARS} characters "
            f"(TrustMark carries 61 bits — 7-bit ASCII)."
        )
    if not all(32 <= ord(c) < 127 for c in text):
        return "Watermark text must be printable ASCII (7-bit)."
    return None


def _get_trustmark():
    """Load the TrustMark model once per process. Caller holds _tm_lock."""
    global _tm, _tm_unavailable
    if _tm is not None or _tm_unavailable:
        return _tm
    try:
        from trustmark import TrustMark

        # Q variant: the robustness/quality default. CPU is explicit — the
        # gateway host has no GPU and implicit device selection surprises.
        # loadRemover=False skips a third model this path never uses.
        _tm = TrustMark(
            verbose=False,
            model_type="Q",
            encoding_type=TrustMark.Encoding.BCH_5,
            device="cpu",
            loadRemover=False,
        )
        # TrustMark's load_model swallows download/checksum errors and
        # returns None instead of raising, so a missing-weights image
        # "constructs" fine and would crash (or no-op) at encode time.
        # Treat a half-built model as unavailable.
        if getattr(_tm, "encoder", None) is None or getattr(_tm, "decoder", None) is None:
            _tm = None
            raise RuntimeError("TrustMark constructed without model weights")
        logger.info("trustmark_loaded", model_type="Q")
    except Exception:
        # Missing package, missing model weights, torch failure — mark
        # unavailable so every request doesn't retry a doomed load.
        _tm_unavailable = True
        logger.exception("trustmark_unavailable_watermarking_disabled")
    return _tm


def _encode_sync(image_bytes: bytes, text: str) -> bytes:
    """Watermark PNG/JPEG bytes, returning PNG bytes. Runs in a worker thread."""
    from PIL import Image

    with _tm_lock:
        tm = _get_trustmark()
        if tm is None:
            return image_bytes
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        marked = tm.encode(img, text)
        out = io.BytesIO()
        marked.save(out, format="PNG")
        return out.getvalue()


async def apply_watermark_b64(b64_image: str, text: str) -> str:
    """Watermark a base64 image, returning base64 PNG.

    Fails OPEN: on any error the ORIGINAL image is returned and the error
    logged — an unmarked image beats a failed generation. The GPU seconds
    are already spent by the time this runs.
    """
    try:
        raw = base64.b64decode(b64_image)
        marked = await asyncio.to_thread(_encode_sync, raw, text)
        if marked is raw:
            return b64_image
        return base64.b64encode(marked).decode("ascii")
    except Exception:
        logger.exception("watermark_failed_returning_unmarked_image")
        return b64_image
