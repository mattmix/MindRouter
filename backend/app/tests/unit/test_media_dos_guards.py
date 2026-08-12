"""
Tests for the ws7 media resource-exhaustion guards.

Covers:
- ocr._image_bytes_to_pages caps the multi-frame decode loop at
  settings.ocr_max_frames (unbounded-frame DoS).
- ocr.Image.MAX_IMAGE_PIXELS is pinned (decompression-bomb guard).
- chat._ooxml_uncompressed_bytes sums the declared uncompressed size of an
  OOXML package and returns -1 for non-zip input (zip-bomb guard).

Both modules import the DB chain at module scope, so we pre-mock the heavy
deps in sys.modules (Fix B from MEMORY) and load the files via
spec_from_file_location, keeping PIL real.
"""

import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]


def _install_stub(name: str, module: ModuleType) -> None:
    sys.modules[name] = module


def _stub_backend_chain(max_frames: int = 500):
    """Pre-mock the backend.app.* modules ocr.py / chat.py import at load time."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("backend")}

    # Intermediate packages so `from backend.app.x import y` resolves.
    for pkg in ("backend", "backend.app", "backend.app.core",
                "backend.app.core.translators", "backend.app.core.telemetry",
                "backend.app.db", "backend.app.services", "backend.app.dashboard",
                "backend.app.storage"):
        mod = ModuleType(pkg)
        mod.__path__ = []  # mark as package
        _install_stub(pkg, mod)

    # canonical_schemas: the four names ocr.py imports.
    canon = ModuleType("backend.app.core.canonical_schemas")
    canon.CanonicalChatRequest = MagicMock(name="CanonicalChatRequest")
    canon.CanonicalMessage = MagicMock(name="CanonicalMessage")
    canon.ImageUrlContent = MagicMock(name="ImageUrlContent")
    canon.TextContent = MagicMock(name="TextContent")
    canon.CanonicalModelInfo = MagicMock(name="CanonicalModelInfo")
    _install_stub("backend.app.core.canonical_schemas", canon)

    crud = ModuleType("backend.app.db.crud")
    _install_stub("backend.app.db.crud", crud)

    logging_config = ModuleType("backend.app.logging_config")
    logging_config.get_logger = lambda *a, **k: MagicMock()
    _install_stub("backend.app.logging_config", logging_config)

    settings_mod = ModuleType("backend.app.settings")
    settings_obj = SimpleNamespace(
        ocr_max_frames=max_frames,
        chat_upload_max_uncompressed_mb=100,
        app_version="test",
    )
    settings_mod.get_settings = lambda: settings_obj
    _install_stub("backend.app.settings", settings_mod)

    return saved


def _load_module(rel_path: str, mod_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ocr_module():
    saved = _stub_backend_chain(max_frames=3)
    try:
        yield _load_module("backend/app/services/ocr.py", "ocr_under_test")
    finally:
        for k in [k for k in sys.modules if k.startswith("backend")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _make_multiframe_gif(num_frames: int) -> bytes:
    frames = [Image.new("RGB", (8, 8), (i * 10 % 256, 0, 0)) for i in range(num_frames)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0)
    return buf.getvalue()


def test_frame_loop_capped_at_max_frames(ocr_module):
    """A 10-frame GIF with max_frames=3 yields exactly 3 pages."""
    gif = _make_multiframe_gif(10)
    pages = ocr_module._image_bytes_to_pages(gif, max_frames=3)
    assert len(pages) == 3
    assert all(isinstance(p, (bytes, bytearray)) and p for p in pages)


def test_frame_loop_returns_all_frames_when_under_cap(ocr_module):
    """A 2-frame GIF under the cap returns both frames (normal input intact)."""
    gif = _make_multiframe_gif(2)
    pages = ocr_module._image_bytes_to_pages(gif, max_frames=500)
    assert len(pages) == 2


def test_single_frame_image_yields_one_page(ocr_module):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (0, 128, 0)).save(buf, format="PNG")
    pages = ocr_module._image_bytes_to_pages(buf.getvalue(), max_frames=500)
    assert len(pages) == 1


def test_max_image_pixels_is_bounded(ocr_module):
    """The decompression-bomb guard is pinned to a finite bound."""
    assert ocr_module.Image.MAX_IMAGE_PIXELS is not None
    assert ocr_module.Image.MAX_IMAGE_PIXELS <= 256_000_000


# ---------------------------------------------------------------------------
# chat._ooxml_uncompressed_bytes
# ---------------------------------------------------------------------------

@pytest.fixture
def chat_module():
    saved = _stub_backend_chain()
    # chat.py imports many more backend modules; stub the ones it names.
    extra = {
        "backend.app.core.latex_normalize": {"normalize_latex": lambda x: x},
        "backend.app.core.telemetry.registry": {"get_registry": lambda: MagicMock()},
        "backend.app.core.translators.openai_in": {"OpenAIInTranslator": MagicMock()},
        "backend.app.db.chat_crud": {},
        "backend.app.db.models": {"ApiKey": MagicMock(), "User": MagicMock()},
        "backend.app.db.session": {
            "get_async_db": MagicMock(),
            "get_async_db_context": MagicMock(),
        },
        "backend.app.dashboard.routes": {
            "get_effective_user_id": MagicMock(),
            "get_session_user_id": MagicMock(),
            "get_masquerade_user_id": MagicMock(),
        },
        "backend.app.services.inference": {"InferenceService": MagicMock()},
        "backend.app.services.web_search": {
            "brave_web_search": MagicMock(),
            "format_search_results": MagicMock(),
        },
        "backend.app.services.branding": {"get_branding": lambda *a, **k: {}},
        "backend.app.services.feature_access": {"image_access": lambda *a, **k: True},
        "backend.app.db.crud": {"get_config_json": MagicMock()},
    }
    for name, attrs in extra.items():
        # Build a FRESH module rather than mutating the real one in place: the
        # finally-block restores real modules from ``saved`` by reference, so an
        # in-place setattr on a real module would leak past teardown and pollute
        # unrelated tests. Copy the real module's namespace first so chat.py's
        # imports still resolve, then overlay the mocks on the copy.
        real = sys.modules.get(name)
        mod = ModuleType(name)
        if real is not None:
            mod.__dict__.update(real.__dict__)
        for k, v in attrs.items():
            setattr(mod, k, v)
        _install_stub(name, mod)
    try:
        yield _load_module("backend/app/dashboard/chat.py", "chat_under_test")
    finally:
        for k in [k for k in sys.modules if k.startswith("backend")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _make_ooxml(uncompressed_len: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Highly compressible payload: small on disk, large uncompressed.
        zf.writestr("[Content_Types].xml", b"a" * uncompressed_len)
    return buf.getvalue()


def test_ooxml_uncompressed_bytes_sums_members(chat_module):
    data = _make_ooxml(50_000)
    total = chat_module._ooxml_uncompressed_bytes(data)
    assert total >= 50_000
    # The compressed archive is far smaller than the declared uncompressed size.
    assert len(data) < total


def test_ooxml_uncompressed_bytes_bad_zip_returns_negative(chat_module):
    assert chat_module._ooxml_uncompressed_bytes(b"not a zip file at all") == -1


def test_ooxml_zip_bomb_exceeds_typical_cap(chat_module):
    """A tiny archive that inflates past a 100MB cap is detectable up front."""
    data = _make_ooxml(120 * 1024 * 1024)
    total = chat_module._ooxml_uncompressed_bytes(data)
    assert total > 100 * 1024 * 1024
    assert len(data) < 1 * 1024 * 1024  # compressed footprint stays tiny
