############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_image_watermark.py: Invisible TrustMark provenance
#     watermark on generated images.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Every generated image gets an invisible TrustMark watermark.

The service must fail OPEN (an unmarked image beats a failed generation —
the GPU seconds are already spent), the payload is capped at 8 printable
ASCII chars (TrustMark Q/BCH_5 = 61 bits), and the wiring must sit in
``_proxy_image_request`` — the single choke point through which both
``/v1/images/generations`` and ``/v1/images/edits`` and the gallery's
stored copy all flow. Because watermarking needs bytes, the gateway
forces ``b64_json`` from the backend whenever marking is on — otherwise
the API's DEFAULT ``response_format="url"`` would bypass the mark
entirely and leave an unmarked copy on the GPU node's public mount.

The trustmark package itself is NOT required here: the service treats an
un-importable trustmark as "watermarking unavailable" and these tests
exercise that path plus a stubbed happy path. All source guards match
COMMENT-STRIPPED text — this repo has twice proven that raw-source
substring guards are satisfied by commented-out code.
"""

import base64
import importlib.util
import io
import re
import sys
import tokenize
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP = Path(__file__).resolve().parents[2]
_ROOT = _APP.parents[1]


def _strip_py_comments(source: str) -> str:
    """Blank out Python comment tokens so commented-out code can't satisfy
    a guard (same approach as test_model_catalog_filter.py)."""
    lines = source.splitlines(keepends=True)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                line = lines[row - 1]
                keep = line[:col]
                lines[row - 1] = keep + "\n" if line.endswith("\n") else keep
    except (tokenize.TokenError, IndentationError):
        pass
    return "".join(lines)


def _strip_hash_comments(text: str) -> str:
    """Dockerfile/TOML-style: truncate every line at its first '#'.

    Full-line stripping is not enough: inside a multi-line RUN, a '#'
    inserted MID-line shell-comments out the rest of that command while
    the text remains — the guard must not keep matching it."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _strip_html_comments(html: str) -> str:
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def _load_image_watermark():
    """Direct-load image_watermark.py with logging stubbed; restore
    sys.modules afterwards (see MEMORY.md sys.modules hygiene rules)."""
    _KEYS = ("backend.app.logging_config",)
    saved = {k: sys.modules.get(k) for k in _KEYS}
    try:
        sys.modules["backend.app.logging_config"] = MagicMock(
            get_logger=MagicMock(return_value=MagicMock())
        )
        spec = importlib.util.spec_from_file_location(
            "image_watermark_under_test", _APP / "services" / "image_watermark.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


iw = _load_image_watermark()


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """The module caches the loaded model per process; tests must not
    inherit each other's cache or unavailable-flag."""
    iw._tm = None
    iw._tm_unavailable = False
    yield
    iw._tm = None
    iw._tm_unavailable = False


def _png_b64(color=(200, 30, 30), size=(32, 32)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_fake_trustmark_class(encoder=True, decoder=True):
    """A FRESH stand-in class per call, so per-test mutations (e.g. a
    raising encode) can never leak into later tests."""

    class _FakeTrustMark:
        class Encoding:
            BCH_5 = 1

        init_kwargs = None

        def __init__(self, **kwargs):
            type(self).init_kwargs = kwargs
            self.encoder = object() if encoder else None
            self.decoder = object() if decoder else None

        def encode(self, img, text):
            from PIL import Image

            return Image.new("RGB", img.size, (1, 2, 3))

    return _FakeTrustMark


@pytest.fixture
def fake_trustmark():
    cls = _make_fake_trustmark_class()
    saved = sys.modules.get("trustmark")
    stub = types.ModuleType("trustmark")
    stub.TrustMark = cls
    sys.modules["trustmark"] = stub
    try:
        yield cls
    finally:
        if saved is None:
            sys.modules.pop("trustmark", None)
        else:
            sys.modules["trustmark"] = saved


class TestPayloadValidation:
    @pytest.mark.parametrize("text", ["A", "UIMR-AI", "12345678", "a b c d "[:8]])
    def test_valid_payloads(self, text):
        assert iw.validate_watermark_text(text) is None

    def test_empty_rejected(self):
        assert iw.validate_watermark_text("") is not None

    def test_over_capacity_rejected(self):
        """9 chars exceeds TrustMark's 61-bit / 8-char ASCII capacity."""
        assert iw.validate_watermark_text("123456789") is not None

    @pytest.mark.parametrize("text", ["Idahø", "水印", "tab\there", "nl\n"])
    def test_non_printable_ascii_rejected(self, text):
        """encode_text_ascii packs 7-bit ASCII; anything else corrupts."""
        assert iw.validate_watermark_text(text) is not None

    @pytest.mark.parametrize("value", [123, True, None, ["UIMR"]])
    def test_non_string_rejected_not_crashed(self, value):
        """get_config_json can hand back a non-string (a bare 123 in the DB
        row parses as int); validation must reject it, not TypeError — a
        crash here would 500 every image request."""
        assert iw.validate_watermark_text(value) is not None

    def test_default_payload_is_valid(self):
        assert iw.validate_watermark_text(iw.WATERMARK_DEFAULT_TEXT) is None
        assert len(iw.WATERMARK_DEFAULT_TEXT) <= iw.WATERMARK_MAX_CHARS


class TestApplyWatermark:
    @pytest.mark.asyncio
    async def test_happy_path_returns_marked_png(self, fake_trustmark):
        original = _png_b64()
        out = await iw.apply_watermark_b64(original, "UIMR-AI")
        assert out != original
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(out)))
        assert img.format == "PNG"
        # model configured the way the module promises: CPU, Q variant,
        # BCH_5 (the 61-bit basis of the 8-char payload cap), no remover
        kw = fake_trustmark.init_kwargs
        assert kw["device"] == "cpu"
        assert kw["model_type"] == "Q"
        assert kw["loadRemover"] is False
        assert kw["encoding_type"] == fake_trustmark.Encoding.BCH_5

    @pytest.mark.asyncio
    async def test_missing_trustmark_fails_open(self):
        """No trustmark package: the ORIGINAL image comes back unchanged.

        sys.modules[name] = None makes the import raise deterministically,
        so this passes identically on machines that DO have trustmark
        installed (e.g. a dev box mirroring the prod image)."""
        saved = sys.modules.get("trustmark")
        sys.modules["trustmark"] = None
        try:
            original = _png_b64()
            assert await iw.apply_watermark_b64(original, "UIMR-AI") == original
            assert iw._tm_unavailable is True
        finally:
            if saved is None:
                sys.modules.pop("trustmark", None)
            else:
                sys.modules["trustmark"] = saved

    @pytest.mark.asyncio
    async def test_half_built_model_fails_open(self):
        """TrustMark swallows weight-download failures and constructs with
        encoder=None; the service must treat that as unavailable, not cache
        a doomed model that crashes at encode time."""
        cls = _make_fake_trustmark_class(encoder=False)
        saved = sys.modules.get("trustmark")
        stub = types.ModuleType("trustmark")
        stub.TrustMark = cls
        sys.modules["trustmark"] = stub
        try:
            original = _png_b64()
            assert await iw.apply_watermark_b64(original, "UIMR-AI") == original
            assert iw._tm_unavailable is True
            assert iw._tm is None
        finally:
            if saved is None:
                sys.modules.pop("trustmark", None)
            else:
                sys.modules["trustmark"] = saved

    @pytest.mark.asyncio
    async def test_unavailable_flag_short_circuits(self, fake_trustmark):
        """Once marked unavailable, later requests must not retry the load
        (a doomed import per image would stall every generation)."""
        iw._tm_unavailable = True
        original = _png_b64()
        assert await iw.apply_watermark_b64(original, "UIMR-AI") == original
        assert iw._tm is None

    @pytest.mark.asyncio
    async def test_encode_exception_fails_open(self, fake_trustmark):
        def _boom(self, img, text):
            raise RuntimeError("torch exploded")

        fake_trustmark.encode = _boom  # fresh class per test — no leakage
        original = _png_b64()
        assert await iw.apply_watermark_b64(original, "UIMR-AI") == original

    @pytest.mark.asyncio
    async def test_garbage_input_fails_open(self, fake_trustmark):
        assert await iw.apply_watermark_b64("not-base64!!!", "UIMR-AI") == "not-base64!!!"


class TestWiring:
    """The watermark must sit at the single image choke point, and the
    install must stay --no-deps pinned. Every source guard here matches
    comment-stripped text and anchors on CALL sites, not names — an
    import line or a commented-out block must never satisfy a guard."""

    def _proxy_segment(self) -> str:
        src = _strip_py_comments((_APP / "services" / "inference.py").read_text())
        i = src.index("async def _proxy_image_request")
        j = src.index("async def", i + 10)
        return src[i:j]

    def test_proxy_image_request_applies_watermark(self):
        seg = self._proxy_segment()
        assert "image_watermark.apply_watermark_b64(" in seg
        assert '"img.watermark_enabled"' in seg
        assert '"img.watermark_text"' in seg

    def test_config_read_is_fail_safe(self):
        """A DB blip must fall back to marking with defaults — never 500 a
        request whose GPU work already happened. The except handler must
        actually swallow: a `raise` appended to it would pass a
        presence-only guard while re-500ing every config failure."""
        seg = self._proxy_segment()
        assert "watermark_config_read_failed_using_defaults" in seg
        gate = seg.index("get_async_db_context()")
        assert "try:" in seg[:gate], "config read must sit inside try/except"
        exc = seg.index("except Exception:", gate)
        handler_end = seg.index("if image_watermark.validate_watermark_text", exc)
        assert "raise" not in seg[exc:handler_end], (
            "the config-read except handler must swallow, not re-raise"
        )
        # an invalid stored payload falls back to the default, not a crash
        assert "wm_text = image_watermark.WATERMARK_DEFAULT_TEXT" in seg

    def test_url_bypass_is_closed(self):
        """response_format='url' is the public API DEFAULT; without forcing
        b64 from the backend BEFORE dispatch, default API calls would
        return unmarked images (and strand an unmarked copy on the GPU
        node). Ordering matters: the same statement after client.post is
        dead code."""
        seg = self._proxy_segment()
        force = seg.index('payload["response_format"] = "b64_json"')
        post = seg.index("client.post(url")
        assert force < post, "b64 forcing must happen before dispatch"

    def test_admin_get_exposes_watermark_config(self):
        src = _strip_py_comments((_APP / "dashboard" / "routes.py").read_text())
        i = src.index("async def admin_images_config(")
        j = src.index("async def admin_images_config_post(")
        seg = src[i:j]
        assert '"img.watermark_enabled"' in seg
        assert '"img.watermark_text"' in seg
        assert "watermark_max_chars" in seg

    def test_admin_post_validates_before_writing(self):
        """Validation must be ENFORCED, not merely called: the reject
        branch (if wm_error -> redirect return) must sit between the call
        and the write — deleting it would leave call ordering intact while
        writing invalid payloads."""
        src = _strip_py_comments((_APP / "dashboard" / "routes.py").read_text())
        i = src.index("async def admin_images_config_post(")
        seg = src[i:i + 6000]
        # anchor on the CALL, not the import line
        v = seg.index("wm_error = validate_watermark_text(")
        e = seg.index("if wm_error:", v)
        r = seg.index("return RedirectResponse", e)
        w = seg.index('set_config(db, "img.watermark_text"')
        assert v < e < r < w, (
            "validate -> reject-and-return -> write, in that order"
        )

    def test_admin_post_saves_toggle_off_with_empty_text(self):
        """Turning the watermark OFF with a cleared text field must save,
        not bounce the whole images-config form. The enabled-flag write
        must sit OUTSIDE the `if wm_on or wm_text:` gate — asserted via
        exact indentation (8 spaces = save_config level, 12 would be the
        gate body), because nesting it makes the toggle impossible to
        turn off."""
        src = _strip_py_comments((_APP / "dashboard" / "routes.py").read_text())
        i = src.index("async def admin_images_config_post(")
        seg = src[i:i + 6000]
        assert "if wm_on or wm_text:" in seg
        assert (
            '\n        await crud.set_config(db, "img.watermark_enabled", wm_on)'
            in seg
        ), "enabled-flag write must be at save_config level, not inside the gate"

    def test_template_has_watermark_controls(self):
        html = _strip_html_comments(
            (_APP / "dashboard" / "templates" / "admin" / "images_config.html").read_text()
        )
        assert 'name="watermark_enabled"' in html
        assert 'name="watermark_text"' in html
        # the length cap must be BOUND to the service constant, not hardcoded
        assert 'maxlength="{{ watermark_max_chars }}"' in html

    def test_dockerfile_pins_no_deps_install_and_proves_the_bake(self):
        """A well-meaning 'fix' to a normal `pip install trustmark` would
        let its numpy<2 pin downgrade numpy for the entire image; and
        because TrustMark swallows download failures, the bake must PROVE
        the weights work with a roundtrip, not just import."""
        docker = _strip_hash_comments((_ROOT / "Dockerfile").read_text())
        assert "--no-deps trustmark==" in docker
        assert "tm.decode(tm.encode(" in docker, "bake must roundtrip, not just import"

    def test_pyproject_carries_the_real_runtime_deps(self):
        py = _strip_hash_comments((_ROOT / "pyproject.toml").read_text())
        for dep in ("torchvision", "einops", "omegaconf", "lightning"):
            assert dep in py, f"trustmark runtime dep {dep} missing from pyproject"
