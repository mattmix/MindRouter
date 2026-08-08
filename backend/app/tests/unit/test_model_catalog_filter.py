############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_model_catalog_filter.py: Non-text models stay out of
#     the LLM catalogs (2.9.10)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The general model catalogs must publish text models only.

Until 2.9.10 `/v1/models` iterated every healthy backend's models with no
modality filter, so `black-forest-labs/FLUX.2-klein-9B` and
`lightricks/ltx-2.3-distilled` were advertised alongside chat models (verified
against production: both appeared among 73 entries). A client selecting one for
`/v1/chat/completions` gets a confusing failure, and registering voice backends
would have added Kokoro and whisper to the same list.

Image, video and speech each have their own discovery endpoint instead:
`/v1/images/models`, `/videos/models`, `/v1/audio/voices`.

Source/behaviour split: `is_catalog_model` is pure and imported directly; the
endpoints are checked structurally because importing them pulls the DB chain
(see MEMORY.md "Import Chain Gotcha").
"""

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP = Path(__file__).resolve().parents[2]


def _load_models_api():
    """Direct-load models_api for its pure helper, stubbing the heavy imports."""
    saved = {k: sys.modules.get(k) for k in (
        "backend", "backend.app", "backend.app.api", "backend.app.api.auth",
        "backend.app.core", "backend.app.core.canonical_schemas",
        "backend.app.core.telemetry", "backend.app.core.telemetry.registry",
        "backend.app.db", "backend.app.db.models", "backend.app.db.session",
    )}

    class _Modality:
        CHAT = "chat"
        COMPLETION = "completion"
        MULTIMODAL = "multimodal"
        EMBEDDING = "embedding"
        RERANKING = "reranking"
        IMAGE_GENERATION = "image_generation"
        VIDEO_GENERATION = "video_generation"
        TTS = "tts"
        STT = "stt"

    for name in ("backend", "backend.app", "backend.app.api", "backend.app.core",
                 "backend.app.core.telemetry", "backend.app.db"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["backend.app.api.auth"] = MagicMock()
    # FastAPI validates the response_model annotation when @router.get runs,
    # so CanonicalModelList must be the real pydantic class, not a MagicMock.
    _cs_spec = importlib.util.spec_from_file_location(
        "backend.app.core.canonical_schemas",
        _APP / "core" / "canonical_schemas.py", submodule_search_locations=[],
    )
    _cs = importlib.util.module_from_spec(_cs_spec)
    _cs_spec.loader.exec_module(_cs)
    sys.modules["backend.app.core.canonical_schemas"] = _cs
    sys.modules["backend.app.core.telemetry.registry"] = MagicMock()
    sys.modules["backend.app.db.session"] = MagicMock()
    sys.modules["backend.app.db.models"] = MagicMock(
        Modality=_Modality, ApiKey=MagicMock(), User=MagicMock()
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "models_api_under_test", _APP / "api" / "models_api.py",
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, _Modality
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def api():
    return _load_models_api()


def _model(modality):
    m = MagicMock()
    m.modality = modality
    return m


class TestIsCatalogModel:
    @pytest.mark.parametrize("attr", ["CHAT", "COMPLETION", "MULTIMODAL",
                                      "EMBEDDING", "RERANKING"])
    def test_text_modalities_are_published(self, api, attr):
        mod, Modality = api
        assert mod.is_catalog_model(_model(getattr(Modality, attr))) is True

    @pytest.mark.parametrize("attr", ["IMAGE_GENERATION", "VIDEO_GENERATION",
                                      "TTS", "STT"])
    def test_non_text_modalities_are_excluded(self, api, attr):
        mod, Modality = api
        assert mod.is_catalog_model(_model(getattr(Modality, attr))) is False

    def test_unknown_modality_is_published(self, api):
        """Fail open: a discovery gap must never hide a working chat model."""
        mod, _ = api
        assert mod.is_catalog_model(_model(None)) is True

    def test_embeddings_and_rerankers_stay(self, api):
        """These are text models with OpenAI precedent for being listed; the
        request was to hide image/video/voice, not to empty the catalog."""
        mod, Modality = api
        assert mod.is_catalog_model(_model(Modality.EMBEDDING)) is True
        assert mod.is_catalog_model(_model(Modality.RERANKING)) is True


class TestEveryCatalogAppliesTheFilter:
    """Three endpoints publish the catalog. If one forgets the filter, the
    models reappear on that surface only — a drift that is easy to miss."""

    def test_v1_models_filters(self):
        src = (_APP / "api" / "models_api.py").read_text()
        assert "if not is_catalog_model(model):" in src

    def test_api_tags_filters(self):
        src = (_APP / "api" / "ollama_api.py").read_text()
        assert "is_catalog_model" in src, "/api/tags must apply the same filter"

    def test_anthropic_models_delegates(self):
        """It reuses list_models, so it inherits the filter — assert the
        delegation rather than a duplicated filter."""
        src = (_APP / "api" / "anthropic_api.py").read_text()
        i = src.index("async def anthropic_models")
        assert "list_models" in src[i:i + 600]

    def test_no_other_endpoint_lists_all_backend_models(self):
        """Catch a fourth catalog appearing without the filter."""
        offenders = []
        for path in (_APP / "api").glob("*.py"):
            src = path.read_text()
            if "get_backend_models" not in src:
                continue
            if path.name in ("models_api.py", "ollama_api.py", "health.py"):
                continue  # these apply is_catalog_model
            if path.name == "telemetry_api.py":
                # admin operational telemetry: seeing every backend's models,
                # including image/video/voice, is the point
                continue
            # video/image discovery filter by engine instead, which is fine
            if re.search(r"engine.*!=.*BackendEngine\.(VIDEO|DIFFUSION)", src):
                continue
            offenders.append(path.name)
        assert not offenders, (
            f"these list backend models without a modality or engine filter: {offenders}"
        )


class TestPerModalityDiscoveryExists:
    """Filtering is only acceptable because each excluded family has its own
    endpoint — otherwise clients have no way to discover those models."""

    def test_images_have_a_discovery_endpoint(self):
        src = (_APP / "api" / "v1_openai.py").read_text()
        assert '@router.get("/images/models")' in src

    def test_videos_have_a_discovery_endpoint(self):
        src = (_APP / "api" / "video_api.py").read_text()
        assert '@router.get("/videos/models")' in src

    def test_image_endpoint_selects_by_diffusion_engine(self):
        src = (_APP / "api" / "v1_openai.py").read_text()
        i = src.index('@router.get("/images/models")')
        block = src[i:i + 1400]
        assert "BackendEngine.DIFFUSION" in block

    def test_docs_no_longer_point_images_at_v1_models(self):
        html = (_APP / "dashboard" / "templates" / "public" / "documentation.html").read_text()
        assert "Image models appear in <code>GET /v1/models</code>" not in html
        assert "/v1/images/models" in html
