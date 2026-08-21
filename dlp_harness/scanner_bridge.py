############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/scanner_bridge.py: Load and drive the
# production DLP scanner (dlp_scanner.py) standalone,
# locally or inside the app container.
#
############################################################

"""Standalone bridge to the production DLP scanner module.

``backend/app/services/dlp_scanner.py`` is a pure-logic module, but importing
it through the package pulls the DB/telemetry chain (services/__init__).  The
bridge loads the single .py FILE via spec_from_file_location with the three
``backend*`` module names temporarily stubbed, then restores sys.modules so
the process stays clean (suite hygiene rule).  This is the ONLY harness module
allowed to touch app code.
"""

import asyncio
import importlib.util
import os
import sys
import time
import types
from typing import Any, Dict, Iterable, List, Optional

try:
    from dlp_harness import constants
except ImportError:  # dlp_harness dir copied loose (in-container fallback)
    import constants

# Where the Dockerfile puts the scanner source (WORKDIR /app + COPY backend/ backend/).
CONTAINER_SCANNER_PATH = "/app/backend/app/services/dlp_scanner.py"

_SUPPORTED_SCANNERS = ("regex", "gliner")

# The module names load_scanner_module must stub for dlp_scanner.py's single
# app import (`from backend.app.logging_config import get_logger`).
_STUBBED_MODULES = ("backend", "backend.app", "backend.app.logging_config")

# resolved path -> loaded module, so repeat calls are cheap.
_module_cache: Dict[str, Any] = {}


class _NullLogger:
    """Accepts structlog-style calls (positional event + kwargs), does nothing."""

    def _noop(self, *args, **kwargs):
        return None

    debug = info = warning = error = exception = _noop


def resolve_scanner_path(path: Optional[str] = None) -> str:
    """Resolve which dlp_scanner.py file to load.

    Precedence: explicit ``path`` arg > DLP_SCANNER_PATH env var > repo-root
    default derived from this file's location.  The env var is how the
    container runner points the bridge at /app/... when the harness directory
    is copied to /tmp inside the container.
    """
    if path:
        return os.path.abspath(path)
    env = os.environ.get("DLP_SCANNER_PATH")
    if env:
        return os.path.abspath(env)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "backend", "app", "services", "dlp_scanner.py")


def load_scanner_module(path: Optional[str] = None):
    """Load dlp_scanner.py standalone and return the module (cached per path).

    Pre-seeds sys.modules with stub ``backend`` packages so the module's
    logging import resolves without pulling the DB/telemetry chain, then
    restores the prior sys.modules state (delete if absent before) — the
    loaded module keeps its own references to the stubs, so the process's
    sys.modules must not stay polluted.
    """
    resolved = resolve_scanner_path(path)
    cached = _module_cache.get(resolved)
    if cached is not None:
        return cached

    prior = {name: sys.modules[name] for name in _STUBBED_MODULES if name in sys.modules}

    backend_pkg = types.ModuleType("backend")
    app_pkg = types.ModuleType("backend.app")
    logging_stub = types.ModuleType("backend.app.logging_config")
    logging_stub.get_logger = lambda name=None: _NullLogger()
    backend_pkg.app = app_pkg
    app_pkg.logging_config = logging_stub

    try:
        sys.modules["backend"] = backend_pkg
        sys.modules["backend.app"] = app_pkg
        sys.modules["backend.app.logging_config"] = logging_stub

        spec = importlib.util.spec_from_file_location(
            "mindrouter_dlp_scanner_standalone", resolved,
            submodule_search_locations=[],
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in _STUBBED_MODULES:
            if name in prior:
                sys.modules[name] = prior[name]
            else:
                sys.modules.pop(name, None)

    _module_cache[resolved] = module
    return module


def gliner_available() -> bool:
    """True when the gliner package is importable.  Loads no weights."""
    try:
        return importlib.util.find_spec("gliner") is not None
    except (ImportError, ValueError):
        return False


def _finding_to_dict(f) -> Dict[str, Any]:
    return {
        "scanner": f.scanner,
        "category": f.category,   # raw label; canonicalization is the eval module's job
        "text": f.text,
        "confidence": float(f.confidence),
        "start": f.start,
        "end": f.end,
    }


def scan_documents(
    docs_texts: Dict[str, str],
    scanners: Iterable[str] = ("regex",),
    gliner_threshold: float = 0.5,
    gliner_categories: Optional[List[str]] = None,
    gliner_max_chars: int = 10_000,
    regex_patterns: Optional[List[Dict[str, str]]] = None,
    regex_keywords: Optional[List[str]] = None,
    apply_global_cap: bool = True,
    include_builtins: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Scan a batch of documents with the production scanner code.

    Returns ``{doc_id: {"findings": [...], "latency_ms": {scanner: float},
    "errors": {scanner: str}}}`` in docs_texts order.  Mirrors run_dlp_scan
    semantics: the MAX_SCAN_CHARS global truncation is applied before any
    scanner when ``apply_global_cap``; a gliner failure (model missing, load
    or predict error) lands in errors["gliner"] while regex findings are
    still returned.  All gliner scans share ONE event loop per call, and the
    model itself is cached module-globally inside dlp_scanner, so it loads
    once per process regardless of batching; a discarded warmup call absorbs
    that one-time load BEFORE per-doc timing, so latency samples never
    include it.
    """
    selected = tuple(scanners)
    unknown = [s for s in selected if s not in _SUPPORTED_SCANNERS]
    if unknown:
        raise ValueError(f"unsupported scanners {unknown}; supported: {_SUPPORTED_SCANNERS}")

    mod = load_scanner_module()

    texts: Dict[str, str] = {}
    results: Dict[str, Dict[str, Any]] = {}
    for doc_id, text in docs_texts.items():
        t = text or ""
        if apply_global_cap and len(t) > mod.MAX_SCAN_CHARS:
            t = t[: mod.MAX_SCAN_CHARS]
        texts[doc_id] = t
        results[doc_id] = {"findings": [], "latency_ms": {}, "errors": {}}

    if "regex" in selected:
        for doc_id, t in texts.items():
            t0 = time.perf_counter()
            # include_builtins=False mirrors a prod config whose saved rule
            # list already contains the built-ins (dlp.regex.builtins_in_list).
            found = mod.scan_regex(t, regex_patterns, regex_keywords, include_builtins)
            results[doc_id]["latency_ms"]["regex"] = (time.perf_counter() - t0) * 1000.0
            results[doc_id]["findings"].extend(_finding_to_dict(f) for f in found)

    if "gliner" in selected:
        async def _gliner_batch():
            # One discarded, untimed warmup call so the first timed sample
            # excludes the one-time model load (can take minutes); the model
            # cache is module-global, so repeat warmups return instantly.
            try:
                await mod.scan_gliner("warmup", categories=gliner_categories,
                                      threshold=gliner_threshold,
                                      max_chars=gliner_max_chars)
            except (mod.DlpScannerError, ImportError):
                pass  # the real error resurfaces per-doc with attribution
            for doc_id, t in texts.items():
                t0 = time.perf_counter()
                found = []
                try:
                    found = await mod.scan_gliner(
                        t,
                        categories=gliner_categories,
                        threshold=gliner_threshold,
                        max_chars=gliner_max_chars,
                    )
                except (mod.DlpScannerError, ImportError) as e:
                    results[doc_id]["errors"]["gliner"] = str(e)
                results[doc_id]["latency_ms"]["gliner"] = (time.perf_counter() - t0) * 1000.0
                results[doc_id]["findings"].extend(_finding_to_dict(f) for f in found)

        asyncio.run(_gliner_batch())

    return results


def prod_default_config() -> Dict[str, Any]:
    """Config dict in the exact shape run_dlp_scan expects, production defaults.

    Regex + GLiNER on, LLM off (prod state; flip keys as needed).  Severity
    rules are keyed by RAW scanner labels because classify_severity looks up
    finding.category verbatim; constants.SEVERITY_RULES_OVERRIDE builds them
    from the constants severity model through the alias table so every label
    a scanner can emit maps to the severity the worker would assign (unlisted
    labels fall back to "moderate" inside classify_severity, matching
    production).
    """
    return {
        "regex.enabled": True,
        "regex.patterns": None,
        "regex.keywords": None,
        "gliner.enabled": True,
        "gliner.threshold": 0.5,   # admin-config default (dlp.gliner.threshold)
        "gliner.categories": None,  # None -> scan_gliner's 9-label default list
        "gliner.max_scan_chars": constants.GLINER_DEFAULT_MAX_CHARS,
        "llm.enabled": False,
        "llm.model": "",
        "llm.system_prompt": "",
        "severity_rules": dict(constants.SEVERITY_RULES_OVERRIDE),
    }
