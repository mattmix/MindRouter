############################################################
# test_web_search_audit.py: the web-search audit entity
############################################################
"""Every outbound web search is recorded as a first-class audit row.

Three layers are covered here:
  1. the providers, which must report the FULL round-trip (url, params,
     headers, HTTP status, verbatim body) on success AND on failure;
  2. the audit service, which redacts, caps, and persists — and must never
     break the search it observes;
  3. the wiring, so no call site can quietly bypass the log.
"""
import asyncio
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_SEARCH_DIR = _APP_DIR / "services" / "search"
_DASHBOARD_DIR = _APP_DIR / "dashboard"
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"
_MIGRATIONS_DIR = _APP_DIR / "db" / "migrations" / "versions"

ROUTES_SRC = (_DASHBOARD_DIR / "routes.py").read_text()
AUDIT_SRC = (_SEARCH_DIR / "audit.py").read_text()
CRUD_SRC = (_APP_DIR / "db" / "crud.py").read_text()
RETENTION_SRC = (_APP_DIR / "services" / "retention.py").read_text()


def _code_only(src: str) -> str:
    """Source with comments and docstrings removed.

    Assertions about what the CODE does must not be satisfied — or defeated —
    by prose that merely mentions the thing.
    """
    import io
    import tokenize

    out = []
    prev_type = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, None
            ):
                continue  # docstring
            out.append(tok.string)
            if tok.type not in (tokenize.NL, tokenize.NEWLINE):
                prev_type = tok.type
            else:
                prev_type = tok.type
    except tokenize.TokenError:
        return src
    return " ".join(out)


from backend.app.services.search.base import (  # noqa: E402
    SearchExchange,
    SearchResult,
    attach_exchange,
    exchange_from_exception,
    response_meta,
)
from backend.app.services.search.brave import BraveSearchProvider  # noqa: E402
from backend.app.services.search.searxng import SearXNGSearchProvider  # noqa: E402
from backend.app.services import search as _search_pkg  # noqa: E402
from backend.app.services.search import audit as A  # noqa: E402


# ------------------------------------------------------------------
# httpx stand-in: the providers build their own AsyncClient, so the
# class is swapped rather than a transport injected.
# ------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, body="{}", headers=None):
        self.status_code = status_code
        self.text = body
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=MagicMock(), response=self
            )


def _fake_client(response=None, raise_exc=None, captured=None):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            if captured is not None:
                captured.update({"url": url, "params": params, "headers": headers})
            if raise_exc is not None:
                raise raise_exc
            return response

    return _Client


BRAVE_OK = json.dumps({
    "query": {"original": "cats", "altered": None},
    "web": {"results": [
        {"title": "T1", "url": "https://a", "description": "D1", "page_age": "2026"},
        {"title": "T2", "url": "https://b", "description": "D2"},
    ]},
})


class TestProviderExchange:
    """A provider must surface what an auditor needs, not just the results."""

    def test_brave_success_captures_the_round_trip(self, monkeypatch):
        import backend.app.services.search.brave as bm

        captured = {}
        monkeypatch.setattr(
            bm.httpx, "AsyncClient",
            _fake_client(_FakeResponse(200, BRAVE_OK, {
                "content-type": "application/json",
                "x-ratelimit-remaining": "42",
                "set-cookie": "should-not-be-kept",
            }), captured=captured),
        )
        cfg = {"search.brave.api_key": "SECRET", "search.brave.endpoint": "https://api.example/search"}
        ex = asyncio.run(BraveSearchProvider().search_exchange("cats", max_results=2, config=cfg))

        assert [r.title for r in ex.results] == ["T1", "T2"]
        assert ex.request_url == "https://api.example/search"
        assert ex.request_params == {"q": "cats", "count": 2}
        # the token is carried through verbatim — the audit layer redacts it
        assert ex.request_headers["X-Subscription-Token"] == "SECRET"
        assert ex.http_status == 200
        assert ex.response_body == BRAVE_OK
        hdrs = ex.response_meta["response_headers"]
        assert hdrs.get("x-ratelimit-remaining") == "42"
        assert "set-cookie" not in hdrs, "only the allowlisted headers are kept"
        assert ex.response_meta["provider_query"]["original"] == "cats"
        assert ex.response_meta["total_results_reported"] == 2

    def test_brave_http_error_still_carries_status_and_body(self, monkeypatch):
        import httpx

        import backend.app.services.search.brave as bm

        body = '{"error":"quota exhausted"}'
        monkeypatch.setattr(
            bm.httpx, "AsyncClient", _fake_client(_FakeResponse(429, body))
        )
        cfg = {"search.brave.api_key": "K"}
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            asyncio.run(BraveSearchProvider().search_exchange("x", config=cfg))
        ex = exchange_from_exception(excinfo.value)
        assert ex is not None
        assert ex.http_status == 429
        assert ex.response_body == body, "the body is WHY it failed — it must survive"

    def test_brave_missing_key_attaches_exchange(self):
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(BraveSearchProvider().search_exchange("x", config={}))
        ex = exchange_from_exception(excinfo.value)
        assert ex is not None and ex.http_status is None

    def test_brave_search_still_returns_plain_results(self, monkeypatch):
        import backend.app.services.search.brave as bm

        monkeypatch.setattr(bm.httpx, "AsyncClient", _fake_client(_FakeResponse(200, BRAVE_OK)))
        out = asyncio.run(
            BraveSearchProvider().search("cats", max_results=2, config={"search.brave.api_key": "K"})
        )
        assert isinstance(out, list) and all(isinstance(r, SearchResult) for r in out)

    def test_searxng_captures_engines_and_body(self, monkeypatch):
        import backend.app.services.search.searxng as sm

        body = json.dumps({"results": [
            {"title": "A", "url": "u", "content": "c", "engine": "google"},
            {"title": "B", "url": "u2", "content": "c2", "engine": "bing"},
        ]})
        monkeypatch.setattr(sm.httpx, "AsyncClient", _fake_client(_FakeResponse(200, body)))
        ex = asyncio.run(SearXNGSearchProvider().search_exchange(
            "q", max_results=5, config={"search.searxng.endpoint": "https://sx/"}
        ))
        assert ex.request_url == "https://sx/search"
        assert ex.http_status == 200 and ex.response_body == body
        assert ex.response_meta["engines"] == ["bing", "google"]

    def test_searxng_missing_endpoint_attaches_exchange(self):
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(SearXNGSearchProvider().search_exchange("x", config={}))
        assert exchange_from_exception(excinfo.value) is not None

    def test_base_default_exchange_keeps_unaware_providers_working(self):
        """A provider that never heard of the audit log still logs a row."""
        from backend.app.services.search.base import SearchProvider

        class Legacy(SearchProvider):
            provider_key = "legacy"

            async def search(self, query, *, max_results=5, config=None):
                return [SearchResult(title="t", url="u", snippet="s")]

            async def health_check(self, config=None):
                return True, "OK"

        ex = asyncio.run(Legacy().search_exchange("q"))
        assert len(ex.results) == 1
        assert ex.http_status is None and ex.request_url is None

    def test_attach_exchange_survives_slotted_exceptions(self):
        class Slotted(Exception):
            __slots__ = ()

        e = Slotted("x")
        attach_exchange(e, SearchExchange())  # must not raise
        assert exchange_from_exception(e) in (None, exchange_from_exception(e))

    def test_response_meta_tolerates_a_broken_headers_object(self):
        bad = MagicMock()
        type(bad).headers = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        assert response_meta(bad) == {"response_headers": {}}


class TestRedactionAndCaps:
    def test_credentials_redacted_by_name_not_value(self):
        out = A.redact_mapping({
            "q": "my secret question",       # the QUERY is the point — keep it
            "count": 5,
            "X-Subscription-Token": "abc",
            "api_key": "abc",
            "Authorization": "Bearer x",
            "Cookie": "s=1",
            "nested": {"password": "p", "safe": 1},
        })
        assert out["q"] == "my secret question"
        assert out["count"] == 5
        for k in ("X-Subscription-Token", "api_key", "Authorization", "Cookie"):
            assert out[k] == A.REDACTED, k
        assert out["nested"] == {"password": A.REDACTED, "safe": 1}

    def test_redact_none_and_empty(self):
        assert A.redact_mapping(None) is None
        assert A.redact_mapping({}) is None

    def test_truncate(self):
        assert A._truncate(None, 10) == (None, False)
        assert A._truncate("abc", 10) == ("abc", False)
        assert A._truncate("abcdef", 3) == ("abc", True)
        # limit 0 means "no cap" at this helper; the caller decides to skip
        assert A._truncate("abc", 0) == ("abc", False)

    def test_results_payload_normalizes(self):
        out = A._results_payload([
            SearchResult(title="t", url="u", snippet="s"),
            {"title": "raw"},
        ])
        assert out[0]["title"] == "t" and out[1] == {"title": "raw"}
        assert A._results_payload(None) is None


class _CapturedSession:
    """Stands in for get_async_db_context(): remembers what was added."""

    def __init__(self, fail_first_commit=False):
        self.added = []
        self.commits = 0
        self.fail_first_commit = fail_first_commit
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        if self.fail_first_commit and self.commits == 1:
            raise RuntimeError("FK violation")

    async def rollback(self):
        self.rolled_back = True


def _patch_session(monkeypatch, session):
    mod = types.ModuleType("backend.app.db.session")
    mod.get_async_db_context = lambda: session
    monkeypatch.setitem(sys.modules, "backend.app.db.session", mod)


class TestRecordSearch:
    _CFG = {"enabled": True, "store_body": True, "max_body_chars": 20}

    def test_success_row_fields(self, monkeypatch):
        sess = _CapturedSession()
        _patch_session(monkeypatch, sess)
        ex = SearchExchange(
            results=[SearchResult(title="t", url="u", snippet="s")],
            request_url="https://api/search",
            request_params={"q": "cats", "count": 3},
            request_headers={"X-Subscription-Token": "SECRET"},
            http_status=200,
            response_body="x" * 100,
            response_meta={"response_headers": {"content-type": "application/json"}},
        )
        uuid = asyncio.run(A.record_search(
            query="cats", source="search_api", provider="brave", exchange=ex,
            latency_ms=123, max_results=3, user_id=7, api_key_id=9,
            request_id=None, client_ip="10.0.0.1", audit_config=self._CFG,
        ))
        assert uuid and sess.commits == 1
        row = sess.added[0]
        assert row.status == "success" and row.http_status == 200
        assert row.provider == "brave" and row.source == "search_api"
        assert row.latency_ms == 123 and row.result_count == 1
        assert row.user_id == 7 and row.api_key_id == 9 and row.client_ip == "10.0.0.1"
        # credentials never reach the table
        assert row.request_headers["X-Subscription-Token"] == A.REDACTED
        assert row.request_params["q"] == "cats"
        # body capped and flagged
        assert len(row.response_body) == 20 and row.response_truncated is True
        assert row.results[0]["title"] == "t"

    def test_error_row_records_type_message_and_status(self, monkeypatch):
        sess = _CapturedSession()
        _patch_session(monkeypatch, sess)
        err = RuntimeError("boom")
        attach_exchange(err, SearchExchange(http_status=502, response_body="bad gateway"))
        asyncio.run(A.record_search(
            query="q", source="mcp", provider="brave", error=err,
            latency_ms=5, audit_config=self._CFG,
        ))
        row = sess.added[0]
        assert row.status == "error"
        assert row.error_type == "RuntimeError" and "boom" in row.error_message
        assert row.http_status == 502 and row.response_body == "bad gateway"

    def test_disabled_writes_nothing(self, monkeypatch):
        sess = _CapturedSession()
        _patch_session(monkeypatch, sess)
        out = asyncio.run(A.record_search(
            query="q", source="mcp", provider="brave",
            audit_config={"enabled": False}, latency_ms=1,
        ))
        assert out is None and sess.added == []

    def test_store_body_off_drops_the_body_only(self, monkeypatch):
        sess = _CapturedSession()
        _patch_session(monkeypatch, sess)
        ex = SearchExchange(http_status=200, response_body="hello", results=[])
        asyncio.run(A.record_search(
            query="q", source="mcp", provider="brave", exchange=ex, latency_ms=1,
            audit_config={"enabled": True, "store_body": False, "max_body_chars": 999},
        ))
        row = sess.added[0]
        assert row.response_body is None
        assert row.http_status == 200, "metadata is still logged"

    def test_fk_failure_retries_without_the_request_link(self, monkeypatch):
        """An audit row without the back-link beats no audit row."""
        sess = _CapturedSession(fail_first_commit=True)
        _patch_session(monkeypatch, sess)
        uuid = asyncio.run(A.record_search(
            query="q", source="responses_api", provider="brave",
            exchange=SearchExchange(http_status=200), latency_ms=1,
            request_id=12345, audit_config=self._CFG,
        ))
        assert uuid is not None
        assert sess.rolled_back is True and sess.commits == 2
        assert sess.added[0].request_id == 12345
        assert sess.added[1].request_id is None

    def test_never_raises_when_the_database_is_gone(self, monkeypatch):
        mod = types.ModuleType("backend.app.db.session")

        def _boom():
            raise RuntimeError("no db")

        mod.get_async_db_context = _boom
        monkeypatch.setitem(sys.modules, "backend.app.db.session", mod)
        assert asyncio.run(A.record_search(
            query="q", source="mcp", provider="brave", latency_ms=1,
            audit_config=self._CFG,
        )) is None

    def test_query_and_error_are_capped(self, monkeypatch):
        sess = _CapturedSession()
        _patch_session(monkeypatch, sess)
        asyncio.run(A.record_search(
            query="q" * (A.MAX_QUERY_CHARS + 500), source="mcp", provider="brave",
            error=RuntimeError("e" * (A.MAX_ERROR_CHARS + 500)),
            latency_ms=1, audit_config=self._CFG,
        ))
        row = sess.added[0]
        assert len(row.query) == A.MAX_QUERY_CHARS
        assert len(row.error_message) == A.MAX_ERROR_CHARS


class TestAuditConfig:
    def test_defaults_and_clamping(self, monkeypatch):
        from backend.app.db import crud as real_crud

        vals = {}

        async def cfg(db, key, default=None):
            return vals.get(key, default)

        monkeypatch.setattr(real_crud, "get_config_json", cfg)
        out = asyncio.run(A.load_audit_config(MagicMock()))
        assert out == {
            "enabled": True, "store_body": True,
            "max_body_chars": A.DEFAULT_MAX_BODY_CHARS,
        }

        vals["search.audit.max_body_chars"] = 10 ** 12  # absurd
        out = asyncio.run(A.load_audit_config(MagicMock()))
        assert out["max_body_chars"] == A.ABSOLUTE_MAX_BODY_CHARS

        vals["search.audit.max_body_chars"] = "not-a-number"
        out = asyncio.run(A.load_audit_config(MagicMock()))
        assert out["max_body_chars"] == A.DEFAULT_MAX_BODY_CHARS

    def test_unreadable_config_falls_back_not_off(self, monkeypatch):
        from backend.app.db import crud as real_crud

        async def boom(db, key, default=None):
            raise RuntimeError("db down")

        monkeypatch.setattr(real_crud, "get_config_json", boom)
        out = asyncio.run(A.load_audit_config(MagicMock()))
        assert out["enabled"] is True, "a config outage must not silently stop auditing"


class TestRunLoggedSearch:
    def _provider(self, exchange=None, raises=None):
        p = MagicMock()
        p.provider_key = "brave"

        async def _ex(query, *, max_results=5, config=None):
            if raises is not None:
                raise raises
            return exchange

        p.search_exchange = _ex
        return p

    def _patch_common(self, monkeypatch, recorded):
        async def _rec(**kwargs):
            recorded.append(kwargs)
            return "uuid"

        monkeypatch.setattr(A, "record_search", _rec)
        monkeypatch.setattr(A, "load_audit_config", AsyncMock(return_value={
            "enabled": True, "store_body": True, "max_body_chars": 1000}))

    def test_success_returns_results_and_logs(self, monkeypatch):
        recorded = []
        self._patch_common(monkeypatch, recorded)
        ex = SearchExchange(results=[SearchResult(title="t", url="u", snippet="s")])
        out = asyncio.run(A.run_logged_search(
            MagicMock(), "cats", source="search_api", max_results=3,
            config={"search.provider": "brave"}, provider=self._provider(ex),
            user_id=1, api_key_id=2, request_id=3, client_ip="1.2.3.4",
        ))
        assert [r.title for r in out] == ["t"]
        assert len(recorded) == 1
        rec = recorded[0]
        assert rec["source"] == "search_api" and rec["provider"] == "brave"
        assert rec.get("error") is None and rec["exchange"] is ex
        assert rec["user_id"] == 1 and rec["request_id"] == 3
        assert rec["latency_ms"] is not None

    def test_failure_logs_then_reraises_the_same_exception(self, monkeypatch):
        recorded = []
        self._patch_common(monkeypatch, recorded)
        boom = ValueError("nope")
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(A.run_logged_search(
                MagicMock(), "q", source="mcp",
                config={"search.provider": "brave"},
                provider=self._provider(raises=boom),
            ))
        assert excinfo.value is boom, "callers' except clauses must be unaffected"
        assert len(recorded) == 1 and recorded[0]["error"] is boom

    def test_unknown_provider_is_logged_and_raised(self, monkeypatch):
        recorded = []
        self._patch_common(monkeypatch, recorded)
        registry = types.ModuleType("backend.app.services.search.registry")
        registry.PROVIDERS = {}
        registry.get_search_config = AsyncMock(return_value={"search.provider": "ghost"})
        monkeypatch.setitem(sys.modules, "backend.app.services.search.registry", registry)
        with pytest.raises(ValueError):
            asyncio.run(A.run_logged_search(MagicMock(), "q", source="mcp"))
        assert len(recorded) == 1 and recorded[0]["provider"] == "ghost"


class TestCallSitesAreWired:
    """No surface may reach a provider without going through the log."""

    CALL_SITES = {
        "api/search_api.py": "WebSearchSource.SEARCH_API",
        "api/mcp_server.py": "WebSearchSource.MCP",
        "services/responses_websearch.py": "WebSearchSource.RESPONSES_API",
        "dashboard/routes.py": "WebSearchSource.ADMIN_TEST",
    }

    @pytest.mark.parametrize("relpath,source_const", sorted(CALL_SITES.items()))
    def test_uses_run_logged_search(self, relpath, source_const):
        src = (_APP_DIR / relpath).read_text()
        assert "run_logged_search(" in src, relpath
        assert source_const in src, relpath

    def test_no_surface_calls_provider_search_directly(self):
        """provider.search(...) outside the search package would bypass the log.

        Comments are stripped first: several call sites explain that
        run_logged_search is a drop-in for provider.search(), and that prose
        must not read as a violation.
        """
        offenders = []
        checked = list(self.CALL_SITES) + [
            "services/model_enrichment.py", "dashboard/chat.py",
            "services/search/registry.py",
        ]
        for rel in checked:
            if "provider.search(" in _code_only((_APP_DIR / rel).read_text()):
                offenders.append(rel)
        assert offenders == [], f"these bypass the audit log: {offenders}"

    def test_registry_convenience_wrapper_is_audited(self):
        """registry.search() is public API — a future caller must not slip past."""
        src = (_SEARCH_DIR / "registry.py").read_text()
        fn = src[src.index("async def search("):]
        assert "run_logged_search(" in fn
        assert "provider.search(" not in _code_only(fn)

    def test_legacy_helper_audits_when_given_a_source(self):
        src = (_APP_DIR / "services" / "web_search.py").read_text()
        assert "source: str | None = None" in src
        assert "record_search(" in src
        # and both of its callers pass one
        chat = (_APP_DIR / "dashboard" / "chat.py").read_text()
        enrich = (_APP_DIR / "services" / "model_enrichment.py").read_text()
        assert "WebSearchSource.CHAT_UI.value" in chat
        assert "WebSearchSource.MODEL_ENRICHMENT.value" in enrich

    def test_legacy_helper_still_swallows_errors(self):
        """Its callers rely on [] rather than an exception."""
        src = (_APP_DIR / "services" / "web_search.py").read_text()
        body = src[src.index("async def brave_web_search"):src.index("def _new_exchange")]
        assert "return []" in body and "raise" not in body.replace("raise_for_status", "")


class TestAuditViewer:
    def test_kind_filter_selects_the_web_search_log(self):
        assert 'AUDIT_KINDS = ("requests", "web_search")' in ROUTES_SRC
        fn = ROUTES_SRC[ROUTES_SRC.index("async def admin_audit("):ROUTES_SRC.index("            \"kind\": \"requests\",")]
        assert 'if kind == "web_search":' in fn
        assert "_render_web_search_audit(" in fn

    def test_detail_endpoint_is_admin_read_gated(self):
        assert '@dashboard_router.get("/admin/audit/web-search/{search_uuid}/detail")' in ROUTES_SRC
        fn = ROUTES_SRC[ROUTES_SRC.index("async def admin_web_search_detail"):]
        fn = fn[:fn.index("def _web_search_record")]
        assert "has_admin_read" in fn
        assert "status_code=401" in fn and "status_code=403" in fn
        assert "include_body=True" in fn

    def test_export_supports_web_search_csv_and_json(self):
        fn = ROUTES_SRC[ROUTES_SRC.index("async def _export_web_search_audit"):]
        fn = fn[:fn.index('@dashboard_router.get("/admin/audit/export")')]
        assert "text/csv" in fn and "application/json" in fn
        assert "include_body=include_content" in fn
        assert "WEB_SEARCH_EXPORT_MAX_ROWS" in fn
        # structured columns must not break the CSV grid
        assert 'json.dumps(row[col]' in fn

    def test_filters_are_shared_between_page_and_export(self):
        """A CSV must not disagree with the table it came from."""
        assert ROUTES_SRC.count("_web_search_filters(") >= 3  # def + page + export

    def test_date_end_bound_covers_the_whole_day(self):
        from backend.app.dashboard.routes import _parse_audit_date

        start = _parse_audit_date("2026-08-22")
        end = _parse_audit_date("2026-08-22", end=True)
        assert start.day == 22 and start.hour == 0
        assert end.day == 23 and end.hour == 0
        assert _parse_audit_date("") is None and _parse_audit_date("nope") is None

    def test_search_text_avoids_the_mediumtext_column(self):
        """A LIKE over response_body would table-scan megabytes per row."""
        fn = CRUD_SRC[CRUD_SRC.index("async def search_web_search_logs"):]
        fn = fn[:fn.index("async def get_web_search_log_by_uuid")]
        assert "WebSearchLog.query.ilike" in fn
        assert "WebSearchLog.error_message.ilike" in fn
        assert "response_body" not in _code_only(fn), "a LIKE over MEDIUMTEXT would scan megabytes"

    def test_templates_render_the_web_search_view(self):
        audit = (_TEMPLATES_DIR / "admin" / "audit.html").read_text()
        partial = (_TEMPLATES_DIR / "admin" / "_audit_web_search.html").read_text()
        assert '{% include "admin/_audit_web_search.html" %}' in audit
        assert 'href="/admin/audit?kind=web_search"' in audit
        assert "ws-reveal-btn" in audit and "/admin/audit/web-search/" in audit
        for control in ('name="provider_filter"', 'name="source_filter"',
                        'name="http_status_filter"', 'name="status_filter"',
                        'name="start_date"', 'name="end_date"'):
            assert control in partial, control
        assert 'value="web_search"' in partial, "filters must stay on this log"


class TestPersistenceAndLifecycle:
    def test_model_columns(self, ):
        from backend.app.db.models import WebSearchLog, WebSearchSource, WebSearchStatus

        cols = WebSearchLog.__table__.columns
        for name in ("search_uuid", "created_at", "source", "provider", "query",
                     "status", "http_status", "latency_ms", "result_count",
                     "request_url", "request_params", "request_headers",
                     "results", "response_body", "response_meta",
                     "error_type", "error_message", "user_id", "api_key_id",
                     "request_id", "client_ip", "response_truncated"):
            assert name in cols, name
        assert WebSearchStatus.SUCCESS.value == "success"
        assert "responses_api" in [s.value for s in WebSearchSource]

    def test_indexes_lead_with_filter_and_end_on_time(self):
        from backend.app.db.models import WebSearchLog

        idx = {i.name: [c.name for c in i.columns] for i in WebSearchLog.__table__.indexes}
        assert idx["ix_web_search_logs_provider_time"] == ["provider", "created_at"]
        assert idx["ix_web_search_logs_status_time"] == ["status", "created_at"]
        assert idx["ix_web_search_logs_source_time"] == ["source", "created_at"]

    def test_migration_081(self):
        mig = (_MIGRATIONS_DIR / "20260822_000000_081_web_search_logs.py").read_text()
        assert 'revision = "081"' in mig and 'down_revision = "080"' in mig
        assert 'op.create_table(\n        "web_search_logs"' in mig
        assert "search.audit.enabled" in mig
        assert "retention.web_search_logs_days" in mig
        assert "op.drop_table" in mig

    def test_audit_rows_outlive_users_keys_and_requests(self):
        """Deleting the actor must not erase the record of the outbound call."""
        fn = CRUD_SRC[CRUD_SRC.index("async def delete_user"):]
        fn = fn[:fn.index("async def ", 100)]
        assert "update(WebSearchLog)" in fn
        assert ".values(user_id=None)" in fn
        assert ".values(request_id=None)" in fn
        assert ".values(api_key_id=None)" in fn
        # and the retention purge detaches before deleting requests
        assert "WebSearchLog.request_id.in_(request_ids)" in RETENTION_SRC

    def test_retention_policy_and_purge_category(self):
        assert '"retention.web_search_logs_days": 0' in RETENTION_SRC
        assert '"web_search_logs",' in RETENTION_SRC
        assert "async def cleanup_expired_web_search_logs" in RETENTION_SRC
        fn = RETENTION_SRC[RETENTION_SRC.index("async def cleanup_expired_web_search_logs"):]
        fn = fn[:fn.index("# Categories an admin may purge")]
        # pure ORM: no text() SQL that could become an injection finding
        assert "text(" not in _code_only(fn)
        assert "delete(WebSearchLog)" in fn
        assert '"retention.web_search_logs_days"' in ROUTES_SRC
