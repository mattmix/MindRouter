############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# web_search.py: Brave Search API integration for web search
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Brave Web Search integration for injecting live web results into chat context."""

import time

import httpx

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def brave_web_search(
    query: str,
    num_results: int = 5,
    api_key: str | None = None,
    *,
    source: str | None = None,
    user_id: int | None = None,
    api_key_id: int | None = None,
    request_id: int | None = None,
    client_ip: str | None = None,
) -> list[dict]:
    """Search the web via Brave Search API.

    Args:
        query: Search query string.
        num_results: Number of results to return.
        api_key: Optional API key override. Falls back to settings if not provided.
        source: When given, the round-trip is written to the web-search audit
            log under this source (see WebSearchSource). Omitted for callers
            with no auditable context.
        user_id, api_key_id, request_id, client_ip: audit attribution.

    Returns a list of dicts with keys: title, url, description.
    Returns empty list on any failure (missing key, timeout, API error).

    This is the pre-registry helper, kept because two callers depend on its
    swallow-everything contract. It records the SAME audit row as the provider
    path so "every outbound search is logged" holds with no exceptions.
    """
    if not api_key:
        settings = get_settings()
        api_key = settings.brave_search_api_key
    if not api_key:
        if source:
            await _audit(
                query=query, source=source, num_results=num_results,
                exchange=None, error=ValueError("Brave Search API key is not configured"),
                latency_ms=0, user_id=user_id, api_key_id=api_key_id,
                request_id=request_id, client_ip=client_ip,
            )
        return []

    params = {"q": query, "count": num_results}
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,  # redacted by the audit layer
    }
    exchange = _new_exchange(BRAVE_SEARCH_URL, params, headers)
    started = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(BRAVE_SEARCH_URL, params=params, headers=headers)
            if exchange is not None:
                exchange.http_status = resp.status_code
                exchange.response_body = resp.text
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            })
        if source:
            if exchange is not None:
                exchange.results = results
            await _audit(
                query=query, source=source, num_results=num_results,
                exchange=exchange, error=None,
                latency_ms=int((time.monotonic() - started) * 1000),
                user_id=user_id, api_key_id=api_key_id,
                request_id=request_id, client_ip=client_ip,
            )
        return results

    except Exception as e:
        logger.warning("brave_web_search failed", exc_info=True)
        if source:
            await _audit(
                query=query, source=source, num_results=num_results,
                exchange=exchange, error=e,
                latency_ms=int((time.monotonic() - started) * 1000),
                user_id=user_id, api_key_id=api_key_id,
                request_id=request_id, client_ip=client_ip,
            )
        return []


def _new_exchange(url, params, headers):
    """A SearchExchange to fill in, or None if the search package is absent."""
    try:
        from backend.app.services.search.base import SearchExchange

        return SearchExchange(
            request_url=url, request_params=dict(params), request_headers=dict(headers)
        )
    except Exception:  # pragma: no cover — import cycle safety net
        return None


async def _audit(*, query, source, num_results, exchange, error, latency_ms,
                 user_id, api_key_id, request_id, client_ip) -> None:
    """Write the audit row; never raises (record_search swallows its own)."""
    try:
        from backend.app.services.search.audit import record_search

        await record_search(
            query=query, source=source, provider="brave",
            exchange=exchange, error=error, latency_ms=latency_ms,
            max_results=num_results, user_id=user_id, api_key_id=api_key_id,
            request_id=request_id, client_ip=client_ip,
        )
    except Exception:  # pragma: no cover
        logger.warning("brave_web_search audit failed", exc_info=True)


def format_search_results(results: list[dict]) -> str:
    """Format search results into a context block for system prompt injection."""
    if not results:
        return ""

    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r["description"]:
            lines.append(f"   {r['description']}")
    lines.append(
        "\nUse the above web search results to inform your answer. "
        "Cite sources with URLs when relevant. If the search results are not "
        "relevant to the user's question, you may ignore them."
    )
    return "\n".join(lines)
