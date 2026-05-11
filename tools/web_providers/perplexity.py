"""Perplexity Search API backend.

Thin search-only client for ``POST https://api.perplexity.ai/search``.

The Perplexity Search API returns full ranked web results (URL + title +
snippet + content) directly — no LLM in the loop — making it a drop-in
replacement for Exa, Tavily, Parallel, or Brave for the ``web_search``
capability.

Auth via ``PERPLEXITY_API_KEY`` (with ``PPLX_API_KEY`` accepted as a
fallback). Every request carries an ``X-Pplx-Integration: hermes-agent``
header for usage attribution.

This module is search-only by design. ``web_extract`` continues to use the
configured extract backend (Firecrawl by default), per the per-capability
selection pattern in ``tools/web_tools.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
_INTEGRATION_HEADER = "X-Pplx-Integration"
# Bumping this version on shipped behavior changes lets Perplexity see which
# Hermes builds are sending traffic.
_INTEGRATION_VALUE = "hermes-agent/1.0"
_DEFAULT_TIMEOUT_SECONDS = 30.0


def _get_perplexity_api_key() -> str:
    """Return the configured Perplexity API key, accepting either env name."""
    key = (os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Get your API key at https://www.perplexity.ai/account/api/keys"
        )
    return key


def _perplexity_search_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST a search payload to the Perplexity Search API and return JSON."""
    headers = {
        "Authorization": f"Bearer {_get_perplexity_api_key()}",
        "Content-Type": "application/json",
        _INTEGRATION_HEADER: _INTEGRATION_VALUE,
    }
    response = httpx.post(
        _PERPLEXITY_SEARCH_URL,
        json=payload,
        headers=headers,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _normalize_perplexity_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Perplexity Search API response to Hermes' web_search shape."""
    items = raw.get("results") or []
    web_results = []
    for i, item in enumerate(items):
        snippet = item.get("snippet") or item.get("content") or ""
        web_results.append(
            {
                "url": item.get("url") or "",
                "title": item.get("title") or "",
                "description": snippet,
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def perplexity_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search the web via the Perplexity Search API.

    Parameters
    ----------
    query : str
        Search query string.
    limit : int
        Maximum number of results to return (clamped to 1..20 to match other
        Hermes web backends).

    Returns
    -------
    dict
        ``{"success": True, "data": {"web": [{"url", "title", "description", "position"}, ...]}}``
    """
    from tools.interrupt import is_interrupted

    if is_interrupted():
        return {"error": "Interrupted", "success": False}

    try:
        requested_limit = int(limit) if limit is not None else 10
    except (TypeError, ValueError):
        requested_limit = 10
    bounded = max(1, min(requested_limit, 20))
    logger.info("Perplexity search: '%s' (limit=%d)", query, bounded)

    payload: Dict[str, Any] = {
        "query": query,
        "max_results": bounded,
    }

    raw = _perplexity_search_request(payload)
    return _normalize_perplexity_results(raw)


class PerplexitySearchProvider(WebSearchProvider):
    """Search-only provider for Perplexity's Search API."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity"

    def is_available(self) -> bool:
        return bool((os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY") or "").strip())

    def supports_search(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        try:
            return perplexity_search(query, limit)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except httpx.HTTPError as exc:
            logger.warning("Perplexity search HTTP error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Perplexity search error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity",
            "badge": "★ recommended",
            "tag": "Perplexity Search API — ranked web results, no LLM in the loop",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": "https://www.perplexity.ai/account/api/keys",
                },
            ],
        }
