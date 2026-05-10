"""TinyFish web search and fetch provider.

TinyFish offers free Search and Fetch APIs for all developers — no credit card
required. See https://tinyfish.ai for details.

This provider implements both ``WebSearchProvider`` (via the Search API) and
``WebExtractProvider`` (via the Fetch API). Authentication is via the
``X-API-Key`` header.

Configuration::

    # ~/.hermes/.env
    TINYFISH_API_KEY=tf-your-key-here

    # ~/.hermes/config.yaml
    web:
      search_backend: "tinyfish"
      extract_backend: "tinyfish"

Rate limits:
    - Search: 5 requests per minute
    - Fetch: 1 credit = 15 URL fetches (credit consumption varies by plan)

See https://docs.tinyfish.ai/search-api/reference and
https://docs.tinyfish.ai/fetch-api/reference for full API docs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from tools.web_providers.base import WebExtractProvider, WebSearchProvider

logger = logging.getLogger(__name__)

_TINYFISH_SEARCH_ENDPOINT = "https://api.search.tinyfish.ai"
_TINYFISH_FETCH_ENDPOINT = "https://api.fetch.tinyfish.ai"

_MAX_FETCH_URLS = 10  # Fetch API max URLs per request


class TinyFishSearchProvider(WebSearchProvider):
    """Search via the TinyFish Search API.

    Requires ``TINYFISH_API_KEY`` to be set. The value is passed as the
    ``X-API-Key`` header. Returns up to 10 results per query (fixed by the API).
    """

    def provider_name(self) -> str:
        return "tinyfish"

    def is_configured(self) -> bool:
        """Return True when ``TINYFISH_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("TINYFISH_API_KEY", "").strip())

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the TinyFish Search API.

        Returns normalized results::

            {
                "success": True,
                "data": {
                    "web": [
                        {
                            "title": str,
                            "url": str,
                            "description": str,
                            "position": int,
                        },
                        ...
                    ]
                }
            }

        On failure returns ``{"success": False, "error": str}``.
        """
        import httpx

        api_key = os.getenv("TINYFISH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "TINYFISH_API_KEY is not set"}

        safe_limit = max(1, int(limit))

        try:
            resp = httpx.get(
                _TINYFISH_SEARCH_ENDPOINT,
                params={"query": query},
                headers={
                    "X-API-Key": api_key,
                    "Accept": "application/json",
                },
                timeout=15,
            )
            # 429 rate limit gets special handling
            if resp.status_code == 429:
                logger.warning("TinyFish Search rate limited (429)")
                return {
                    "success": False,
                    "error": (
                        "TinyFish Search rate limit exceeded (5 req/min). "
                        "Retry after 15s with exponential backoff."
                    ),
                }
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("TinyFish Search HTTP error: %s", exc)
            status = exc.response.status_code
            detail = _search_error_detail(status)
            return {"success": False, "error": f"TinyFish Search returned HTTP {status}: {detail}"}
        except httpx.RequestError as exc:
            logger.warning("TinyFish Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach TinyFish Search: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("TinyFish Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse TinyFish Search response as JSON"}

        raw_results = data.get("results", []) or []
        truncated = raw_results[:safe_limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("snippet", "")),
                "position": int(r.get("position", i + 1)),
            }
            for i, r in enumerate(truncated)
        ]

        logger.info(
            "TinyFish Search '%s': %d results (limit %d, total %d)",
            query,
            len(web_results),
            limit,
            data.get("total_results", 0),
        )
        return {"success": True, "data": {"web": web_results}}


class TinyFishExtractProvider(WebExtractProvider):
    """Extract page content via the TinyFish Fetch API.

    Requires ``TINYFISH_API_KEY`` to be set. Uses a real browser to render
    JavaScript-heavy pages and returns clean extracted text. Up to 10 URLs
    per request.

    Supports ``format`` parameter in ``extract()`` kwargs (``"markdown"``,
    ``"html"``, or ``"json"``). Defaults to ``"markdown"`` (best for LLMs).
    """

    def provider_name(self) -> str:
        return "tinyfish"

    def is_configured(self) -> bool:
        """Return True when ``TINYFISH_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("TINYFISH_API_KEY", "").strip())

    def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Extract content from the given URLs using the TinyFish Fetch API.

        Returns normalized results::

            {
                "success": True,
                "data": [
                    {
                        "url": str,
                        "title": str,
                        "content": str,
                        "raw_content": str,
                        "metadata": dict,
                    },
                    ...
                ]
            }

        On failure returns ``{"success": False, "error": str}``.
        """
        import httpx

        api_key = os.getenv("TINYFISH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "TINYFISH_API_KEY is not set"}

        if not urls:
            return {"success": True, "data": []}

        # Fetch API max: 10 URLs per request
        fetch_urls = urls[: _MAX_FETCH_URLS]

        format = kwargs.get("format", "markdown")
        if format not in ("markdown", "html", "json"):
            format = "markdown"

        payload = {
            "urls": fetch_urls,
            "format": format,
        }

        try:
            resp = httpx.post(
                _TINYFISH_FETCH_ENDPOINT,
                json=payload,
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=60,
            )
            if resp.status_code == 429:
                logger.warning("TinyFish Fetch rate limited (429)")
                return {
                    "success": False,
                    "error": "TinyFish Fetch rate limit exceeded. Retry with backoff.",
                }
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("TinyFish Fetch HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"TinyFish Fetch returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("TinyFish Fetch request error: %s", exc)
            return {"success": False, "error": f"Could not reach TinyFish Fetch: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("TinyFish Fetch response parse error: %s", exc)
            return {"success": False, "error": "Could not parse TinyFish Fetch response as JSON"}

        results: List[Dict[str, Any]] = []
        raw_results = data.get("results", []) or []
        for item in raw_results:
            url = str(item.get("url", ""))
            final_url = str(item.get("final_url", url))
            title = str(item.get("title") or "")
            text = item.get("text", "")

            # text can be string or object depending on format
            if isinstance(text, dict):
                content = str(text)
            else:
                content = str(text or "")

            metadata = {
                "final_url": final_url,
                "description": str(item.get("description") or ""),
                "language": str(item.get("language") or ""),
                "author": str(item.get("author") or ""),
                "published_date": str(item.get("published_date") or ""),
                "latency_ms": item.get("latency_ms"),
            }

            results.append({
                "url": url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": metadata,
            })

        # Report per-URL errors from the API
        fetch_errors = data.get("errors", []) or []
        for err in fetch_errors:
            err_url = err.get("url", "unknown")
            err_type = err.get("error", "unknown")
            logger.info("TinyFish Fetch error for %s: %s", err_url, err_type)
            # Add error entries for URLs that failed
            if not any(r["url"] == err_url for r in results):
                results.append({
                    "url": err_url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"TinyFish Fetch failed: {err_type}",
                    "metadata": {},
                })

        logger.info(
            "TinyFish Fetch: %d/%d URLs extracted successfully",
            len(raw_results),
            len(fetch_urls),
        )
        return {"success": True, "data": results}


def _search_error_detail(status: int) -> str:
    """Return a human-readable detail message for known Search API HTTP status codes."""
    details = {
        400: "Missing or invalid query parameter",
        401: "Missing or invalid API key",
        402: "Active subscription or credits needed",
        403: "Search API access not enabled for this account",
        404: "Search API not available",
        429: "Rate limit exceeded (5 req/min)",
        500: "Internal server error",
        503: "Search service unavailable — retry with backoff",
    }
    return details.get(status, "Unknown error")
