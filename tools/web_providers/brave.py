"""Brave Search web search provider.

Brave Search uses its own independent index (not Google or Bing), making it
a privacy-focused alternative to Tavily/Exa/Parallel/Firecrawl.

A free tier is available at https://brave.com/search/api/ — 2,000 queries/month
with no credit card required.

Configuration::

    # ~/.hermes/.env  (or export in shell)
    BRAVE_API_KEY=your-key-here

    # ~/.hermes/config.yaml
    web:
      search_backend: "brave"
      # extract_backend defaults to firecrawl when brave is the search backend
      # (Brave Search does not offer a content-extraction endpoint)

Get an API key at: https://api.search.brave.com/app/keys
Pricing: https://brave.com/search/api/
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider(WebSearchProvider):
    """Search via the Brave Search API.

    Requires ``BRAVE_API_KEY`` to be set.
    Uses ``X-Subscription-Token`` header authentication (no OAuth complexity).
    Brave has its own independent web index — results differ from Google/Bing.

    Extract/crawl are not supported by the Brave Search API.  When ``brave``
    is selected as ``search_backend``, the extract backend falls back to
    whichever extract provider is configured (defaulting to ``firecrawl``).
    """

    def provider_name(self) -> str:
        return "brave"

    def is_configured(self) -> bool:
        """Return True when ``BRAVE_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("BRAVE_API_KEY", "").strip())

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search via the Brave Search API.

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

        api_key = os.getenv("BRAVE_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BRAVE_API_KEY is not set"}

        params: Dict[str, Any] = {
            "q": query,
            "count": min(limit, 20),  # Brave API max is 20 per request
            "text_decorations": "false",
            "search_lang": "en",
            "country": "us",
            "safesearch": "moderate",
        }

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

        try:
            resp = httpx.get(
                _BRAVE_SEARCH_ENDPOINT,
                params=params,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                return {"success": False, "error": "BRAVE_API_KEY is invalid or expired"}
            if status == 422:
                return {"success": False, "error": "Brave Search rejected the query (422)"}
            if status == 429:
                return {"success": False, "error": "Brave Search rate limit exceeded — try again later"}
            logger.warning("Brave Search HTTP error: %s", exc)
            return {"success": False, "error": f"Brave Search returned HTTP {status}"}
        except httpx.RequestError as exc:
            logger.warning("Brave Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Brave Search API: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Brave Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Brave Search response as JSON"}

        raw_results = data.get("web", {}).get("results", [])

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("description", "")),
                "position": i + 1,
            }
            for i, r in enumerate(raw_results[:limit])
        ]

        logger.info(
            "Brave Search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}
