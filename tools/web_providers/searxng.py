"""SearXNG web search provider.

SearXNG is a free, self-hosted, privacy-respecting metasearch engine.
It implements ``WebSearchProvider`` only — there is no extract capability.

Configuration::

    # ~/.hermes/.env
    SEARXNG_URL=http://localhost:8080

    # Use SearXNG for search, pair with any extract provider:
    # ~/.hermes/config.yaml
    web:
      search_backend: "searxng"
      extract_backend: "firecrawl"

Public SearXNG instances are listed at https://searx.space/ but self-hosting
is recommended for production use (rate limits and availability vary per
public instance).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)


class SearXNGSearchProvider(WebSearchProvider):
    """Search via a SearXNG instance.

    Requires ``SEARXNG_URL`` to be set (e.g. ``http://localhost:8080``).
    No API key needed — SearXNG is open-source and self-hosted.

    Uses the SearXNG JSON API (``/search?format=json``).  Results are
    sorted by SearXNG's own score and truncated to *limit*.
    """

    def provider_name(self) -> str:
        return "searxng"

    def is_configured(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set to a non-empty value."""
        return bool(os.getenv("SEARXNG_URL", "").strip())

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance.

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

        base_url = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }

        try:
            resp = httpx.get(
                f"{base_url}/search",
                params=params,
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {"success": False, "error": f"SearXNG returned HTTP {exc.response.status_code}"}
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {"success": False, "error": f"Could not reach SearXNG at {base_url}: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG response parse error: %s", exc)
            return {"success": False, "error": "Could not parse SearXNG response as JSON"}

        raw_results = data.get("results", [])

        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(
            raw_results,
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}
