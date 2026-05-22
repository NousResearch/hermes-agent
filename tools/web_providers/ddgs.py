"""DuckDuckGo web search provider via the ``ddgs`` Python package.

DuckDuckGo does not provide an official programmatic search API.  The
community-maintained `ddgs <https://pypi.org/project/ddgs/>`_ package (the
renamed successor of ``duckduckgo-search``) scrapes DuckDuckGo's HTML results
page and normalizes them.  It implements ``WebSearchProvider`` only — there is
no extract capability.

Configuration::

    # No API key required. Enable by installing the package and pointing the
    # web backend at ddgs:
    pip install ddgs

    # ~/.hermes/config.yaml
    web:
      search_backend: "ddgs"
      extract_backend: "firecrawl"    # pair with an extract provider if needed

Rate limits are enforced server-side by DuckDuckGo.  Expect intermittent
``DuckDuckGoSearchException`` / 202 responses under heavy use; this provider
surfaces them as ``{"success": False, "error": ...}`` rather than crashing
the tool call.

See https://duckduckgo.com/?q=duckduckgo+tos for terms of use.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)


class DDGSSearchProvider(WebSearchProvider):
    """Search via the ``ddgs`` package (DuckDuckGo HTML scrape).

    No API key required.  The provider is considered "configured" when the
    ``ddgs`` package is importable — there is nothing else to set up.
    """

    def provider_name(self) -> str:
        return "ddgs"

    def is_configured(self) -> bool:
        """Return True when the ``ddgs`` package is importable.

        Called at tool-registration time; must not perform network I/O.
        """
        try:
            import ddgs  # noqa: F401
            return True
        except ImportError:
            return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a DuckDuckGo search and return normalized results.

        Returns ``{"success": True, "data": {"web": [...]}}`` on success or
        ``{"success": False, "error": str}`` on failure (missing package,
        rate-limited, network error, etc.).
        """
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            return {
                "success": False,
                "error": "ddgs package is not installed — run `pip install ddgs`",
            }

        # DDGS().text yields at most `max_results` items; we cap defensively
        # in case the package ignores the hint.
        safe_limit = max(1, int(limit))

        try:
            web_results = []
            with DDGS() as client:
                for i, hit in enumerate(client.text(query, max_results=safe_limit)):
                    if i >= safe_limit:
                        break
                    url = str(hit.get("href") or hit.get("url") or "")
                    web_results.append(
                        {
                            "title": str(hit.get("title", "")),
                            "url": url,
                            "description": str(hit.get("body", "")),
                            "position": i + 1,
                        }
                    )
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

        logger.info("DDGS search '%s': %d results (limit %d)", query, len(web_results), limit)
        return {"success": True, "data": {"web": web_results}}
