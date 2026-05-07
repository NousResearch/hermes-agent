"""SearXNG web search provider.

SearXNG is a free, self-hosted, privacy-respecting metasearch engine.
It implements ``WebSearchProvider`` only; there is no extract capability.

Configuration lives in ``config.yaml`` because the endpoint URL is not a
secret::

    web:
      search_backend: "searxng"
      extract_backend: "firecrawl"
      searxng:
        base_url: "http://127.0.0.1:8080"

``SEARXNG_URL`` is still accepted as a backwards-compatible non-secret mirror.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)


def _searxng_config() -> Dict[str, Any]:
    """Load SearXNG config, with ``SEARXNG_URL`` as a legacy mirror."""
    try:
        from hermes_cli.config import load_config

        web_cfg = (load_config() or {}).get("web") or {}
    except Exception:
        web_cfg = {}
    raw = web_cfg.get("searxng") if isinstance(web_cfg, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    env_base_url = os.getenv("SEARXNG_URL", "").strip()
    configured_base_url = str(raw.get("base_url") or env_base_url).strip()
    return {
        "enabled": bool(raw.get("enabled", True)),
        "base_url": configured_base_url.rstrip("/"),
        "timeout": float(raw.get("timeout") or 15),
        "categories": str(raw.get("categories") or ""),
        "language": str(raw.get("language") or "auto"),
        "safesearch": raw.get("safesearch", 0),
    }


class SearXNGSearchProvider(WebSearchProvider):
    """Search via a configured SearXNG instance."""

    def provider_name(self) -> str:
        return "searxng"

    def is_configured(self) -> bool:
        cfg = _searxng_config()
        return bool(cfg["enabled"] and cfg["base_url"])

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search and return normalized ``data.web`` results."""
        import httpx

        cfg = _searxng_config()
        base_url = str(cfg["base_url"] or "").strip().rstrip("/")
        if not cfg["enabled"] or not base_url:
            return {
                "success": False,
                "error": "SearXNG is not configured. Set web.searxng.base_url in config.yaml or SEARXNG_URL.",
            }

        params: Dict[str, Any] = {"q": query, "format": "json", "pageno": 1}
        if cfg["categories"]:
            params["categories"] = cfg["categories"]
        if cfg["language"] and str(cfg["language"]).lower() != "auto":
            params["language"] = cfg["language"]
        if cfg["safesearch"] not in (None, ""):
            params["safesearch"] = cfg["safesearch"]

        try:
            resp = httpx.get(
                f"{base_url}/search",
                params=params,
                timeout=float(cfg["timeout"]),
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"SearXNG returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("SearXNG response parse error: %s", exc)
            return {"success": False, "error": "Could not parse SearXNG response as JSON"}

        raw_results = data.get("results", [])
        sorted_results = sorted(
            [r for r in raw_results if isinstance(r, dict)],
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
                "source_backend": "searxng",
                "engine": r.get("engine") or r.get("engines") or "",
                "category": r.get("category") or "",
                "score": r.get("score"),
                "published_date": r.get("publishedDate")
                or r.get("published_date")
                or "",
            }
            for i, r in enumerate(sorted_results)
            if r.get("url")
        ]

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results, "backend": "searxng"}}
