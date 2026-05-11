"""MiniMax web search provider."""

from __future__ import annotations

import logging
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)


class MiniMaxSearchProvider(WebSearchProvider):
    """Search via MiniMax Coding Plan API.

    Requires MiniMax OAuth credentials.
    """

    def provider_name(self) -> str:
        return "minimax"

    def is_configured(self) -> bool:
        """Return True when MiniMax OAuth credentials are available."""
        try:
            from hermes_cli.auth import get_provider_auth_state
            state = get_provider_auth_state("minimax-oauth")
            return bool(state and state.get("access_token"))
        except Exception:
            return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against MiniMax Coding Plan API."""
        import httpx
        from hermes_cli.auth import resolve_minimax_oauth_runtime_credentials

        try:
            creds = resolve_minimax_oauth_runtime_credentials()
            api_key = creds["api_key"]
            from agent.minimax_client import _parse_root_url
            root_url = _parse_root_url(creds["base_url"])
        except Exception as exc:
            return {"success": False, "error": f"Failed to resolve MiniMax credentials: {exc}"}

        search_url = f"{root_url}/v1/coding_plan/search"

        try:
            resp = httpx.post(
                search_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={"q": query},
                timeout=30,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("MiniMax Search HTTP error: %s", exc)
            return {"success": False, "error": f"MiniMax returned HTTP {exc.response.status_code}"}
        except httpx.RequestError as exc:
            logger.warning("MiniMax Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach MiniMax Search at {search_url}: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("MiniMax Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse MiniMax response as JSON"}

        # MiniMax results format: [{"title": "...", "link": "...", "snippet": "..."}, ...]
        raw_results = data if isinstance(data, list) else data.get("results", [])
        if not isinstance(raw_results, list):
            # Sometimes wrapped in a 'results' or 'data' key depending on internal proxying
            raw_results = data.get("data", []) if isinstance(data, dict) else []

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("link", r.get("url", ""))),
                "description": str(r.get("snippet", r.get("description", ""))),
                "position": i + 1,
            }
            for i, r in enumerate(raw_results[:limit])
        ]

        logger.info("MiniMax search '%s': %d results", query, len(web_results))
        return {"success": True, "data": {"web": web_results}}
