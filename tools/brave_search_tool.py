"""Advanced Brave Search API tool."""

from __future__ import annotations

import json
from typing import Any, Dict

from plugins.web.brave_search.client import BraveSearchApiClient, is_brave_search_configured
from tools.registry import registry

_MODES = {"both", "web", "llm", "images", "news", "videos", "discussions", "suggest", "raw"}
_MAX_QUERY_CHARS = 500
_MAX_OPTION_CHARS = 64


def _safe_limit(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 5
    return max(1, min(number, 20))


def _empty_data() -> Dict[str, Any]:
    return {
        "web": [],
        "news": [],
        "videos": [],
        "images": [],
        "discussions": [],
        "faq": [],
        "infobox": [],
        "locations": [],
        "llm_context": [],
        "suggestions": [],
    }


def _parse_raw(value: Any) -> bool:
    return value is True


def _validated_text(value: Any, *, default: str = "", max_length: int = _MAX_OPTION_CHARS) -> str:
    text = str(value or default).strip()
    return text[:max_length]


def _merge_data(target: Dict[str, Any], source: Dict[str, Any], *, raw_key: str | None = None) -> None:
    for key, value in source.items():
        if key == "raw" and raw_key:
            target[raw_key] = value
            continue
        if key in target and isinstance(target[key], list) and isinstance(value, list):
            target[key] = value
        else:
            target[key] = value


def brave_search_tool(args: Dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"success": False, "error": "query is required"})
    if len(query) > _MAX_QUERY_CHARS:
        return json.dumps({"success": False, "error": f"query must be {_MAX_QUERY_CHARS} characters or fewer"})

    mode = str(args.get("mode") or "both").strip().lower()
    raw_flag = _parse_raw(args.get("raw", False))
    if mode not in _MODES:
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported Brave Search API mode: {mode}",
            }
        )

    limit = _safe_limit(args.get("limit", 5))
    country = _validated_text(args.get("country"), default="ALL")
    search_lang = _validated_text(args.get("search_lang"), default="en")
    safesearch = _validated_text(args.get("safesearch"), default="moderate")
    if safesearch not in {"off", "moderate", "strict"}:
        return json.dumps({"success": False, "error": "safesearch must be one of: off, moderate, strict"})
    freshness = _validated_text(args.get("freshness"), default="") or None
    try:
        max_llm_tokens = int(args.get("max_llm_tokens", 8192))
    except (TypeError, ValueError):
        max_llm_tokens = 8192

    client = BraveSearchApiClient()
    options = {
        "country": country,
        "search_lang": search_lang,
        "safesearch": safesearch,
        "freshness": freshness,
        "raw": raw_flag or mode == "raw",
    }

    selected_mode = "both" if mode == "raw" else mode
    data = _empty_data()
    errors = []

    def merge_response(response: Dict[str, Any], endpoint: str, *, allow_partial: bool = False) -> bool:
        if not response.get("success"):
            if allow_partial:
                errors.append({"endpoint": endpoint, "error": response.get("error", "unknown error")})
                return False
            raise RuntimeError(response.get("error", "unknown error"))
        _merge_data(data, response.get("data", {}), raw_key=f"raw_{endpoint}")
        return True

    if selected_mode in {"both", "web"}:
        response = client.search_web(query, limit, **options)
        try:
            merge_response(response, "web", allow_partial=selected_mode == "both")
        except RuntimeError as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    if selected_mode in {"both", "llm"}:
        response = client.llm_context(
            query,
            limit,
            country=country,
            search_lang=search_lang,
            safesearch=safesearch,
            freshness=freshness,
            max_llm_tokens=max_llm_tokens,
            raw=raw_flag or mode == "raw",
        )
        try:
            merge_response(response, "llm_context", allow_partial=selected_mode == "both")
        except RuntimeError as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    if selected_mode == "images":
        response = client.search_images(query, limit, **options)
        if not response.get("success"):
            return json.dumps(response, ensure_ascii=False)
        _merge_data(data, response.get("data", {}), raw_key="raw_images")

    if selected_mode == "news":
        response = client.search_news(query, limit, **options)
        if not response.get("success"):
            return json.dumps(response, ensure_ascii=False)
        _merge_data(data, response.get("data", {}), raw_key="raw_news")

    if selected_mode == "videos":
        response = client.search_videos(query, limit, **options)
        if not response.get("success"):
            return json.dumps(response, ensure_ascii=False)
        _merge_data(data, response.get("data", {}), raw_key="raw_videos")

    if selected_mode == "discussions":
        response = client.search_discussions(query, limit, **options)
        if not response.get("success"):
            return json.dumps(response, ensure_ascii=False)
        _merge_data(data, response.get("data", {}), raw_key="raw_discussions")

    if selected_mode == "suggest":
        response = client.suggest(query, country=country, raw=raw_flag or mode == "raw")
        if not response.get("success"):
            return json.dumps(response, ensure_ascii=False)
        _merge_data(data, response.get("data", {}), raw_key="raw_suggest")

    if selected_mode == "both" and errors:
        data["errors"] = errors

    return json.dumps(
        {
            "success": selected_mode != "both" or len(errors) < 2,
            "mode": mode,
            "query": query,
            "data": data,
        },
        ensure_ascii=False,
    )


BRAVE_SEARCH_SCHEMA = {
    "name": "brave_search",
    "description": (
        "Search Brave Search API with paid and grounding modes. Default both "
        "returns web results plus grounding context. Use directly for news, "
        "videos, images, discussions, suggestions, and Brave context mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "mode": {
                "type": "string",
                "enum": ["both", "web", "llm", "images", "news", "videos", "discussions", "suggest", "raw"],
                "default": "both",
            },
            "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            "country": {"type": "string", "default": "ALL"},
            "search_lang": {"type": "string", "default": "en"},
            "safesearch": {
                "type": "string",
                "enum": ["off", "moderate", "strict"],
                "default": "moderate",
            },
            "freshness": {
                "type": "string",
                "description": "Optional Brave freshness value, such as pd, pw, pm, py, or a supported date range.",
            },
            "max_llm_tokens": {"type": "integer", "default": 8192, "minimum": 1024, "maximum": 32768},
            "raw": {"type": "boolean", "default": False},
        },
        "required": ["query"],
    },
}


registry.register(
    name="brave_search",
    toolset="web",
    schema=BRAVE_SEARCH_SCHEMA,
    handler=lambda args, **kw: brave_search_tool(args),
    check_fn=is_brave_search_configured,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"],
    emoji="🔎",
    max_result_size_chars=100_000,
)
