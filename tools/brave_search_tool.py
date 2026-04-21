#!/usr/bin/env python3
"""Brave Search native tools.

This module exposes Brave's web search, query suggestions, and AI answers
as first-class Hermes tools. It is intentionally self-contained so it can be
imported by the normal tool registry auto-discovery path.

Brave API docs used by this implementation:
- Web search: GET /res/v1/web/search
- Suggestions: GET /res/v1/suggest/search
- AI answers: POST /res/v1/chat/completions
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_BRAVE_BASE_URL = "https://api.search.brave.com/res/v1"
_BRAVE_DEFAULT_MODEL = "brave-pro"


def _brave_api_key() -> str:
    """Return the configured Brave API key, honoring both supported env names."""
    return (
        os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        or os.getenv("BRAVE_API_KEY", "").strip()
    )


def check_brave_api_key() -> bool:
    """Return True when Brave Search credentials are available."""
    return bool(_brave_api_key())


def _brave_headers() -> Dict[str, str]:
    api_key = _brave_api_key()
    if not api_key:
        raise ValueError(
            "BRAVE_SEARCH_API_KEY environment variable not set. "
            "Get your Brave Search API key at https://api-dashboard.search.brave.com"
        )
    return {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _safe_int(value: Any, default: int, minimum: int = 1, maximum: int = 20) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _normalize_result_item(item: Dict[str, Any], position: int) -> Dict[str, Any]:
    description = (
        item.get("description")
        or item.get("snippet")
        or item.get("extra_snippets")
        or item.get("summary")
        or ""
    )
    if isinstance(description, list):
        description = "\n".join(str(x) for x in description if x)
    return {
        "title": str(item.get("title") or item.get("name") or ""),
        "url": str(item.get("url") or item.get("link") or ""),
        "description": str(description),
        "position": position,
    }


def _normalize_search_response(payload: Dict[str, Any], count: int) -> Dict[str, Any]:
    web = payload.get("web") if isinstance(payload.get("web"), dict) else {}
    results = (
        web.get("results")
        or payload.get("results")
        or payload.get("web_results")
        or []
    )
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(results[:count], start=1):
        if isinstance(item, dict):
            normalized.append(_normalize_result_item(item, idx))
    data: Dict[str, Any] = {"web": normalized}

    summary = web.get("summary") or payload.get("summary")
    if summary:
        data["summary"] = summary

    return {"success": True, "data": data}


def brave_search(
    query: str,
    count: int = 10,
    country: str = "us",
    freshness: Optional[str] = None,
    extra_snippets: bool = False,
    summary: bool = False,
    search_lang: Optional[str] = None,
) -> str:
    """Search the web using Brave Search."""
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    count = _safe_int(count, default=10, minimum=1, maximum=20)
    params: Dict[str, Any] = {
        "q": query,
        "count": count,
        "country": (country or "us").strip() or "us",
    }
    if freshness:
        params["freshness"] = freshness
    if extra_snippets:
        params["extra_snippets"] = True
    if summary:
        params["summary"] = True
    if search_lang:
        params["search_lang"] = search_lang

    try:
        response = httpx.get(
            f"{_BRAVE_BASE_URL}/web/search",
            params=params,
            headers=_brave_headers(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return _json_dumps(_normalize_search_response(payload, count))
    except Exception as exc:
        logger.exception("Brave search failed: %s", exc)
        return tool_error(f"Brave search failed: {type(exc).__name__}: {exc}")


def _normalize_suggestions(payload: Dict[str, Any], count: int) -> List[str]:
    items = (
        payload.get("suggestions")
        or payload.get("results")
        or payload.get("web", {}).get("results")
        or payload.get("data")
        or []
    )
    suggestions: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("query")
                or item.get("text")
                or item.get("suggestion")
                or item.get("title")
                or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            suggestions.append(text)
        if len(suggestions) >= count:
            break
    return suggestions


def brave_suggest(query: str, count: int = 10, lang: Optional[str] = None) -> str:
    """Return Brave Search query suggestions."""
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    count = _safe_int(count, default=10, minimum=1, maximum=20)
    params: Dict[str, Any] = {"q": query, "count": count}
    if lang:
        params["lang"] = lang

    try:
        response = httpx.get(
            f"{_BRAVE_BASE_URL}/suggest/search",
            params=params,
            headers=_brave_headers(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        suggestions = _normalize_suggestions(payload, count)
        return _json_dumps({"success": True, "data": {"query": query, "suggestions": suggestions}})
    except Exception as exc:
        logger.exception("Brave suggest failed: %s", exc)
        return tool_error(f"Brave suggest failed: {type(exc).__name__}: {exc}")


def _extract_answer_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            joined = "".join(parts).strip()
            if joined:
                return joined
        text = choice.get("text")
        if isinstance(text, str):
            return text.strip()
    return str(payload.get("answer") or payload.get("content") or "").strip()


def _extract_sources(payload: Dict[str, Any]) -> List[Any]:
    for key in ("citations", "sources", "references", "source", "reference_urls"):
        value = payload.get(key)
        if not value:
            continue
        if isinstance(value, list):
            sources: List[Any] = []
            for item in value:
                if isinstance(item, dict):
                    source_url = item.get("url") or item.get("link") or item.get("source")
                    if source_url:
                        sources.append({
                            "url": source_url,
                            "title": item.get("title") or item.get("name") or "",
                        })
                    else:
                        sources.append(item)
                else:
                    sources.append(item)
            if sources:
                return sources
        elif isinstance(value, dict):
            return [value]
        else:
            return [value]
    return []


def brave_answers(query: str, model: str = _BRAVE_DEFAULT_MODEL) -> str:
    """Get an AI-generated answer from Brave Search.

    Brave exposes this as an OpenAI-compatible chat completion endpoint.
    """
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    model = (model or _BRAVE_DEFAULT_MODEL).strip() or _BRAVE_DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
    }

    try:
        response = httpx.post(
            f"{_BRAVE_BASE_URL}/chat/completions",
            json=payload,
            headers={**_brave_headers(), "Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = _extract_answer_text(data)
        sources = _extract_sources(data)
        result: Dict[str, Any] = {
            "success": True,
            "data": {
                "query": query,
                "model": model,
                "answer": answer,
            },
        }
        if sources:
            result["data"]["sources"] = sources
        usage = data.get("usage")
        if usage:
            result["data"]["usage"] = usage
        return _json_dumps(result)
    except Exception as exc:
        logger.exception("Brave answers failed: %s", exc)
        return tool_error(f"Brave answers failed: {type(exc).__name__}: {exc}")


BRAVE_SEARCH_SCHEMA = {
    "name": "brave_search",
    "description": "Search the web with Brave Search. Supports country, freshness, extra snippets, and optional summary results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to send to Brave Search",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Maximum number of results to return (1-20)",
            },
            "country": {
                "type": "string",
                "default": "us",
                "description": "Two-letter country code used for result localization",
            },
            "freshness": {
                "type": "string",
                "enum": ["pd", "pw", "pm", "py"],
                "description": "Freshness filter: pd=day, pw=week, pm=month, py=year",
            },
            "extra_snippets": {
                "type": "boolean",
                "default": False,
                "description": "Request extra snippets in the Brave response",
            },
            "summary": {
                "type": "boolean",
                "default": False,
                "description": "Request a Brave summary result when available",
            },
            "search_lang": {
                "type": "string",
                "description": "Optional search language override",
            },
        },
        "required": ["query"],
    },
}

BRAVE_SUGGEST_SCHEMA = {
    "name": "brave_suggest",
    "description": "Get Brave Search autocomplete suggestions for a query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query prefix to autocomplete",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Maximum number of suggestions to return (1-20)",
            },
            "lang": {
                "type": "string",
                "description": "Optional language override",
            },
        },
        "required": ["query"],
    },
}

BRAVE_ANSWERS_SCHEMA = {
    "name": "brave_answers",
    "description": "Get an AI-generated answer from Brave Search using the OpenAI-compatible chat completions endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question to answer",
            },
            "model": {
                "type": "string",
                "default": _BRAVE_DEFAULT_MODEL,
                "description": "Brave answer model name",
            },
        },
        "required": ["query"],
    },
}

registry.register(
    name="brave_search",
    toolset="web",
    schema=BRAVE_SEARCH_SCHEMA,
    handler=lambda args, **kw: brave_search(
        args.get("query", ""),
        count=args.get("count", 10),
        country=args.get("country", "us"),
        freshness=args.get("freshness"),
        extra_snippets=bool(args.get("extra_snippets", False)),
        summary=bool(args.get("summary", False)),
        search_lang=args.get("search_lang"),
    ),
    check_fn=check_brave_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"],
    emoji="🔎",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_suggest",
    toolset="web",
    schema=BRAVE_SUGGEST_SCHEMA,
    handler=lambda args, **kw: brave_suggest(
        args.get("query", ""),
        count=args.get("count", 10),
        lang=args.get("lang"),
    ),
    check_fn=check_brave_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"],
    emoji="💡",
    max_result_size_chars=50_000,
)

registry.register(
    name="brave_answers",
    toolset="web",
    schema=BRAVE_ANSWERS_SCHEMA,
    handler=lambda args, **kw: brave_answers(
        args.get("query", ""),
        model=args.get("model", _BRAVE_DEFAULT_MODEL),
    ),
    check_fn=check_brave_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"],
    emoji="🧠",
    max_result_size_chars=100_000,
)
