#!/usr/bin/env python3
"""Brave Search native tools.

This module exposes Brave's web, news, image, video, local, suggestion, and answer APIs
as first-class Hermes tools. It is intentionally self-contained so it can be
imported by the normal tool registry auto-discovery path.

Brave API docs used by this implementation:
- Web search: GET /res/v1/web/search
- News search: GET /res/v1/news/search
- Image search: GET /res/v1/images/search
- Video search: GET /res/v1/videos/search
- Local POIs: GET /res/v1/local/pois
- Local descriptions: GET /res/v1/local/descriptions
- Suggestions: GET /res/v1/suggest/search
- AI answers: POST /res/v1/chat/completions
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import httpx

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_BRAVE_BASE_URL = "https://api.search.brave.com/res/v1"
_BRAVE_DEFAULT_MODEL = "brave-pro"


def _brave_search_api_key() -> str:
    """Return the configured Brave Search key for search-style endpoints."""
    return (
        os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        or os.getenv("BRAVE_FREE_API_KEY", "").strip()
        or os.getenv("BRAVE_API_KEY", "").strip()
    )


def _brave_answers_api_key() -> str:
    """Return the configured Brave Answers key, with legacy fallback."""
    return os.getenv("BRAVE_ANSWERS_API_KEY", "").strip() or _brave_search_api_key()


def _brave_autosuggest_api_key() -> str:
    """Return the configured Brave Autosuggest key, with legacy fallback."""
    return os.getenv("BRAVE_AUTOSUGGEST_API_KEY", "").strip() or _brave_search_api_key()


def check_brave_search_api_key() -> bool:
    """Return True when Brave Search credentials are available."""
    return bool(_brave_search_api_key())


def check_brave_answers_api_key() -> bool:
    """Return True when Brave Answers credentials are available."""
    return bool(_brave_answers_api_key())


def check_brave_autosuggest_api_key() -> bool:
    """Return True when Brave Autosuggest credentials are available."""
    return bool(_brave_autosuggest_api_key())


def _brave_headers(api_key: str, *, missing_message: str) -> Dict[str, str]:
    if not api_key:
        raise ValueError(missing_message)
    return {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }


def _brave_search_headers() -> Dict[str, str]:
    return _brave_headers(
        _brave_search_api_key(),
        missing_message=(
            "BRAVE_SEARCH_API_KEY or BRAVE_FREE_API_KEY environment variable not set. "
            "Get your Brave Search API key at https://api-dashboard.search.brave.com"
        ),
    )


def _brave_answers_headers() -> Dict[str, str]:
    return _brave_headers(
        _brave_answers_api_key(),
        missing_message=(
            "BRAVE_ANSWERS_API_KEY environment variable not set. "
            "Get your Brave Answers API key at https://api-dashboard.search.brave.com"
        ),
    )


def _brave_autosuggest_headers() -> Dict[str, str]:
    return _brave_headers(
        _brave_autosuggest_api_key(),
        missing_message=(
            "BRAVE_AUTOSUGGEST_API_KEY environment variable not set. "
            "Get your Brave Autosuggest API key at https://api-dashboard.search.brave.com"
        ),
    )


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _safe_int(value: Any, default: int, minimum: int = 1, maximum: int = 20) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _add_param(params: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        params[key] = value


def _normalize_string_list(value: Any, maximum: int = 20) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    normalized: List[str] = []
    for item in raw_values:
        text = str(item).strip()
        if text:
            normalized.append(text)
        if len(normalized) >= maximum:
            break
    return normalized


def _brave_http_error(exc: httpx.HTTPStatusError, *, label: str) -> str:
    response = exc.response
    payload: Dict[str, Any] = {}
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = {}

    error_info = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    brave_error: Dict[str, Any] = {}
    for key in ("code", "detail", "id"):
        value = error_info.get(key)
        if value not in (None, ""):
            brave_error[key] = value

    meta = error_info.get("meta") if isinstance(error_info.get("meta"), dict) else {}
    component = meta.get("component")
    if component:
        brave_error["component"] = component

    status = error_info.get("status") or getattr(response, "status_code", None)
    if status not in (None, ""):
        brave_error["status"] = status

    detail = brave_error.get("detail")
    code = brave_error.get("code")
    if code and detail:
        message = f"{label} failed: {code} - {detail}"
    elif code:
        message = f"{label} failed: {code}"
    elif detail:
        message = f"{label} failed: {detail}"
    elif status is not None:
        message = f"{label} failed: HTTP {status}"
    else:
        message = f"{label} failed: {exc}"

    if brave_error:
        return tool_error(message, brave_error=brave_error)
    return tool_error(message)


def _brave_get_json(
    path: str,
    *,
    params: Dict[str, Any],
    label: str,
    headers_factory: Callable[[], Dict[str, str]],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = httpx.get(
            f"{_BRAVE_BASE_URL}{path}",
            params=params,
            headers=headers_factory(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload, None
        return None, tool_error(f"{label} failed: unexpected response payload")
    except httpx.HTTPStatusError as exc:
        logger.warning("%s returned HTTP %s", label, exc.response.status_code if exc.response else "?")
        return None, _brave_http_error(exc, label=label)
    except Exception as exc:
        logger.exception("%s failed: %s", label, exc)
        return None, tool_error(f"{label} failed: {type(exc).__name__}: {exc}")


def _normalize_ranked_results(results: Any, *, count: Optional[int] = None, offset: int = 0) -> List[Any]:
    if not isinstance(results, list):
        return []

    limited = results[:count] if count is not None else results
    normalized: List[Any] = []
    start_position = max(0, offset) + 1
    for idx, item in enumerate(limited, start=start_position):
        if isinstance(item, dict):
            enriched = dict(item)
            enriched.setdefault("position", idx)
            normalized.append(enriched)
        else:
            normalized.append({"value": item, "position": idx})
    return normalized


def _normalize_collection_response(
    payload: Dict[str, Any],
    *,
    key: str,
    count: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        key: _normalize_ranked_results(payload.get("results"), count=count, offset=offset)
    }
    for extra_key in ("query", "extra", "type"):
        if payload.get(extra_key) is not None:
            data[extra_key] = payload[extra_key]
    return {"success": True, "data": data}


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
    normalized: Dict[str, Any] = {
        "title": str(item.get("title") or item.get("name") or ""),
        "url": str(item.get("url") or item.get("link") or ""),
        "description": str(description),
        "position": position,
    }
    if item.get("img"):
        normalized["img"] = item["img"]
    return normalized


def _normalize_search_response(payload: Dict[str, Any], count: int, offset: int = 0) -> Dict[str, Any]:
    web = payload.get("web") if isinstance(payload.get("web"), dict) else {}
    results = (
        web.get("results")
        or payload.get("results")
        or payload.get("web_results")
        or []
    )
    normalized: List[Dict[str, Any]] = []
    start_position = max(0, offset) + 1
    for idx, item in enumerate(results[:count], start=start_position):
        if isinstance(item, dict):
            normalized.append(_normalize_result_item(item, idx))
    data: Dict[str, Any] = {"web": normalized}

    if payload.get("query") is not None:
        data["query"] = payload["query"]
    for key in ("news", "videos", "summarizer", "rich"):
        if payload.get(key) is not None:
            data[key] = payload[key]

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
    ui_lang: Optional[str] = None,
    safesearch: Optional[str] = None,
    offset: Optional[int] = None,
    text_decorations: Optional[bool] = None,
    spellcheck: Optional[bool] = None,
    result_filter: Optional[str] = None,
    goggles: Optional[str] = None,
    goggles_id: Optional[str] = None,
    units: Optional[str] = None,
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
    _add_param(params, "freshness", freshness)
    _add_param(params, "search_lang", search_lang)
    _add_param(params, "ui_lang", ui_lang)
    _add_param(params, "safesearch", safesearch)
    if offset is not None:
        params["offset"] = _safe_int(offset, default=0, minimum=0, maximum=9)
    if text_decorations is not None:
        params["text_decorations"] = bool(text_decorations)
    if spellcheck is not None:
        params["spellcheck"] = bool(spellcheck)
    _add_param(params, "result_filter", result_filter)
    _add_param(params, "goggles", goggles)
    _add_param(params, "goggles_id", goggles_id)
    _add_param(params, "units", units)
    if extra_snippets:
        params["extra_snippets"] = True
    if summary:
        params["summary"] = True

    payload, error = _brave_get_json(
        "/web/search",
        params=params,
        label="Brave search",
        headers_factory=_brave_search_headers,
    )
    if error:
        return error

    normalized_offset = params.get("offset", 0)
    return _json_dumps(_normalize_search_response(payload or {}, count, offset=normalized_offset))


def _normalize_suggestions(payload: Dict[str, Any], count: int) -> List[Any]:
    items = (
        payload.get("suggestions")
        or payload.get("results")
        or payload.get("web", {}).get("results")
        or payload.get("data")
        or []
    )
    suggestions: List[Any] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                suggestions.append(text)
        elif isinstance(item, dict):
            if set(item.keys()) <= {"query"} and isinstance(item.get("query"), str):
                text = item["query"].strip()
                if text:
                    suggestions.append(text)
            else:
                suggestions.append(dict(item))
        else:
            text = str(item).strip()
            if text:
                suggestions.append(text)
        if len(suggestions) >= count:
            break
    return suggestions


def brave_suggest(
    query: str,
    count: int = 5,
    country: str = "US",
    lang: Optional[str] = None,
    rich: bool = False,
) -> str:
    """Return Brave Search query suggestions."""
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    count = _safe_int(count, default=5, minimum=1, maximum=20)
    params: Dict[str, Any] = {"q": query, "count": count, "country": country or "US"}
    if lang:
        params["lang"] = lang
    if rich:
        params["rich"] = True

    payload, error = _brave_get_json(
        "/suggest/search",
        params=params,
        label="Brave suggest",
        headers_factory=_brave_autosuggest_headers,
    )
    if error:
        return error

    payload = payload or {}
    suggestions = _normalize_suggestions(payload, count)
    query_value = payload.get("query", query)
    return _json_dumps({"success": True, "data": {"query": query_value, "suggestions": suggestions}})


def brave_news(
    query: str,
    count: int = 20,
    country: str = "US",
    search_lang: Optional[str] = None,
    ui_lang: Optional[str] = None,
    safesearch: Optional[str] = None,
    offset: Optional[int] = None,
    spellcheck: Optional[bool] = None,
    freshness: Optional[str] = None,
    extra_snippets: Optional[bool] = None,
    goggles: Optional[str] = None,
    include_fetch_metadata: Optional[bool] = None,
    operators: Optional[bool] = None,
) -> str:
    """Search news content using Brave Search."""
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    count = _safe_int(count, default=20, minimum=1, maximum=50)
    params: Dict[str, Any] = {"q": query, "count": count, "country": (country or "US").strip() or "US"}
    _add_param(params, "search_lang", search_lang)
    _add_param(params, "ui_lang", ui_lang)
    _add_param(params, "safesearch", safesearch)
    if offset is not None:
        params["offset"] = _safe_int(offset, default=0, minimum=0, maximum=9)
    _add_param(params, "spellcheck", spellcheck)
    _add_param(params, "freshness", freshness)
    _add_param(params, "extra_snippets", extra_snippets)
    _add_param(params, "goggles", goggles)
    _add_param(params, "include_fetch_metadata", include_fetch_metadata)
    _add_param(params, "operators", operators)

    payload, error = _brave_get_json(
        "/news/search",
        params=params,
        label="Brave news",
        headers_factory=_brave_search_headers,
    )
    if error:
        return error

    return _json_dumps(_normalize_collection_response(payload or {}, key="news", count=count, offset=params.get("offset", 0)))


def brave_images(
    query: str,
    count: int = 50,
    country: str = "US",
    search_lang: Optional[str] = None,
    safesearch: Optional[str] = None,
    spellcheck: Optional[bool] = None,
) -> str:
    """Search image content using Brave Search."""
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    count = _safe_int(count, default=50, minimum=1, maximum=200)
    params: Dict[str, Any] = {"q": query, "count": count, "country": (country or "US").strip() or "US"}
    _add_param(params, "search_lang", search_lang)
    _add_param(params, "safesearch", safesearch)
    _add_param(params, "spellcheck", spellcheck)

    payload, error = _brave_get_json(
        "/images/search",
        params=params,
        label="Brave images",
        headers_factory=_brave_search_headers,
    )
    if error:
        return error

    return _json_dumps(_normalize_collection_response(payload or {}, key="images", count=count))


def brave_videos(
    query: str,
    count: int = 20,
    country: str = "US",
    search_lang: Optional[str] = None,
    ui_lang: Optional[str] = None,
    safesearch: Optional[str] = None,
    offset: Optional[int] = None,
    spellcheck: Optional[bool] = None,
    freshness: Optional[str] = None,
    include_fetch_metadata: Optional[bool] = None,
    operators: Optional[bool] = None,
) -> str:
    """Search video content using Brave Search."""
    query = (query or "").strip()
    if not query:
        return tool_error("Query is required")

    count = _safe_int(count, default=20, minimum=1, maximum=50)
    params: Dict[str, Any] = {"q": query, "count": count, "country": (country or "US").strip() or "US"}
    _add_param(params, "search_lang", search_lang)
    _add_param(params, "ui_lang", ui_lang)
    _add_param(params, "safesearch", safesearch)
    if offset is not None:
        params["offset"] = _safe_int(offset, default=0, minimum=0, maximum=9)
    _add_param(params, "spellcheck", spellcheck)
    _add_param(params, "freshness", freshness)
    _add_param(params, "include_fetch_metadata", include_fetch_metadata)
    _add_param(params, "operators", operators)

    payload, error = _brave_get_json(
        "/videos/search",
        params=params,
        label="Brave videos",
        headers_factory=_brave_search_headers,
    )
    if error:
        return error

    return _json_dumps(_normalize_collection_response(payload or {}, key="videos", count=count, offset=params.get("offset", 0)))


def brave_local_pois(
    ids: Any,
    search_lang: Optional[str] = None,
    ui_lang: Optional[str] = None,
    units: Optional[str] = None,
) -> str:
    """Fetch Brave local POI details for location ids from web search."""
    location_ids = _normalize_string_list(ids)
    if not location_ids:
        return tool_error("At least one location id is required")

    params: Dict[str, Any] = {"ids": location_ids}
    _add_param(params, "search_lang", search_lang)
    _add_param(params, "ui_lang", ui_lang)
    _add_param(params, "units", units)

    payload, error = _brave_get_json(
        "/local/pois",
        params=params,
        label="Brave local POIs",
        headers_factory=_brave_search_headers,
    )
    if error:
        return error

    return _json_dumps(_normalize_collection_response(payload or {}, key="pois"))


def brave_local_descriptions(ids: Any) -> str:
    """Fetch Brave local descriptions for location ids from web search."""
    location_ids = _normalize_string_list(ids)
    if not location_ids:
        return tool_error("At least one location id is required")

    payload, error = _brave_get_json(
        "/local/descriptions",
        params={"ids": location_ids},
        label="Brave local descriptions",
        headers_factory=_brave_search_headers,
    )
    if error:
        return error

    return _json_dumps(_normalize_collection_response(payload or {}, key="descriptions"))


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


def _extract_streaming_answer(payload_text: str) -> Dict[str, Any]:
    parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None

    for raw_line in payload_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        chunk_text = line[5:].strip()
        if not chunk_text or chunk_text == "[DONE]":
            continue
        try:
            chunk = json.loads(chunk_text)
        except json.JSONDecodeError:
            continue

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))

    return {
        "answer": "".join(parts).strip(),
        "usage": usage,
    }


def brave_answers(
    query: Optional[str] = None,
    model: str = _BRAVE_DEFAULT_MODEL,
    messages: Optional[List[Dict[str, Any]]] = None,
    stream: Optional[bool] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    enable_entities: Optional[bool] = None,
    enable_citations: Optional[bool] = None,
) -> str:
    """Get an AI-generated answer from Brave Search.

    Brave exposes this as an OpenAI-compatible chat completion endpoint.
    """
    query = (query or "").strip()
    model = (model or _BRAVE_DEFAULT_MODEL).strip() or _BRAVE_DEFAULT_MODEL

    request_messages = messages if messages is not None else None
    if request_messages is None:
        if not query:
            return tool_error("Query is required")
        request_messages = [{"role": "user", "content": query}]
    elif not request_messages:
        return tool_error("Messages are required")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": request_messages,
    }
    payload["stream"] = False if stream is None else stream
    if country is not None:
        payload["country"] = country
    if language is not None:
        payload["language"] = language
    if enable_entities is not None:
        payload["enable_entities"] = enable_entities
    if enable_citations is not None:
        payload["enable_citations"] = enable_citations

    try:
        response = httpx.post(
            f"{_BRAVE_BASE_URL}/chat/completions",
            json=payload,
            headers={**_brave_answers_headers(), "Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            streamed = _extract_streaming_answer(response.text)
            answer = streamed["answer"]
            sources = []
            usage = streamed.get("usage")
        else:
            data = response.json()
            answer = _extract_answer_text(data)
            sources = _extract_sources(data)
            usage = data.get("usage")
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
        if usage:
            result["data"]["usage"] = usage
        return _json_dumps(result)
    except httpx.HTTPStatusError as exc:
        logger.warning("Brave answers returned HTTP %s", exc.response.status_code if exc.response else "?")
        return _brave_http_error(exc, label="Brave answers")
    except Exception as exc:
        logger.exception("Brave answers failed: %s", exc)
        return tool_error(f"Brave answers failed: {type(exc).__name__}: {exc}")


BRAVE_NEWS_SCHEMA = {
    "name": "brave_news",
    "description": "Search recent news articles with Brave Search.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "News query to send to Brave Search"},
            "count": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20, "description": "Maximum number of news results to return (1-50)"},
            "country": {"type": "string", "default": "US", "description": "Two-letter country code used for result localization"},
            "search_lang": {"type": "string", "description": "News result language"},
            "ui_lang": {"type": "string", "description": "UI language"},
            "safesearch": {"type": "string", "description": "Safe search mode"},
            "offset": {"type": "integer", "minimum": 0, "maximum": 9, "default": 0, "description": "Result offset (0-9)"},
            "spellcheck": {"type": "boolean", "description": "Enable or disable spellcheck"},
            "freshness": {"type": "string", "description": "Freshness filter or custom date range"},
            "extra_snippets": {"type": "boolean", "description": "Request extra snippets in the Brave response"},
            "goggles": {"type": "string", "description": "Named goggles filter"},
            "include_fetch_metadata": {"type": "boolean", "description": "Include fetch metadata"},
            "operators": {"type": "boolean", "description": "Apply Brave search operators"},
        },
        "required": ["query"],
    },
}

BRAVE_IMAGES_SCHEMA = {
    "name": "brave_images",
    "description": "Search image results with Brave Search.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Image query to send to Brave Search"},
            "count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50, "description": "Maximum number of image results to return (1-200)"},
            "country": {"type": "string", "default": "US", "description": "Two-letter country code used for result localization"},
            "search_lang": {"type": "string", "description": "Image result language"},
            "safesearch": {"type": "string", "description": "Safe search mode"},
            "spellcheck": {"type": "boolean", "description": "Enable or disable spellcheck"},
        },
        "required": ["query"],
    },
}

BRAVE_VIDEOS_SCHEMA = {
    "name": "brave_videos",
    "description": "Search video results with Brave Search.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Video query to send to Brave Search"},
            "count": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20, "description": "Maximum number of video results to return (1-50)"},
            "country": {"type": "string", "default": "US", "description": "Two-letter country code used for result localization"},
            "search_lang": {"type": "string", "description": "Video result language"},
            "ui_lang": {"type": "string", "description": "UI language"},
            "safesearch": {"type": "string", "description": "Safe search mode"},
            "offset": {"type": "integer", "minimum": 0, "maximum": 9, "default": 0, "description": "Result offset (0-9)"},
            "spellcheck": {"type": "boolean", "description": "Enable or disable spellcheck"},
            "freshness": {"type": "string", "description": "Freshness filter or custom date range"},
            "include_fetch_metadata": {"type": "boolean", "description": "Include fetch metadata"},
            "operators": {"type": "boolean", "description": "Apply Brave search operators"},
        },
        "required": ["query"],
    },
}

BRAVE_LOCAL_POIS_SCHEMA = {
    "name": "brave_local_pois",
    "description": "Fetch Brave local POI details for location ids returned by Brave web search.",
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
                "description": "Location ids returned by Brave web search location results",
            },
            "search_lang": {"type": "string", "description": "Search language preference"},
            "ui_lang": {"type": "string", "description": "UI language"},
            "units": {"type": "string", "description": "Measurement units: metric or imperial"},
        },
        "required": ["ids"],
    },
}

BRAVE_LOCAL_DESCRIPTIONS_SCHEMA = {
    "name": "brave_local_descriptions",
    "description": "Fetch Brave local descriptions for location ids returned by Brave web search.",
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
                "description": "Location ids returned by Brave web search location results",
            },
        },
        "required": ["ids"],
    },
}


BRAVE_SEARCH_SCHEMA = {
    "name": "brave_search",
    "description": "Search the web with Brave Search. Supports country, freshness, search language, UI language, safe search, offsets, goggles, and optional summary results.",
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
            "search_lang": {
                "type": "string",
                "description": "Search result language",
            },
            "ui_lang": {
                "type": "string",
                "description": "UI language",
            },
            "safesearch": {
                "type": "string",
                "description": "Safe search mode",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "maximum": 9,
                "default": 0,
                "description": "Result offset (0-9)",
            },
            "text_decorations": {
                "type": "boolean",
                "description": "Enable or disable text decorations",
            },
            "spellcheck": {
                "type": "boolean",
                "description": "Enable or disable spellcheck",
            },
            "result_filter": {
                "type": "string",
                "description": "Filter the returned result types",
            },
            "goggles": {
                "type": "string",
                "description": "Named goggles filter",
            },
            "goggles_id": {
                "type": "string",
                "description": "Goggles identifier",
            },
            "units": {
                "type": "string",
                "description": "Units preference",
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

        },
        "required": ["query"],
    },
}

BRAVE_SUGGEST_SCHEMA = {
    "name": "brave_suggest",
    "description": "Get Brave Search autocomplete suggestions for a query. Works best with BRAVE_AUTOSUGGEST_API_KEY.",
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
                "default": 5,
                "description": "Maximum number of suggestions to return (1-20)",
            },
            "country": {
                "type": "string",
                "default": "US",
                "description": "Two-letter country code used for suggestions",
            },
            "lang": {
                "type": "string",
                "description": "Optional language override",
            },
            "rich": {
                "type": "boolean",
                "default": False,
                "description": "Request rich suggestions when available",
            },
        },
        "required": ["query"],
    },
}

BRAVE_ANSWERS_SCHEMA = {
    "name": "brave_answers",
    "description": "Get an AI-generated answer from Brave Search using the OpenAI-compatible chat completions endpoint. Works best with BRAVE_ANSWERS_API_KEY.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question to answer",
            },
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                },
                "description": "OpenAI-style chat messages; if provided, they are sent directly to Brave.",
            },
            "model": {
                "type": "string",
                "default": _BRAVE_DEFAULT_MODEL,
                "description": "Brave answer model name",
            },
            "stream": {
                "type": "boolean",
                "description": "Enable streaming responses",
            },
            "country": {
                "type": "string",
                "description": "Country hint",
            },
            "language": {
                "type": "string",
                "description": "Language hint",
            },
            "enable_entities": {
                "type": "boolean",
                "description": "Enable entity extraction",
            },
            "enable_citations": {
                "type": "boolean",
                "description": "Enable citations",
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
        ui_lang=args.get("ui_lang"),
        safesearch=args.get("safesearch"),
        offset=args.get("offset"),
        text_decorations=args.get("text_decorations"),
        spellcheck=args.get("spellcheck"),
        result_filter=args.get("result_filter"),
        goggles=args.get("goggles"),
        goggles_id=args.get("goggles_id"),
        units=args.get("units"),
    ),
    check_fn=check_brave_search_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="🔎",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_news",
    toolset="web",
    schema=BRAVE_NEWS_SCHEMA,
    handler=lambda args, **kw: brave_news(
        args.get("query", ""),
        count=args.get("count", 20),
        country=args.get("country", "US"),
        search_lang=args.get("search_lang"),
        ui_lang=args.get("ui_lang"),
        safesearch=args.get("safesearch"),
        offset=args.get("offset"),
        spellcheck=args.get("spellcheck"),
        freshness=args.get("freshness"),
        extra_snippets=args.get("extra_snippets"),
        goggles=args.get("goggles"),
        include_fetch_metadata=args.get("include_fetch_metadata"),
        operators=args.get("operators"),
    ),
    check_fn=check_brave_search_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="📰",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_images",
    toolset="web",
    schema=BRAVE_IMAGES_SCHEMA,
    handler=lambda args, **kw: brave_images(
        args.get("query", ""),
        count=args.get("count", 50),
        country=args.get("country", "US"),
        search_lang=args.get("search_lang"),
        safesearch=args.get("safesearch"),
        spellcheck=args.get("spellcheck"),
    ),
    check_fn=check_brave_search_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="🖼️",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_videos",
    toolset="web",
    schema=BRAVE_VIDEOS_SCHEMA,
    handler=lambda args, **kw: brave_videos(
        args.get("query", ""),
        count=args.get("count", 20),
        country=args.get("country", "US"),
        search_lang=args.get("search_lang"),
        ui_lang=args.get("ui_lang"),
        safesearch=args.get("safesearch"),
        offset=args.get("offset"),
        spellcheck=args.get("spellcheck"),
        freshness=args.get("freshness"),
        include_fetch_metadata=args.get("include_fetch_metadata"),
        operators=args.get("operators"),
    ),
    check_fn=check_brave_search_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="🎬",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_local_pois",
    toolset="web",
    schema=BRAVE_LOCAL_POIS_SCHEMA,
    handler=lambda args, **kw: brave_local_pois(
        args.get("ids"),
        search_lang=args.get("search_lang"),
        ui_lang=args.get("ui_lang"),
        units=args.get("units"),
    ),
    check_fn=check_brave_search_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="📍",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_local_descriptions",
    toolset="web",
    schema=BRAVE_LOCAL_DESCRIPTIONS_SCHEMA,
    handler=lambda args, **kw: brave_local_descriptions(args.get("ids")),
    check_fn=check_brave_search_api_key,
    requires_env=["BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="🗺️",
    max_result_size_chars=100_000,
)

registry.register(
    name="brave_suggest",
    toolset="web",
    schema=BRAVE_SUGGEST_SCHEMA,
    handler=lambda args, **kw: brave_suggest(
        args.get("query", ""),
        count=args.get("count", 5),
        country=args.get("country", "US"),
        lang=args.get("lang"),
        rich=bool(args.get("rich", False)),
    ),
    check_fn=check_brave_autosuggest_api_key,
    requires_env=["BRAVE_AUTOSUGGEST_API_KEY", "BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="💡",
    max_result_size_chars=50_000,
)

registry.register(
    name="brave_answers",
    toolset="web",
    schema=BRAVE_ANSWERS_SCHEMA,
    handler=lambda args, **kw: brave_answers(
        args.get("query"),
        model=args.get("model", _BRAVE_DEFAULT_MODEL),
        messages=args.get("messages"),
        stream=args.get("stream"),
        country=args.get("country"),
        language=args.get("language"),
        enable_entities=args.get("enable_entities"),
        enable_citations=args.get("enable_citations"),
    ),
    check_fn=check_brave_answers_api_key,
    requires_env=["BRAVE_ANSWERS_API_KEY", "BRAVE_SEARCH_API_KEY", "BRAVE_FREE_API_KEY", "BRAVE_API_KEY"],
    emoji="🧠",
    max_result_size_chars=100_000,
)
