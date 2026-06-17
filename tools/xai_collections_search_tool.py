#!/usr/bin/env python3
"""xAI Collections Search / RAG via the Responses API ``file_search`` tool."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_COLLECTIONS_SEARCH_MODEL = "grok-4.3"
DEFAULT_XAI_COLLECTIONS_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_XAI_COLLECTIONS_SEARCH_RETRIES = 2
DEFAULT_XAI_COLLECTIONS_SEARCH_MAX_RESULTS = 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_xai_collections_search_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return load_config().get("xai_collections_search", {}) or {}
    except Exception:
        return {}


def _get_xai_collections_search_model() -> str:
    cfg = _load_xai_collections_search_config()
    return (
        str(cfg.get("model") or "").strip()
        or DEFAULT_XAI_COLLECTIONS_SEARCH_MODEL
    )


def _get_xai_collections_search_timeout_seconds() -> int:
    cfg = _load_xai_collections_search_config()
    raw_value = cfg.get(
        "timeout_seconds",
        DEFAULT_XAI_COLLECTIONS_SEARCH_TIMEOUT_SECONDS,
    )
    try:
        return max(30, int(raw_value))
    except Exception:
        return DEFAULT_XAI_COLLECTIONS_SEARCH_TIMEOUT_SECONDS


def _get_xai_collections_search_retries() -> int:
    cfg = _load_xai_collections_search_config()
    raw_value = cfg.get("retries", DEFAULT_XAI_COLLECTIONS_SEARCH_RETRIES)
    try:
        return max(0, int(raw_value))
    except Exception:
        return DEFAULT_XAI_COLLECTIONS_SEARCH_RETRIES


def _get_xai_collections_search_max_num_results() -> int:
    cfg = _load_xai_collections_search_config()
    raw_value = cfg.get(
        "max_num_results",
        DEFAULT_XAI_COLLECTIONS_SEARCH_MAX_RESULTS,
    )
    try:
        return max(1, int(raw_value))
    except Exception:
        return DEFAULT_XAI_COLLECTIONS_SEARCH_MAX_RESULTS


def _get_xai_collections_search_collection_ids() -> List[str]:
    cfg = _load_xai_collections_search_config()
    return _normalize_string_list(
        cfg.get("collection_ids") or cfg.get("vector_store_ids"),
        "collection_ids",
    )


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

def _resolve_xai_bearer() -> Tuple[str, str, str]:
    """Return ``(api_key, base_url, source)`` or raise on missing credentials."""
    creds = resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "No xAI credentials available. Run `hermes auth add xai-oauth` "
            "to sign in with your SuperGrok subscription, or set XAI_API_KEY."
        )
    base_url = str(creds.get("base_url") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    source = str(creds.get("provider") or "xai")
    return api_key, base_url, source


def check_xai_collections_search_requirements() -> bool:
    """Return True when xAI credentials are available and non-empty."""
    try:
        creds = resolve_xai_http_credentials()
        return bool(str(creds.get("api_key") or "").strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_string_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raise ValueError(f"{field_name} must be a string or list of strings")

    cleaned: List[str] = []
    seen = set()
    for item in raw_items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    return cleaned


def _normalize_max_num_results(value: Optional[int]) -> int:
    if value is None:
        return _get_xai_collections_search_max_num_results()
    try:
        normalized = int(value)
    except Exception as exc:
        raise ValueError("max_num_results must be a positive integer") from exc
    if normalized < 1:
        raise ValueError("max_num_results must be a positive integer")
    return normalized


def _extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

    parts: List[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if ctype in ("output_text", "text"):
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_inline_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if not isinstance(annotation, dict):
                    continue
                citation_type = annotation.get("type")
                if citation_type not in ("url_citation", "file_citation"):
                    continue
                citations.append(
                    {
                        "type": citation_type,
                        "url": annotation.get("url", ""),
                        "title": annotation.get("title", ""),
                        "file_id": annotation.get("file_id"),
                        "filename": annotation.get("filename"),
                        "start_index": annotation.get("start_index"),
                        "end_index": annotation.get("end_index"),
                    }
                )
    return citations


def _extract_file_search_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in ("file_search_call", "collections_search_call"):
            calls.append(item)
    return calls


def _extract_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_calls = payload.get("tool_calls") or []
    if isinstance(raw_calls, dict):
        return [raw_calls]
    if isinstance(raw_calls, list):
        return [call for call in raw_calls if isinstance(call, dict)]
    return []


def _http_error_message(exc: requests.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        code = str(payload.get("code") or "").strip()
        error_value = payload.get("error")
        if isinstance(error_value, dict):
            error = str(error_value.get("message") or "").strip()
        else:
            error = str(error_value or "").strip()
        message = error or str(payload)
        if code and code not in message:
            message = f"{code}: {message}"
        return message or str(exc)

    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text[:500]
    return str(exc)


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

def xai_collections_search_tool(
    query: str,
    collection_ids: Optional[List[str]] = None,
    max_num_results: Optional[int] = None,
) -> str:
    if not query or not query.strip():
        return tool_error("query is required for xai_collections_search")

    try:
        api_key, base_url, source = _resolve_xai_bearer()
    except RuntimeError as exc:
        return tool_error(str(exc))

    try:
        resolved_collection_ids = _normalize_string_list(
            collection_ids,
            "collection_ids",
        ) or _get_xai_collections_search_collection_ids()
        if not resolved_collection_ids:
            return tool_error(
                "collection_ids is required for xai_collections_search "
                "(pass collection_ids or configure xai_collections_search.collection_ids)"
            )

        resolved_max_num_results = _normalize_max_num_results(max_num_results)
        tool_def = {
            "type": "file_search",
            "vector_store_ids": resolved_collection_ids,
            "max_num_results": resolved_max_num_results,
        }

        payload = {
            "model": _get_xai_collections_search_model(),
            "input": [
                {
                    "role": "user",
                    "content": query.strip(),
                }
            ],
            "tools": [tool_def],
            "store": False,
        }

        timeout_seconds = _get_xai_collections_search_timeout_seconds()
        max_retries = _get_xai_collections_search_retries()
        response: Optional[requests.Response] = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": hermes_xai_user_agent(),
                    },
                    json=payload,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                break
            except requests.HTTPError as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code is None or status_code < 500 or attempt >= max_retries:
                    raise
                logger.warning(
                    "xai_collections_search upstream failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    _http_error_message(e),
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "xai_collections_search transient failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))

        if response is None:
            raise RuntimeError("xai_collections_search request did not return a response")

        data = response.json()
        answer = _extract_response_text(data)

        return json.dumps(
            {
                "success": True,
                "provider": "xai",
                "credential_source": source,
                "tool": "xai_collections_search",
                "xai_tool": "file_search",
                "xai_sdk_tool": "collections_search",
                "model": payload["model"],
                "query": query.strip(),
                "collection_ids": resolved_collection_ids,
                "max_num_results": resolved_max_num_results,
                "answer": answer,
                "citations": list(data.get("citations") or []),
                "inline_citations": _extract_inline_citations(data),
                "tool_calls": _extract_tool_calls(data),
                "file_search_calls": _extract_file_search_calls(data),
                "server_side_tool_usage": data.get("server_side_tool_usage") or {},
                "usage": data.get("usage") or {},
            },
            ensure_ascii=False,
        )
    except requests.HTTPError as e:
        logger.error("xai_collections_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_collections_search",
                "xai_tool": "file_search",
                "error": _http_error_message(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except requests.ReadTimeout as e:
        logger.error("xai_collections_search timed out: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_collections_search",
                "xai_tool": "file_search",
                "error": (
                    "xAI collections search timed out after "
                    f"{_get_xai_collections_search_timeout_seconds()} seconds"
                ),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("xai_collections_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_collections_search",
                "xai_tool": "file_search",
                "error": str(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )


XAI_COLLECTIONS_SEARCH_SCHEMA = {
    "name": "xai_collections_search",
    "description": (
        "Search existing xAI Collections / vector stores using xAI's hosted "
        "Responses API file_search tool. Use this for RAG over documents "
        "already uploaded to xAI. This does not create or upload collections."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question or search query to answer from the xAI collection.",
            },
            "collection_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Existing xAI collection / vector store IDs to search. "
                    "Can be omitted only if configured in xai_collections_search.collection_ids."
                ),
            },
            "max_num_results": {
                "type": "integer",
                "description": "Maximum number of matching chunks to retrieve. Defaults to config value or 10.",
                "minimum": 1,
                "default": DEFAULT_XAI_COLLECTIONS_SEARCH_MAX_RESULTS,
            },
        },
        "required": ["query"],
    },
}


def _handle_xai_collections_search(args, **kw):
    if not isinstance(args, dict):
        args = {}
    return xai_collections_search_tool(
        query=args.get("query", ""),
        collection_ids=args.get("collection_ids"),
        max_num_results=args.get("max_num_results"),
    )


registry.register(
    name="xai_collections_search",
    toolset="xai_collections_search",
    schema=XAI_COLLECTIONS_SEARCH_SCHEMA,
    handler=_handle_xai_collections_search,
    check_fn=check_xai_collections_search_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🗂️",
    max_result_size_chars=100_000,
)
