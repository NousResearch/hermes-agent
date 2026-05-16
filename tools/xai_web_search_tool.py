#!/usr/bin/env python3
"""Web Search tool backed by xAI's built-in ``web_search`` Responses API tool.

This is intentionally separate from Hermes' provider-agnostic ``web_search``
tool. It exposes xAI's server-side Responses tool directly for users who want
Grok to search and browse the web through xAI, while leaving existing web
search backends unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_WEB_SEARCH_MODEL = "grok-4.3"
DEFAULT_XAI_WEB_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_XAI_WEB_SEARCH_RETRIES = 2
MAX_DOMAINS = 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_xai_web_search_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return load_config().get("xai_web_search", {}) or {}
    except Exception:
        return {}


def _get_xai_web_search_model() -> str:
    cfg = _load_xai_web_search_config()
    return (str(cfg.get("model") or "").strip() or DEFAULT_XAI_WEB_SEARCH_MODEL)


def _get_xai_web_search_timeout_seconds() -> int:
    cfg = _load_xai_web_search_config()
    raw_value = cfg.get("timeout_seconds", DEFAULT_XAI_WEB_SEARCH_TIMEOUT_SECONDS)
    try:
        return max(30, int(raw_value))
    except Exception:
        return DEFAULT_XAI_WEB_SEARCH_TIMEOUT_SECONDS


def _get_xai_web_search_retries() -> int:
    cfg = _load_xai_web_search_config()
    raw_value = cfg.get("retries", DEFAULT_XAI_WEB_SEARCH_RETRIES)
    try:
        return max(0, int(raw_value))
    except Exception:
        return DEFAULT_XAI_WEB_SEARCH_RETRIES


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


def check_xai_web_search_requirements() -> bool:
    """Return True when xAI credentials are available and non-empty."""
    try:
        creds = resolve_xai_http_credentials()
        return bool(str(creds.get("api_key") or "").strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_domains(domains: Optional[List[str]], field_name: str) -> List[str]:
    cleaned: List[str] = []
    for domain in domains or []:
        raw = str(domain or "").strip()
        if not raw:
            continue
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or parsed.netloc or parsed.path).split("/")[0].strip().lower()
        host = host.lstrip(".")
        if host:
            cleaned.append(host)
    if len(cleaned) > MAX_DOMAINS:
        raise ValueError(f"{field_name} supports at most {MAX_DOMAINS} domains")
    return cleaned


def _extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

    parts: List[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            ctype = content.get("type")
            if ctype in ("output_text", "text"):
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_inline_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") != "url_citation":
                    continue
                citations.append(
                    {
                        "url": annotation.get("url", ""),
                        "title": annotation.get("title", ""),
                        "start_index": annotation.get("start_index"),
                        "end_index": annotation.get("end_index"),
                    }
                )
    return citations


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
        error = str(payload.get("error") or "").strip()
        if not error and isinstance(payload.get("error"), dict):
            error = str(payload["error"].get("message") or "").strip()
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


def xai_web_search_tool(
    query: str,
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    enable_image_understanding: bool = False,
) -> str:
    if not query or not query.strip():
        return tool_error("query is required for xai_web_search")

    try:
        api_key, base_url, source = _resolve_xai_bearer()
    except RuntimeError as exc:
        return tool_error(str(exc))

    try:
        allowed = _normalize_domains(allowed_domains, "allowed_domains")
        excluded = _normalize_domains(excluded_domains, "excluded_domains")
        if allowed and excluded:
            return tool_error("allowed_domains and excluded_domains cannot be used together")

        tool_def: Dict[str, Any] = {"type": "web_search"}
        filters: Dict[str, Any] = {}
        if allowed:
            filters["allowed_domains"] = allowed
        if excluded:
            filters["excluded_domains"] = excluded
        if filters:
            tool_def["filters"] = filters
        if enable_image_understanding:
            tool_def["enable_image_understanding"] = True

        payload = {
            "model": _get_xai_web_search_model(),
            "input": [
                {
                    "role": "user",
                    "content": query.strip(),
                }
            ],
            "tools": [tool_def],
            "store": False,
        }

        timeout_seconds = _get_xai_web_search_timeout_seconds()
        max_retries = _get_xai_web_search_retries()
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
                    "xai_web_search upstream failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    _http_error_message(e),
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "xai_web_search transient failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))

        if response is None:
            raise RuntimeError("xai_web_search request did not return a response")

        data = response.json()
        answer = _extract_response_text(data)
        citations = list(data.get("citations") or [])
        inline_citations = _extract_inline_citations(data)

        return json.dumps(
            {
                "success": True,
                "provider": "xai",
                "credential_source": source,
                "tool": "xai_web_search",
                "xai_tool": "web_search",
                "model": payload["model"],
                "query": query.strip(),
                "answer": answer,
                "citations": citations,
                "inline_citations": inline_citations,
                "server_side_tool_usage": data.get("server_side_tool_usage") or {},
            },
            ensure_ascii=False,
        )
    except requests.HTTPError as e:
        logger.error("xai_web_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_web_search",
                "xai_tool": "web_search",
                "error": _http_error_message(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except requests.ReadTimeout as e:
        logger.error("xai_web_search timed out: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_web_search",
                "xai_tool": "web_search",
                "error": (
                    "xAI web_search timed out after "
                    f"{_get_xai_web_search_timeout_seconds()} seconds"
                ),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("xai_web_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_web_search",
                "xai_tool": "web_search",
                "error": str(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )


XAI_WEB_SEARCH_SCHEMA = {
    "name": "xai_web_search",
    "description": (
        "Search and browse the live web using xAI's built-in web_search "
        "Responses API tool. Use this when the user specifically wants Grok/xAI "
        "web search or when xAI server-side citations are useful."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up on the web.",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domains to search exclusively (max 5).",
            },
            "excluded_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domains to exclude from search (max 5).",
            },
            "enable_image_understanding": {
                "type": "boolean",
                "description": "Whether xAI should analyze images encountered while browsing.",
                "default": False,
            },
        },
        "required": ["query"],
    },
}


def _handle_xai_web_search(args, **kw):
    return xai_web_search_tool(
        query=args.get("query", ""),
        allowed_domains=args.get("allowed_domains"),
        excluded_domains=args.get("excluded_domains"),
        enable_image_understanding=bool(args.get("enable_image_understanding", False)),
    )


registry.register(
    name="xai_web_search",
    toolset="xai_web_search",
    schema=XAI_WEB_SEARCH_SCHEMA,
    handler=_handle_xai_web_search,
    check_fn=check_xai_web_search_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🌐",
    max_result_size_chars=100_000,
)
