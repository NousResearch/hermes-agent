"""xAI web_search tool — search the live web via xAI's Responses API
``web_search`` built-in tool, returns answer + citations.

This is the web counterpart of ``x_search`` (which targets X / Twitter):
``web_search`` lets the model search the live web, scrape pages it finds,
and ground its answer with citations. Implemented as a Responses API
call with ``tools: [{"type": "web_search", ...}]`` so the model knows it
must use the built-in web tool to answer.

The xAI Responses API doc currently states: *"only functions and web
search are supported as tools"* — so this is the canonical way to wire
xAI-native web search into Hermes without going through a third-party
search provider.

Reference: https://docs.x.ai/docs/api-reference#responses-create
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent


logger = logging.getLogger(__name__)


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_WEB_SEARCH_MODEL = "grok-4.3"
DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_WEB_SEARCH_RETRIES = 2
MAX_WEBSITES = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_xai_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_web_search_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("web_search", {}) or {}
    except Exception:
        return {}


def _get_web_search_model() -> str:
    cfg = _load_web_search_config()
    return (cfg.get("model") or DEFAULT_WEB_SEARCH_MODEL).strip()


def _get_web_search_timeout_seconds() -> int:
    cfg = _load_web_search_config()
    raw = cfg.get("timeout_seconds", DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS)
    try:
        return max(30, int(raw))
    except Exception:
        return DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS


def _get_web_search_retries() -> int:
    cfg = _load_web_search_config()
    raw = cfg.get("retries", DEFAULT_WEB_SEARCH_RETRIES)
    try:
        return max(0, int(raw))
    except Exception:
        return DEFAULT_WEB_SEARCH_RETRIES


def check_web_search_requirements() -> bool:
    return bool(os.getenv("XAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_websites(websites: Optional[List[str]], field_name: str) -> List[str]:
    """Normalize a list of website hostnames (strip protocol + path)."""
    cleaned: List[str] = []
    for site in websites or []:
        s = str(site or "").strip()
        # Strip http(s)://
        for prefix in ("https://", "http://"):
            if s.startswith(prefix):
                s = s[len(prefix):]
        # Strip path
        s = s.split("/", 1)[0]
        # Strip leading www. for stable matching
        if s.startswith("www."):
            s = s[4:]
        s = s.strip()
        if s:
            cleaned.append(s)
    if len(cleaned) > MAX_WEBSITES:
        raise ValueError(f"{field_name} supports at most {MAX_WEBSITES} websites")
    return cleaned


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Best-effort extraction of plain answer text from Responses API body."""
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
    """Pull url_citation annotations out of output[*].content[*].annotations."""
    citations: List[Dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") == "url_citation":
                    citations.append({
                        "url": annotation.get("url", ""),
                        "title": annotation.get("title", ""),
                        "start_index": annotation.get("start_index"),
                        "end_index": annotation.get("end_index"),
                    })
    return citations


def _http_error_message(exc: requests.HTTPError) -> str:
    resp = getattr(exc, "response", None)
    if resp is None:
        return str(exc)
    try:
        body = resp.text or ""
    except Exception:
        body = ""
    code = getattr(resp, "status_code", "?")
    return f"HTTP {code}: {body[:300]}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def web_search_tool(
    query: str,
    allowed_websites: Optional[List[str]] = None,
    excluded_websites: Optional[List[str]] = None,
    from_date: str = "",
    to_date: str = "",
    country: str = "",
) -> str:
    """Search the live web via xAI's web_search built-in tool.

    Parameters
    ----------
    query : str
        Required search query.
    allowed_websites : list of str, optional
        Whitelist of hostnames the search must restrict to (max 10).
        Mutually exclusive with ``excluded_websites``.
    excluded_websites : list of str, optional
        Blacklist of hostnames the search must avoid (max 10).
    from_date, to_date : str, optional
        ISO-8601 ``YYYY-MM-DD`` date filters.
    country : str, optional
        ISO-3166 alpha-2 country code (e.g. ``"FR"``, ``"US"``) to localize
        results.

    Returns
    -------
    str
        JSON-encoded ``{success, provider, tool, model, query, answer,
        citations, inline_citations}`` on success, or
        ``{success: false, error, error_type}`` on failure.
    """
    if not query or not query.strip():
        return tool_error("query is required for web_search", success=False, provider="xai", tool="web_search")

    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return tool_error("XAI_API_KEY is not set", success=False, provider="xai", tool="web_search")

    try:
        allowed = _normalize_websites(allowed_websites, "allowed_websites")
        excluded = _normalize_websites(excluded_websites, "excluded_websites")
        if allowed and excluded:
            return tool_error(
                "allowed_websites and excluded_websites cannot be used together",
                success=False, provider="xai", tool="web_search",
            )

        tool_def: Dict[str, Any] = {"type": "web_search"}
        if allowed:
            tool_def["allowed_websites"] = allowed
        if excluded:
            tool_def["excluded_websites"] = excluded
        if from_date.strip():
            tool_def["from_date"] = from_date.strip()
        if to_date.strip():
            tool_def["to_date"] = to_date.strip()
        if country.strip():
            tool_def["country"] = country.strip().upper()

        payload = {
            "model": _get_web_search_model(),
            "input": [
                {
                    "role": "user",
                    "content": query.strip(),
                }
            ],
            "tools": [tool_def],
            "store": False,
        }

        timeout_seconds = _get_web_search_timeout_seconds()
        max_retries = _get_web_search_retries()
        response: Optional[requests.Response] = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{_get_xai_base_url()}/responses",
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
                    "web_search upstream failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    _http_error_message(e),
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "web_search transient failure on attempt %s/%s: %s",
                    attempt + 1, max_retries + 1, e,
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))

        if response is None:
            raise RuntimeError("web_search request did not return a response")

        data = response.json()
        answer = _extract_response_text(data)
        citations = list(data.get("citations") or [])
        inline_citations = _extract_inline_citations(data)

        return json.dumps(
            {
                "success": True,
                "provider": "xai",
                "tool": "web_search",
                "model": payload["model"],
                "query": query.strip(),
                "answer": answer,
                "citations": citations,
                "inline_citations": inline_citations,
            },
            ensure_ascii=False,
        )
    except requests.HTTPError as e:
        logger.error("web_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "web_search",
                "error": _http_error_message(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except requests.ReadTimeout as e:
        logger.error("web_search timed out: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "web_search",
                "error": f"xAI web_search timed out after {_get_web_search_timeout_seconds()} seconds",
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("web_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "web_search",
                "error": str(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

WEB_SEARCH_SCHEMA = {
    "name": "xai_web_search",
    "description": (
        "Search the live web via xAI's web_search built-in tool. Returns "
        "an answer grounded in citations. Optional: allow/exclude lists of "
        "websites (max 10 each, mutually exclusive), date range filters, "
        "country code for localization."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Required search query.",
            },
            "allowed_websites": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hostnames the search must restrict to (max 10).",
            },
            "excluded_websites": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hostnames the search must avoid (max 10).",
            },
            "from_date": {
                "type": "string",
                "description": "ISO-8601 YYYY-MM-DD lower bound on result dates.",
            },
            "to_date": {
                "type": "string",
                "description": "ISO-8601 YYYY-MM-DD upper bound on result dates.",
            },
            "country": {
                "type": "string",
                "description": "ISO-3166 alpha-2 country code (e.g. 'FR').",
            },
        },
        "required": ["query"],
    },
}


def _handle_web_search_tool_call(args: Dict[str, Any], **_kw: Any) -> str:
    return web_search_tool(
        query=args.get("query", ""),
        allowed_websites=args.get("allowed_websites"),
        excluded_websites=args.get("excluded_websites"),
        from_date=str(args.get("from_date") or ""),
        to_date=str(args.get("to_date") or ""),
        country=str(args.get("country") or ""),
    )


registry.register(
    name="xai_web_search",
    toolset="xai_web_search",
    schema=WEB_SEARCH_SCHEMA,
    handler=_handle_web_search_tool_call,
    check_fn=check_web_search_requirements,
    emoji="🌐",
)
