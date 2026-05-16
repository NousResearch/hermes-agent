#!/usr/bin/env python3
"""X Search tool backed by xAI's built-in ``x_search`` Responses API tool.

Authentication
--------------
The tool registers when **either** xAI credential path is available:

* ``XAI_API_KEY`` is set in ``~/.hermes/.env`` or the process environment
  (paid xAI API key), OR
* The user is signed in via xAI Grok OAuth — SuperGrok subscription —
  i.e. ``hermes auth add xai-oauth`` has been run and the stored refresh
  token still works.

Credential preference at call time matches
:func:`tools.xai_http.resolve_xai_http_credentials`: SuperGrok OAuth first,
direct OAuth resolver second, ``XAI_API_KEY`` last. That helper also
auto-refreshes the OAuth access token when it's within the refresh skew
window, so a ``True`` from :func:`check_x_search_requirements` means the
bearer is fetchable AND non-empty.

Salvaged from PR #10786 (originally by @Jaaneek); credential resolution
reworked to honor both auth modes per Teknium's design.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_X_SEARCH_MODEL = "grok-4.20-reasoning"
DEFAULT_X_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_X_SEARCH_RETRIES = 2
MAX_HANDLES = 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_x_search_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return load_config().get("x_search", {}) or {}
    except Exception:
        return {}


def _get_x_search_model() -> str:
    cfg = _load_x_search_config()
    return (str(cfg.get("model") or "").strip() or DEFAULT_X_SEARCH_MODEL)


def _get_x_search_timeout_seconds() -> int:
    cfg = _load_x_search_config()
    raw_value = cfg.get("timeout_seconds", DEFAULT_X_SEARCH_TIMEOUT_SECONDS)
    try:
        return max(30, int(raw_value))
    except Exception:
        return DEFAULT_X_SEARCH_TIMEOUT_SECONDS


def _get_x_search_retries() -> int:
    cfg = _load_x_search_config()
    raw_value = cfg.get("retries", DEFAULT_X_SEARCH_RETRIES)
    try:
        return max(0, int(raw_value))
    except Exception:
        return DEFAULT_X_SEARCH_RETRIES


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

def _resolve_xai_bearer() -> Tuple[str, str, str]:
    """Return ``(api_key, base_url, source)``.

    ``source`` is one of ``"xai-oauth"`` or ``"xai"`` so callers (and tests)
    can tell which credential path won. Raises ``RuntimeError`` if no usable
    credential is available — the registered :func:`check_x_search_requirements`
    gate makes that case unreachable in normal operation, but the runtime
    check exists so a credential that expires between registration and
    invocation produces a clean tool error instead of a 401.
    """
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


def check_x_search_requirements() -> bool:
    """Return True when xAI credentials are available AND valid.

    ``resolve_xai_http_credentials`` calls
    :func:`hermes_cli.auth.resolve_xai_oauth_runtime_credentials` which
    auto-refreshes the OAuth access token if it's expiring; a successful
    return therefore implies a usable bearer.
    """
    try:
        creds = resolve_xai_http_credentials()
        return bool(str(creds.get("api_key") or "").strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_handles(handles: Optional[List[str]], field_name: str) -> List[str]:
    cleaned: List[str] = []
    for handle in handles or []:
        normalized = str(handle or "").strip().lstrip("@")
        if normalized:
            cleaned.append(normalized)
    if len(cleaned) > MAX_HANDLES:
        raise ValueError(f"{field_name} supports at most {MAX_HANDLES} handles")
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

def x_search_tool(
    query: str,
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: str = "",
    to_date: str = "",
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
) -> str:
    if not query or not query.strip():
        return tool_error("query is required for x_search")

    try:
        api_key, base_url, source = _resolve_xai_bearer()
    except RuntimeError as exc:
        return tool_error(str(exc))

    try:
        allowed = _normalize_handles(allowed_x_handles, "allowed_x_handles")
        excluded = _normalize_handles(excluded_x_handles, "excluded_x_handles")
        if allowed and excluded:
            return tool_error("allowed_x_handles and excluded_x_handles cannot be used together")

        tool_def: Dict[str, Any] = {"type": "x_search"}
        if allowed:
            tool_def["allowed_x_handles"] = allowed
        if excluded:
            tool_def["excluded_x_handles"] = excluded
        if from_date.strip():
            tool_def["from_date"] = from_date.strip()
        if to_date.strip():
            tool_def["to_date"] = to_date.strip()
        if enable_image_understanding:
            tool_def["enable_image_understanding"] = True
        if enable_video_understanding:
            tool_def["enable_video_understanding"] = True

        payload = {
            "model": _get_x_search_model(),
            "input": [
                {
                    "role": "user",
                    "content": query.strip(),
                }
            ],
            "tools": [tool_def],
            "store": False,
        }

        timeout_seconds = _get_x_search_timeout_seconds()
        max_retries = _get_x_search_retries()
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
                    "x_search upstream failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    _http_error_message(e),
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "x_search transient failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))

        if response is None:
            raise RuntimeError("x_search request did not return a response")

        data = response.json()

        answer = _extract_response_text(data)
        citations = list(data.get("citations") or [])
        inline_citations = _extract_inline_citations(data)

        return json.dumps(
            {
                "success": True,
                "provider": "xai",
                "credential_source": source,
                "tool": "x_search",
                "model": payload["model"],
                "query": query.strip(),
                "answer": answer,
                "citations": citations,
                "inline_citations": inline_citations,
            },
            ensure_ascii=False,
        )
    except requests.HTTPError as e:
        logger.error("x_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "x_search",
                "error": _http_error_message(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except requests.ReadTimeout as e:
        logger.error("x_search timed out: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "x_search",
                "error": f"xAI x_search timed out after {_get_x_search_timeout_seconds()} seconds",
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("x_search failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "x_search",
                "error": str(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )


X_SEARCH_SCHEMA = {
    "name": "x_search",
    "description": (
        "Search X (Twitter) posts, profiles, and threads using xAI's built-in "
        "X Search tool. Use this for current discussion, reactions, or claims "
        "on X rather than general web pages. Available when xAI credentials "
        "are configured (SuperGrok OAuth or XAI_API_KEY)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up on X.",
            },
            "allowed_x_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of X handles to include exclusively (max 10).",
            },
            "excluded_x_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of X handles to exclude (max 10).",
            },
            "from_date": {
                "type": "string",
                "description": "Optional start date in YYYY-MM-DD format.",
            },
            "to_date": {
                "type": "string",
                "description": "Optional end date in YYYY-MM-DD format.",
            },
            "enable_image_understanding": {
                "type": "boolean",
                "description": "Whether xAI should analyze images attached to matching X posts.",
                "default": False,
            },
            "enable_video_understanding": {
                "type": "boolean",
                "description": "Whether xAI should analyze videos attached to matching X posts.",
                "default": False,
            },
        },
        "required": ["query"],
    },
}


def _handle_x_search(args, **kw):
    return x_search_tool(
        query=args.get("query", ""),
        allowed_x_handles=args.get("allowed_x_handles"),
        excluded_x_handles=args.get("excluded_x_handles"),
        from_date=args.get("from_date", ""),
        to_date=args.get("to_date", ""),
        enable_image_understanding=bool(args.get("enable_image_understanding", False)),
        enable_video_understanding=bool(args.get("enable_video_understanding", False)),
    )


registry.register(
    name="x_search",
    toolset="x_search",
    schema=X_SEARCH_SCHEMA,
    handler=_handle_x_search,
    check_fn=check_x_search_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🐦",
    max_result_size_chars=100_000,
)
