#!/usr/bin/env python3
"""X Search tool backed by xAI's built-in ``x_search`` Responses API tool.

Registers when either xAI credential path is available (``XAI_API_KEY`` or ``hermes auth add
xai-oauth``). At call time an explicit ``XAI_API_KEY`` wins (``prefer_api_key=True``): x_search
is API-metered and the subscription OAuth bearer answers ``/v1/responses`` without citations.
Date filters are validated client-side so malformed windows fail fast instead of burning a
billable call. Results carry ``degraded``: True when a narrowing filter was active AND xAI
returned no citations in either channel (answer came from model knowledge, not the X index).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from tools.registry import registry, tool_error
from tools.xai_http import DEFAULT_XAI_BASE_URL, hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

DEFAULT_X_SEARCH_MODEL = "grok-4.5"
DEFAULT_X_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_X_SEARCH_RETRIES = 2
X_SEARCH_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
MAX_HANDLES = 10


def _load_x_search_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("x_search", {}) or {}
    except Exception:
        return {}


def _get_x_search_reasoning_effort() -> Optional[str]:
    raw_value = _load_x_search_config().get("reasoning_effort")
    effort = str(raw_value).strip().lower() if raw_value is not None else ""
    if effort and effort not in X_SEARCH_REASONING_EFFORTS:
        allowed = ", ".join(X_SEARCH_REASONING_EFFORTS)
        raise ValueError(f"x_search.reasoning_effort must be one of: {allowed} (got {raw_value!r})")
    return effort or None


def _get_x_search_int(key: str, default: int, floor: int) -> int:
    try:
        return max(floor, int(_load_x_search_config().get(key, default)))
    except Exception:
        return default


def _resolve_xai_bearer() -> Tuple[str, str, str]:
    """Return ``(api_key, base_url, source)``; ``source`` is ``"xai-oauth"`` or ``"xai"``. Raises RuntimeError
    when no credential is usable (expiry between registration and call -> clean tool error, not a 401).

    x_search is API-index access: when a subscription OAuth credential is configured alongside a paid
    ``XAI_API_KEY``, the OAuth path authorizes but answers ``/v1/responses`` in a degraded Grok explanatory
    mode with no citations, while the API key returns real posts (#88040). Pass ``prefer_api_key=True`` so
    the shared resolver checks the explicit API key first — same root cause as the TTS fix for #87045
    (#87081) — keeping OAuth as the fallback when no API key is configured.
    """
    creds = resolve_xai_http_credentials(prefer_api_key=True)
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "No xAI credentials available. Run `hermes auth add xai-oauth` "
            "to sign in with your SuperGrok subscription, or set XAI_API_KEY."
        )
    base_url = str(creds.get("base_url") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    return api_key, base_url, str(creds.get("provider") or "xai")


def check_x_search_requirements() -> bool:
    """True when xAI credentials resolve to a non-empty bearer (OAuth auto-refreshed)."""
    try:
        return bool(str(resolve_xai_http_credentials().get("api_key") or "").strip())
    except Exception:
        return False


def _normalize_handles(handles: Optional[List[str]], field_name: str) -> List[str]:
    cleaned = [h for h in (str(handle or "").strip().lstrip("@") for handle in handles or []) if h]
    if len(cleaned) > MAX_HANDLES:
        raise ValueError(f"{field_name} supports at most {MAX_HANDLES} handles")
    return cleaned


def _parse_iso_date(value: str, field_name: str) -> Optional[date]:
    """Strict YYYY-MM-DD or None for blank (xAI silently accepts malformed dates and returns no citations)."""
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (got {raw!r})") from exc


def _validate_date_range(from_date: str, to_date: str) -> None:
    """Both parse as YYYY-MM-DD; from <= to; from not after today UTC (to may be in the future)."""
    parsed_from, parsed_to = _parse_iso_date(from_date, "from_date"), _parse_iso_date(to_date, "to_date")
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise ValueError(
            f"from_date ({parsed_from.isoformat()}) must be on or before to_date ({parsed_to.isoformat()})"
        )
    today_utc = datetime.now(timezone.utc).date()
    if parsed_from is not None and parsed_from > today_utc:
        raise ValueError(
            f"from_date ({parsed_from.isoformat()}) is in the future; "
            f"X Search only indexes past posts (today UTC is {today_utc.isoformat()})"
        )


def _message_contents(payload: Dict[str, Any]):
    for item in payload.get("output", []) or []:
        if item.get("type") == "message":
            yield from item.get("content", []) or []


def _extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text
    contents = (c for c in _message_contents(payload) if c.get("type") in {"output_text", "text"})
    parts = (str(c.get("text") or "").strip() for c in contents)
    return "\n\n".join(p for p in parts if p).strip()


def _extract_inline_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "url": a.get("url", ""), "title": a.get("title", ""),
            "start_index": a.get("start_index"), "end_index": a.get("end_index"),
        }
        for content in _message_contents(payload)
        for a in content.get("annotations", []) or []
        if a.get("type") == "url_citation"
    ]


def _http_error_message(exc: requests.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        payload = response.json()
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        text = str(getattr(response, "text", "") or "").strip()
        return text[:500] if text else str(exc)
    code = str(payload.get("code") or "").strip()
    message = str(payload.get("error") or "").strip() or str(payload)
    return (f"{code}: {message}" if code and code not in message else message) or str(exc)


def _error_json(error: str, exc: BaseException) -> str:
    body = {"success": False, "provider": "xai", "tool": "x_search", "error": error}
    return json.dumps({**body, "error_type": type(exc).__name__}, ensure_ascii=False)


def _post_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> requests.Response:
    """POST with retries on 5xx / timeout / connection errors; re-raises the last failure."""
    timeout_seconds = _get_x_search_int("timeout_seconds", DEFAULT_X_SEARCH_TIMEOUT_SECONDS, 30)
    max_retries = _get_x_search_int("retries", DEFAULT_X_SEARCH_RETRIES, 0)
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code is None or status_code < 500 or attempt >= max_retries:
                raise
            kind, detail = "upstream", _http_error_message(e)
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            if attempt >= max_retries:
                raise
            kind, detail = "transient", e
        logger.warning("x_search %s failure on attempt %s/%s: %s", kind, attempt + 1, max_retries + 1, detail)
        time.sleep(min(5.0, 1.5 * (attempt + 1)))
    raise RuntimeError("x_search request did not return a response")


def _build_x_search_tool_def(
    allowed_x_handles, excluded_x_handles, from_date: str, to_date: str,
    enable_image_understanding: bool, enable_video_understanding: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``(tool_def, active_filters)``; raises ValueError on invalid filters."""
    allowed = _normalize_handles(allowed_x_handles, "allowed_x_handles")
    excluded = _normalize_handles(excluded_x_handles, "excluded_x_handles")
    if allowed and excluded:
        raise ValueError("allowed_x_handles and excluded_x_handles cannot be used together")
    _validate_date_range(from_date, to_date)

    tool_def: Dict[str, Any] = {"type": "x_search"}
    active_filters: List[str] = []
    filters = (("allowed_x_handles", allowed), ("excluded_x_handles", excluded),
               ("from_date", from_date.strip()), ("to_date", to_date.strip()))
    for key, value in filters:
        if value:
            tool_def[key] = value
            active_filters.append(key)
    if enable_image_understanding:
        tool_def["enable_image_understanding"] = True
    if enable_video_understanding:
        tool_def["enable_video_understanding"] = True
    return tool_def, active_filters


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
        tool_def, active_filters = _build_x_search_tool_def(
            allowed_x_handles, excluded_x_handles, from_date, to_date,
            enable_image_understanding, enable_video_understanding,
        )
        reasoning_effort = _get_x_search_reasoning_effort()
    except ValueError as exc:
        return tool_error(str(exc))

    try:
        payload = {
            "model": str(_load_x_search_config().get("model") or "").strip() or DEFAULT_X_SEARCH_MODEL,
            "input": [{"role": "user", "content": query.strip()}],
            "tools": [tool_def],
            "store": False,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        headers = {
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        }
        data = _post_with_retries(f"{base_url}/responses", headers, payload).json()
        citations = list(data.get("citations") or [])
        inline_citations = _extract_inline_citations(data)
        # xAI returns 200 with a synthesized answer even when no posts match the narrowing
        # filters; with both citation channels empty the answer came from training data.
        degraded = bool(active_filters) and not citations and not inline_citations
        result = {
            "success": True, "provider": "xai", "credential_source": source, "tool": "x_search",
            "model": payload["model"], "query": query.strip(), "answer": _extract_response_text(data),
            "citations": citations, "inline_citations": inline_citations, "degraded": degraded,
            "degraded_reason": (
                f"no citations returned despite filters: {', '.join(active_filters)}" if degraded else None
            ),
        }
        return json.dumps(result, ensure_ascii=False)
    except requests.HTTPError as e:
        logger.error("x_search failed: %s", e, exc_info=True)
        return _error_json(_http_error_message(e), e)
    except requests.ReadTimeout as e:
        logger.error("x_search timed out: %s", e, exc_info=True)
        timeout = _get_x_search_int("timeout_seconds", DEFAULT_X_SEARCH_TIMEOUT_SECONDS, 30)
        return _error_json(f"xAI x_search timed out after {timeout} seconds", e)
    except Exception as e:
        logger.error("x_search failed: %s", e, exc_info=True)
        return _error_json(str(e), e)


X_SEARCH_SCHEMA = {
    "name": "x_search",
    "description": (
        "Search X (Twitter) posts, profiles, and threads using xAI's built-in "
        "X Search tool. Read-only discovery only: use this for current "
        "discussion, reactions, or claims on public X rather than general web "
        "pages. Do not use it to post, reply, like, DM, upload media, delete, "
        "or inspect the user's authenticated X account — those require a "
        "separate authenticated X API surface outside this tool. Available "
        "when xAI credentials are configured (SuperGrok OAuth or XAI_API_KEY)."
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
        args.get("query", ""), args.get("allowed_x_handles"), args.get("excluded_x_handles"),
        args.get("from_date", ""), args.get("to_date", ""),
        bool(args.get("enable_image_understanding", False)),
        bool(args.get("enable_video_understanding", False)),
    )


registry.register(
    name="x_search", toolset="x_search", schema=X_SEARCH_SCHEMA, handler=_handle_x_search,
    check_fn=check_x_search_requirements, requires_env=["XAI_API_KEY"], emoji="🐦",
    max_result_size_chars=100_000,
)
