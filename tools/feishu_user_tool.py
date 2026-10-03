"""User-scoped Feishu / Lark message tools backed by a ``user_access_token``.

``feishu_message_search`` is the whole point of the user grant: Feishu's cross-chat full-text search
(``POST /open-apis/im/v1/messages/search``) accepts a ``user_access_token`` and nothing else, so the
bot's ``tenant_access_token`` cannot reach it at all. ``feishu_message_list`` reads one chat's
history *as the user*, which sees conversations — and other bots' messages in shared chats — that the
Hermes bot is not itself a party to.

Both tools disappear from the schema until ``hermes feishu login`` has stored a grant
(``feishu_user_auth.has_user_token`` is the ``check_fn``), so an install that never authorized user
access pays nothing for them.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from tools.feishu_user_auth import has_user_token, resolve_user_access_token
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_SEARCH_PATH = "/open-apis/im/v1/messages/search"
_LIST_PATH = "/open-apis/im/v1/messages"

# Feishu's own caps: search tops out at 30 per page, message list at 50.
_SEARCH_PAGE_SIZE_MAX = 30
_LIST_PAGE_SIZE_MAX = 50
_DEFAULT_PAGE_SIZE = 15
_REQUEST_TIMEOUT_S = 20.0

# ``display_info`` wraps matched terms in <h>…</h>; the model already knows the query, so the
# markers are noise that also reads as broken HTML.
_HIGHLIGHT_RE = re.compile(r"</?h>")

# A message body can be an arbitrarily large post/card payload; one page of 30 of them would
# dominate the turn. Kept per-message rather than truncating the whole result so every row survives.
_BODY_PREVIEW_CHARS = 500


def _clamp_page_size(raw: Any, maximum: int) -> int:
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PAGE_SIZE
    return max(1, min(size, maximum))


def _string_list(raw: Any) -> List[str]:
    """Accept a list or a comma-separated string (XML tool-calling mode delivers arrays as text)."""
    if isinstance(raw, str):
        raw = [part for part in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _call(method: str, path: str, *, queries: Dict[str, Any], body: Optional[dict] = None) -> Dict[str, Any]:
    """One Open API call with a live UAT. Returns ``data``; raises RuntimeError on a non-zero code."""
    import httpx
    credentials = resolve_user_access_token()
    url = f"{credentials['base_url']}{path}"
    headers = {
        "Authorization": f"{credentials['token_type']} {credentials['access_token']}",
        "Content-Type": "application/json; charset=utf-8",
    }
    params = {k: v for k, v in queries.items() if v not in (None, "")}
    response = httpx.request(
        method, url, params=params, headers=headers, json=body, timeout=_REQUEST_TIMEOUT_S)
    try:
        payload = response.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    code = payload.get("code")
    if code != 0:
        detail = str(payload.get("msg") or response.text.strip() or f"HTTP {response.status_code}")
        # 99991672 / 99991679: the grant lacks the scope. Naming the fix beats echoing Feishu's code.
        if code in (99991672, 99991679):
            detail += (" — the user grant is missing a required scope; re-run "
                       "`hermes feishu login` (it requests search:message and the "
                       "im:message.*_msg:get_as_user pair).")
        raise RuntimeError(detail)
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _summarize_body(item: Dict[str, Any]) -> str:
    """Readable text for one listed message: the ``text`` field when there is one, else raw content."""
    content = ((item.get("body") or {}) if isinstance(item.get("body"), dict) else {}).get("content")
    if not isinstance(content, str) or not content:
        return ""
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return content[:_BODY_PREVIEW_CHARS]
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return parsed["text"][:_BODY_PREVIEW_CHARS]
    return content[:_BODY_PREVIEW_CHARS]


def _search_row(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("meta_data") if isinstance(item.get("meta_data"), dict) else {}
    return {
        "message_id": meta.get("message_id") or item.get("id"),
        "chat_id": meta.get("chat_id"),
        "from_id": meta.get("from_id"),
        "msg_type": meta.get("type"),
        "create_time": meta.get("create_time"),
        "thread_id": meta.get("thread_id"),
        "is_p2p_chat": meta.get("is_p2p_chat"),
        "snippet": _HIGHLIGHT_RE.sub("", str(item.get("display_info") or "")),
    }


def _list_row(item: Dict[str, Any]) -> Dict[str, Any]:
    sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
    return {
        "message_id": item.get("message_id"),
        "chat_id": item.get("chat_id"),
        "sender_id": sender.get("id"),
        "sender_type": sender.get("sender_type"),
        "msg_type": item.get("msg_type"),
        "create_time": item.get("create_time"),
        "root_id": item.get("root_id"),
        "deleted": item.get("deleted"),
        "text": _summarize_body(item),
    }


FEISHU_MESSAGE_SEARCH_SCHEMA = {
    "name": "feishu_message_search",
    "description": (
        "Full-text search across the authorized user's Feishu/Lark message history — every chat "
        "they can see, including conversations and other bots' messages the Hermes bot is not part "
        "of. Returns matching messages with chat_id, sender, timestamp and a plain-text snippet; "
        "follow up with feishu_message_list on a chat_id to read the surrounding conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords."},
            "chat_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Restrict the search to these chat IDs (oc_...). Omit to search everywhere.",
            },
            "from_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Restrict to messages sent by these open IDs (ou_...).",
            },
            "start_time": {
                "type": "string",
                "description": "Only messages at or after this RFC 3339 timestamp (e.g. 2026-03-21T16:15:30+08:00).",
            },
            "end_time": {"type": "string", "description": "Only messages at or before this RFC 3339 timestamp."},
            "page_size": {
                "type": "integer",
                "description": f"Results per page, 1-{_SEARCH_PAGE_SIZE_MAX} (default {_DEFAULT_PAGE_SIZE}).",
            },
            "page_token": {"type": "string", "description": "page_token from a previous call, to fetch the next page."},
        },
        "required": ["query"],
    },
}

FEISHU_MESSAGE_LIST_SCHEMA = {
    "name": "feishu_message_list",
    "description": (
        "List messages in one Feishu/Lark chat as the authorized user, newest or oldest first. "
        "Unlike the bot's own view this includes messages the Hermes bot did not send or receive, "
        "so it can be used to read what other agents said in a shared chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {"type": "string", "description": "The chat to read (oc_...)."},
            "start_time": {"type": "string", "description": "Unix seconds; only messages at or after this time."},
            "end_time": {"type": "string", "description": "Unix seconds; only messages at or before this time."},
            "sort_type": {
                "type": "string", "enum": ["ByCreateTimeAsc", "ByCreateTimeDesc"],
                "description": "Oldest-first (default) or newest-first.",
            },
            "page_size": {
                "type": "integer",
                "description": f"Messages per page, 1-{_LIST_PAGE_SIZE_MAX} (default {_DEFAULT_PAGE_SIZE}).",
            },
            "page_token": {"type": "string", "description": "page_token from a previous call."},
        },
        "required": ["chat_id"],
    },
}


def _handle_feishu_message_search(args: dict, **kwargs) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")
    message_filter: Dict[str, Any] = {}
    for key in ("chat_ids", "from_ids"):
        values = _string_list(args.get(key))
        if values:
            message_filter[key] = values
    time_range = {
        key: str(args[key]).strip()
        for key in ("start_time", "end_time") if str(args.get(key) or "").strip()
    }
    if time_range:
        message_filter["time_range"] = time_range

    body: Dict[str, Any] = {"query": query}
    if message_filter:
        body["filter"] = message_filter
    try:
        data = _call(
            "POST", _SEARCH_PATH,
            queries={
                "page_size": _clamp_page_size(args.get("page_size"), _SEARCH_PAGE_SIZE_MAX),
                "page_token": str(args.get("page_token") or "").strip(),
            },
            body=body)
    except Exception as exc:
        return tool_error(f"Feishu message search failed: {exc}")
    items = data.get("items") if isinstance(data.get("items"), list) else []
    return tool_result(
        success=True, count=len(items), total=data.get("total"),
        messages=[_search_row(item) for item in items if isinstance(item, dict)],
        has_more=bool(data.get("has_more")), page_token=data.get("page_token") or None,
        notice=data.get("notice") or None)


def _handle_feishu_message_list(args: dict, **kwargs) -> str:
    chat_id = str(args.get("chat_id") or "").strip()
    if not chat_id:
        return tool_error("chat_id is required")
    sort_type = str(args.get("sort_type") or "").strip()
    try:
        data = _call(
            "GET", _LIST_PATH,
            queries={
                "container_id_type": "chat", "container_id": chat_id,
                "start_time": str(args.get("start_time") or "").strip(),
                "end_time": str(args.get("end_time") or "").strip(),
                "sort_type": sort_type if sort_type in {"ByCreateTimeAsc", "ByCreateTimeDesc"} else "",
                "page_size": _clamp_page_size(args.get("page_size"), _LIST_PAGE_SIZE_MAX),
                "page_token": str(args.get("page_token") or "").strip(),
            })
    except Exception as exc:
        return tool_error(f"Feishu message list failed: {exc}")
    items = data.get("items") if isinstance(data.get("items"), list) else []
    return tool_result(
        success=True, chat_id=chat_id, count=len(items),
        messages=[_list_row(item) for item in items if isinstance(item, dict)],
        has_more=bool(data.get("has_more")), page_token=data.get("page_token") or None)


registry.register(
    name="feishu_message_search", toolset="feishu_user", schema=FEISHU_MESSAGE_SEARCH_SCHEMA,
    handler=_handle_feishu_message_search, check_fn=has_user_token, requires_env=[],
    is_async=False, description="Search the user's Feishu/Lark message history", emoji="\U0001f50d")

registry.register(
    name="feishu_message_list", toolset="feishu_user", schema=FEISHU_MESSAGE_LIST_SCHEMA,
    handler=_handle_feishu_message_list, check_fn=has_user_token, requires_env=[],
    is_async=False, description="List a Feishu/Lark chat's messages as the user", emoji="\U0001f4dc")
