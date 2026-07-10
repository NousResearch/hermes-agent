"""Slack Web API tool for thread-safe workspace operations.

This tool is intentionally scoped to Slack's Web API primitives that an agent
needs for reliable workspace work: channel discovery, message history, thread
replies, file metadata, posting replies, and reactions.  It is exposed only in
Slack-oriented toolsets (not the global core toolset) so the model-tool surface
stays narrow.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text
from tools.registry import registry

_SLACK_API_BASE = "https://slack.com/api/"
_CHANNEL_RE = re.compile(r"^[CGD][A-Z0-9]{8,}$")
_TS_RE = re.compile(r"^\d{10}\.\d{6}$")
_SAFE_CHANNEL_TYPES = "public_channel,private_channel,im,mpim"


def check_slack_api_requirements() -> bool:
    """Expose the tool only when a Slack bot token is configured."""
    return bool(os.getenv("SLACK_BOT_TOKEN"))


def _token() -> str:
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is not configured")
    return token


def _clean_error(text: Any) -> str:
    return redact_sensitive_text(str(text))


def _post_form(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    body = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}).encode()
    req = urllib.request.Request(
        _SLACK_API_BASE + method,
        data=body,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps({k: v for k, v in payload.items() if v is not None}).encode()
    req = urllib.request.Request(
        _SLACK_API_BASE + method,
        data=body,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    profile = user.get("profile") or {}
    return {
        "id": user.get("id"),
        "name": user.get("name"),
        "real_name": user.get("real_name") or profile.get("real_name"),
        "display_name": profile.get("display_name"),
        "is_bot": user.get("is_bot"),
        "deleted": user.get("deleted"),
        "team_id": user.get("team_id"),
    }


def _public_channel(ch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": ch.get("id"),
        "name": ch.get("name"),
        "is_channel": ch.get("is_channel"),
        "is_group": ch.get("is_group"),
        "is_im": ch.get("is_im"),
        "is_mpim": ch.get("is_mpim"),
        "is_private": ch.get("is_private"),
        "is_member": ch.get("is_member"),
        "is_archived": ch.get("is_archived"),
        "num_members": ch.get("num_members"),
        "topic": (ch.get("topic") or {}).get("value"),
        "purpose": (ch.get("purpose") or {}).get("value"),
    }


def _public_file(f: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f.get("id"),
        "name": f.get("name"),
        "title": f.get("title"),
        "mimetype": f.get("mimetype"),
        "filetype": f.get("filetype"),
        "size": f.get("size"),
        "url_private": f.get("url_private"),
        "created": f.get("created"),
        "user": f.get("user"),
    }


def _public_message(m: Dict[str, Any], include_blocks: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": m.get("type"),
        "subtype": m.get("subtype"),
        "user": m.get("user"),
        "bot_id": m.get("bot_id"),
        "username": m.get("username"),
        "ts": m.get("ts"),
        "thread_ts": m.get("thread_ts"),
        "reply_count": m.get("reply_count"),
        "latest_reply": m.get("latest_reply"),
        "text": m.get("text"),
    }
    if m.get("files"):
        out["files"] = [_public_file(f) for f in m.get("files", [])]
    if include_blocks and m.get("blocks"):
        out["blocks"] = m.get("blocks")
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def _require_channel(channel: Optional[str]) -> str:
    if not channel or not _CHANNEL_RE.match(channel):
        raise ValueError("channel must be a Slack conversation ID starting with C/G/D")
    return channel


def _require_ts(ts: Optional[str], name: str = "ts") -> str:
    if not ts or not _TS_RE.match(ts):
        raise ValueError(f"{name} must be a Slack timestamp like 1783504790.412029")
    return ts


def _paginate(method: str, base: Dict[str, Any], collection_key: str, limit: int, max_pages: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    cursor = None
    pages = 0
    while pages < max_pages and len(items) < limit:
        page_limit = min(200, max(1, limit - len(items)))
        params = dict(base, limit=page_limit)
        if cursor:
            params["cursor"] = cursor
        res = _post_form(method, params)
        if not res.get("ok"):
            raise RuntimeError(f"Slack API {method} failed: {res.get('error')}")
        items.extend(res.get(collection_key) or [])
        cursor = (res.get("response_metadata") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break
    return items[:limit]


def slack_api_tool(
    action: str,
    channel: Optional[str] = None,
    thread_ts: Optional[str] = None,
    ts: Optional[str] = None,
    text: Optional[str] = None,
    query: Optional[str] = None,
    user: Optional[str] = None,
    emoji: Optional[str] = None,
    limit: int = 50,
    include_blocks: bool = False,
    types: str = _SAFE_CHANNEL_TYPES,
    max_pages: int = 5,
    oldest: Optional[str] = None,
    latest: Optional[str] = None,
) -> str:
    """Handle Slack Web API actions and return compact JSON."""
    try:
        limit = max(1, min(int(limit or 50), 200))
        max_pages = max(1, min(int(max_pages or 5), 10))
        action = (action or "").strip()

        if action == "auth_test":
            res = _post_form("auth.test", {})
            return json.dumps({k: res.get(k) for k in ("ok", "team", "team_id", "user", "user_id", "bot_id", "url")}, ensure_ascii=False)

        if action == "list_channels":
            channels = _paginate(
                "conversations.list",
                {"types": types or _SAFE_CHANNEL_TYPES, "exclude_archived": "true"},
                "channels",
                limit,
                max_pages,
            )
            if query:
                q = query.lower()
                channels = [c for c in channels if q in (c.get("name") or "").lower() or q in (c.get("id") or "").lower()]
            return json.dumps({"ok": True, "channels": [_public_channel(c) for c in channels]}, ensure_ascii=False)

        if action == "channel_info":
            res = _post_form("conversations.info", {"channel": _require_channel(channel)})
            if not res.get("ok"):
                raise RuntimeError(res.get("error"))
            return json.dumps({"ok": True, "channel": _public_channel(res.get("channel") or {})}, ensure_ascii=False)

        if action == "history":
            messages = _paginate(
                "conversations.history",
                {"channel": _require_channel(channel), "oldest": oldest, "latest": latest, "inclusive": "true"},
                "messages",
                limit,
                max_pages,
            )
            return json.dumps({"ok": True, "channel": channel, "messages": [_public_message(m, include_blocks) for m in messages]}, ensure_ascii=False)

        if action == "replies":
            messages = _paginate(
                "conversations.replies",
                {"channel": _require_channel(channel), "ts": _require_ts(thread_ts or ts, "thread_ts"), "oldest": oldest, "latest": latest, "inclusive": "true"},
                "messages",
                limit,
                max_pages,
            )
            return json.dumps({"ok": True, "channel": channel, "thread_ts": thread_ts or ts, "messages": [_public_message(m, include_blocks) for m in messages]}, ensure_ascii=False)

        if action == "send":
            if not text:
                raise ValueError("text is required for send")
            res = _post_json("chat.postMessage", {"channel": _require_channel(channel), "thread_ts": thread_ts, "text": text})
            return json.dumps({"ok": res.get("ok"), "channel": res.get("channel"), "ts": res.get("ts"), "thread_ts": thread_ts, "error": res.get("error")}, ensure_ascii=False)

        if action == "react":
            if not emoji:
                raise ValueError("emoji is required for react")
            res = _post_form("reactions.add", {"channel": _require_channel(channel), "timestamp": _require_ts(ts), "name": emoji.strip(":")})
            return json.dumps({"ok": res.get("ok"), "error": res.get("error")}, ensure_ascii=False)

        if action == "user_info":
            if not user:
                raise ValueError("user is required for user_info")
            res = _post_form("users.info", {"user": user})
            if not res.get("ok"):
                raise RuntimeError(res.get("error"))
            return json.dumps({"ok": True, "user": _public_user(res.get("user") or {})}, ensure_ascii=False)

        if action == "files":
            params: Dict[str, Any] = {"channel": channel, "user": user, "ts_from": oldest, "ts_to": latest}
            res = _post_form("files.list", {k: v for k, v in params.items() if v})
            if not res.get("ok"):
                raise RuntimeError(res.get("error"))
            files = (res.get("files") or [])[:limit]
            return json.dumps({"ok": True, "files": [_public_file(f) for f in files]}, ensure_ascii=False)

        raise ValueError(f"Unknown action: {action}")
    except Exception as exc:
        return json.dumps({"ok": False, "error": _clean_error(exc)}, ensure_ascii=False)


SLACK_API_SCHEMA = {
    "name": "slack_api",
    "description": (
        "Use Slack Web API for thread-safe workspace work: list channels, read "
        "channel history, read thread replies, send parent/thread messages, add "
        "reactions, inspect users, and list files. Always call replies when a "
        "message has reply_count or when auditing a Slack thread."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["auth_test", "list_channels", "channel_info", "history", "replies", "send", "react", "user_info", "files"],
                "description": "Slack operation to perform.",
            },
            "channel": {"type": "string", "description": "Slack conversation ID (C..., G..., or D...)."},
            "thread_ts": {"type": "string", "description": "Parent message timestamp for thread replies or posting into a thread."},
            "ts": {"type": "string", "description": "Message timestamp for reactions or as a thread timestamp fallback."},
            "text": {"type": "string", "description": "Message text for action='send'."},
            "query": {"type": "string", "description": "Case-insensitive channel name/id filter for list_channels."},
            "user": {"type": "string", "description": "Slack user ID for user_info or files filtering."},
            "emoji": {"type": "string", "description": "Emoji name for react, with or without colons."},
            "limit": {"type": "integer", "description": "Maximum items to return (1-200).", "default": 50},
            "include_blocks": {"type": "boolean", "description": "Include raw Slack blocks in message output; normally false to save tokens.", "default": False},
            "types": {"type": "string", "description": "Conversation types for list_channels.", "default": _SAFE_CHANNEL_TYPES},
            "max_pages": {"type": "integer", "description": "Pagination pages to fetch (1-10).", "default": 5},
            "oldest": {"type": "string", "description": "Slack timestamp lower bound for history/replies/files."},
            "latest": {"type": "string", "description": "Slack timestamp upper bound for history/replies/files."},
        },
        "required": ["action"],
    },
}


registry.register(
    name="slack_api",
    toolset="slack",
    schema=SLACK_API_SCHEMA,
    handler=lambda args, **kw: slack_api_tool(
        action=args.get("action", ""),
        channel=args.get("channel"),
        thread_ts=args.get("thread_ts"),
        ts=args.get("ts"),
        text=args.get("text"),
        query=args.get("query"),
        user=args.get("user"),
        emoji=args.get("emoji"),
        limit=args.get("limit", 50),
        include_blocks=bool(args.get("include_blocks", False)),
        types=args.get("types", _SAFE_CHANNEL_TYPES),
        max_pages=args.get("max_pages", 5),
        oldest=args.get("oldest"),
        latest=args.get("latest"),
    ),
    check_fn=check_slack_api_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    description="Slack Web API read/write/thread tool",
    emoji="💼",
    max_result_size_chars=24000,
)
