"""Slack workspace intelligence tool.

Provides read-only Slack channel/history access for Slack-connected Hermes
profiles. Access is intentionally scoped to conversations the bot is already a
member of; the tool never joins channels or attempts admin actions.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.registry import registry

try:  # pragma: no cover - exercised via tests with monkeypatches
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional extra
    WebClient = None  # type: ignore[assignment]
    SlackApiError = Exception  # type: ignore[assignment]
    SLACK_SDK_AVAILABLE = False


_CHANNEL_ID_RE = re.compile(r"^[CGD][A-Z0-9]{8,}$")
_CHANNEL_MENTION_RE = re.compile(r"^<#([CGD][A-Z0-9]{8,})(?:\|[^>]+)?>$")
_RELATIVE_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_MAX_HISTORY_LIMIT = 200
_MAX_SEARCH_CHANNELS = 20
_USER_CACHE: Dict[str, str] = {}


SLACK_SCHEMA = {
    "name": "slack",
    "description": (
        "Read Slack conversations the bot is already a member of. Use this for "
        "workspace/project updates, decisions, blockers, open questions, and "
        "channel summaries. The tool cannot join channels or bypass Slack scopes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_channels", "read_channel", "read_thread", "search_messages"],
                "description": "Action to perform.",
            },
            "channel": {
                "type": "string",
                "description": "Slack channel name (#product), ID (C...), DM ID (D...), or channel mention (<#C...|name>).",
            },
            "channels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Channels to scan for search_messages. If omitted, scans joined channels up to max_channels.",
            },
            "thread_ts": {
                "type": "string",
                "description": "Slack thread timestamp for read_thread.",
            },
            "query": {
                "type": "string",
                "description": "Case-insensitive text query for search_messages.",
            },
            "since": {
                "type": "string",
                "description": "Optional lower bound: Slack ts, Unix seconds, ISO datetime/date, or relative like 24h, 7d, 2w.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Maximum messages/results to return. Default 50; hard-capped at 200.",
            },
            "max_channels": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "For search_messages without explicit channels, max joined channels to scan. Default 10.",
            },
            "include_threads": {
                "type": "boolean",
                "description": "For read_channel, include a small preview of replies for messages with threads. Default false to avoid rate limits.",
            },
        },
        "required": ["action"],
    },
}


def check_slack_tool_requirements() -> bool:
    """Return True when Slack SDK and a bot token are configured."""
    return SLACK_SDK_AVAILABLE and bool(_get_bot_token())


def _get_bot_token() -> Optional[str]:
    token = (os.environ.get("SLACK_BOT_TOKEN") or "").strip()
    return token or None


def _client() -> Any:
    token = _get_bot_token()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is not configured")
    if not SLACK_SDK_AVAILABLE or WebClient is None:
        raise RuntimeError("slack-sdk is not installed; install hermes-agent[slack]")
    return WebClient(token=token)


def _json_ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _json_error(message: str, **payload: Any) -> str:
    return json.dumps({"ok": False, "error": message, **payload}, ensure_ascii=False)


def _slack_error_details(exc: Exception) -> Tuple[str, Dict[str, Any]]:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            err = response.get("error") or str(exc)
            needed = response.get("needed")
            provided = response.get("provided")
            extra = {k: v for k, v in {"needed": needed, "provided": provided}.items() if v}
            return f"Slack API error: {err}", extra
        except Exception:
            pass
    return f"Slack API error: {exc}", {}


def _slack_error_message(exc: Exception) -> str:
    message, extra = _slack_error_details(exc)
    if extra:
        details = ", ".join(f"{key}: {value}" for key, value in extra.items())
        return f"{message} ({details})"
    return message


def _slack_error(exc: Exception) -> str:
    message, extra = _slack_error_details(exc)
    return _json_error(message, **extra)


def _normalize_channel(value: str) -> str:
    raw = (value or "").strip()
    mention = _CHANNEL_MENTION_RE.fullmatch(raw)
    if mention:
        return mention.group(1)
    return raw.lstrip("#").strip()


def _channel_type(ch: Dict[str, Any]) -> str:
    if ch.get("is_im"):
        return "dm"
    if ch.get("is_mpim"):
        return "group_dm"
    if ch.get("is_private") or ch.get("is_group"):
        return "private"
    return "public"


def _parse_since(value: Optional[str]) -> Optional[str]:
    """Parse user-friendly lower bounds into Slack oldest timestamps."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    lowered = raw.lower()
    now = datetime.now(timezone.utc)
    if lowered in {"today", "start of today"}:
        dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return str(dt.timestamp())
    if lowered == "yesterday":
        dt = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return str(dt.timestamp())

    rel = _RELATIVE_RE.fullmatch(raw)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower()
        delta = {
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
        }[unit]
        return str((now - delta).timestamp())

    # Slack timestamps and Unix seconds are accepted as-is.
    try:
        float(raw)
        return raw
    except ValueError:
        pass

    # YYYY-MM-DD is common in prompts; treat as UTC midnight.
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            dt = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return str(dt.timestamp())
    except ValueError:
        return None


def _ts_to_iso(ts: str) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _directory_channels() -> List[Dict[str, Any]]:
    try:
        from gateway.channel_directory import load_directory

        directory = load_directory()
        return list(directory.get("platforms", {}).get("slack", []) or [])
    except Exception:
        return []


def _directory_lookup(channel: str) -> Optional[str]:
    try:
        from gateway.channel_directory import resolve_channel_name

        return resolve_channel_name("slack", channel)
    except Exception:
        return None


def _list_joined_channels(client: Any, *, limit: int = 200) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return channels the bot user belongs to via users.conversations.

    We ask for private channels first because work bots often need them. If the
    app lacks ``groups:read``, Slack rejects the mixed request; retry public/DM
    discovery and return a warning so public channels still work.
    """
    warnings: List[str] = []

    def _fetch(types: str, remaining: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        channels: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while remaining > 0:
            page_limit = min(200, remaining)
            try:
                response = client.users_conversations(
                    types=types,
                    exclude_archived=True,
                    limit=page_limit,
                    cursor=cursor,
                )
            except Exception as exc:
                return channels, _slack_error_message(exc)

            if not response.get("ok", True):
                return channels, response.get("error", "users_conversations_failed")
            for ch in response.get("channels", []) or []:
                cid = ch.get("id")
                if not cid:
                    continue
                channels.append({
                    "id": cid,
                    "name": ch.get("name") or ch.get("user") or cid,
                    "type": _channel_type(ch),
                    "is_member": ch.get("is_member", True),
                    "num_members": ch.get("num_members"),
                })
                remaining -= 1
                if remaining <= 0:
                    break
            cursor = (response.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        return channels, None

    requested_limit = max(1, min(limit, 1000))
    channels, warning = _fetch("public_channel,private_channel,im,mpim", requested_limit)
    if warning and ("missing_scope" in warning or "groups:read" in warning):
        warnings.append(warning)
        public_channels, public_warning = _fetch("public_channel", requested_limit)
        if public_channels:
            channels = public_channels
        if public_warning:
            warnings.append(public_warning)
    elif warning:
        warnings.append(warning)

    return channels, "; ".join(warnings) if warnings else None


def _resolve_channel_id(client: Any, channel: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = _normalize_channel(channel)
    if not normalized:
        return None, "channel is required"
    if _CHANNEL_ID_RE.fullmatch(normalized):
        return normalized, None

    directory_id = _directory_lookup(normalized)
    if directory_id:
        return directory_id, None

    channels, warning = _list_joined_channels(client, limit=1000)
    for ch in channels:
        if str(ch.get("name", "")).lower() == normalized.lower():
            return str(ch["id"]), None

    suffix = f"; Slack warning: {warning}" if warning else ""
    return None, f"Slack channel not found or bot is not a member: {channel}{suffix}"


def _known_member_channel(channel_id: str) -> bool:
    for ch in _directory_channels():
        if ch.get("id") == channel_id:
            return True
    return False


def _ensure_bot_is_member(client: Any, channel_id: str) -> Optional[str]:
    if channel_id.startswith("D"):
        return None
    if _known_member_channel(channel_id):
        return None
    try:
        info = client.conversations_info(channel=channel_id)
        ch = info.get("channel", {}) if info else {}
        if ch and ch.get("is_member") is False:
            return "Bot is not a member of that channel. Invite it first, then retry."
    except Exception as exc:
        # Keep the real Slack failure visible. Most non-members/private channels
        # fail here before history can be read anyway.
        return json.loads(_slack_error(exc)).get("error", str(exc))
    return None


def _resolve_user_name(client: Any, user_id: str) -> str:
    if not user_id:
        return "unknown"
    if user_id in _USER_CACHE:
        return _USER_CACHE[user_id]
    try:
        response = client.users_info(user=user_id)
        user = response.get("user", {}) if response else {}
        profile = user.get("profile", {}) or {}
        name = profile.get("display_name") or profile.get("real_name") or user.get("name") or user_id
    except Exception:
        name = user_id
    _USER_CACHE[user_id] = name
    return name


def _message_text(msg: Dict[str, Any]) -> str:
    text = (msg.get("text") or "").strip()
    if text:
        return text
    # Lightweight fallback for messages with only files/attachments.
    files = msg.get("files") or []
    if files:
        titles = [f.get("title") or f.get("name") for f in files if f.get("title") or f.get("name")]
        if titles:
            return "[files] " + ", ".join(titles)
    return ""


def _message_item(client: Any, msg: Dict[str, Any], *, channel_id: str, channel_name: str = "") -> Dict[str, Any]:
    user_id = msg.get("user") or msg.get("bot_id") or ""
    username = msg.get("username") or (msg.get("bot_profile") or {}).get("name")
    user_name = username or _resolve_user_name(client, user_id)
    ts = str(msg.get("ts", ""))
    item = {
        "ts": ts,
        "time": _ts_to_iso(ts),
        "channel_id": channel_id,
        "channel_name": channel_name,
        "user_id": user_id,
        "user_name": user_name,
        "text": _message_text(msg),
    }
    if msg.get("thread_ts") and msg.get("thread_ts") != msg.get("ts"):
        item["thread_ts"] = msg.get("thread_ts")
    if msg.get("reply_count"):
        item["reply_count"] = msg.get("reply_count")
        item["thread_ts"] = msg.get("thread_ts") or msg.get("ts")
    return item


def _history(client: Any, channel_id: str, *, limit: int, since: Optional[str]) -> List[Dict[str, Any]]:
    oldest = _parse_since(since)
    remaining = max(1, min(int(limit or 50), _MAX_HISTORY_LIMIT))
    messages: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while remaining > 0:
        page_limit = min(100, remaining)
        kwargs = {"channel": channel_id, "limit": page_limit, "cursor": cursor}
        if oldest:
            kwargs["oldest"] = oldest
        response = client.conversations_history(**kwargs)
        page = response.get("messages", []) if response else []
        messages.extend(page)
        remaining -= len(page)
        cursor = (response.get("response_metadata") or {}).get("next_cursor") if response else None
        if not cursor or not page:
            break
    return messages


def _list_channels_action(limit: int = 200) -> str:
    client = _client()
    channels, warning = _list_joined_channels(client, limit=limit)
    if not channels and not warning:
        channels = _directory_channels()[: max(1, min(limit, 1000))]
    return _json_ok(
        channels=channels,
        count=len(channels),
        warning=warning if isinstance(warning, str) else None,
        scope="channels the bot user has joined or cached from gateway sessions",
    )


def _read_channel_action(channel: str, limit: int = 50, since: Optional[str] = None, include_threads: bool = False) -> str:
    client = _client()
    channel_id, error = _resolve_channel_id(client, channel)
    if error:
        return _json_error(error)
    assert channel_id is not None
    membership_error = _ensure_bot_is_member(client, channel_id)
    if membership_error:
        return _json_error(membership_error)

    try:
        messages = _history(client, channel_id, limit=limit, since=since)
        items = [_message_item(client, msg, channel_id=channel_id, channel_name=channel) for msg in messages]
        if include_threads:
            for item in items:
                if not item.get("reply_count") or not item.get("thread_ts"):
                    continue
                try:
                    replies = client.conversations_replies(channel=channel_id, ts=item["thread_ts"], limit=10)
                    item["thread_preview"] = [
                        _message_item(client, r, channel_id=channel_id, channel_name=channel)
                        for r in (replies.get("messages", []) or [])
                        if r.get("ts") != item.get("ts")
                    ][:9]
                except Exception as exc:
                    item["thread_error"] = json.loads(_slack_error(exc)).get("error")
        return _json_ok(channel_id=channel_id, messages=items, count=len(items), since=_parse_since(since))
    except Exception as exc:
        return _slack_error(exc)


def _read_thread_action(channel: str, thread_ts: str, limit: int = 50) -> str:
    client = _client()
    channel_id, error = _resolve_channel_id(client, channel)
    if error:
        return _json_error(error)
    assert channel_id is not None
    membership_error = _ensure_bot_is_member(client, channel_id)
    if membership_error:
        return _json_error(membership_error)
    if not thread_ts:
        return _json_error("thread_ts is required for read_thread")

    try:
        response = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=max(1, min(int(limit or 50), _MAX_HISTORY_LIMIT)),
        )
        messages = response.get("messages", []) if response else []
        return _json_ok(
            channel_id=channel_id,
            thread_ts=thread_ts,
            messages=[_message_item(client, msg, channel_id=channel_id, channel_name=channel) for msg in messages],
            count=len(messages),
        )
    except Exception as exc:
        return _slack_error(exc)


def _coerce_channels(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _search_messages_action(
    query: str,
    channels: Any = None,
    limit: int = 50,
    since: Optional[str] = None,
    max_channels: int = 10,
) -> str:
    client = _client()
    needle = (query or "").strip().lower()
    if not needle:
        return _json_error("query is required for search_messages")

    requested = _coerce_channels(channels)
    channel_ids: List[Tuple[str, str]] = []
    warnings: List[str] = []

    if requested:
        for ch in requested:
            cid, error = _resolve_channel_id(client, ch)
            if error:
                warnings.append(error)
                continue
            assert cid is not None
            membership_error = _ensure_bot_is_member(client, cid)
            if membership_error:
                warnings.append(f"{ch}: {membership_error}")
                continue
            channel_ids.append((cid, ch))
    else:
        joined, warning = _list_joined_channels(client, limit=max(1, min(max_channels or 10, _MAX_SEARCH_CHANNELS)))
        if warning:
            warnings.append(str(warning))
        if not joined and not warning:
            joined = _directory_channels()[: max(1, min(max_channels or 10, _MAX_SEARCH_CHANNELS))]
        channel_ids = [(str(ch["id"]), str(ch.get("name") or ch["id"])) for ch in joined if ch.get("id")]

    max_results = max(1, min(int(limit or 50), _MAX_HISTORY_LIMIT))
    matches: List[Dict[str, Any]] = []
    for cid, name in channel_ids[:_MAX_SEARCH_CHANNELS]:
        if len(matches) >= max_results:
            break
        try:
            for msg in _history(client, cid, limit=100, since=since):
                item = _message_item(client, msg, channel_id=cid, channel_name=name)
                if needle in item.get("text", "").lower():
                    matches.append(item)
                    if len(matches) >= max_results:
                        break
        except Exception as exc:
            warnings.append(f"{name}: {json.loads(_slack_error(exc)).get('error')}")

    return _json_ok(
        query=query,
        matches=matches,
        count=len(matches),
        scanned_channels=[{"id": cid, "name": name} for cid, name in channel_ids[:_MAX_SEARCH_CHANNELS]],
        since=_parse_since(since),
        warnings=warnings,
        scope="history scan over channels the bot user has joined; not Slack global search",
    )


def slack_handler(
    action: str,
    channel: str = "",
    channels: Any = None,
    thread_ts: str = "",
    query: str = "",
    since: Optional[str] = None,
    limit: int = 50,
    max_channels: int = 10,
    include_threads: bool = False,
) -> str:
    """Execute a Slack read/search action."""
    action = (action or "").strip()
    if action == "list_channels":
        return _list_channels_action(limit=limit or 200)
    if action == "read_channel":
        return _read_channel_action(channel=channel, limit=limit, since=since, include_threads=include_threads)
    if action == "read_thread":
        return _read_thread_action(channel=channel, thread_ts=thread_ts, limit=limit)
    if action == "search_messages":
        return _search_messages_action(query=query, channels=channels, limit=limit, since=since, max_channels=max_channels)
    return _json_error(
        "Unknown action. Use one of: list_channels, read_channel, read_thread, search_messages."
    )


_HANDLER_DEFAULTS = {
    "action": "",
    "channel": "",
    "channels": None,
    "thread_ts": "",
    "query": "",
    "since": None,
    "limit": 50,
    "max_channels": 10,
    "include_threads": False,
}


registry.register(
    name="slack",
    toolset="slack",
    schema=SLACK_SCHEMA,
    handler=lambda args, **kw: slack_handler(
        **{key: args.get(key, default) for key, default in _HANDLER_DEFAULTS.items()}
    ),
    check_fn=check_slack_tool_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="💬",
    max_result_size_chars=60000,
)
