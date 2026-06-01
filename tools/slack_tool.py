"""Slack context tools for gateway-hosted Hermes agents.

These tools are read-only. They reuse the live Slack gateway client when the
agent is running inside Slack, and fall back to ``SLACK_BOT_TOKEN`` for CLI
debugging. They are intentionally scoped to channel/thread context because that
is the minimum surface Hermes needs to verify Slack reports before concluding.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _response_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return data
    try:
        return dict(response)
    except Exception:
        return {}


async def _call_slack(client: Any, method_name: str, **kwargs: Any) -> Dict[str, Any]:
    method = getattr(client, method_name)
    try:
        return _response_dict(await method(**kwargs))
    except Exception as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            return _response_dict(response)
        return {"ok": False, "error": str(exc) or exc.__class__.__name__}


def _get_slack_client() -> Optional[Any]:
    """Return an AsyncWebClient from the live gateway adapter or env token."""
    try:
        from gateway import runtime as gateway_runtime

        adapter = gateway_runtime.get_slack_adapter()
        app = getattr(adapter, "_app", None)
        client = getattr(app, "client", None)
        if client is not None:
            return client
    except Exception:
        logger.debug("Slack tool could not read live gateway adapter", exc_info=True)

    token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        return None

    try:
        from slack_sdk.web.async_client import AsyncWebClient

        return AsyncWebClient(token=token)
    except Exception:
        logger.debug("Slack SDK is unavailable for env-token fallback", exc_info=True)
        return None


def check_slack_tool_requirements() -> bool:
    return _get_slack_client() is not None


def _slack_error(response: Any, *, fallback: str = "slack_api_error") -> Dict[str, Any]:
    response_data = _response_dict(response)
    error = fallback
    if response_data:
        error = response_data.get("error") or fallback
    detail: Dict[str, Any] = {"ok": False, "error": error}
    for key in ("needed", "provided"):
        if response_data.get(key):
            detail[key] = response_data.get(key)
    if error == "missing_scope":
        detail["fix"] = (
            "Add the missing Slack OAuth scope, reinstall the Hermes Slack app, "
            "then restart the gateway."
        )
    return detail


def _format_ts(ts: Optional[str]) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(ts)


def _message_summary(message: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    text = str(message.get("text") or "")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return {
        "ts": message.get("ts"),
        "time": _format_ts(message.get("ts")),
        "user": message.get("user") or message.get("bot_id") or "unknown",
        "text": text,
        "thread_ts": message.get("thread_ts"),
        "reply_count": message.get("reply_count", 0),
    }


def _sort_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(message: Dict[str, Any]) -> float:
        try:
            return float(message.get("ts") or 0)
        except (TypeError, ValueError):
            return 0.0

    return sorted(messages, key=_key)


async def _resolve_channel_id(client: Any, channel: str) -> Dict[str, Any]:
    channel = (channel or "").strip()
    if not channel:
        return {"ok": False, "error": "channel is required"}
    if not channel.startswith("#"):
        return {"ok": True, "channel_id": channel}

    response = await _call_slack(
        client,
        "conversations_list",
        limit=200,
        types="public_channel,private_channel",
        exclude_archived=True,
    )
    if not response.get("ok"):
        return _slack_error(response)

    wanted = channel[1:]
    for item in response.get("channels", []):
        if item.get("name") == wanted:
            return {"ok": True, "channel_id": item.get("id"), "channel_name": wanted}
    return {"ok": False, "error": "channel_not_found", "channel": channel}


async def slack_list_channels(limit: int = 100, include_private: bool = True) -> Dict[str, Any]:
    """List Slack channels the bot can see."""
    client = _get_slack_client()
    if client is None:
        return {"ok": False, "error": "No Slack client available"}

    limit = max(1, min(int(limit or 100), 200))
    channel_types = "public_channel,private_channel" if include_private else "public_channel"
    response = await _call_slack(
        client,
        "conversations_list",
        limit=limit,
        types=channel_types,
        exclude_archived=True,
    )
    if not response.get("ok"):
        return _slack_error(response)

    channels = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "is_private": bool(item.get("is_private")),
            "is_member": bool(item.get("is_member")),
            "member_count": item.get("num_members", 0),
        }
        for item in response.get("channels", [])
    ]
    return {"ok": True, "count": len(channels), "channels": channels}


async def slack_get_messages(
    channel: str,
    limit: int = 30,
    max_chars_per_message: int = 500,
) -> Dict[str, Any]:
    """Fetch recent messages from a Slack channel by ID or ``#name``."""
    client = _get_slack_client()
    if client is None:
        return {"ok": False, "error": "No Slack client available"}

    resolved = await _resolve_channel_id(client, channel)
    if not resolved.get("ok"):
        return resolved

    limit = max(1, min(int(limit or 30), 100))
    max_chars = max(80, min(int(max_chars_per_message or 500), 2000))
    response = await _call_slack(
        client,
        "conversations_history",
        channel=resolved["channel_id"],
        limit=limit,
    )
    if not response.get("ok"):
        return _slack_error(response)

    messages = [
        _message_summary(item, max_chars)
        for item in _sort_messages(response.get("messages", []))
    ]
    return {
        "ok": True,
        "channel": resolved["channel_id"],
        "channel_name": resolved.get("channel_name"),
        "messages": messages,
    }


async def slack_get_thread(
    channel: str,
    thread_ts: str,
    limit: int = 50,
    max_chars_per_message: int = 1000,
) -> Dict[str, Any]:
    """Fetch replies for a Slack thread by channel ID/name and root ``thread_ts``."""
    client = _get_slack_client()
    if client is None:
        return {"ok": False, "error": "No Slack client available"}
    if not str(thread_ts or "").strip():
        return {"ok": False, "error": "thread_ts is required"}

    resolved = await _resolve_channel_id(client, channel)
    if not resolved.get("ok"):
        return resolved

    limit = max(1, min(int(limit or 50), 100))
    max_chars = max(80, min(int(max_chars_per_message or 1000), 4000))
    response = await _call_slack(
        client,
        "conversations_replies",
        channel=resolved["channel_id"],
        ts=str(thread_ts).strip(),
        limit=limit,
    )
    if not response.get("ok"):
        return _slack_error(response)

    messages = [
        _message_summary(item, max_chars)
        for item in _sort_messages(response.get("messages", []))
    ]
    return {
        "ok": True,
        "channel": resolved["channel_id"],
        "channel_name": resolved.get("channel_name"),
        "thread_ts": str(thread_ts).strip(),
        "messages": messages,
    }


async def slack_list_channels_tool(args: Dict[str, Any], **_kwargs: Any) -> str:
    return _json(
        await slack_list_channels(
            limit=args.get("limit", 100),
            include_private=args.get("include_private", True),
        )
    )


async def slack_get_messages_tool(args: Dict[str, Any], **_kwargs: Any) -> str:
    return _json(
        await slack_get_messages(
            channel=str(args.get("channel") or ""),
            limit=args.get("limit", 30),
            max_chars_per_message=args.get("max_chars_per_message", 500),
        )
    )


async def slack_get_thread_tool(args: Dict[str, Any], **_kwargs: Any) -> str:
    return _json(
        await slack_get_thread(
            channel=str(args.get("channel") or ""),
            thread_ts=str(args.get("thread_ts") or ""),
            limit=args.get("limit", 50),
            max_chars_per_message=args.get("max_chars_per_message", 1000),
        )
    )


SLACK_LIST_CHANNELS_SCHEMA = {
    "name": "slack_list_channels",
    "description": "List Slack channels the bot can see. Use this to resolve #channel names to channel IDs.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 200},
            "include_private": {"type": "boolean", "default": True},
        },
    },
}

SLACK_GET_MESSAGES_SCHEMA = {
    "name": "slack_get_messages",
    "description": "Fetch recent Slack channel messages by channel ID or #name before answering from Slack context.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Slack channel ID like C123 or a #channel-name."},
            "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 100},
            "max_chars_per_message": {"type": "integer", "default": 500, "minimum": 80, "maximum": 2000},
        },
        "required": ["channel"],
    },
}

SLACK_GET_THREAD_SCHEMA = {
    "name": "slack_get_thread",
    "description": "Fetch a Slack thread by channel ID/name and root thread_ts. Use this to verify the full thread before debugging.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Slack channel ID like C123 or a #channel-name."},
            "thread_ts": {"type": "string", "description": "Root Slack message ts/thread_ts, e.g. 1717000000.123456."},
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 100},
            "max_chars_per_message": {"type": "integer", "default": 1000, "minimum": 80, "maximum": 4000},
        },
        "required": ["channel", "thread_ts"],
    },
}

registry.register(
    name="slack_list_channels",
    toolset="slack",
    schema=SLACK_LIST_CHANNELS_SCHEMA,
    handler=slack_list_channels_tool,
    check_fn=check_slack_tool_requirements,
    is_async=True,
    emoji="💬",
    max_result_size_chars=60_000,
)

registry.register(
    name="slack_get_messages",
    toolset="slack",
    schema=SLACK_GET_MESSAGES_SCHEMA,
    handler=slack_get_messages_tool,
    check_fn=check_slack_tool_requirements,
    is_async=True,
    emoji="💬",
    max_result_size_chars=80_000,
)

registry.register(
    name="slack_get_thread",
    toolset="slack",
    schema=SLACK_GET_THREAD_SCHEMA,
    handler=slack_get_thread_tool,
    check_fn=check_slack_tool_requirements,
    is_async=True,
    emoji="💬",
    max_result_size_chars=120_000,
)
