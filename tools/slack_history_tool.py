"""Slack History Tool -- read conversation history and thread replies.

Allows the agent to fetch messages from Slack channels and threads using
the Slack Web API (conversations.history and conversations.replies).

Requires:
  - SLACK_BOT_TOKEN environment variable
  - Bot token scopes: channels:history, groups:history, im:history, mpim:history
  - The bot must be a member of the channel (invite it or use conversations.join)
"""

import json
import logging
import os
from datetime import datetime

from tools.registry import registry

logger = logging.getLogger(__name__)


def check_requirements() -> bool:
    """Check if Slack bot token is available."""
    return bool(os.getenv("SLACK_BOT_TOKEN"))


SLACK_HISTORY_SCHEMA = {
    "name": "slack_history",
    "description": (
        "Read Slack conversation history. Fetch messages from a channel or thread.\n\n"
        "Use this to read back through a Slack channel's recent messages or to fetch "
        "all replies in a specific thread. Requires the bot to be a member of the channel.\n\n"
        "Parameters:\n"
        "- channel: The channel ID (e.g., 'C081ZMSRBT8'). Required.\n"
        "- thread_ts: If provided, fetches replies in that thread instead of channel history.\n"
        "- limit: Number of messages to fetch (default: 50, max: 200).\n"
        "- latest: Only fetch messages before this Unix timestamp.\n"
        "- oldest: Only fetch messages after this Unix timestamp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Slack channel ID (e.g., 'C081ZMSRBT8'). Use the channel ID from the current conversation context."
            },
            "thread_ts": {
                "type": "string",
                "description": "Thread timestamp to fetch replies for. If omitted, fetches channel-level messages."
            },
            "limit": {
                "type": "integer",
                "description": "Number of messages to fetch (default: 50, max: 200)."
            },
            "latest": {
                "type": "string",
                "description": "Only fetch messages before this Unix timestamp."
            },
            "oldest": {
                "type": "string",
                "description": "Only fetch messages after this Unix timestamp."
            },
        },
        "required": ["channel"],
    },
}


async def _fetch_slack_history(args: dict) -> str:
    """Fetch Slack conversation history or thread replies."""
    channel = args.get("channel", "")
    thread_ts = args.get("thread_ts")
    limit = min(args.get("limit", 50), 200)
    latest = args.get("latest")
    oldest = args.get("oldest")

    if not channel:
        return json.dumps({"error": "channel is required"})

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return json.dumps({"error": "SLACK_BOT_TOKEN not set"})

    try:
        import aiohttp
    except ImportError:
        return json.dumps({"error": "aiohttp not installed. Run: pip install aiohttp"})

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    # Choose endpoint: conversations.replies for threads, conversations.history for channels
    if thread_ts:
        url = "https://slack.com/api/conversations.replies"
        params = {"channel": channel, "ts": thread_ts, "limit": limit}
    else:
        url = "https://slack.com/api/conversations.history"
        params = {"channel": channel, "limit": limit}

    if latest:
        params["latest"] = latest
    if oldest:
        params["oldest"] = oldest

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()

        if not data.get("ok"):
            error = data.get("error", "unknown")
            hints = {
                "not_in_channel": "The bot is not a member of this channel. Invite it first.",
                "channel_not_found": "Channel ID not found. Double-check the channel ID.",
                "missing_scope": "The bot token is missing required scopes (channels:history, groups:history).",
                "invalid_auth": "The SLACK_BOT_TOKEN is invalid or expired.",
            }
            hint = hints.get(error, "")
            return json.dumps({"error": f"Slack API error: {error}", "hint": hint})

        messages = data.get("messages", [])

        # Resolve user names for better readability
        user_cache = {}
        formatted = []
        for msg in messages:
            user_id = msg.get("user", "")
            user_display = user_cache.get(user_id)

            if user_id and not user_display:
                # Try to resolve user name
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            "https://slack.com/api/users.info",
                            headers=headers,
                            params={"user": user_id},
                        ) as user_resp:
                            user_data = await user_resp.json()
                            if user_data.get("ok"):
                                profile = user_data["user"].get("profile", {})
                                user_display = (
                                    profile.get("display_name")
                                    or profile.get("real_name")
                                    or user_data["user"].get("real_name")
                                    or user_data["user"].get("name")
                                    or user_id
                                )
                            else:
                                user_display = user_id
                except Exception:
                    user_display = user_id
                user_cache[user_id] = user_display

            # Format timestamp
            ts = msg.get("ts", "")
            try:
                dt = datetime.fromtimestamp(float(ts))
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError, OSError):
                time_str = ts

            text = msg.get("text", "")

            # Replace user mentions <@UXXXXX> with resolved names
            import re
            def _replace_mention(m):
                uid = m.group(1)
                return f"@{user_cache.get(uid, uid)}"
            text = re.sub(r"<@(U[A-Z0-9]+)>", _replace_mention, text)

            entry = {
                "user": user_display or msg.get("bot_id", "bot"),
                "text": text,
                "ts": ts,
                "time": time_str,
            }

            # Include thread info if present
            if msg.get("thread_ts") and msg.get("reply_count"):
                entry["thread_ts"] = msg["thread_ts"]
                entry["reply_count"] = msg["reply_count"]

            # Include reactions if present
            if msg.get("reactions"):
                entry["reactions"] = [
                    {"emoji": r["name"], "count": r["count"]}
                    for r in msg["reactions"]
                ]

            # Include file attachments info
            if msg.get("files"):
                entry["files"] = [
                    {"name": f.get("name", ""), "type": f.get("mimetype", "")}
                    for f in msg["files"]
                ]

            formatted.append(entry)

        # Reverse so oldest messages come first (Slack returns newest-first for history)
        if not thread_ts:
            formatted.reverse()

        result = {
            "channel": channel,
            "message_count": len(formatted),
            "messages": formatted,
        }

        if data.get("has_more"):
            result["has_more"] = True
            result["hint"] = "More messages available. Use 'oldest' or 'latest' params to paginate."

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Failed to fetch Slack history: {e}"})


def slack_history_tool(args, **kw):
    """Synchronous wrapper — dispatched by the registry."""
    from model_tools import _run_async
    return _run_async(_fetch_slack_history(args))


# Register with the tool registry
registry.register(
    name="slack_history",
    toolset="slack_history",
    schema=SLACK_HISTORY_SCHEMA,
    handler=lambda args, **kw: slack_history_tool(args, **kw),
    check_fn=check_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    description="Read Slack conversation history and thread replies",
    emoji="💬",
)
