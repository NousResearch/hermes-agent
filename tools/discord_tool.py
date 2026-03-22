"""Discord Tool -- fetch messages and channel history from Discord.

Allows the agent to retrieve a Discord message by ID, fetch recent messages
from a channel, and look up basic channel/server metadata. Requires the bot
to have Read Message History permission in the target channel.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"


DISCORD_FETCH_MESSAGE_SCHEMA = {
    "name": "discord_fetch_message",
    "description": (
        "Fetch a Discord message by ID, or retrieve recent messages from a channel.\n\n"
        "Use this when the user pastes a Discord message link "
        "(e.g. https://discord.com/channels/GUILD/CHANNEL/MESSAGE) "
        "or asks you to read a specific Discord message. "
        "The bot must be in the server and have Read Message History permission.\n\n"
        "Actions:\n"
        "- 'fetch': retrieve a single message by message_id + channel_id\n"
        "- 'history': retrieve recent messages from a channel (up to 100)\n"
        "- 'parse_link': extract channel_id and message_id from a discord.com URL"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["fetch", "history", "parse_link"],
                "description": (
                    "Action to perform: "
                    "'fetch' retrieves a single message, "
                    "'history' retrieves recent messages from a channel, "
                    "'parse_link' extracts IDs from a discord.com URL."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": "Discord channel ID (required for fetch and history).",
            },
            "message_id": {
                "type": "string",
                "description": "Discord message ID (required for fetch).",
            },
            "limit": {
                "type": "integer",
                "description": "Number of messages to retrieve for history (1-100, default 20).",
            },
            "before": {
                "type": "string",
                "description": "Return messages before this message ID (for history pagination).",
            },
            "url": {
                "type": "string",
                "description": "Full discord.com message URL to parse (for parse_link action).",
            },
        },
        "required": ["action"],
    },
}


def discord_fetch_message_tool(args, **kw):
    """Handle discord_fetch_message tool calls."""
    action = args.get("action", "fetch")

    if action == "parse_link":
        return _handle_parse_link(args)

    if action == "history":
        return _handle_history(args)

    return _handle_fetch(args)


def _handle_parse_link(args):
    """Extract guild_id, channel_id, message_id from a discord.com URL."""
    import re

    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "url is required for parse_link action"})

    # https://discord.com/channels/{guild_id}/{channel_id}/{message_id}
    match = re.search(
        r"discord\.com/channels/(\d+)/(\d+)(?:/(\d+))?", url
    )
    if not match:
        return json.dumps({
            "error": f"Could not parse Discord URL: {url}. "
            "Expected format: https://discord.com/channels/GUILD_ID/CHANNEL_ID/MESSAGE_ID"
        })

    result = {
        "guild_id": match.group(1),
        "channel_id": match.group(2),
    }
    if match.group(3):
        result["message_id"] = match.group(3)

    return json.dumps(result)


def _handle_fetch(args):
    """Fetch a single Discord message by channel_id + message_id."""
    channel_id = args.get("channel_id", "").strip()
    message_id = args.get("message_id", "").strip()

    if not channel_id or not message_id:
        return json.dumps({
            "error": "Both channel_id and message_id are required for fetch action. "
            "If you have a discord.com URL, use action='parse_link' first."
        })

    token = _get_discord_token()
    if not token:
        return json.dumps({
            "error": "Discord bot token not configured. "
            "Set DISCORD_BOT_TOKEN in your environment or ~/.hermes/config.yaml."
        })

    try:
        from model_tools import _run_async
        return _run_async(_fetch_message(token, channel_id, message_id))
    except Exception as e:
        return json.dumps({"error": f"discord_fetch_message failed: {e}"})


def _handle_history(args):
    """Fetch recent messages from a Discord channel."""
    channel_id = args.get("channel_id", "").strip()
    if not channel_id:
        return json.dumps({"error": "channel_id is required for history action"})

    limit = int(args.get("limit", 20))
    limit = max(1, min(100, limit))
    before = args.get("before", "").strip() or None

    token = _get_discord_token()
    if not token:
        return json.dumps({
            "error": "Discord bot token not configured. "
            "Set DISCORD_BOT_TOKEN in your environment or ~/.hermes/config.yaml."
        })

    try:
        from model_tools import _run_async
        return _run_async(_fetch_history(token, channel_id, limit, before))
    except Exception as e:
        return json.dumps({"error": f"discord_fetch_message history failed: {e}"})


def _get_discord_token():
    """Resolve the Discord bot token from env or gateway config."""
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if token:
        return token
    # Try to load from gateway config (same source used by send_message_tool)
    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.DISCORD)
        if pconfig and pconfig.enabled and pconfig.token:
            return pconfig.token
    except Exception:
        pass
    return None


async def _fetch_message(token, channel_id, message_id):
    """Retrieve a single message via Discord REST API."""
    try:
        import aiohttp
    except ImportError:
        return json.dumps({"error": "aiohttp not installed. Run: pip install aiohttp"})

    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages/{message_id}"
    headers = {"Authorization": f"Bot {token}"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    return json.dumps({
                        "error": f"Message {message_id} not found in channel {channel_id}. "
                        "It may have been deleted, or the bot may not have access."
                    })
                if resp.status == 403:
                    return json.dumps({
                        "error": f"Bot lacks permission to read channel {channel_id}. "
                        "Ensure the bot has 'Read Message History' permission."
                    })
                if resp.status != 200:
                    body = await resp.text()
                    return json.dumps({
                        "error": f"Discord API error ({resp.status}): {body}"
                    })
                data = await resp.json()

        return json.dumps(_format_message(data))
    except Exception as e:
        return json.dumps({"error": f"Discord fetch failed: {e}"})


async def _fetch_history(token, channel_id, limit, before):
    """Retrieve recent messages from a channel via Discord REST API."""
    try:
        import aiohttp
    except ImportError:
        return json.dumps({"error": "aiohttp not installed. Run: pip install aiohttp"})

    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {token}"}
    params = {"limit": limit}
    if before:
        params["before"] = before

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 403:
                    return json.dumps({
                        "error": f"Bot lacks permission to read channel {channel_id}. "
                        "Ensure the bot has 'Read Message History' permission."
                    })
                if resp.status != 200:
                    body = await resp.text()
                    return json.dumps({
                        "error": f"Discord API error ({resp.status}): {body}"
                    })
                data = await resp.json()

        messages = [_format_message(m) for m in data]
        return json.dumps({
            "channel_id": channel_id,
            "count": len(messages),
            "messages": messages,
        })
    except Exception as e:
        return json.dumps({"error": f"Discord history fetch failed: {e}"})


def _format_message(data):
    """Normalize a raw Discord message object into a clean dict."""
    author = data.get("author", {})
    attachments = [
        {
            "id": a.get("id"),
            "filename": a.get("filename"),
            "url": a.get("url"),
            "content_type": a.get("content_type"),
            "size": a.get("size"),
        }
        for a in data.get("attachments", [])
    ]
    embeds = [
        {
            "title": e.get("title"),
            "description": e.get("description"),
            "url": e.get("url"),
        }
        for e in data.get("embeds", [])
        if e.get("title") or e.get("description")
    ]
    result = {
        "message_id": data.get("id"),
        "channel_id": data.get("channel_id"),
        "content": data.get("content", ""),
        "author": {
            "id": author.get("id"),
            "username": author.get("username"),
            "display_name": author.get("global_name") or author.get("username"),
            "bot": author.get("bot", False),
        },
        "timestamp": data.get("timestamp"),
        "edited_timestamp": data.get("edited_timestamp"),
        "type": data.get("type", 0),
    }
    if attachments:
        result["attachments"] = attachments
    if embeds:
        result["embeds"] = embeds
    referenced = data.get("referenced_message")
    if referenced:
        ref_author = referenced.get("author", {})
        result["reply_to"] = {
            "message_id": referenced.get("id"),
            "author": ref_author.get("username"),
            "content": referenced.get("content", "")[:200],
        }
    return result


def _check_discord_fetch_message():
    """Gate the tool on Discord being configured."""
    if _get_discord_token():
        return True
    return False


# --- Registry ---
from tools.registry import registry

registry.register(
    name="discord_fetch_message",
    toolset="messaging",
    schema=DISCORD_FETCH_MESSAGE_SCHEMA,
    handler=discord_fetch_message_tool,
    check_fn=_check_discord_fetch_message,
    emoji="💬",
)
