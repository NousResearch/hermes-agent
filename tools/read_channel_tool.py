"""Read Channel Tool -- read message history from Discord channels/threads.

Fetches recent messages from a Discord channel or thread, returning them
as formatted text. Supports Discord channel IDs, URLs, and human-friendly
channel names via the channel directory.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Match Discord URLs: discord.com/channels/guild_id/channel_id[/message_id]
_DISCORD_URL_RE = re.compile(
    r"(?:https?://)?(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/"
    r"(\d+)/(\d+)(?:/(\d+))?"
)

READ_CHANNEL_SCHEMA = {
    "name": "read_channel",
    "description": (
        "Read recent messages from a Discord channel or thread.\n\n"
        "Accepts a Discord channel ID, thread ID, channel URL, or #channel-name.\n"
        "Returns formatted message history with author, timestamp, and content.\n"
        "Attachments are listed by filename and URL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Channel to read. Accepts:\n"
                    "- Channel/thread ID: '1234567890'\n"
                    "- Discord URL: 'https://discord.com/channels/guild/channel'\n"
                    "- Channel name: '#general' or 'general'\n"
                    "- 'discord:#channel-name'"
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Number of messages to fetch (1-50, default 20)",
                "default": 20,
            },
            "before": {
                "type": "string",
                "description": "Fetch messages before this message ID (for pagination)",
            },
            "around": {
                "type": "string",
                "description": "Fetch messages around this message ID",
            },
        },
        "required": ["target"],
    },
}


def read_channel_tool(args, **kw):
    """Handle read_channel tool calls."""
    target = args.get("target", "").strip()
    limit = min(max(int(args.get("limit", 20)), 1), 50)
    before = args.get("before")
    around = args.get("around")

    if not target:
        return json.dumps({"error": "target is required"})

    # Extract channel ID from Discord URL
    url_match = _DISCORD_URL_RE.search(target)
    if url_match:
        channel_id = url_match.group(2)
    elif target.lstrip("-").isdigit():
        channel_id = target
    else:
        # Try resolving channel name via directory
        clean_name = target.lstrip("#").strip()
        # Strip platform prefix if present
        if clean_name.startswith("discord:"):
            clean_name = clean_name[8:].lstrip("#").strip()
        channel_id = _resolve_channel_name(clean_name)
        if not channel_id:
            return json.dumps({
                "error": f"Could not resolve '{target}' to a channel ID. "
                "Use a numeric channel ID or Discord URL instead.",
            })

    # Get Discord bot token from gateway config
    token = _get_discord_token()
    if not token:
        return json.dumps({
            "error": "Discord not configured. No bot token found.",
        })

    try:
        from model_tools import _run_async
        result = _run_async(_fetch_messages(token, channel_id, limit, before, around))
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Failed to read channel: {e}"})


def _resolve_channel_name(name: str) -> str | None:
    """Resolve a human-friendly channel name to a Discord channel ID."""
    try:
        from gateway.channel_directory import resolve_channel_name
        resolved = resolve_channel_name("discord", name)
        if resolved and resolved.isdigit():
            return resolved
        # Try with # prefix
        resolved = resolve_channel_name("discord", f"#{name}")
        if resolved and resolved.isdigit():
            return resolved
    except Exception:
        pass

    # Fallback: scan channel_directory.json directly
    try:
        import pathlib
        dir_path = pathlib.Path.home() / ".hermes" / "channel_directory.json"
        if dir_path.exists():
            data = json.loads(dir_path.read_text())
            for entry in data.get("discord", []):
                entry_name = entry.get("name", "").lstrip("#").lower()
                if entry_name == name.lower():
                    return str(entry.get("id", ""))
    except Exception:
        pass

    return None


def _get_discord_token() -> str | None:
    """Get the Discord bot token from gateway config or environment."""
    # Try environment first
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if token:
        return token

    # Try gateway config
    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.DISCORD)
        if pconfig and pconfig.token:
            return pconfig.token
    except Exception:
        pass

    return None


async def _fetch_messages(token: str, channel_id: str, limit: int,
                          before: str | None, around: str | None) -> dict:
    """Fetch messages from Discord REST API and format them."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed"}

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    params = {"limit": str(limit)}
    if before:
        params["before"] = before
    if around:
        params["around"] = around

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            # First, get channel info for context
            channel_info = None
            try:
                async with session.get(
                    f"https://discord.com/api/v10/channels/{channel_id}",
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        channel_info = await resp.json()
            except Exception:
                pass

            # Fetch messages
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 403:
                    return {"error": f"No access to channel {channel_id}. Bot may not have permission."}
                if resp.status == 404:
                    return {"error": f"Channel {channel_id} not found."}
                if resp.status != 200:
                    body = await resp.text()
                    return {"error": f"Discord API error ({resp.status}): {body}"}

                messages = await resp.json()

    except Exception as e:
        return {"error": f"HTTP request failed: {e}"}

    if not messages:
        return {
            "channel_id": channel_id,
            "message_count": 0,
            "messages": "No messages found.",
        }

    # Format messages (oldest first)
    messages.reverse()
    formatted_lines = []

    channel_name = ""
    if channel_info:
        name = channel_info.get("name", "")
        ch_type = channel_info.get("type", 0)
        if ch_type == 11:  # Thread
            channel_name = f"Thread: {name}"
        else:
            channel_name = f"#{name}"
        topic = channel_info.get("topic")
        if topic:
            channel_name += f" — {topic}"

    if channel_name:
        formatted_lines.append(f"📍 {channel_name}")
        formatted_lines.append(f"Messages: {len(messages)} (most recent)")
        formatted_lines.append("─" * 40)

    for msg in messages:
        author = msg.get("author", {})
        username = author.get("global_name") or author.get("username", "Unknown")
        is_bot = author.get("bot", False)
        if is_bot:
            username = f"🤖 {username}"

        # Parse timestamp
        timestamp = msg.get("timestamp", "")
        if timestamp:
            # Simplify ISO timestamp to readable format
            timestamp = timestamp.replace("T", " ")[:19]

        content = msg.get("content", "")
        attachments = msg.get("attachments", [])
        embeds = msg.get("embeds", [])
        msg_id = msg.get("id", "")

        # Replace <@user_id> mentions with readable names
        for mention in msg.get("mentions", []):
            mention_id = mention.get("id", "")
            mention_name = mention.get("global_name") or mention.get("username", "")
            if mention_id and mention_name:
                content = content.replace(f"<@{mention_id}>", f"@{mention_name}")
                content = content.replace(f"<@!{mention_id}>", f"@{mention_name}")

        # Build message line
        line = f"[{timestamp}] {username}: {content}"

        # Add attachment info
        if attachments:
            att_info = []
            for att in attachments:
                fname = att.get("filename", "file")
                att_url = att.get("url", "")
                size = att.get("size", 0)
                size_str = f" ({size // 1024}KB)" if size else ""
                att_info.append(f"  📎 {fname}{size_str}: {att_url}")
            line += "\n" + "\n".join(att_info)

        # Add embed info (titles only)
        if embeds:
            for emb in embeds:
                title = emb.get("title", "")
                desc = emb.get("description", "")[:100] if emb.get("description") else ""
                if title:
                    line += f"\n  🔗 Embed: {title}"
                    if desc:
                        line += f" — {desc}"

        # Reference (reply-to)
        ref_msg = msg.get("referenced_message")
        if ref_msg:
            ref_author = ref_msg.get("author", {})
            ref_name = ref_author.get("global_name") or ref_author.get("username", "?")
            ref_content = (ref_msg.get("content", "") or "")[:80]
            line = f"  ↳ replying to {ref_name}: \"{ref_content}\"\n{line}"

        formatted_lines.append(line)

    return {
        "channel_id": channel_id,
        "message_count": len(messages),
        "messages": "\n\n".join(formatted_lines),
    }


def _check_read_channel():
    """Gate read_channel on Discord being available."""
    token = _get_discord_token()
    return bool(token)


# --- Registry ---
from tools.registry import registry

registry.register(
    name="read_channel",
    toolset="messaging",
    schema=READ_CHANNEL_SCHEMA,
    handler=read_channel_tool,
    check_fn=_check_read_channel,
    emoji="📖",
)
