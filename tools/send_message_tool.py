"""Send Message Tool -- cross-channel messaging via platform APIs.

Sends a message to a user or channel on any connected messaging platform
(Telegram, Discord, Slack). Works in both CLI and gateway contexts.
"""

import json
import logging

logger = logging.getLogger(__name__)


SEND_MESSAGE_SCHEMA = {
    "name": "send_message",
    "description": "Send a message to a user or channel on any connected messaging platform. Use this when the user asks you to send something to a different platform, or when delivering notifications/alerts to a specific destination.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Delivery target. Format: 'platform' (uses home channel) or 'platform:chat_id' (specific chat). Examples: 'telegram', 'discord:123456789', 'slack:C01234ABCDE'"
            },
            "message": {
                "type": "string",
                "description": "The message text to send"
            }
        },
        "required": ["target", "message"]
    }
}


def send_message_tool(args, **kw):
    """Handle cross-channel send_message tool calls.

    Sends a message directly to the target platform using its API.
    Works in both CLI and gateway contexts -- does not require the
    gateway to be running. Loads credentials from the gateway config
    (env vars / ~/.hermes/gateway.json).
    """
    target = args.get("target", "")
    message = args.get("message", "")
    if not target or not message:
        return json.dumps({"error": "Both 'target' and 'message' are required"})

    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    chat_id = parts[1].strip() if len(parts) > 1 else None

    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
    except Exception as e:
        return json.dumps({"error": f"Failed to load gateway config: {e}"})

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
    }
    platform = platform_map.get(platform_name)
    if not platform:
        avail = ", ".join(platform_map.keys())
        return json.dumps({"error": f"Unknown platform: {platform_name}. Available: {avail}"})

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        return json.dumps({"error": f"Platform '{platform_name}' is not configured. Set up credentials in ~/.hermes/gateway.json or environment variables."})

    used_home_channel = False
    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = home.chat_id
            used_home_channel = True
        else:
            return json.dumps({"error": f"No home channel set for {platform_name} to determine where to send the message. Either specify a channel directly with '{platform_name}:CHANNEL_ID', or set a home channel via: hermes config set {platform_name.upper()}_HOME_CHANNEL <channel_id>"})

    try:
        from model_tools import _run_async
        result = _run_async(_send_to_platform(platform, pconfig, chat_id, message))
        if used_home_channel and isinstance(result, dict) and result.get("success"):
            result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Send failed: {e}"})


async def _send_to_platform(platform, pconfig, chat_id, message):
    """Route a message to the appropriate platform sender."""
    from gateway.config import Platform
    if platform == Platform.TELEGRAM:
        return await _send_telegram(pconfig.token, chat_id, message)
    elif platform == Platform.DISCORD:
        return await _send_discord(pconfig.token, chat_id, message)
    elif platform == Platform.SLACK:
        return await _send_slack(pconfig.token, chat_id, message)
    return {"error": f"Direct sending not yet implemented for {platform.value}"}


async def _send_telegram(token, chat_id, message):
    """Send via Telegram Bot API (one-shot, no polling needed)."""
    try:
        from telegram import Bot
        bot = Bot(token=token)
        msg = await bot.send_message(chat_id=int(chat_id), text=message)
        return {"success": True, "platform": "telegram", "chat_id": chat_id, "message_id": str(msg.message_id)}
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: pip install python-telegram-bot"}
    except Exception as e:
        return {"error": f"Telegram send failed: {e}"}


async def _send_discord(token, chat_id, message):
    """Send via Discord REST API (no websocket client needed)."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        url = f"https://discord.com/api/v10/channels/{chat_id}/messages"
        headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
        chunks = [message[i:i+2000] for i in range(0, len(message), 2000)]
        message_ids = []
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                async with session.post(url, headers=headers, json={"content": chunk}) as resp:
                    if resp.status not in (200, 201):
                        body = await resp.text()
                        return {"error": f"Discord API error ({resp.status}): {body}"}
                    data = await resp.json()
                    message_ids.append(data.get("id"))
        return {"success": True, "platform": "discord", "chat_id": chat_id, "message_ids": message_ids}
    except Exception as e:
        return {"error": f"Discord send failed: {e}"}


async def _send_slack(token, chat_id, message):
    """Send via Slack Web API."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"channel": chat_id, "text": message}) as resp:
                data = await resp.json()
                if data.get("ok"):
                    return {"success": True, "platform": "slack", "chat_id": chat_id, "message_id": data.get("ts")}
                return {"error": f"Slack API error: {data.get('error', 'unknown')}"}
    except Exception as e:
        return {"error": f"Slack send failed: {e}"}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="send_message",
    toolset="messaging",
    schema=SEND_MESSAGE_SCHEMA,
    handler=send_message_tool,
)
