"""Send Message Tool -- cross-channel messaging via platform APIs.

Sends a message to a user or channel on any connected messaging platform
(Telegram, Discord, Slack). Supports listing available targets and resolving
human-friendly channel names to IDs. Works in both CLI and gateway contexts.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


SEND_MESSAGE_SCHEMA = {
    "name": "send_message",
    "description": (
        "Send a message to a connected messaging platform, or list available targets.\n\n"
        "IMPORTANT: When the user asks to send to a specific channel or person "
        "(not just a bare platform name), call send_message(action='list') FIRST to see "
        "available targets, then send to the correct one.\n"
        "If the user just says a platform name like 'send to telegram', send directly "
        "to the home channel without listing first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "list"],
                "description": "Action to perform. 'send' (default) sends a message. 'list' returns all available channels/contacts across connected platforms.",
            },
            "target": {
                "type": "string",
                "description": "Delivery target. Format: 'platform' (uses home channel), 'platform:#channel-name', or 'platform:chat_id'. Examples: 'telegram', 'discord:#bot-home', 'slack:#engineering'",
            },
            "message": {"type": "string", "description": "The message text to send"},
        },
        "required": [],
    },
}


def send_message_tool(args, **kw):
    """Handle cross-channel send_message tool calls."""
    action = args.get("action", "send")

    if action == "list":
        return _handle_list()

    return _handle_send(args)


def _handle_list():
    """Return formatted list of available messaging targets."""
    try:
        from gateway.channel_directory import format_directory_for_display

        return json.dumps({"targets": format_directory_for_display()})
    except Exception as e:
        return json.dumps({"error": f"Failed to load channel directory: {e}"})


def _handle_send(args):
    """Send a message to a platform target."""
    target = args.get("target", "")
    message = args.get("message", "")
    if not target or not message:
        return json.dumps({"error": "Both 'target' and 'message' are required when action='send'"})

    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    chat_id = parts[1].strip() if len(parts) > 1 else None

    # Resolve human-friendly channel names to numeric IDs
    if chat_id and not chat_id.lstrip("-").isdigit():
        try:
            from gateway.channel_directory import resolve_channel_name

            resolved = resolve_channel_name(platform_name, chat_id)
            if resolved:
                chat_id = resolved
            else:
                return json.dumps(
                    {
                        "error": f"Could not resolve '{chat_id}' on {platform_name}. "
                        f"Use send_message(action='list') to see available targets."
                    }
                )
        except Exception:
            return json.dumps(
                {"error": f"Could not resolve '{chat_id}' on {platform_name}. Try using a numeric channel ID instead."}
            )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return json.dumps({"error": "Interrupted"})

    try:
        from gateway.config import Platform, load_gateway_config

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
        return json.dumps(
            {
                "error": f"Platform '{platform_name}' is not configured. Set up credentials in ~/.hermes/gateway.json or environment variables."
            }
        )

    used_home_channel = False
    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = home.chat_id
            used_home_channel = True
        else:
            return json.dumps(
                {
                    "error": f"No home channel set for {platform_name} to determine where to send the message. "
                    f"Either specify a channel directly with '{platform_name}:CHANNEL_NAME', "
                    f"or set a home channel via: hermes config set {platform_name.upper()}_HOME_CHANNEL <channel_id>"
                }
            )

    try:
        from model_tools import _run_async

        result = _run_async(_send_to_platform(platform, pconfig, chat_id, message))
        if used_home_channel and isinstance(result, dict) and result.get("success"):
            result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"

        # Mirror the sent message into the target's gateway session
        if isinstance(result, dict) and result.get("success"):
            try:
                from gateway.mirror import mirror_to_session

                source_label = os.getenv("HERMES_SESSION_PLATFORM", "cli")
                if mirror_to_session(platform_name, chat_id, message, source_label=source_label):
                    result["mirrored"] = True
            except Exception:
                pass

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
        chunks = [message[i : i + 2000] for i in range(0, len(message), 2000)]
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


def _check_send_message():
    """Gate send_message on gateway running (always available on messaging platforms)."""
    platform = os.getenv("HERMES_SESSION_PLATFORM", "")
    if platform and platform != "local":
        return True
    try:
        from gateway.status import is_gateway_running

        return is_gateway_running()
    except Exception:
        return False


# --- Registry ---
from tools.registry import registry

registry.register(
    name="send_message",
    toolset="messaging",
    schema=SEND_MESSAGE_SCHEMA,
    handler=send_message_tool,
    check_fn=_check_send_message,
)
