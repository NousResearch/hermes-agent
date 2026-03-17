"""Send Message Tool -- cross-channel messaging via platform APIs.

Sends a message to a user or channel on any connected messaging platform
(Telegram, Discord, Slack). Supports listing available targets and resolving
human-friendly channel names to IDs. Works in both CLI and gateway contexts.
"""

import json
import logging
import os
import re
import ssl
import time

logger = logging.getLogger(__name__)

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a"}
_VOICE_EXTS = {".ogg", ".opus"}


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
                "description": "Action to perform. 'send' (default) sends a message. 'list' returns all available channels/contacts across connected platforms."
            },
            "target": {
                "type": "string",
                "description": "Delivery target. Format: 'platform' (uses home channel), 'platform:#channel-name', 'platform:chat_id', or Telegram topic 'telegram:chat_id:thread_id'. Examples: 'telegram', 'telegram:-1001234567890:17585', 'discord:#bot-home', 'slack:#engineering', 'signal:+15551234567'"
            },
            "message": {
                "type": "string",
                "description": "The message text to send"
            }
        },
        "required": []
    }
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
    target_ref = parts[1].strip() if len(parts) > 1 else None
    chat_id = None
    thread_id = None

    if target_ref:
        chat_id, thread_id, is_explicit = _parse_target_ref(platform_name, target_ref)
    else:
        is_explicit = False

    # Resolve human-friendly channel names to numeric IDs
    if target_ref and not is_explicit:
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(platform_name, target_ref)
            if resolved:
                chat_id, thread_id, _ = _parse_target_ref(platform_name, resolved)
            else:
                return json.dumps({
                    "error": f"Could not resolve '{target_ref}' on {platform_name}. "
                    f"Use send_message(action='list') to see available targets."
                })
        except Exception:
            return json.dumps({
                "error": f"Could not resolve '{target_ref}' on {platform_name}. "
                f"Try using a numeric channel ID instead."
            })

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return json.dumps({"error": "Interrupted"})

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
        "signal": Platform.SIGNAL,
        "email": Platform.EMAIL,
        "sms": Platform.SMS,
    }
    platform = platform_map.get(platform_name)
    if not platform:
        avail = ", ".join(platform_map.keys())
        return json.dumps({"error": f"Unknown platform: {platform_name}. Available: {avail}"})

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        return json.dumps({"error": f"Platform '{platform_name}' is not configured. Set up credentials in ~/.hermes/gateway.json or environment variables."})

    from gateway.platforms.base import BasePlatformAdapter

    media_files, cleaned_message = BasePlatformAdapter.extract_media(message)
    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)

    used_home_channel = False
    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = home.chat_id
            used_home_channel = True
        else:
            return json.dumps({
                "error": f"No home channel set for {platform_name} to determine where to send the message. "
                f"Either specify a channel directly with '{platform_name}:CHANNEL_NAME', "
                f"or set a home channel via: hermes config set {platform_name.upper()}_HOME_CHANNEL <channel_id>"
            })

    duplicate_skip = _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id)
    if duplicate_skip:
        return json.dumps(duplicate_skip)

    try:
        from model_tools import _run_async
        result = _run_async(
            _send_to_platform(
                platform,
                pconfig,
                chat_id,
                cleaned_message,
                thread_id=thread_id,
                media_files=media_files,
            )
        )
        if used_home_channel and isinstance(result, dict) and result.get("success"):
            result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"

        # Mirror the sent message into the target's gateway session
        if isinstance(result, dict) and result.get("success") and mirror_text:
            try:
                from gateway.mirror import mirror_to_session
                source_label = os.getenv("HERMES_SESSION_PLATFORM", "cli")
                if mirror_to_session(platform_name, chat_id, mirror_text, source_label=source_label, thread_id=thread_id):
                    result["mirrored"] = True
            except Exception:
                pass

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Send failed: {e}"})


def _parse_target_ref(platform_name: str, target_ref: str):
    """Parse a tool target into chat_id/thread_id and whether it is explicit."""
    if platform_name == "telegram":
        match = _TELEGRAM_TOPIC_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
    if target_ref.lstrip("-").isdigit():
        return target_ref, None, True
    return None, None, False


def _describe_media_for_mirror(media_files):
    """Return a human-readable mirror summary when a message only contains media."""
    if not media_files:
        return ""
    if len(media_files) == 1:
        media_path, is_voice = media_files[0]
        ext = os.path.splitext(media_path)[1].lower()
        if is_voice and ext in _VOICE_EXTS:
            return "[Sent voice message]"
        if ext in _IMAGE_EXTS:
            return "[Sent image attachment]"
        if ext in _VIDEO_EXTS:
            return "[Sent video attachment]"
        if ext in _AUDIO_EXTS:
            return "[Sent audio attachment]"
        return "[Sent document attachment]"
    return f"[Sent {len(media_files)} media attachments]"


def _get_cron_auto_delivery_target():
    """Return the cron scheduler's auto-delivery target for the current run, if any."""
    platform = os.getenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip().lower()
    chat_id = os.getenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
    if not platform or not chat_id:
        return None
    thread_id = os.getenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip() or None
    return {
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
    }


def _maybe_skip_cron_duplicate_send(platform_name: str, chat_id: str, thread_id: str | None):
    """Skip redundant cron send_message calls when the scheduler will auto-deliver there."""
    auto_target = _get_cron_auto_delivery_target()
    if not auto_target:
        return None

    same_target = (
        auto_target["platform"] == platform_name
        and str(auto_target["chat_id"]) == str(chat_id)
        and auto_target.get("thread_id") == thread_id
    )
    if not same_target:
        return None

    target_label = f"{platform_name}:{chat_id}"
    if thread_id is not None:
        target_label += f":{thread_id}"

    return {
        "success": True,
        "skipped": True,
        "reason": "cron_auto_delivery_duplicate_target",
        "target": target_label,
        "note": (
            f"Skipped send_message to {target_label}. This cron job will already auto-deliver "
            "its final response to that same target. Put the intended user-facing content in "
            "your final response instead, or use a different target if you want an additional message."
        ),
    }


async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None):
    """Route a message to the appropriate platform sender.

    Long messages are automatically chunked to fit within platform limits
    using the same smart-splitting algorithm as the gateway adapters
    (preserves code-block boundaries, adds part indicators).
    """
    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.telegram import TelegramAdapter
    from gateway.platforms.discord import DiscordAdapter
    from gateway.platforms.slack import SlackAdapter

    media_files = media_files or []

    # Platform message length limits (from adapter class attributes)
    _MAX_LENGTHS = {
        Platform.TELEGRAM: TelegramAdapter.MAX_MESSAGE_LENGTH,
        Platform.DISCORD: DiscordAdapter.MAX_MESSAGE_LENGTH,
        Platform.SLACK: SlackAdapter.MAX_MESSAGE_LENGTH,
    }

    # Smart-chunk the message to fit within platform limits.
    # For short messages or platforms without a known limit this is a no-op.
    max_len = _MAX_LENGTHS.get(platform)
    if max_len:
        chunks = BasePlatformAdapter.truncate_message(message, max_len)
    else:
        chunks = [message]

    # --- Telegram: special handling for media attachments ---
    if platform == Platform.TELEGRAM:
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _send_telegram(
                pconfig.token,
                chat_id,
                chunk,
                media_files=media_files if is_last else [],
                thread_id=thread_id,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Non-Telegram platforms ---
    if media_files and not message.strip():
        return {
            "error": (
                f"send_message MEDIA delivery is currently only supported for telegram; "
                f"target {platform.value} had only media attachments"
            )
        }
    warning = None
    if media_files:
        warning = (
            f"MEDIA attachments were omitted for {platform.value}; "
            "native send_message media delivery is currently only supported for telegram"
        )

    last_result = None
    for chunk in chunks:
        if platform == Platform.DISCORD:
            result = await _send_discord(pconfig.token, chat_id, chunk)
        elif platform == Platform.SLACK:
            result = await _send_slack(pconfig.token, chat_id, chunk)
        elif platform == Platform.SIGNAL:
            result = await _send_signal(pconfig.extra, chat_id, chunk)
        elif platform == Platform.EMAIL:
            result = await _send_email(pconfig.extra, chat_id, chunk)
        elif platform == Platform.SMS:
            result = await _send_sms(pconfig.api_key, chat_id, chunk)
        else:
            result = {"error": f"Direct sending not yet implemented for {platform.value}"}

        if isinstance(result, dict) and result.get("error"):
            return result
        last_result = result

    if warning and isinstance(last_result, dict) and last_result.get("success"):
        warnings = list(last_result.get("warnings", []))
        warnings.append(warning)
        last_result["warnings"] = warnings
    return last_result


async def _send_telegram(token, chat_id, message, media_files=None, thread_id=None):
    """Send via Telegram Bot API (one-shot, no polling needed).

    Applies markdown→MarkdownV2 formatting (same as the gateway adapter)
    so that bold, links, and headers render correctly.
    """
    try:
        from telegram import Bot
        from telegram.constants import ParseMode

        # Reuse the gateway adapter's format_message for markdown→MarkdownV2
        try:
            from gateway.platforms.telegram import TelegramAdapter, _escape_mdv2, _strip_mdv2
            _adapter = TelegramAdapter.__new__(TelegramAdapter)
            formatted = _adapter.format_message(message)
        except Exception:
            # Fallback: send as-is if formatting unavailable
            formatted = message

        bot = Bot(token=token)
        int_chat_id = int(chat_id)
        media_files = media_files or []
        thread_kwargs = {}
        if thread_id is not None:
            thread_kwargs["message_thread_id"] = int(thread_id)

        last_msg = None
        warnings = []

        if formatted.strip():
            try:
                last_msg = await bot.send_message(
                    chat_id=int_chat_id, text=formatted,
                    parse_mode=ParseMode.MARKDOWN_V2, **thread_kwargs
                )
            except Exception as md_error:
                # MarkdownV2 failed, fall back to plain text
                if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                    logger.warning("MarkdownV2 parse failed in _send_telegram, falling back to plain text: %s", md_error)
                    try:
                        from gateway.platforms.telegram import _strip_mdv2
                        plain = _strip_mdv2(formatted)
                    except Exception:
                        plain = message
                    last_msg = await bot.send_message(
                        chat_id=int_chat_id, text=plain,
                        parse_mode=None, **thread_kwargs
                    )
                else:
                    raise

        for media_path, is_voice in media_files:
            if not os.path.exists(media_path):
                warning = f"Media file not found, skipping: {media_path}"
                logger.warning(warning)
                warnings.append(warning)
                continue

            ext = os.path.splitext(media_path)[1].lower()
            try:
                with open(media_path, "rb") as f:
                    if ext in _IMAGE_EXTS:
                        last_msg = await bot.send_photo(
                            chat_id=int_chat_id, photo=f, **thread_kwargs
                        )
                    elif ext in _VIDEO_EXTS:
                        last_msg = await bot.send_video(
                            chat_id=int_chat_id, video=f, **thread_kwargs
                        )
                    elif ext in _VOICE_EXTS and is_voice:
                        last_msg = await bot.send_voice(
                            chat_id=int_chat_id, voice=f, **thread_kwargs
                        )
                    elif ext in _AUDIO_EXTS:
                        last_msg = await bot.send_audio(
                            chat_id=int_chat_id, audio=f, **thread_kwargs
                        )
                    else:
                        last_msg = await bot.send_document(
                            chat_id=int_chat_id, document=f, **thread_kwargs
                        )
            except Exception as e:
                warning = f"Failed to send media {media_path}: {e}"
                logger.error(warning)
                warnings.append(warning)

        if last_msg is None:
            error = "No deliverable text or media remained after processing MEDIA tags"
            if warnings:
                return {"error": error, "warnings": warnings}
            return {"error": error}

        result = {
            "success": True,
            "platform": "telegram",
            "chat_id": chat_id,
            "message_id": str(last_msg.message_id),
        }
        if warnings:
            result["warnings"] = warnings
        return result
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: pip install python-telegram-bot"}
    except Exception as e:
        return {"error": f"Telegram send failed: {e}"}


async def _send_discord(token, chat_id, message):
    """Send a single message via Discord REST API (no websocket client needed).

    Chunking is handled by _send_to_platform() before this is called.
    """
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        url = f"https://discord.com/api/v10/channels/{chat_id}/messages"
        headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"content": message}) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    return {"error": f"Discord API error ({resp.status}): {body}"}
                data = await resp.json()
        return {"success": True, "platform": "discord", "chat_id": chat_id, "message_id": data.get("id")}
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


async def _send_signal(extra, chat_id, message):
    """Send via signal-cli JSON-RPC API."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}
    try:
        http_url = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/")
        account = extra.get("account", "")
        if not account:
            return {"error": "Signal account not configured"}

        params = {"account": account, "message": message}
        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        payload = {
            "jsonrpc": "2.0",
            "method": "send",
            "params": params,
            "id": f"send_{int(time.time() * 1000)}",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{http_url}/api/v1/rpc", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return {"error": f"Signal RPC error: {data['error']}"}
            return {"success": True, "platform": "signal", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"Signal send failed: {e}"}


async def _send_email(extra, chat_id, message):
    """Send via SMTP (one-shot, no persistent connection needed)."""
    import smtplib
    from email.mime.text import MIMEText

    address = extra.get("address") or os.getenv("EMAIL_ADDRESS", "")
    password = os.getenv("EMAIL_PASSWORD", "")
    smtp_host = extra.get("smtp_host") or os.getenv("EMAIL_SMTP_HOST", "")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not all([address, password, smtp_host]):
        return {"error": "Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)"}

    try:
        msg = MIMEText(message, "plain", "utf-8")
        msg["From"] = address
        msg["To"] = chat_id
        msg["Subject"] = "Hermes Agent"

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls(context=ssl.create_default_context())
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"Email send failed: {e}"}


async def _send_sms(api_key, chat_id, message):
    """Send via Telnyx SMS REST API (one-shot, no persistent connection needed)."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        from_number = os.getenv("TELNYX_FROM_NUMBERS", "").split(",")[0].strip()
        if not from_number:
            return {"error": "TELNYX_FROM_NUMBERS not configured"}
        if not api_key:
            api_key = os.getenv("TELNYX_API_KEY", "")
        if not api_key:
            return {"error": "TELNYX_API_KEY not configured"}

        # Strip markdown for SMS
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", message, flags=re.DOTALL)
        text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"```[a-z]*\n?", "", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Chunk to 1600 chars
        chunks = [text[i:i+1600] for i in range(0, len(text), 1600)] if len(text) > 1600 else [text]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        message_ids = []
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                payload = {"from": from_number, "to": chat_id, "text": chunk}
                async with session.post(
                    "https://api.telnyx.com/v2/messages",
                    json=payload,
                    headers=headers,
                ) as resp:
                    body = await resp.json()
                    if resp.status >= 400:
                        return {"error": f"Telnyx API error ({resp.status}): {body}"}
                    message_ids.append(body.get("data", {}).get("id", ""))
        return {"success": True, "platform": "sms", "chat_id": chat_id, "message_ids": message_ids}
    except Exception as e:
        return {"error": f"SMS send failed: {e}"}


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
    emoji="📨",
)
