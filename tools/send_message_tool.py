"""Send Message Tool -- cross-channel messaging via platform APIs.

Sends a message to a user or channel on any connected messaging platform
(Telegram, Discord, Slack). Supports listing available targets and resolving
human-friendly channel names to IDs. Works in both CLI and gateway contexts.
"""

from dataclasses import dataclass
import json
import logging
import os
import re
import ssl
import time
from urllib.parse import quote

from gateway.kasia_config import is_kasia_address_authorized

logger = logging.getLogger(__name__)

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a"}
_VOICE_EXTS = {".ogg", ".opus"}


class SendMessageError(RuntimeError):
    """Raised when send_message cannot resolve or deliver a target."""


@dataclass(slots=True)
class ParsedSendTarget:
    platform_name: str
    target_ref: str | None
    chat_id: str | None
    thread_id: str | None
    is_explicit: bool


@dataclass(slots=True)
class SendContext:
    platform_name: str
    platform: object
    pconfig: object
    chat_id: str
    thread_id: str | None
    cleaned_message: str
    media_files: list[tuple[str, bool]]
    mirror_text: str
    used_home_channel: bool
    action: str
    display_name: str | None
    retry: bool


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
                "enum": ["send", "list", "initiate"],
                "description": "Action to perform. 'send' (default) sends a message. 'list' returns all available channels/contacts across connected platforms. 'initiate' explicitly starts a new Kasia conversation with a peer."
            },
            "target": {
                "type": "string",
                "description": "Delivery target. Format: 'platform' (uses home channel), 'platform:#channel-name', 'platform:chat_id', or Telegram topic 'telegram:chat_id:thread_id'. Examples: 'telegram', 'telegram:-1001234567890:17585', 'discord:#bot-home', 'slack:#engineering', 'signal:+15551234567', 'kasia:kaspa:qp...'"
            },
            "message": {
                "type": "string",
                "description": "The message text to send"
            },
            "display_name": {
                "type": "string",
                "description": "Optional display label to attach when initiating a new Kasia conversation."
            },
            "retry": {
                "type": "boolean",
                "description": "When action='initiate', retry an existing pending Kasia handshake instead of returning the stored pending state."
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

    return _handle_send(args, action=action)


def _handle_list():
    """Return formatted list of available messaging targets."""
    try:
        from gateway.channel_directory import format_directory_for_display
        return json.dumps({"targets": format_directory_for_display()})
    except Exception as e:
        return json.dumps({"error": f"Failed to load channel directory: {e}"})


def _handle_send(args, action="send"):
    """Send a message to a platform target."""
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return json.dumps({"error": "Interrupted"})

    try:
        send_context = _build_send_context(args, action)
    except SendMessageError as error:
        return json.dumps({"error": str(error)})

    duplicate_skip = _maybe_skip_cron_duplicate_send(
        send_context.platform_name,
        send_context.chat_id,
        send_context.thread_id,
    )
    if duplicate_skip:
        return json.dumps(duplicate_skip)

    try:
        result = _dispatch_send(send_context)
        _annotate_send_result(result, send_context)
        _maybe_mirror_send_result(result, send_context)
        return json.dumps(result)
    except SendMessageError as error:
        return json.dumps({"error": str(error)})


def _build_send_context(args, action: str) -> SendContext:
    """Resolve tool args into a concrete platform send request."""
    target = _require_send_target(args)
    message = _require_send_message(args, action)
    parsed_target = _parse_send_target(target)
    gateway_config, platform, pconfig = _load_send_platform(parsed_target.platform_name)
    media_files, cleaned_message, mirror_text = _extract_send_message_parts(message)
    chat_id, thread_id, used_home_channel = _resolve_send_destination(
        parsed_target,
        platform,
        gateway_config,
    )
    return SendContext(
        platform_name=parsed_target.platform_name,
        platform=platform,
        pconfig=pconfig,
        chat_id=chat_id,
        thread_id=thread_id,
        cleaned_message=cleaned_message,
        media_files=media_files,
        mirror_text=mirror_text,
        used_home_channel=used_home_channel,
        action=action,
        display_name=args.get("display_name"),
        retry=bool(args.get("retry")),
    )


def _require_send_target(args) -> str:
    target = str(args.get("target", "")).strip()
    if not target:
        raise SendMessageError("A 'target' is required")
    return target


def _require_send_message(args, action: str) -> str:
    message = str(args.get("message", ""))
    if action == "send" and not message:
        raise SendMessageError(
            "Both 'target' and 'message' are required when action='send'"
        )
    return message


def _parse_send_target(target: str) -> ParsedSendTarget:
    platform_name, _, raw_target_ref = target.partition(":")
    normalized_platform = platform_name.strip().lower()
    normalized_target_ref = raw_target_ref.strip() or None
    chat_id = None
    thread_id = None
    is_explicit = False

    if normalized_target_ref:
        chat_id, thread_id, is_explicit = _parse_target_ref(
            normalized_platform,
            normalized_target_ref,
        )

    return ParsedSendTarget(
        platform_name=normalized_platform,
        target_ref=normalized_target_ref,
        chat_id=chat_id,
        thread_id=thread_id,
        is_explicit=is_explicit,
    )


def _load_send_platform(platform_name: str):
    try:
        from gateway.config import Platform, load_gateway_config
    except ImportError as error:
        raise SendMessageError(f"Failed to load gateway config: {error}") from error

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
        "signal": Platform.SIGNAL,
        "matrix": Platform.MATRIX,
        "mattermost": Platform.MATTERMOST,
        "homeassistant": Platform.HOMEASSISTANT,
        "dingtalk": Platform.DINGTALK,
        "kasia": Platform.KASIA,
        "email": Platform.EMAIL,
        "sms": Platform.SMS,
    }
    platform = platform_map.get(platform_name)
    if not platform:
        available_platforms = ", ".join(platform_map.keys())
        raise SendMessageError(
            f"Unknown platform: {platform_name}. Available: {available_platforms}"
        )

    try:
        gateway_config = load_gateway_config()
    except Exception as error:
        raise SendMessageError(f"Failed to load gateway config: {error}") from error

    platform_config = gateway_config.platforms.get(platform)
    if not platform_config or not platform_config.enabled:
        raise SendMessageError(
            f"Platform '{platform_name}' is not configured. "
            "Set up credentials in ~/.hermes/config.yaml or environment variables."
        )

    return gateway_config, platform, platform_config


def _extract_send_message_parts(message: str):
    from gateway.platforms.base import BasePlatformAdapter

    media_files, cleaned_message = BasePlatformAdapter.extract_media(message)
    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)
    return media_files, cleaned_message, mirror_text


def _resolve_send_destination(parsed_target: ParsedSendTarget, platform, gateway_config):
    chat_id = parsed_target.chat_id
    thread_id = parsed_target.thread_id

    if parsed_target.target_ref and not parsed_target.is_explicit:
        chat_id, thread_id = _resolve_named_target(
            parsed_target.platform_name,
            parsed_target.target_ref,
        )

    used_home_channel = False
    if not chat_id:
        home_channel = gateway_config.get_home_channel(platform)
        if not home_channel:
            raise SendMessageError(
                f"No home channel set for {parsed_target.platform_name} to determine where to send the message. "
                f"Either specify a channel directly with '{parsed_target.platform_name}:CHANNEL_NAME', "
                f"or set a home channel via: hermes config set {parsed_target.platform_name.upper()}_HOME_CHANNEL <channel_id>"
            )
        chat_id = home_channel.chat_id
        used_home_channel = True

    return str(chat_id), thread_id, used_home_channel


def _resolve_named_target(platform_name: str, target_ref: str):
    try:
        from gateway.channel_directory import resolve_channel_name
    except ImportError as error:
        raise SendMessageError(
            f"Could not resolve '{target_ref}' on {platform_name}. "
            "Channel directory support is unavailable."
        ) from error

    resolved_target = resolve_channel_name(platform_name, target_ref)
    if not resolved_target:
        raise SendMessageError(
            f"Could not resolve '{target_ref}' on {platform_name}. "
            "Use send_message(action='list') to see available targets."
        )

    resolved_chat_id, resolved_thread_id, _ = _parse_target_ref(
        platform_name,
        resolved_target,
    )
    return resolved_chat_id, resolved_thread_id


def _dispatch_send(send_context: SendContext) -> dict:
    from model_tools import _run_async

    dispatch_kwargs = _send_action_kwargs(send_context)
    try:
        return _run_async(
            _send_to_platform(
                send_context.platform,
                send_context.pconfig,
                send_context.chat_id,
                send_context.cleaned_message,
                thread_id=send_context.thread_id,
                media_files=send_context.media_files,
                **dispatch_kwargs,
            )
        )
    except Exception as error:
        raise SendMessageError(f"Send failed: {error}") from error


def _send_action_kwargs(send_context: SendContext) -> dict:
    if (
        send_context.action == "send"
        and not send_context.display_name
        and not send_context.retry
    ):
        return {}
    return {
        "action": send_context.action,
        "display_name": send_context.display_name,
        "retry": send_context.retry,
    }


def _annotate_send_result(result: dict, send_context: SendContext) -> None:
    if (
        send_context.used_home_channel
        and isinstance(result, dict)
        and result.get("success")
    ):
        result["note"] = (
            f"Sent to {send_context.platform_name} home channel "
            f"(chat_id: {send_context.chat_id})"
        )


def _maybe_mirror_send_result(result: dict, send_context: SendContext) -> None:
    if not (
        isinstance(result, dict)
        and result.get("success")
        and send_context.mirror_text
    ):
        return

    try:
        from gateway.mirror import mirror_to_session
    except ImportError as error:
        logger.debug("Mirror unavailable for %s: %s", send_context.platform_name, error)
        return

    source_label = os.getenv("HERMES_SESSION_PLATFORM", "cli")
    mirror_chat_id = str(result.get("chat_id") or send_context.chat_id)
    mirrored = mirror_to_session(
        send_context.platform_name,
        mirror_chat_id,
        send_context.mirror_text,
        source_label=source_label,
        thread_id=send_context.thread_id,
    )
    if mirrored:
        result["mirrored"] = True


def _parse_target_ref(platform_name: str, target_ref: str):
    """Parse a tool target into chat_id/thread_id and whether it is explicit."""
    if platform_name == "telegram":
        match = _TELEGRAM_TOPIC_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
    if platform_name == "kasia":
        lowered = target_ref.lower()
        if lowered.startswith(("kaspa:", "kaspatest:", "kaspasim:")):
            return target_ref, None, True
        if lowered.endswith(".kas"):
            return target_ref, None, True
        if lowered.startswith("broadcast:"):
            return f"broadcast:{target_ref.split(':', 1)[1]}", None, True
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


async def _send_to_platform(
    platform,
    pconfig,
    chat_id,
    message,
    thread_id=None,
    media_files=None,
    action="send",
    display_name=None,
    retry=False,
):
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

    if action != "send" and platform != Platform.KASIA:
        return {"error": f"Action '{action}' is only supported for Kasia right now"}

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
        elif platform == Platform.WHATSAPP:
            result = await _send_whatsapp(pconfig.extra, chat_id, chunk)
        elif platform == Platform.SIGNAL:
            result = await _send_signal(pconfig.extra, chat_id, chunk)
        elif platform == Platform.KASIA:
            if action == "send" and not display_name and not retry:
                result = await _send_kasia(pconfig.extra, chat_id, chunk)
            else:
                result = await _send_kasia(
                    pconfig.extra,
                    chat_id,
                    chunk,
                    action=action,
                    display_name=display_name,
                    retry=retry,
                )
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
    so that bold, links, and headers render correctly.  If the message
    already contains HTML tags, it is sent with ``parse_mode='HTML'``
    instead, bypassing MarkdownV2 conversion.
    """
    try:
        from telegram import Bot
        from telegram.constants import ParseMode

        # Auto-detect HTML tags — if present, skip MarkdownV2 and send as HTML.
        # Inspired by github.com/ashaney — PR #1568.
        _has_html = bool(re.search(r'<[a-zA-Z/][^>]*>', message))

        if _has_html:
            formatted = message
            send_parse_mode = ParseMode.HTML
        else:
            # Reuse the gateway adapter's format_message for markdown→MarkdownV2
            try:
                from gateway.platforms.telegram import TelegramAdapter, _escape_mdv2, _strip_mdv2
                _adapter = TelegramAdapter.__new__(TelegramAdapter)
                formatted = _adapter.format_message(message)
            except Exception:
                # Fallback: send as-is if formatting unavailable
                formatted = message
            send_parse_mode = ParseMode.MARKDOWN_V2

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
                    parse_mode=send_parse_mode, **thread_kwargs
                )
            except Exception as md_error:
                # Parse failed, fall back to plain text
                if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower() or "html" in str(md_error).lower():
                    logger.warning("Parse mode %s failed in _send_telegram, falling back to plain text: %s", send_parse_mode, md_error)
                    if not _has_html:
                        try:
                            from gateway.platforms.telegram import _strip_mdv2
                            plain = _strip_mdv2(formatted)
                        except Exception:
                            plain = message
                    else:
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


async def _send_whatsapp(extra, chat_id, message):
    """Send via the local WhatsApp bridge HTTP API."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        bridge_port = extra.get("bridge_port", 3000)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{bridge_port}/send",
                json={"chatId": chat_id, "message": message},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "success": True,
                        "platform": "whatsapp",
                        "chat_id": chat_id,
                        "message_id": data.get("messageId"),
                    }
                body = await resp.text()
                return {"error": f"WhatsApp bridge error ({resp.status}): {body}"}
    except Exception as e:
        return {"error": f"WhatsApp send failed: {e}"}


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


async def _send_kasia(extra, chat_id, message, action="send", display_name=None, retry=False):
    """Send via the local Kasia bridge HTTP API."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    try:
        bridge_port = _read_kasia_int_setting(extra, "bridge_port", 3010)
        send_wait_ms = _read_kasia_int_setting(extra, "send_wait_ms", 5000)
    except ValueError as error:
        return {"error": str(error)}

    try:
        async with aiohttp.ClientSession() as session:
            resolved_chat_id = chat_id
            should_resolve_target = action == "initiate" or _is_kasia_kns_target(chat_id)
            if should_resolve_target:
                resolved_target = await _resolve_kasia_target(
                    session,
                    bridge_port,
                    chat_id,
                    aiohttp,
                )
                if "error" in resolved_target:
                    return resolved_target
                resolved_chat_id = resolved_target.get("chatId") or chat_id
            if action == "initiate":
                if resolved_target.get("kind") != "dm":
                    return {
                        "error": "Handshake initiation is only supported for direct Kasia conversations"
                    }
                if not _is_kasia_target_authorized(resolved_chat_id):
                    return {
                        "error": f"Kasia initiation is not authorized for {resolved_chat_id}"
                    }

            endpoint, payload, timeout_total = _build_kasia_request(
                chat_id=resolved_chat_id if should_resolve_target else chat_id,
                message=message,
                action=action,
                display_name=display_name,
                retry=retry,
                send_wait_ms=send_wait_ms,
            )
            data = await _post_kasia_request(
                session,
                bridge_port,
                endpoint,
                payload,
                timeout_total,
                aiohttp,
            )
            if "error" in data:
                return data

            return _build_kasia_result(
                data,
                action=action,
                fallback_chat_id=resolved_chat_id if should_resolve_target else chat_id,
            )
    except Exception as e:
        return {"error": f"Kasia send failed: {e}"}


def _read_kasia_int_setting(extra, key: str, default: int) -> int:
    raw_value = extra.get(key, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Kasia {key}: {raw_value!r}") from error


def _is_kasia_kns_target(chat_id: str) -> bool:
    return str(chat_id or "").strip().lower().endswith(".kas")


async def _resolve_kasia_target(session, bridge_port: int, chat_id: str, aiohttp):
    async with session.get(
        f"http://127.0.0.1:{bridge_port}/resolve-target/{quote(str(chat_id), safe='')}",
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resolve_resp:
        if resolve_resp.status != 200:
            body = await resolve_resp.text()
            return {
                "error": f"Kasia target resolution failed ({resolve_resp.status}): {body}"
            }
        return await resolve_resp.json()


def _build_kasia_request(
    *,
    chat_id: str,
    message: str,
    action: str,
    display_name,
    retry: bool,
    send_wait_ms: int,
):
    if action == "initiate":
        return (
            "/handshakes/initiate",
            {
                "chatId": chat_id,
                "displayName": display_name,
                "retry": retry,
            },
            30,
        )
    if str(chat_id).startswith("broadcast:"):
        return (
            "/broadcasts/send",
            {
                "channelName": str(chat_id).split(":", 1)[1],
                "message": message,
                "waitMs": send_wait_ms,
            },
            max(10, int(send_wait_ms / 1000) + 10),
        )
    return (
        "/send",
        {
            "chatId": chat_id,
            "message": message,
            "waitMs": send_wait_ms,
        },
        max(10, int(send_wait_ms / 1000) + 10),
    )


async def _post_kasia_request(
    session,
    bridge_port: int,
    endpoint: str,
    payload: dict,
    timeout_total: int,
    aiohttp,
):
    async with session.post(
        f"http://127.0.0.1:{bridge_port}{endpoint}",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_total),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            return {"error": f"Kasia bridge error ({resp.status}): {body}"}
        return await resp.json()


def _build_kasia_result(data: dict, *, action: str, fallback_chat_id: str) -> dict:
    if data.get("status") in {"failed", "rejected"}:
        return {"error": data.get("error") or "Kasia send failed"}
    if action == "initiate":
        return {
            "success": True,
            "platform": "kasia",
            "chat_id": data.get("chatId") or fallback_chat_id,
            "message_id": data.get("txId"),
            "status": data.get("status"),
        }
    return {
        "success": True,
        "platform": "kasia",
        "chat_id": data.get("chatId") or fallback_chat_id,
        "message_id": data.get("jobId") or data.get("txId") or data.get("messageId"),
        "job_id": data.get("jobId"),
        "status": data.get("status"),
        "status_message": data.get("statusMessage"),
        "part_count": data.get("partCount"),
        "completed_parts": data.get("completedParts"),
        "indexed_parts": data.get("indexedParts"),
        "submitted_ms": data.get("submittedMs"),
        "indexed_ms": data.get("indexedMs"),
    }


def _is_kasia_target_authorized(address: str) -> bool:
    return is_kasia_address_authorized(address)


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


async def _send_sms(auth_token, chat_id, message):
    """Send a single SMS via Twilio REST API.

    Uses HTTP Basic auth (Account SID : Auth Token) and form-encoded POST.
    Chunking is handled by _send_to_platform() before this is called.
    """
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    import base64

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    from_number = os.getenv("TWILIO_PHONE_NUMBER", "")
    if not account_sid or not auth_token or not from_number:
        return {"error": "SMS not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER required)"}

    # Strip markdown — SMS renders it as literal characters
    message = re.sub(r"\*\*(.+?)\*\*", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"\*(.+?)\*", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"__(.+?)__", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"_(.+?)_", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"```[a-z]*\n?", "", message)
    message = re.sub(r"`(.+?)`", r"\1", message)
    message = re.sub(r"^#{1,6}\s+", "", message, flags=re.MULTILINE)
    message = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", message)
    message = re.sub(r"\n{3,}", "\n\n", message)
    message = message.strip()

    try:
        creds = f"{account_sid}:{auth_token}"
        encoded = base64.b64encode(creds.encode("ascii")).decode("ascii")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        headers = {"Authorization": f"Basic {encoded}"}

        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field("From", from_number)
            form_data.add_field("To", chat_id)
            form_data.add_field("Body", message)

            async with session.post(url, data=form_data, headers=headers) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    error_msg = body.get("message", str(body))
                    return {"error": f"Twilio API error ({resp.status}): {error_msg}"}
                msg_sid = body.get("sid", "")
                return {"success": True, "platform": "sms", "chat_id": chat_id, "message_id": msg_sid}
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
