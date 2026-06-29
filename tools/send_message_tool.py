"""Send Message Tool -- cross-channel messaging via platform APIs.

Sends a message to a user or channel on any connected messaging platform
(Telegram, Discord, Slack). Supports listing available targets and resolving
human-friendly channel names to IDs. Works in both CLI and gateway contexts.
"""

import asyncio
import json
import logging
import os
import re
import ssl
import time
from email.utils import formatdate

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_FEISHU_TARGET_RE = re.compile(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$")
# Slack conversation IDs: C (public channel), G (private/group channel), D (DM).
# Must be uppercase alphanumeric, 9+ chars. User IDs (U...) and workspace IDs
# (W...) are NOT valid chat.postMessage channel values — posting to them fails
# because the API requires a conversation ID. To DM a user you must first call
# conversations.open to obtain a D... ID. Without this gate, Slack IDs fall
# through to channel-name resolution, which only matches by name and fails.
_SLACK_TARGET_RE = re.compile(r"^\s*([CGDU][A-Z0-9]{8,})\s*$")
# Session-derived Slack thread targets use "<conversation_id>:<thread_ts>".
_SLACK_THREAD_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,}):([^\s:]+)\s*$")
_WEIXIN_TARGET_RE = re.compile(r"^\s*((?:wxid|gh|v\d+|wm|wb)_[A-Za-z0-9_-]+|[A-Za-z0-9._-]+@chatroom|filehelper)\s*$")
_YUANBAO_TARGET_RE = re.compile(r"^\s*((?:group|direct):[^:]+)\s*$")
# Discord snowflake IDs are numeric, same regex pattern as Telegram topic targets.
_NUMERIC_TOPIC_RE = _TELEGRAM_TOPIC_TARGET_RE
# Platforms that address recipients by phone number and accept E.164 format
# (with a leading '+'). Without this, "+15551234567" fails the isdigit() check
# below and falls through to channel-name resolution, which has no way to
# resolve a raw phone number. Keeping the '+' preserves the E.164 form that
# downstream adapters (signal, etc.) expect.
_PHONE_PLATFORMS = frozenset({"photon", "signal", "sms", "whatsapp"})
_E164_TARGET_RE = re.compile(r"^\s*\+(\d{7,15})\s*$")
# WhatsApp JIDs: group chats (<digits>@g.us), individual users
# (<phone>@s.whatsapp.net), linked identities (<id>@lid), and broadcast /
# newsletter chats. These are explicit native targets the bridge accepts
# verbatim — they must NOT fall through to home-channel resolution.
_WHATSAPP_JID_RE = re.compile(
    r"^\s*[\w-]+@(?:g\.us|s\.whatsapp\.net|lid|broadcast|newsletter)\s*$",
    re.IGNORECASE,
)
# Email addresses — a valid email like "user@domain.com" should be treated as
# an explicit target for the email platform, not fall through to channel-name
# resolution which has no way to resolve a raw address.
_EMAIL_TARGET_RE = re.compile(r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$")
# Most platforms read their home channel from "<PLATFORM>_HOME_CHANNEL", but a
# few diverge. Email reads EMAIL_HOME_ADDRESS (see gateway/config.py), so the
# generic "<PLATFORM>_HOME_CHANNEL" hint would point users at a variable that is
# never read. Map the exceptions so the error guidance is actually actionable.
_HOME_CHANNEL_ENV_OVERRIDES = {"email": "EMAIL_HOME_ADDRESS"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}
_VOICE_EXTS = {".ogg", ".opus"}
# Telegram's Bot API sendAudio only accepts MP3 / M4A. Other audio
# formats either route through sendVoice (Opus/OGG) or fall back to
# document delivery.
_TELEGRAM_SEND_AUDIO_EXTS = {".mp3", ".m4a"}
_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\s]+)",
    re.IGNORECASE,
)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\b(access_token|api[_-]?key|auth[_-]?token|signature|sig)\s*=\s*([^\s,;]+)",
    re.IGNORECASE,
)


def _sanitize_error_text(text) -> str:
    """Redact secrets from error text before surfacing it to users/models."""
    redacted = redact_sensitive_text(text)
    redacted = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", redacted)
    redacted = _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", redacted)
    return redacted


def _error(message: str) -> dict:
    """Build a standardized error payload with redacted content."""
    return {"error": _sanitize_error_text(message)}


def _display_chat_id(platform_name: str, chat_id: str) -> str:
    """Return a result-safe chat identifier for tool transcripts/log consumers."""
    if platform_name == "signal" and str(chat_id).startswith("group:"):
        return "group:***"
    return chat_id


def _telegram_retry_delay(exc: Exception, attempt: int) -> float | None:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return 1.0

    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return None
    if (
        "bad gateway" in text
        or "502" in text
        or "too many requests" in text
        or "429" in text
        or "service unavailable" in text
        or "503" in text
        or "gateway timeout" in text
        or "504" in text
    ):
        return float(2 ** attempt)
    return None


async def _send_telegram_message_with_retry(bot, *, attempts: int = 3, **kwargs):
    for attempt in range(attempts):
        try:
            return await bot.send_message(**kwargs)
        except Exception as exc:
            delay = _telegram_retry_delay(exc, attempt)
            if delay is None or attempt >= attempts - 1:
                raise
            logger.warning(
                "Transient Telegram send failure (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                attempts,
                delay,
                _sanitize_error_text(exc),
            )
            await asyncio.sleep(delay)


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
                "enum": ["send", "list", "react", "unreact"],
                "description": "Action to perform. 'send' (default) sends a message. 'list' returns all available channels/contacts across connected platforms. 'react' attaches an emoji reaction to a message (platforms that support it, e.g. photon/iMessage tapbacks). 'unreact' retracts a previously-added reaction."
            },
            "target": {
                "type": "string",
                "description": "Delivery target. Format: 'platform' (uses home channel), 'platform:#channel-name', 'platform:chat_id', or 'platform:chat_id:thread_id' for Telegram topics and Discord threads. Examples: 'telegram', 'telegram:-1001234567890:17585', 'discord:999888777:555444333', 'discord:#bot-home', 'slack:#engineering', 'signal:+155****4567', 'matrix:!roomid:server.org', 'matrix:@user:server.org', 'ntfy:alerts-channel' (explicit ntfy topic), 'yuanbao:direct:<account_id>' (DM), 'yuanbao:group:<group_code>' (group chat)"
            },
            "message": {
                "type": "string",
                "description": "The message text to send. To send an image or file, include MEDIA:<local_path> (e.g. 'MEDIA:/tmp/report.pdf') in the message — the platform will deliver it as a native media attachment."
            },
            "emoji": {
                "type": "string",
                "description": "For action='react': the emoji to react with (e.g. '❤️'). On iMessage, ❤️👍👎😂‼️❓ render as native tapbacks; other emoji use custom-emoji reactions."
            },
            "message_id": {
                "type": "string",
                "description": "For action='react'/'unreact': id of the message to react to. Omit to target the most recent message received in that chat (usually the one being replied to)."
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

    if action == "react":
        return _handle_react(args)

    if action == "unreact":
        return _handle_react(args, remove=True)

    return _handle_send(args)


def _handle_list():
    """Return formatted list of available messaging targets."""
    try:
        from gateway.channel_directory import format_directory_for_display
        return json.dumps({"targets": format_directory_for_display()})
    except Exception as e:
        return json.dumps(_error(f"Failed to load channel directory: {e}"))


def _handle_react(args, remove=False):
    """Attach (or with ``remove=True`` retract) an emoji reaction on a message
    via a live gateway adapter.

    Only adapters that expose ``add_reaction(chat_id, emoji, message_id)`` /
    ``remove_reaction(chat_id, message_id)`` coroutines support this (e.g.
    photon/iMessage tapbacks). Requires the gateway to be running in this
    process — there is no standalone fallback, since reacting needs the
    adapter's live message-id state.
    """
    target = args.get("target", "")
    emoji = (args.get("emoji") or "").strip()
    message_id = (args.get("message_id") or "").strip() or None
    if not target or (not remove and not emoji):
        return tool_error(
            "Both 'target' and 'emoji' are required when action='react'"
            if not remove
            else "'target' is required when action='unreact'"
        )

    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    target_ref = parts[1].strip() if len(parts) > 1 else None
    chat_id = None
    if target_ref:
        chat_id, _thread_id, _ = _parse_target_ref(platform_name, target_ref)
        if not chat_id:
            try:
                from gateway.channel_directory import resolve_channel_name
                resolved = resolve_channel_name(platform_name, target_ref)
            except Exception:
                resolved = None
            # Opaque platform-native ids (e.g. photon space GUIDs like
            # 'any;-;+1555...') match no parser pattern and no directory
            # entry — pass them through verbatim; the adapter validates.
            chat_id = resolved or target_ref

    try:
        from gateway.config import Platform, load_gateway_config
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return tool_error(f"Unknown platform: {platform_name}")

    if not chat_id:
        try:
            config = load_gateway_config()
            home = config.get_home_channel(platform)
        except Exception:
            home = None
        if not home:
            return tool_error(
                f"No chat specified and no home channel set for {platform_name}. "
                f"Use '{platform_name}:chat_id'."
            )
        chat_id = home.chat_id

    runner = None
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    adapter = runner.adapters.get(platform) if runner is not None else None
    if adapter is None:
        return tool_error(
            f"Reactions require a live {platform_name} adapter in the running "
            "gateway (not available from cron/standalone contexts)."
        )
    fn_name = "remove_reaction" if remove else "add_reaction"
    react_fn = getattr(adapter, fn_name, None)
    if not callable(react_fn):
        return tool_error(
            f"Platform '{platform_name}' does not support message reactions."
        )

    try:
        from model_tools import _run_async
        if remove:
            result = _run_async(
                react_fn(chat_id=chat_id, message_id=message_id)
            )
        else:
            result = _run_async(
                react_fn(chat_id=chat_id, emoji=emoji, message_id=message_id)
            )
    except Exception as e:
        return json.dumps(_error(f"Reaction failed: {e}"))
    if isinstance(result, dict):
        return json.dumps(result)
    return json.dumps({"success": bool(result)})


def _handle_send(args):
    """Send a message to a platform target."""
    target = args.get("target", "")
    message = args.get("message", "")
    if not target or not message:
        return tool_error("Both 'target' and 'message' are required when action='send'")

    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    target_ref = parts[1].strip() if len(parts) > 1 else None
    chat_id = None
    thread_id = None

    if target_ref:
        chat_id, thread_id, is_explicit = _parse_target_ref(platform_name, target_ref)
    else:
        is_explicit = False

    # Discord threads are channels with their own snowflake IDs, but Hermes
    # lane checks and mirrors need the parent lane plus thread id. If a user or
    # model provides a bare known thread id, normalize it through the channel
    # directory before treating it as a direct channel send.
    if platform_name == "discord" and target_ref and is_explicit and thread_id is None:
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(platform_name, target_ref)
            if resolved and resolved != target_ref:
                chat_id, thread_id, _ = _parse_target_ref(platform_name, resolved)
        except Exception:
            pass

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
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
    except Exception as e:
        return json.dumps(_error(f"Failed to load gateway config: {e}"))

    # Accept any platform name — built-in names resolve to their enum
    # member, plugin platform names create dynamic members via _missing_().
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return tool_error(f"Unknown platform: {platform_name}")

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        # Weixin can be configured purely via .env; synthesize a pconfig so
        # send_message and cron delivery work without a gateway.yaml entry.
        if platform_name == "weixin":
            wx_token = os.getenv("WEIXIN_TOKEN", "").strip()
            wx_account = os.getenv("WEIXIN_ACCOUNT_ID", "").strip()
            if wx_token and wx_account:
                from gateway.config import PlatformConfig
                pconfig = PlatformConfig(
                    enabled=True,
                    token=wx_token,
                    extra={
                        "account_id": wx_account,
                        "base_url": os.getenv("WEIXIN_BASE_URL", "").strip(),
                        "cdn_base_url": os.getenv("WEIXIN_CDN_BASE_URL", "").strip(),
                    },
                )
            else:
                return tool_error(f"Platform '{platform_name}' is not configured. Set up credentials in ~/.hermes/config.yaml or environment variables.")
        else:
            return tool_error(f"Platform '{platform_name}' is not configured. Set up credentials in ~/.hermes/config.yaml or environment variables.")

    from gateway.platforms.base import BasePlatformAdapter

    # Capture [[as_document]] directive before extract_media strips it.
    # Image-extension files in this batch will route through send_document
    # instead of send_photo so the original bytes survive (e.g. info-graph
    # JPGs where Telegram's sendPhoto recompresses to 1280px).
    force_document_attachments = "[[as_document]]" in message

    media_files, cleaned_message = BasePlatformAdapter.extract_media(message)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)

    if platform_name == "discord":
        try:
            from gateway.support_ops_routing import lint_and_resolve_discord_content
            content_lint = lint_and_resolve_discord_content(cleaned_message)
        except Exception as e:
            return json.dumps(_error(f"Discord Support Ops pre-send lint failed: {e}"))
        if not content_lint.ok:
            return json.dumps(
                _error(
                    "Discord Support Ops pre-send lint blocked delivery: "
                    f"{content_lint.blocked_reason}"
                )
            )
        cleaned_message = content_lint.content

    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)

    used_home_channel = False
    if not chat_id:
        home = config.get_home_channel(platform)
        if not home and platform_name == "weixin":
            wx_home = os.getenv("WEIXIN_HOME_CHANNEL", "").strip()
            if wx_home:
                from gateway.config import HomeChannel
                home = HomeChannel(platform=platform, chat_id=wx_home, name="Weixin Home")
        if home:
            chat_id = home.chat_id
            used_home_channel = True
        else:
            home_env = _HOME_CHANNEL_ENV_OVERRIDES.get(
                platform_name, f"{platform_name.upper()}_HOME_CHANNEL"
            )
            return json.dumps({
                "error": f"No home channel set for {platform_name} to determine where to send the message. "
                f"Either specify a channel directly with '{platform_name}:CHANNEL_NAME', "
                f"or set a home channel via: hermes config set {home_env} <channel_id>"
            })

    if platform_name == "discord":
        try:
            from gateway.support_ops_routing import lint_discord_target_for_content
            target_lint = lint_discord_target_for_content(
                cleaned_message,
                chat_id=str(chat_id or ""),
                thread_id=str(thread_id) if thread_id is not None else None,
            )
        except Exception as e:
            return json.dumps(_error(f"Discord Support Ops target lint failed: {e}"))
        if not target_lint.ok:
            expected = (
                f" Expected channel_id: {target_lint.expected_channel_id}."
                if target_lint.expected_channel_id
                else ""
            )
            return json.dumps(
                _error(
                    "Discord Support Ops target lint blocked delivery: "
                    f"{target_lint.blocked_reason}.{expected}"
                )
            )

    duplicate_skip = _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id)
    if duplicate_skip:
        return json.dumps(duplicate_skip)

    # Slack: resolve user IDs (U...) to DM channel IDs via conversations.open
    if platform_name == "slack" and chat_id and chat_id.startswith("U"):
        try:
            import aiohttp
            async def _open_slack_dm(token, user_id):
                url = "https://slack.com/api/conversations.open"
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.post(url, headers=headers, json={"users": [user_id]}) as resp:
                        data = await resp.json()
                        if data.get("ok"):
                            return data["channel"]["id"]
                        return None
            from model_tools import _run_async
            dm_channel = _run_async(_open_slack_dm(pconfig.token, chat_id))
            if dm_channel:
                chat_id = dm_channel
            else:
                return json.dumps({"error": f"Could not open DM with Slack user {chat_id}. Check bot permissions (im:write)."})
        except Exception as e:
            return json.dumps({"error": f"Failed to open Slack DM: {e}"})

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
                force_document=force_document_attachments,
            )
        )
        if used_home_channel and isinstance(result, dict) and result.get("success"):
            result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"

        # Mirror the sent message into the target's gateway session
        if isinstance(result, dict) and result.get("success") and mirror_text:
            try:
                from gateway.mirror import mirror_to_session
                from gateway.session_context import get_session_env
                source_label = get_session_env("HERMES_SESSION_PLATFORM", "cli")
                user_id = get_session_env("HERMES_SESSION_USER_ID", "") or None
                if mirror_to_session(
                    platform_name,
                    chat_id,
                    mirror_text,
                    source_label=source_label,
                    thread_id=thread_id,
                    user_id=user_id,
                ):
                    result["mirrored"] = True
            except Exception:
                pass

        if isinstance(result, dict) and "error" in result:
            result["error"] = _sanitize_error_text(result["error"])
        return json.dumps(result)
    except Exception as e:
        return json.dumps(_error(f"Send failed: {e}"))


def _parse_target_ref(platform_name: str, target_ref: str):
    """Parse a tool target into chat_id/thread_id and whether it is explicit."""
    if platform_name == "telegram":
        match = _TELEGRAM_TOPIC_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
        from plugins.platforms.telegram.telegram_ids import (
            parse_telegram_username_target,
        )

        username = parse_telegram_username_target(target_ref)
        if username:
            return username, None, True
    if platform_name == "feishu":
        match = _FEISHU_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
    if platform_name == "discord":
        match = _NUMERIC_TOPIC_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
    if platform_name == "slack":
        match = _SLACK_THREAD_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
        match = _SLACK_TARGET_RE.fullmatch(target_ref)
        if match:
            chat_id = match.group(1)
            # Slack user IDs (U...) and workspace IDs (W...) are NOT valid
            # explicit send targets — chat.postMessage rejects them. A DM
            # must be opened first via conversations.open to get a D...
            # conversation ID. Caller still gets the chat_id so the U→D
            # resolution path in send_message() can run.
            is_explicit = chat_id[0] not in {"U", "W"}
            return chat_id, None, is_explicit
    if platform_name == "matrix":
        trimmed = target_ref.strip()
        split_idx = trimmed.rfind(":$")
        if split_idx > 0:
            return trimmed[:split_idx], trimmed[split_idx + 1 :], True
    if platform_name == "weixin":
        match = _WEIXIN_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), None, True
    if platform_name == "yuanbao":
        match = _YUANBAO_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), None, True
        if target_ref.strip().isdigit():
            return f"group:{target_ref.strip()}", None, True
        return None, None, False
    if platform_name == "ntfy":
        topic = target_ref.strip()
        if topic:
            return topic, None, True
    if platform_name == "email":
        match = _EMAIL_TARGET_RE.fullmatch(target_ref)
        if match:
            return target_ref.strip(), None, True
    if platform_name == "whatsapp":
        # Native WhatsApp JIDs (group @g.us, user @s.whatsapp.net, @lid, etc.)
        # are explicit targets — pass through verbatim. E.164 '+' numbers fall
        # through to the _PHONE_PLATFORMS handler below.
        if _WHATSAPP_JID_RE.fullmatch(target_ref):
            return target_ref.strip(), None, True
    stripped_target = target_ref.strip()
    if platform_name == "signal" and stripped_target.startswith("group:"):
        group_id = stripped_target[len("group:"):].strip()
        if group_id:
            return f"group:{group_id}", None, True
        return None, None, False
    if platform_name in _PHONE_PLATFORMS:
        match = _E164_TARGET_RE.fullmatch(target_ref)
        if match:
            # Preserve the leading '+' — signal-cli and sms/whatsapp adapters
            # expect E.164 format for direct recipients.
            return target_ref.strip(), None, True
    if target_ref.lstrip("-").isdigit():
        return target_ref, None, True
    # Matrix room IDs (start with !) and user IDs (start with @) are explicit
    if platform_name == "matrix" and (target_ref.startswith("!") or target_ref.startswith("@")):
        return target_ref, None, True
    # XMPP JIDs (user@server or room@conference.server) are explicit
    if platform_name == "xmpp" and "@" in target_ref:
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
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip().lower()
    chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
    if not platform or not chat_id:
        return None
    thread_id = get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip() or None
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


async def _send_via_adapter(
    platform,
    pconfig,
    chat_id,
    chunk,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Send a message via a live gateway adapter, with a standalone fallback
    for out-of-process callers (e.g. cron running separately from the gateway).

    Order of attempts:
      1. Live in-process adapter via ``_gateway_runner_ref()`` (the path that
         existed before this change).
      2. The plugin's ``standalone_sender_fn`` registered on its
         ``PlatformEntry`` (used when the gateway is not in this process, so
         the runner weakref is ``None``).
      3. A descriptive error explaining both options.
    """
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    runner = None
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None

    if runner is not None:
        try:
            adapter = runner.adapters.get(platform)
        except Exception:
            adapter = None
        if adapter is not None:
            try:
                metadata = {}
                if thread_id:
                    metadata["thread_id"] = thread_id
                if platform_name == "ntfy" and chat_id:
                    metadata["publish_topic"] = chat_id
                if not metadata:
                    metadata = None
                result = await adapter.send(chat_id=chat_id, content=chunk, metadata=metadata)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                return {"error": f"Plugin platform send failed: {e}"}
            if result.success:
                return {"success": True, "message_id": result.message_id}
            return {"error": f"Adapter send failed: {result.error}"}

    entry = None
    try:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name)
    except Exception:
        entry = None

    if entry is not None and entry.standalone_sender_fn is not None:
        try:
            result = await entry.standalone_sender_fn(
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files,
                force_document=force_document,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Plugin standalone send for %s raised", platform_name, exc_info=True)
            return {"error": f"Plugin standalone send failed: {e}"}

        if isinstance(result, dict) and (result.get("success") or result.get("error")):
            return result
        return {
            "error": (
                f"Plugin standalone send for '{platform_name}' returned an "
                f"invalid result: expected a dict with 'success' or 'error' "
                f"keys, got {type(result).__name__}"
            )
        }

    return {
        "error": (
            f"No live adapter for platform '{platform_name}'. Is the gateway "
            f"running with this platform connected? For out-of-process delivery "
            f"(e.g. cron in a separate process), the platform plugin must "
            f"register a standalone_sender_fn on its PlatformEntry."
        )
    }


async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False):
    """Route a message to the appropriate platform sender.

    Long messages are automatically chunked to fit within platform limits
    using the same smart-splitting algorithm as the gateway adapters
    (preserves code-block boundaries, adds part indicators).
    """
    from gateway.config import Platform

    media_files = media_files or []

    # Weixin handles text/media delivery inside its native helper and does not
    # need the optional platform adapter imports below. Keep this branch early
    # so a Weixin send is not blocked by unrelated optional dependencies (for
    # example lark-oapi's heavy Feishu import path).
    if platform == Platform.WEIXIN:
        return await _send_weixin(pconfig, chat_id, message, media_files=media_files)

    from gateway.platforms.base import BasePlatformAdapter, utf16_len

    # Telegram adapter import is optional (requires python-telegram-bot)
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
        _telegram_available = True
    except ImportError:
        _telegram_available = False

    # Feishu adapter migrated to a plugin (#41112); its max_message_length
    # (8000) now flows through the registry fallback below.

    media_files = media_files or []

    # Slack mrkdwn formatting is applied inside the slack plugin's
    # _standalone_send (the registry standalone_sender_fn) rather than here —
    # the SlackAdapter moved to plugins/platforms/slack/ in #41112.

    # Platform message length limits (from adapter class attributes for
    # built-in platforms; from PlatformEntry.max_message_length for plugins,
    # resolved via the registry fallback below — covers Slack and Feishu, both
    # migrated to plugins in #41112).
    _MAX_LENGTHS = {
        Platform.TELEGRAM: TelegramAdapter.MAX_MESSAGE_LENGTH if _telegram_available else 4096,
    }

    # Check plugin registry for max_message_length
    if platform not in _MAX_LENGTHS:
        try:
            from gateway.platform_registry import platform_registry
            entry = platform_registry.get(platform.value)
            if entry and entry.max_message_length > 0:
                _MAX_LENGTHS[platform] = entry.max_message_length
        except Exception:
            pass

    # Smart-chunk the message to fit within platform limits.
    # For short messages or platforms without a known limit this is a no-op.
    # Telegram measures length in UTF-16 code units, not Unicode codepoints.
    max_len = _MAX_LENGTHS.get(platform)
    if max_len:
        _len_fn = utf16_len if platform == Platform.TELEGRAM else None
        chunks = BasePlatformAdapter.truncate_message(message, max_len, len_fn=_len_fn)
    else:
        chunks = [message]

    # --- Telegram: special handling for media attachments ---
    if platform == Platform.TELEGRAM:
        last_result = None
        disable_link_previews = bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews"))
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _send_telegram(
                pconfig.token,
                chat_id,
                chunk,
                media_files=media_files if is_last else [],
                thread_id=thread_id,
                disable_link_previews=disable_link_previews,
                force_document=force_document,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Discord: chunked delivery via the registry's standalone_sender_fn.
    # The plugin's ``_standalone_send`` (registered in
    # plugins/platforms/discord/adapter.py) handles forum channels, threads,
    # and multipart media uploads.  ``_send_via_adapter`` tries the live
    # in-process adapter first via ``adapter.send()``, but Discord's elif
    # historically went straight to the HTTP path; we preserve that by
    # explicitly invoking the registry hook here so behavior is unchanged.
    if platform == Platform.DISCORD:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get("discord")
        if entry is None or entry.standalone_sender_fn is None:
            return {"error": "Discord plugin not registered or missing standalone_sender_fn"}
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await entry.standalone_sender_fn(
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files if is_last else [],
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Matrix: use the native adapter helper when media is present ---
    if platform == Platform.MATRIX and media_files:
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _send_matrix_via_adapter(
                pconfig,
                chat_id,
                chunk,
                media_files=media_files if is_last else [],
                thread_id=thread_id,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Signal: native attachment support via JSON-RPC attachments param ---
    if platform == Platform.SIGNAL and media_files:
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _send_signal(
                pconfig.extra,
                chat_id,
                chunk,
                media_files=media_files if is_last else [],
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Yuanbao: native media attachment support via running gateway adapter ---
    if platform == Platform.YUANBAO and media_files:
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _send_yuanbao(
                chat_id,
                chunk,
                media_files=media_files if is_last else None,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Feishu: native media attachment support via the registry's
    # standalone_sender_fn (plugins/platforms/feishu/adapter.py::_standalone_send). #41112
    if platform == Platform.FEISHU and media_files:
        from gateway.platform_registry import platform_registry as _pr_feishu
        from hermes_cli.plugins import discover_plugins as _dp_feishu
        _dp_feishu()
        _feishu_entry = _pr_feishu.get("feishu")
        if _feishu_entry is None or _feishu_entry.standalone_sender_fn is None:
            return {"error": "Feishu plugin not registered or missing standalone_sender_fn"}
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _feishu_entry.standalone_sender_fn(
                pconfig,
                chat_id,
                chunk,
                media_files=media_files if is_last else None,
                thread_id=thread_id,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- WhatsApp: native media attachment support via the registry's
    # standalone_sender_fn (plugins/platforms/whatsapp/adapter.py::_standalone_send).
    # The plugin uploads each file through the local Baileys bridge /send-media
    # endpoint so images/videos/audio arrive as native bubbles, not documents. #41112
    if platform == Platform.WHATSAPP and media_files:
        from gateway.platform_registry import platform_registry as _pr_wa
        from hermes_cli.plugins import discover_plugins as _dp_wa
        _dp_wa()
        _wa_entry = _pr_wa.get("whatsapp")
        if _wa_entry is None or _wa_entry.standalone_sender_fn is None:
            return {"error": "WhatsApp plugin not registered or missing standalone_sender_fn"}
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _wa_entry.standalone_sender_fn(
                pconfig,
                chat_id,
                chunk,
                media_files=media_files if is_last else None,
                thread_id=thread_id,
                force_document=force_document,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Non-media platforms ---
    if media_files and not message.strip():
        return {
            "error": (
                f"send_message MEDIA delivery is currently only supported for telegram, discord, matrix, weixin, signal, yuanbao, feishu and whatsapp; "
                f"target {platform.value} had only media attachments"
            )
        }
    warning = None
    if media_files:
        warning = (
            f"MEDIA attachments were omitted for {platform.value}; "
            "native send_message media delivery is currently only supported for telegram, discord, matrix, weixin, signal, yuanbao, feishu and whatsapp"
        )

    last_result = None
    for chunk in chunks:
        if platform == Platform.SLACK:
            # Slack migrated to a bundled plugin (#41112); delivery flows
            # through the registry's standalone_sender_fn, which applies
            # mrkdwn formatting and posts via the Slack Web API.
            from gateway.platform_registry import platform_registry
            _slack_entry = platform_registry.get("slack")
            if _slack_entry is None or _slack_entry.standalone_sender_fn is None:
                result = {"error": "Slack plugin not registered or missing standalone_sender_fn"}
            else:
                result = await _slack_entry.standalone_sender_fn(
                    pconfig, chat_id, chunk, thread_id=thread_id
                )
        elif platform == Platform.WHATSAPP:
            result = await _registry_standalone_send("whatsapp", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.SIGNAL:
            result = await _send_signal(pconfig.extra, chat_id, chunk)
        elif platform == Platform.EMAIL:
            result = await _registry_standalone_send("email", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.SMS:
            result = await _registry_standalone_send("sms", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.MATRIX:
            result = await _registry_standalone_send("matrix", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.DINGTALK:
            result = await _registry_standalone_send("dingtalk", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.FEISHU:
            result = await _registry_standalone_send("feishu", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.WECOM:
            result = await _registry_standalone_send("wecom", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.BLUEBUBBLES:
            result = await _send_bluebubbles(pconfig.extra, chat_id, chunk)
        elif platform == Platform.QQBOT:
            result = await _send_qqbot(pconfig, chat_id, chunk)
        elif platform == Platform.YUANBAO:
            result = await _send_yuanbao(chat_id, chunk)
        else:
            # Plugin platform: route through the gateway's live adapter if
            # available, otherwise the plugin's standalone_sender_fn.
            result = await _send_via_adapter(
                platform,
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files,
                force_document=force_document,
            )

        if isinstance(result, dict) and result.get("error"):
            return result
        last_result = result

    if warning and isinstance(last_result, dict) and last_result.get("success"):
        warnings = list(last_result.get("warnings", []))
        warnings.append(warning)
        last_result["warnings"] = warnings
    return last_result


def _is_telegram_thread_not_found(error: Exception) -> bool:
    """Check if a Telegram error is a thread-not-found failure.

    Matches the gateway adapter's ``_is_thread_not_found_error`` for
    the standalone ``_send_telegram`` path (issue #27012).
    """
    return "thread not found" in str(error).lower()


async def _send_telegram(token, chat_id, message, media_files=None, thread_id=None, disable_link_previews=False, force_document=False):
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
                from plugins.platforms.telegram.adapter import TelegramAdapter
                _adapter = TelegramAdapter.__new__(TelegramAdapter)
                formatted = _adapter.format_message(message)
            except Exception:
                # Fallback: send as-is if formatting unavailable
                formatted = message
            send_parse_mode = ParseMode.MARKDOWN_V2

        # Honour a configured proxy (telegram.proxy_url in config.yaml, exported
        # as TELEGRAM_PROXY env var by load_gateway_config). Without this, the
        # standalone send path bypasses the proxy and times out in regions
        # where api.telegram.org is blocked. The in-gateway adapter does the
        # same thing in gateway/platforms/telegram.py.
        try:
            from gateway.platforms.base import resolve_proxy_url
            _tg_proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
        except Exception:
            _tg_proxy = None
        if _tg_proxy:
            try:
                from telegram.request import HTTPXRequest
                logger.info("send_message: standalone Telegram send routed through proxy %s", _tg_proxy)
                bot = Bot(
                    token=token,
                    request=HTTPXRequest(proxy=_tg_proxy),
                    get_updates_request=HTTPXRequest(proxy=_tg_proxy),
                )
            except Exception as _proxy_err:
                logger.warning("send_message: failed to attach Telegram proxy (%s), falling back to direct connection", _proxy_err)
                bot = Bot(token=token)
        else:
            bot = Bot(token=token)
        from plugins.platforms.telegram.telegram_ids import (
            normalize_telegram_chat_id,
        )

        # Telegram accepts a numeric chat_id OR an @username string; normalize
        # rather than force-int so username home channels don't crash (#13206).
        int_chat_id = normalize_telegram_chat_id(chat_id)
        media_files = media_files or []
        thread_kwargs = {}
        if thread_id is not None:
            # Reuse the gateway adapter's General-topic mapping: in Telegram
            # forum supergroups, the General topic is addressed as
            # message_thread_id="1" on incoming updates, but Bot API
            # sendMessage rejects message_thread_id=1 with "Message thread
            # not found". The adapter's helper maps "1" to None for that
            # reason; the send_message tool needs the same mapping or a
            # send to a forum group's General topic always errors out
            # (see issue #22267).
            try:
                from plugins.platforms.telegram.adapter import TelegramAdapter
                effective_thread_id = TelegramAdapter._message_thread_id_for_send(
                    str(thread_id)
                )
            except Exception:
                # Fallback: explicit mapping in case the adapter import
                # fails (e.g. python-telegram-bot missing in this venv).
                effective_thread_id = (
                    None if str(thread_id) == "1" else int(thread_id)
                )
            if effective_thread_id is not None:
                thread_kwargs["message_thread_id"] = effective_thread_id
        # disable_web_page_preview is only valid for send_message, not
        # send_photo/send_video/etc.  Keep it separate so media sends
        # don't inherit an invalid parameter (issue #27012).
        text_kwargs = dict(thread_kwargs)
        if disable_link_previews:
            text_kwargs["disable_web_page_preview"] = True

        last_msg = None
        warnings = []

        if formatted.strip():
            try:
                last_msg = await _send_telegram_message_with_retry(
                    bot,
                    chat_id=int_chat_id, text=formatted,
                    parse_mode=send_parse_mode, **text_kwargs
                )
            except Exception as md_error:
                # Thread not found — retry without message_thread_id so the
                # message still delivers (matching the gateway adapter's
                # fallback behaviour, issue #27012).
                if _is_telegram_thread_not_found(md_error) and thread_kwargs:
                    logger.warning(
                        "Thread %s not found in _send_telegram, retrying without message_thread_id",
                        thread_kwargs.get("message_thread_id"),
                    )
                    text_kwargs.pop("message_thread_id", None)
                    last_msg = await _send_telegram_message_with_retry(
                        bot,
                        chat_id=int_chat_id, text=formatted,
                        parse_mode=send_parse_mode, **text_kwargs
                    )
                elif "parse" in str(md_error).lower() or "markdown" in str(md_error).lower() or "html" in str(md_error).lower():
                    logger.warning(
                        "Parse mode %s failed in _send_telegram, falling back to plain text: %s",
                        send_parse_mode,
                        _sanitize_error_text(md_error),
                    )
                    if not _has_html:
                        try:
                            from plugins.platforms.telegram.adapter import _strip_mdv2
                            plain = _strip_mdv2(formatted)
                        except Exception:
                            plain = message
                    else:
                        plain = message
                    last_msg = await _send_telegram_message_with_retry(
                        bot,
                        chat_id=int_chat_id, text=plain,
                        parse_mode=None, **text_kwargs
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
                    media_kwargs = dict(thread_kwargs)
                    try:
                        if ext in _IMAGE_EXTS and not force_document:
                            last_msg = await bot.send_photo(
                                chat_id=int_chat_id, photo=f, **media_kwargs
                            )
                        elif ext in _VIDEO_EXTS:
                            last_msg = await bot.send_video(
                                chat_id=int_chat_id, video=f, **media_kwargs
                            )
                        elif ext in _VOICE_EXTS and is_voice:
                            last_msg = await bot.send_voice(
                                chat_id=int_chat_id, voice=f, **media_kwargs
                            )
                        elif ext in _TELEGRAM_SEND_AUDIO_EXTS:
                            last_msg = await bot.send_audio(
                                chat_id=int_chat_id, audio=f, **media_kwargs
                            )
                        else:
                            last_msg = await bot.send_document(
                                chat_id=int_chat_id, document=f, **media_kwargs
                            )
                    except Exception as media_err:
                        if _is_telegram_thread_not_found(media_err) and media_kwargs.get("message_thread_id"):
                            # Thread not found for media — retry without
                            # message_thread_id (issue #27012).
                            logger.warning(
                                "Thread %s not found for media send, retrying without message_thread_id",
                                media_kwargs["message_thread_id"],
                            )
                            # Re-seek the file since the first attempt consumed it
                            f.seek(0)
                            media_kwargs.pop("message_thread_id", None)
                            if ext in _IMAGE_EXTS and not force_document:
                                last_msg = await bot.send_photo(
                                    chat_id=int_chat_id, photo=f, **media_kwargs
                                )
                            elif ext in _VIDEO_EXTS:
                                last_msg = await bot.send_video(
                                    chat_id=int_chat_id, video=f, **media_kwargs
                                )
                            elif ext in _VOICE_EXTS and is_voice:
                                last_msg = await bot.send_voice(
                                    chat_id=int_chat_id, voice=f, **media_kwargs
                                )
                            elif ext in _TELEGRAM_SEND_AUDIO_EXTS:
                                last_msg = await bot.send_audio(
                                    chat_id=int_chat_id, audio=f, **media_kwargs
                                )
                            else:
                                last_msg = await bot.send_document(
                                    chat_id=int_chat_id, document=f, **media_kwargs
                                )
                        else:
                            raise
            except Exception as e:
                warning = _sanitize_error_text(f"Failed to send media {media_path}: {e}")
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
        return _error(f"Telegram send failed: {e}")


# _send_slack moved to the slack plugin as _standalone_send
# (plugins/platforms/slack/adapter.py), wired via standalone_sender_fn. #41112.


async def _registry_standalone_send(platform_name, pconfig, chat_id, message, thread_id=None):
    """Dispatch a one-shot send through a migrated platform plugin's
    standalone_sender_fn (registry hook).  Used for platforms whose adapter
    moved out of gateway/platforms/ into plugins/platforms/<name>/ (#41112):
    the legacy inline ``_send_<platform>`` helper now lives in the plugin as
    ``_standalone_send`` and is reached via the platform registry.
    """
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    discover_plugins()  # idempotent — ensure the entry is registered
    entry = platform_registry.get(platform_name)
    if entry is None or entry.standalone_sender_fn is None:
        return {"error": f"{platform_name} plugin not registered or missing standalone_sender_fn"}
    return await entry.standalone_sender_fn(pconfig, chat_id, message, thread_id=thread_id)


# _send_whatsapp moved to plugins/platforms/whatsapp/adapter.py::_standalone_send,
# wired via standalone_sender_fn and reached through _registry_standalone_send. #41112.


async def _send_signal(extra, chat_id, message, media_files=None):
    """Send via signal-cli JSON-RPC API.

    Supports both text-only and text-with-attachments (images/audio/documents).
    Multi-attachment sends are chunked into batches of
    SIGNAL_MAX_ATTACHMENTS_PER_MSG and metered by the process-wide
    SignalAttachmentScheduler — same bucket the gateway adapter uses, so
    sends from this tool and inbound-driven replies share rate-limit state.
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}

    from gateway.platforms.signal_rate_limit import (
        SIGNAL_BATCH_PACING_NOTICE_THRESHOLD,
        SIGNAL_MAX_ATTACHMENTS_PER_MSG,
        SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
        _extract_retry_after_seconds,
        _format_wait,
        _is_signal_rate_limit_error,
        _signal_send_timeout,
        get_scheduler,
    )
    from gateway.platforms.signal_format import markdown_to_signal

    try:
        http_url = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/")
        account = extra.get("account", "")
        if not account:
            return {"error": "Signal account not configured"}

        valid_media = media_files or []
        attachment_paths = []
        for media_path, _is_voice in valid_media:
            if os.path.exists(media_path):
                attachment_paths.append(media_path)
            else:
                logger.warning("Signal media file not found, skipping: %s", media_path)

        # Chunk attachments. With no attachments we still emit one batch
        # (text only). With attachments, the text rides on batch #0 so the
        # caption isn't repeated across every chunk.
        if attachment_paths:
            att_batches = [
                attachment_paths[i:i + SIGNAL_MAX_ATTACHMENTS_PER_MSG]
                for i in range(0, len(attachment_paths), SIGNAL_MAX_ATTACHMENTS_PER_MSG)
            ]
        else:
            att_batches = [[]]

        plain_text, text_styles = markdown_to_signal(message)

        async def _post(batch_attachments, batch_message):
            params = {"account": account, "message": batch_message}
            if batch_message and text_styles:
                if len(text_styles) == 1:
                    params["textStyle"] = text_styles[0]
                else:
                    params["textStyles"] = text_styles
            if chat_id.startswith("group:"):
                params["groupId"] = chat_id[6:]
            else:
                params["recipient"] = [chat_id]
            if batch_attachments:
                params["attachments"] = batch_attachments

            payload = {
                "jsonrpc": "2.0",
                "method": "send",
                "params": params,
                "id": f"send_{int(time.time() * 1000)}",
            }
            timeout = _signal_send_timeout(len(batch_attachments) if batch_attachments else 0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{http_url}/api/v1/rpc", json=payload)
                resp.raise_for_status()
                return resp.json()

        async def _send_inline_notice(text: str) -> None:
            """Best-effort one-shot RPC for a user-facing pacing notice."""
            notice_params = {"account": account, "message": text}
            if chat_id.startswith("group:"):
                notice_params["groupId"] = chat_id[6:]
            else:
                notice_params["recipient"] = [chat_id]
            try:
                async with httpx.AsyncClient(timeout=30.0) as _client:
                    await _client.post(
                        f"{http_url}/api/v1/rpc",
                        json={
                            "jsonrpc": "2.0",
                            "method": "send",
                            "params": notice_params,
                            "id": f"notice_{int(time.time() * 1000)}",
                        },
                    )
            except Exception as _e:
                logger.warning("Signal: inline notice failed: %s", _e)

        scheduler = get_scheduler()
        logger.info(
            "send_message Signal: scheduler state=%s, %d attachment(s) in %d batch(es)",
            scheduler.state(), len(attachment_paths), len(att_batches),
        )
        failed_batches: list[int] = []
        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            if n > 0:
                estimated = scheduler.estimate_wait(n)
                if estimated >= SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                    await _send_inline_notice(
                        f"(More images coming — pausing ~{_format_wait(estimated)} "
                        f"for Signal rate limit, batch {idx + 1}/{len(att_batches)}.)"
                    )

            batch_message = plain_text if idx == 0 else ""

            for attempt in range(1, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
                try:
                    await scheduler.acquire(n)
                    _rpc_t0 = time.monotonic()
                    data = await _post(att_batch, batch_message)
                    _rpc_duration = time.monotonic() - _rpc_t0
                    if "error" not in data:
                        await scheduler.report_rpc_duration(_rpc_duration, n)
                        break

                    err = data["error"]

                    if not _is_signal_rate_limit_error(err):
                        return _error(f"Signal RPC error on batch {idx + 1}/{len(att_batches)}: {err}")

                    server_retry_after = _extract_retry_after_seconds(err)
                    scheduler.feedback(server_retry_after, n)

                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        failed_batches.append(idx + 1)
                        logger.error(
                            "Signal: rate-limit retries exhausted on batch %d/%d "
                            "(%d attachments lost, server retry_after=%s)",
                            idx + 1, len(att_batches), n,
                            f"{server_retry_after:.0f}s" if server_retry_after else "unknown",
                        )
                        break
                    logger.warning(
                        "Signal: rate-limited on batch %d/%d "
                        "(attempt %d/%d, server retry_after=%s); "
                        "scheduler will pace the retry",
                        idx + 1, len(att_batches),
                        attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                        f"{server_retry_after:.0f}s" if server_retry_after else "unknown",
                    )
                except Exception as e:
                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        failed_batches.append(idx + 1)
                        logger.error(
                            "Signal: send error on batch %d/%d after %d attempts: %s",
                            idx + 1, len(att_batches), attempt, str(e)
                        )
                        break
                    logger.warning(
                        "Signal: transient error on batch %d/%d (attempt %d/%d): %s; will retry",
                        idx + 1, len(att_batches), attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS, str(e)
                    )

        warnings = []
        if len(attachment_paths) < len(valid_media):
            warnings.append("Some media files were skipped (not found on disk)")
        if failed_batches:
            warnings.append(
                f"Signal rate-limited {len(failed_batches)} batch(es) "
                f"(#{', #'.join(str(b) for b in failed_batches)})"
            )

        if failed_batches and len(failed_batches) == len(att_batches):
            return _error(
                f"Signal: every batch ({len(att_batches)}) hit rate limit; "
                f"no attachments delivered"
            )

        result = {"success": True, "platform": "signal", "chat_id": _display_chat_id("signal", chat_id)}
        if warnings:
            result["warnings"] = warnings
        return result
    except Exception as e:
        return _error(f"Signal send failed: {e}")


# _send_email moved to plugins/platforms/email/adapter.py::_standalone_send;
# _send_sms moved to plugins/platforms/sms/adapter.py::_standalone_send. Both
# wired via standalone_sender_fn, reached through _registry_standalone_send. #41112.


# _send_matrix moved to plugins/platforms/matrix/adapter.py::_standalone_send,
# wired via standalone_sender_fn and reached through _registry_standalone_send. #41112.
# (_send_matrix_via_adapter below stays — it's the native-media upload path.)


async def _send_matrix_via_adapter(pconfig, chat_id, message, media_files=None, thread_id=None):
    """Send via the Matrix adapter so native Matrix media uploads are preserved.

    When a live gateway adapter is available (i.e. the tool runs inside a
    running gateway), the persistent connection is reused — one olm/megolm
    session for all sends.  This avoids per-message E2EE re-init storms
    that exhaust recipient OTKs and silently drop messages (issue #46310).

    Falls back to an ephemeral connect/disconnect cycle only when no gateway
    is running (standalone cron, ``hermes send`` CLI).
    """
    media_files = media_files or []
    metadata = {"thread_id": thread_id} if thread_id else None

    # --- Try the live gateway adapter first (persistent E2EE session) ---
    # Reusing the running gateway's already-connected adapter is the whole
    # point of #46310: it avoids a per-send login + olm/megolm re-init + OTK
    # claim that, under burst sends, exhausts recipient one-time keys and
    # silently drops messages. The import is guarded narrowly (gateway code may
    # be absent in some standalone contexts); a runner that *exists* but whose
    # adapter lookup fails is logged rather than silently swallowed, because a
    # silent fall-through here would re-introduce the exact reconnect storm
    # this fix prevents.
    live_adapter = None
    runner = None
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is not None:
        try:
            from gateway.config import Platform
            live_adapter = runner.adapters.get(Platform.MATRIX)
        except Exception:
            logger.warning(
                "Matrix: live gateway adapter lookup failed; falling back to an "
                "ephemeral connect (may re-init E2EE per send, see #46310)",
                exc_info=True,
            )
            live_adapter = None

    if live_adapter is not None:
        # NOTE: the live adapter is owned by the gateway — we must NOT
        # disconnect it. Correctness here depends on this branch returning
        # before the ephemeral ``adapter`` is constructed below, so the
        # ephemeral ``finally`` disconnect never touches the live session.
        return await _matrix_send_core(
            live_adapter, chat_id, message, media_files, metadata
        )

    # --- Fallback: ephemeral adapter (standalone / cron context) ---
    try:
        from plugins.platforms.matrix.adapter import MatrixAdapter
    except ImportError:
        return {"error": "Matrix dependencies not installed. Run: pip install 'mautrix[encryption]'"}

    adapter = MatrixAdapter(pconfig)
    try:
        connected = await adapter.connect()
        if not connected:
            return _error("Matrix connect failed")
        return await _matrix_send_core(
            adapter, chat_id, message, media_files, metadata
        )
    except Exception as e:
        return _error(f"Matrix send failed: {e}")
    finally:
        try:
            await adapter.disconnect()
        except Exception:
            pass


async def _matrix_send_core(adapter, chat_id, message, media_files, metadata):
    """Core send logic shared by live and ephemeral Matrix adapters."""
    last_result = None

    if message.strip():
        last_result = await adapter.send(chat_id, message, metadata=metadata)
        if not last_result.success:
            return _error(f"Matrix send failed: {last_result.error}")

    for media_path, is_voice in media_files:
        if not os.path.exists(media_path):
            return _error(f"Media file not found: {media_path}")

        ext = os.path.splitext(media_path)[1].lower()
        if ext in _IMAGE_EXTS:
            last_result = await adapter.send_image_file(chat_id, media_path, metadata=metadata)
        elif ext in _VIDEO_EXTS:
            last_result = await adapter.send_video(chat_id, media_path, metadata=metadata)
        elif ext in _VOICE_EXTS and is_voice:
            last_result = await adapter.send_voice(chat_id, media_path, metadata=metadata)
        elif ext in _AUDIO_EXTS:
            last_result = await adapter.send_voice(chat_id, media_path, metadata=metadata)
        else:
            last_result = await adapter.send_document(chat_id, media_path, metadata=metadata)

        if not last_result.success:
            return _error(f"Matrix media send failed: {last_result.error}")

    if last_result is None:
        return {"error": "No deliverable text or media remained after processing MEDIA tags"}

    return {
        "success": True,
        "platform": "matrix",
        "chat_id": chat_id,
        "message_id": last_result.message_id,
    }


# _send_dingtalk moved to plugins/platforms/dingtalk/adapter.py::_standalone_send,
# wired via standalone_sender_fn and reached through _registry_standalone_send. #41112.


# _send_wecom moved to plugins/platforms/wecom/adapter.py::_standalone_send,
# wired via standalone_sender_fn and reached through _registry_standalone_send. #41112.


async def _send_weixin(pconfig, chat_id, message, media_files=None):
    """Send via Weixin iLink using the native adapter helper."""
    try:
        from gateway.platforms.weixin import check_weixin_requirements, send_weixin_direct
        if not check_weixin_requirements():
            return {"error": "Weixin requirements not met. Need aiohttp + cryptography."}
    except ImportError:
        return {"error": "Weixin adapter not available."}

    try:
        return await send_weixin_direct(
            extra=pconfig.extra,
            token=pconfig.token,
            chat_id=chat_id,
            message=message,
            media_files=media_files,
        )
    except Exception as e:
        return _error(f"Weixin send failed: {e}")


async def _send_bluebubbles(extra, chat_id, message):
    """Send via BlueBubbles iMessage server using the adapter's REST API."""
    try:
        from gateway.platforms.bluebubbles import BlueBubblesAdapter, check_bluebubbles_requirements
        if not check_bluebubbles_requirements():
            return {"error": "BlueBubbles requirements not met (need aiohttp + httpx)."}
    except ImportError:
        return {"error": "BlueBubbles adapter not available."}

    try:
        from gateway.config import PlatformConfig
        pconfig = PlatformConfig(extra=extra)
        adapter = BlueBubblesAdapter(pconfig)
        connected = await adapter.connect()
        if not connected:
            return _error("BlueBubbles: failed to connect to server")
        try:
            result = await adapter.send(chat_id, message)
            if not result.success:
                return _error(f"BlueBubbles send failed: {result.error}")
            return {"success": True, "platform": "bluebubbles", "chat_id": chat_id, "message_id": result.message_id}
        finally:
            await adapter.disconnect()
    except Exception as e:
        return _error(f"BlueBubbles send failed: {e}")


# _send_feishu moved to plugins/platforms/feishu/adapter.py::_standalone_send,
# wired via standalone_sender_fn and reached through _registry_standalone_send
# (and the feishu media branch above). #41112.


def _check_send_message():
    """Gate send_message on gateway running (always available on messaging platforms).

    Also passes for kanban workers — the dispatcher sets ``HERMES_KANBAN_TASK``
    on every spawned worker, but those workers run with the assignee profile's
    ``HERMES_HOME`` which has no ``gateway.pid``, so the gateway-running check
    would fail even though the parent gateway is alive. Honoring the env var
    lets workers call ``send_message`` to deliver rich content directly to the
    originating chat (paired with ``kanban_complete`` for the short notifier
    summary), which is the canonical pattern for any worker that needs to
    reply with more than the ~200-char first-line truncation the kanban
    notifier applies.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return True
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if platform and platform != "local":
        return True
    try:
        from gateway.status import is_gateway_running
        return is_gateway_running()
    except Exception:
        return False


async def _send_qqbot(pconfig, chat_id, message):
    """Send via QQBot using the REST API directly (no WebSocket needed).

    Uses the QQ Bot Open Platform REST endpoints to get an access token
    and post a message. Supports guild channels, C2C (private) chats,
    and group chats by trying the appropriate endpoints.
    """
    try:
        import httpx
    except ImportError:
        return _error("QQBot direct send requires httpx. Run: pip install httpx")

    extra = pconfig.extra or {}
    appid = extra.get("app_id") or os.getenv("QQ_APP_ID", "")
    secret = (pconfig.token or extra.get("client_secret")
              or os.getenv("QQ_CLIENT_SECRET", ""))
    if not appid or not secret:
        return _error("QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured.")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Step 1: Get access token
            token_resp = await client.post(
                "https://bots.qq.com/app/getAppAccessToken",
                json={"appId": str(appid), "clientSecret": str(secret)},
            )
            if token_resp.status_code != 200:
                return _error(f"QQBot token request failed: {token_resp.status_code}")
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                return _error(f"QQBot: no access_token in response")

            # Step 2: Send message via REST
            # QQ Bot API has separate endpoints for channels, C2C, and groups.
            # We try them in order: channel first, then fallback to C2C.
            headers = {
                "Authorization": f"QQBot {access_token}",
                "Content-Type": "application/json",
            }
            payload = {"content": message[:4000], "msg_type": 0}

            # Try channel endpoint first (works for guild channels)
            url = f"https://api.sgroup.qq.com/channels/{chat_id}/messages"
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in {200, 201}:
                data = resp.json()
                return {"success": True, "platform": "qqbot", "chat_id": chat_id,
                        "message_id": data.get("id")}

            # If channel endpoint failed (likely "频道不存在"), try C2C endpoint
            url_c2c = f"https://api.sgroup.qq.com/v2/users/{chat_id}/messages"
            resp_c2c = await client.post(url_c2c, json=payload, headers=headers)
            if resp_c2c.status_code in {200, 201}:
                data = resp_c2c.json()
                return {"success": True, "platform": "qqbot", "chat_id": chat_id,
                        "message_id": data.get("id")}

            # If C2C also failed, try group endpoint
            url_group = f"https://api.sgroup.qq.com/v2/groups/{chat_id}/messages"
            resp_group = await client.post(url_group, json=payload, headers=headers)
            if resp_group.status_code in {200, 201}:
                data = resp_group.json()
                return {"success": True, "platform": "qqbot", "chat_id": chat_id,
                        "message_id": data.get("id")}

            # All endpoints failed — return the most informative error
            return _error(f"QQBot send failed: channel={resp.status_code} c2c={resp_c2c.status_code} group={resp_group.status_code}")
    except Exception as e:
        return _error(f"QQBot send failed: {e}")


async def _send_yuanbao(chat_id, message, media_files=None):
    """Send via Yuanbao using the running gateway adapter's WebSocket connection.

    Yuanbao uses a persistent WebSocket — unlike HTTP-based platforms, we
    cannot create a throwaway client.  We obtain the running singleton from
    the adapter module itself (``get_active_adapter``).

    chat_id format:
      - Group: "group:<group_code>"
      - DM:    "direct:<account_id>" or just "<account_id>"
    """
    try:
        from gateway.platforms.yuanbao import get_active_adapter, send_yuanbao_direct
    except ImportError:
        return _error("Yuanbao adapter module not available.")

    adapter = get_active_adapter()
    if adapter is None:
        return _error(
            "Yuanbao adapter is not running. "
            "Start the gateway with yuanbao platform enabled first."
        )

    try:
        return await send_yuanbao_direct(adapter, chat_id, message, media_files=media_files)
    except Exception as e:
        return _error(f"Yuanbao send failed: {e}")


# --- Registry ---
from tools.registry import tool_error

# NOTE: ``send_message`` is intentionally NOT registered as an agent-callable
# model tool. The agent should not decide on its own to fire off cross-platform
# messages or reactions. The send engine in this module (``_send_to_platform``,
# ``_send_via_adapter``, ``_parse_target_ref``, the per-platform ``_send_*``
# helpers) remains the shared transport used by:
#   - cron delivery (cron/scheduler.py)
#   - the ``hermes send`` CLI command (hermes_cli/send_cmd.py)
#   - the gateway kanban notifier (dashboard-toggled, outside agent control)
#   - the standalone MCP server (mcp_serve.py), which is an opt-in surface
# Those callers import the helpers directly; none of them need the registry
# entry.
