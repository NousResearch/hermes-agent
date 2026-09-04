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
import time


from agent.redact import redact_sensitive_text
from agent.secret_scope import get_secret
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Platform transports extracted to send_message_transports.py (physical-line split).
from .send_message_transports import (  # noqa: F401 — re-exported for monkeypatch targets
    _AUDIO_EXTS,
    _CAPTIONABLE_EXTS,
    _DEFAULT_CAPTION_LIMIT,
    _IMAGE_EXTS,
    _TELEGRAM_CAPTION_LIMIT,
    _TELEGRAM_SEND_AUDIO_EXTS,
    _VIDEO_EXTS,
    _VOICE_EXTS,
    _display_chat_id,
    _error,
    _is_telegram_thread_not_found,
    _matrix_send_core,
    _media_caption_split,
    _registry_standalone_send,
    _resolve_slack_user_target,
    _sanitize_error_text,
    _send_bluebubbles,
    _send_matrix_via_adapter,
    _send_qqbot,
    _send_signal,
    _send_telegram,
    _send_telegram_message_with_retry,
    _send_weixin,
    _send_yuanbao,
    _telegram_retry_delay,
)

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_FEISHU_TARGET_RE = re.compile(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$")
# Slack conversation IDs: C (public channel), G (private/group channel), D (DM).
# Must be uppercase alphanumeric, 9+ chars. User IDs (U...) are parsed as
# explicit user targets (``user:U...``) and are converted to D... conversations
# via conversations.open before chat.postMessage — posting directly to a U/W
# ID fails because the API requires a conversation ID. ``@handle`` targets are
# resolved through users.list first (``user_name:...``).
_SLACK_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,})\s*$")
_SLACK_USER_ID_RE = re.compile(r"^\s*(U[A-Z0-9]{8,})\s*$")
_SLACK_USER_NAME_RE = re.compile(r"^\s*@([A-Za-z0-9._-]{1,80})\s*$")
_SLACK_MENTION_RE = re.compile(r"^\s*<@(U[A-Z0-9]{8,})(?:\|[^>]+)?>\s*$")
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
# Photon DM chat GUID (mirrors _DM_CHAT_GUID_RE in the photon adapter).
_PHOTON_DM_GUID_RE = re.compile(r"^any;-;\+\d{6,}$")
# WhatsApp JIDs: group chats (<digits>@g.us), individual users
# (<phone>@s.whatsapp.net), linked identities (<id>@lid), and broadcast /
# newsletter chats. These are explicit native targets the bridge accepts
# verbatim — they must NOT fall through to home-channel resolution.
_WHATSAPP_JID_RE = re.compile(
    r"^\s*[\w-]+@(?:g\.us|s\.whatsapp\.net|lid|broadcast|newsletter)\s*$",
    re.IGNORECASE,
)
# Buzz channels and DMs use native UUID identifiers. They are explicit
# targets and must never substitute the configured home channel.
_BUZZ_UUID_RE = re.compile(
    r"^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$",
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

def prepare_send_message_platforms() -> None:
    """Load enabled standalone plugins before tool schemas/cache keys are built."""
    from hermes_cli.plugins import discover_plugins

    discover_plugins()












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
    prepare_send_message_platforms()
    if target_ref:
        # Platform-native ids (e.g. photon space GUIDs like 'any;-;+1555...')
        # match no parser pattern and no directory entry, so hand them to
        # the adapter unchanged; it validates them.
        chat_id, _thread_id, resolution_error = resolve_send_target(
            platform_name, target_ref, pass_unresolved_references=True
        )
        if resolution_error:
            return tool_error(resolution_error)

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

    prepare_send_message_platforms()
    if target_ref:
        chat_id, thread_id, resolution_error = resolve_send_target(
            platform_name, target_ref
        )
        if resolution_error:
            return tool_error(resolution_error)

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
    except Exception as e:
        return json.dumps(_error(f"Failed to load gateway config: {e}"))

    from gateway.platform_registry import platform_registry

    entry = platform_registry.get(platform_name)
    is_builtin = platform_name in {member.value for member in Platform}
    if not is_builtin and entry is None:
        return tool_error(
            f"Unknown or unregistered plugin platform: {platform_name}"
        )
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return tool_error(f"Unknown platform: {platform_name}")

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        # Weixin can be configured purely via .env; synthesize a pconfig so
        # send_message and cron delivery work without a gateway.yaml entry.
        if platform_name == "weixin":
            wx_token = get_secret("WEIXIN_TOKEN", "").strip()
            wx_account = get_secret("WEIXIN_ACCOUNT_ID", "").strip()
            if wx_token and wx_account:
                from gateway.config import PlatformConfig
                pconfig = PlatformConfig(
                    enabled=True,
                    token=wx_token,
                    extra={
                        "account_id": wx_account,
                        "base_url": get_secret("WEIXIN_BASE_URL", "").strip(),
                        "cdn_base_url": get_secret("WEIXIN_CDN_BASE_URL", "").strip(),
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
    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)

    used_home_channel = False
    if not chat_id:
        # A2A origin rule: a confirmation emitted
        # from an A2A session for a context that was born in a real gateway
        # session (e.g. a Discord thread) must return to that origin's
        # chat/thread — the session that initiated the A2A exchange. The
        # home channel is only the fallback when no origin exists; without
        # this, A2A confirmations post to the platform-wide default channel
        # instead of the originating thread.
        try:
            from plugins.platforms.a2a.tools import _current_a2a_origin_target
            origin_target = _current_a2a_origin_target(platform_name)
        except Exception:
            origin_target = {}
        if origin_target:
            chat_id = origin_target["chat_id"]
            thread_id = origin_target.get("thread_id") or thread_id
            logger.info(
                "send_message: A2A session confirmation routed to origin %s chat %s (thread %s) "
                "instead of the home channel",
                platform_name, chat_id, thread_id,
            )
        else:
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
                return tool_error(
                    f"No home channel set for {platform_name} to determine where to send the message. "
                    f"Either specify a channel directly with '{platform_name}:CHANNEL_NAME', "
                    f"or set a home channel via: hermes config set {home_env} <channel_id>"
                )

    duplicate_skip = _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id)
    if duplicate_skip:
        return json.dumps(duplicate_skip)

    # Slack: resolve user targets to DM channel IDs before sending.
    # _parse_target_ref emits internal ``user:U...`` / ``user_name:@handle``
    # targets; a bare U... id can also arrive from session metadata or the
    # home-channel config. All are opened via conversations.open (fixes #19236).
    if platform_name == "slack" and chat_id:
        _slack_dm_target = chat_id
        if _slack_dm_target.startswith("U") and _SLACK_USER_ID_RE.fullmatch(_slack_dm_target):
            _slack_dm_target = f"user:{_slack_dm_target}"
        if _slack_dm_target.startswith(("user:", "user_name:")):
            from model_tools import _run_async
            _resolved, _resolve_err = _run_async(
                _resolve_slack_user_target(pconfig.token, _slack_dm_target)
            )
            if _resolve_err:
                return json.dumps(_resolve_err)
            chat_id = _resolved

    try:
        from model_tools import _run_async
        send_kwargs = {
            "thread_id": thread_id,
            "media_files": media_files,
            "force_document": force_document_attachments,
        }
        # Preserve the exact built-in call contract; only custom handlers need
        # the complete typed request.
        if entry is not None and entry.send_message_handler is not None:
            send_kwargs["args"] = args
        result = _run_async(
            _send_to_platform(
                platform,
                pconfig,
                chat_id,
                cleaned_message,
                **send_kwargs,
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
            return match.group(1), None, True
        match = _SLACK_USER_ID_RE.fullmatch(target_ref) or _SLACK_MENTION_RE.fullmatch(target_ref)
        if match:
            return f"user:{match.group(1)}", None, True
        match = _SLACK_USER_NAME_RE.fullmatch(target_ref)
        if match:
            return f"user_name:{match.group(1)}", None, True
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
    if platform_name == "buzz" and _BUZZ_UUID_RE.fullmatch(target_ref):
        return target_ref.strip(), None, True
    stripped_target = target_ref.strip()
    if platform_name == "signal" and stripped_target.startswith("group:"):
        group_id = stripped_target[len("group:"):].strip()
        if group_id:
            return f"group:{group_id}", None, True
        return None, None, False
    # WeCom: group IDs start with "wr" or "wc", user IDs start with "wo" or
    # are bare alphanumeric strings. Treat any non-empty WeCom target_ref as
    # an explicit chat_id — the adapter resolves whether to use APP_CMD_RESPONSE
    # (groups) or APP_CMD_SEND (DMs) internally.
    if platform_name == "wecom":
        stripped = target_ref.strip()
        if stripped:
            return stripped, None, True
    if platform_name in _PHONE_PLATFORMS:
        match = _E164_TARGET_RE.fullmatch(target_ref)
        if match:
            # Preserve the leading '+' — signal-cli and sms/whatsapp adapters
            # expect E.164 format for direct recipients.
            return target_ref.strip(), None, True
    if platform_name == "photon":
        # Photon DM chat GUIDs ('any;-;+1555...') are platform-native ids the
        # adapter resolves itself — pass through verbatim instead of bouncing
        # them off the channel directory (mirrors the react handler).
        if _PHOTON_DM_GUID_RE.fullmatch(target_ref.strip()):
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


def resolve_send_target(
    platform_name: str, target_ref: str, *, pass_unresolved_references: bool = False
) -> tuple[str | None, str | None, str | None]:
    """Resolve one send target the same way for every caller (model tool, CLI, cron).

    Channel-directory IDs are trusted. Plugin platforms must explicitly parse
    native target syntax; for the model-facing send tool (the default), a
    target that can't be resolved is an error — the model can read the error
    and pick a listed target instead.

    ``pass_unresolved_references=True`` restores the old pass-through behavior for
    callers that have no model in the loop (cron delivering a stored job's
    output, react/unreact on platform-native message ids): if the target
    can't be resolved and the platform is built in, or is a plugin platform
    that declares no parser, the string is handed to the adapter exactly as
    written and the adapter decides whether it's valid. A plugin platform
    that DOES declare a parser stays strict for every caller — its parser is
    the authority on native syntax.

    The optional validator has the final say over parser-normalized,
    directory-resolved, and passed-through IDs alike.
    """
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry

    entry = platform_registry.get(platform_name)

    def _validate(candidate: str) -> str | None:
        if entry is None or entry.validate_target_ref_fn is None:
            return None
        try:
            verdict = entry.validate_target_ref_fn(candidate)
        except Exception:
            logger.debug(
                "Plugin target validator failed for %s", platform_name, exc_info=True
            )
            return f"Target validator failed for platform '{platform_name}'"
        if verdict is True:
            return None
        if isinstance(verdict, str) and verdict:
            return f"Invalid target '{target_ref}' on {platform_name}: {verdict}"
        return f"Invalid target '{target_ref}' on {platform_name}"

    if entry is not None and entry.parse_target_ref_fn is not None:
        try:
            parsed = entry.parse_target_ref_fn(target_ref)
        except Exception:
            logger.debug(
                "Plugin target parser failed for %s", platform_name, exc_info=True
            )
            return None, None, f"Target parser failed for platform '{platform_name}'"
        if parsed is not None:
            if (
                not isinstance(parsed, tuple)
                or len(parsed) != 2
                or not isinstance(parsed[0], str)
                or not parsed[0]
                or (parsed[1] is not None and not isinstance(parsed[1], str))
            ):
                return (
                    None,
                    None,
                    f"Target parser for platform '{platform_name}' returned an invalid result",
                )
            parsed_chat_id, parsed_thread_id = parsed
            error = _validate(parsed_chat_id)
            return (None, None, error) if error else (
                parsed_chat_id,
                parsed_thread_id,
                None,
            )

    parsed_chat_id, parsed_thread_id, explicit = _parse_target_ref(
        platform_name, target_ref
    )
    if explicit and parsed_chat_id is not None:
        error = _validate(parsed_chat_id)
        return (None, None, error) if error else (
            parsed_chat_id,
            parsed_thread_id,
            None,
        )

    resolution_failed = False
    try:
        from gateway.channel_directory import resolve_channel_name

        resolved = resolve_channel_name(platform_name, target_ref)
    except Exception:
        resolved = None
        resolution_failed = True
    if resolved:
        parsed_chat_id, parsed_thread_id, _ = _parse_target_ref(
            platform_name, resolved
        )
        chat_id = parsed_chat_id or resolved
        error = _validate(chat_id)
        return (None, None, error) if error else (
            chat_id,
            parsed_thread_id,
            None,
        )

    is_builtin = platform_name in {member.value for member in Platform}
    if entry is None and not is_builtin:
        return None, None, f"Unknown or unregistered plugin platform: {platform_name}"

    def _pass_through_unresolved():
        """Hand the raw target to the adapter unchanged (it validates)."""
        error = _validate(target_ref)
        if error:
            return None, None, error
        logger.debug(
            "Handing unresolved target '%s' to the %s adapter unchanged "
            "(the adapter validates it)",
            target_ref, platform_name,
        )
        return target_ref, None, None

    if entry is not None and entry.source == "plugin" and not is_builtin:
        if pass_unresolved_references and entry.parse_target_ref_fn is None:
            return _pass_through_unresolved()
        return (
            None,
            None,
            f"Could not resolve '{target_ref}' on {platform_name}. "
            "The plugin parser did not recognize it and no channel-directory entry matched.",
        )
    if pass_unresolved_references:
        return _pass_through_unresolved()
    hint = (
        "Try using a numeric channel ID instead."
        if resolution_failed
        else "Use send_message(action='list') to see available targets."
    )
    return None, None, f"Could not resolve '{target_ref}' on {platform_name}. {hint}"


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


def _bounded_send_error(detail, max_chars=900):
    """Bound untrusted adapter/plugin error detail returned by send_message."""
    text = str(detail or "send failed")
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


async def _send_live_adapter_media(
    adapter,
    chat_id,
    message,
    media_files,
    *,
    thread_id=None,
    metadata=None,
    force_document=False,
):
    """Deliver text and every media descriptor through adapter media APIs."""
    caption, separate_text = _media_caption_split(
        message, media_files, max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    last_result = None
    if separate_text and separate_text.strip():
        last_result = await adapter.send(
            chat_id=chat_id, content=separate_text, metadata=metadata
        )
        if not last_result.success:
            return {"error": f"Adapter send failed: {_bounded_send_error(last_result.error)}"}

    total = len(media_files)
    for index, descriptor in enumerate(media_files):
        if not isinstance(descriptor, (list, tuple)) or not descriptor:
            return {"error": f"Adapter media send failed: invalid media descriptor {index + 1}/{total}"}
        media_path = descriptor[0]
        is_voice = bool(descriptor[1]) if len(descriptor) > 1 else False
        if not isinstance(media_path, str) or not media_path:
            return {"error": f"Adapter media send failed: invalid media descriptor {index + 1}/{total}"}
        if not os.path.exists(media_path):
            return {"error": f"Adapter media send failed: media file {index + 1}/{total} was not found"}

        ext = os.path.splitext(media_path)[1].lower()
        kwargs = {
            "caption": caption if index == 0 else None,
            "reply_to": thread_id,
            "metadata": metadata,
        }
        if force_document:
            method_name = "send_document"
            media_kind = "document"
        elif ext in _IMAGE_EXTS:
            method_name = "send_image_file"
            media_kind = "image"
        elif ext in _VIDEO_EXTS:
            method_name = "send_video"
            media_kind = "video"
        elif is_voice or ext in _AUDIO_EXTS:
            method_name = "send_voice"
            media_kind = "audio"
        else:
            method_name = "send_document"
            media_kind = "document"

        from gateway.platforms.base import BasePlatformAdapter

        adapter_method = getattr(type(adapter), method_name, None)
        base_fallback = getattr(BasePlatformAdapter, method_name)
        if adapter_method is None or adapter_method is base_fallback:
            return {
                "error": (
                    f"Live adapter does not implement native {media_kind} delivery; "
                    f"media file {index + 1}/{total} was not sent"
                )
            }
        try:
            last_result = await getattr(adapter, method_name)(chat_id, media_path, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "error": (
                    f"Adapter media send failed after {index}/{total} files: "
                    f"{_bounded_send_error(exc)}"
                )
            }
        if not last_result.success:
            detail = _bounded_send_error(last_result.error or "media send failed")
            return {
                "error": f"Adapter media send failed after {index}/{total} files: {detail}"
            }

    if last_result is None:
        return {"error": "No deliverable text or media remained after processing MEDIA tags"}
    return {
        "success": True,
        "message_id": last_result.message_id,
        "media_delivered": True,
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
                # The adapter's send() uses asyncio.Queue + worker tasks bound
                # to the gateway's main event loop.  Calling send() from a
                # different thread/loop (the agent's tool worker thread) causes
                # a cross-loop Future deadlock: the worker loop's selector never
                # gets woken when the gateway loop resolves the future.
                # When on a different loop, dispatch onto the gateway loop via
                # run_coroutine_threadsafe and await the wrapped future.
                gateway_loop = getattr(runner, "_gateway_loop", None)
                try:
                    _current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    _current_loop = None

                _need_cross_loop = (
                    gateway_loop is not None
                    and _current_loop is not gateway_loop
                )

                # Media descriptors route through the adapter's native media
                # APIs (same cross-loop rules apply — the media helper awaits
                # adapter methods bound to the gateway loop).
                if media_files:
                    def _media_coro():
                        return _send_live_adapter_media(
                            adapter,
                            chat_id,
                            chunk,
                            media_files,
                            thread_id=thread_id,
                            metadata=metadata,
                            force_document=force_document,
                        )
                    if _need_cross_loop:
                        if not gateway_loop.is_running():
                            return {"error": "Gateway loop is not running; cannot dispatch adapter send"}
                        from agent.async_utils import safe_schedule_threadsafe
                        media_fut = safe_schedule_threadsafe(
                            _media_coro(),
                            gateway_loop,
                            logger=logger,
                            log_message="send_message: failed to schedule media send on gateway loop",
                        )
                        if media_fut is None:
                            return {"error": "Gateway loop unavailable for send dispatch"}
                        return await asyncio.shield(asyncio.wrap_future(media_fut))
                    return await _media_coro()

                if _need_cross_loop:
                    if not gateway_loop.is_running():
                        return {"error": "Gateway loop is not running; cannot dispatch adapter send"}
                    from agent.async_utils import safe_schedule_threadsafe
                    fut = safe_schedule_threadsafe(
                        adapter.send(chat_id=chat_id, content=chunk, metadata=metadata),
                        gateway_loop,
                        logger=logger,
                        log_message="send_message: failed to schedule on gateway loop",
                    )
                    if fut is None:
                        return {"error": "Gateway loop unavailable for send dispatch"}
                    # Use shield so that if the caller's task is cancelled (e.g.
                    # agent interrupt), the already-enqueued send on the gateway
                    # loop is NOT cancelled — preventing "tool failed but message
                    # still sent later" followed by agent retry causing duplicates.
                    # No explicit timeout here: the adapter's internal request
                    # timeout (15s) and the upper-layer _run_async 300s timeout
                    # provide sufficient protection against hangs.
                    result = await asyncio.shield(asyncio.wrap_future(fut))
                else:
                    # Same loop or no gateway loop (CLI, tests) — direct await.
                    result = await adapter.send(chat_id=chat_id, content=chunk, metadata=metadata)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                return {"error": f"Plugin platform send failed: {_bounded_send_error(e)}"}
            if result.success:
                return {"success": True, "message_id": result.message_id}
            return {"error": f"Adapter send failed: {_bounded_send_error(result.error)}"}

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
            return {"error": f"Plugin standalone send failed: {_bounded_send_error(e)}"}

        if isinstance(result, dict) and (result.get("success") or result.get("error")):
            if result.get("error"):
                return {**result, "error": _bounded_send_error(result["error"])}
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


async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None):
    """Route a message to the appropriate platform sender.

    Long messages are automatically chunked to fit within platform limits
    using the same smart-splitting algorithm as the gateway adapters
    (preserves code-block boundaries, adds part indicators).
    """
    from gateway.config import Platform

    platform_name = platform.value if hasattr(platform, "value") else str(platform)

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

    # Signal's standalone path (_send_signal) speaks raw JSON-RPC and does not
    # go through SignalAdapter.send(), so it never benefits from the adapter's
    # native chunking. Register the platform limit here so the shared
    # truncate_message() pass below splits long sends instead of signal-cli
    # rejecting them. Sourced from the adapter module so the two paths can't
    # drift (credit: @5L-hermes01 in #67279, @lkz-de in #57929).
    try:
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH as _SIGNAL_MAX
        _MAX_LENGTHS[Platform.SIGNAL] = _SIGNAL_MAX
    except ImportError:
        _MAX_LENGTHS[Platform.SIGNAL] = 8000

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
    # _send_telegram now owns text chunking internally — it formats the full
    # message (MarkdownV2/HTML) and then splits the *formatted* text on UTF-16
    # length so escaping inflation can't push a chunk over Telegram's 4096
    # limit (issue #28557). Pass the whole message in one call; media attaches
    # after all text chunks.
    if platform == Platform.TELEGRAM:
        disable_link_previews = bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews"))
        return await _send_telegram(
            pconfig.token,
            chat_id,
            message,
            media_files=media_files,
            thread_id=thread_id,
            disable_link_previews=disable_link_previews,
            force_document=force_document,
        )

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
        # MEDIA:<path> caption: single captionable file + short text rides as
        # the media message content instead of a separate message before the
        # attachment (single enforced decision in _media_caption_split). Cap on
        # the platform's own message limit so the caption is always deliverable.
        _dc_caption, _ = _media_caption_split(
            message, media_files,
            max_caption_len=(max_len or _DEFAULT_CAPTION_LIMIT),
        )
        if _dc_caption is not None:
            result = await entry.standalone_sender_fn(
                pconfig,
                chat_id,
                "",
                thread_id=thread_id,
                media_files=media_files,
                caption=_dc_caption,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            return result
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

    # --- Matrix: route ALL sends through the native adapter so text is
    # encrypted in E2EE rooms too (issue: text-only sends arrived with a red
    # padlock because they took the raw-HTTP standalone path). The adapter
    # reuses the live gateway's E2EE session when available (#46310) and falls
    # back to an encryption-aware ephemeral adapter for standalone/cron. ---
    if platform == Platform.MATRIX:
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

    # --- Slack: native media via files_upload_v2 in the plugin's
    # standalone_sender_fn (plugins/platforms/slack/adapter.py::_standalone_send).
    # Gateway in-channel MEDIA: delivery already worked; send_message previously
    # omitted Slack attachments and told the model media was unsupported.
    if platform == Platform.SLACK and media_files:
        from gateway.platform_registry import platform_registry as _pr_slack
        from hermes_cli.plugins import discover_plugins as _dp_slack
        _dp_slack()
        _slack_entry = _pr_slack.get("slack")
        if _slack_entry is None or _slack_entry.standalone_sender_fn is None:
            return {"error": "Slack plugin not registered or missing standalone_sender_fn"}
        _sl_caption, _ = _media_caption_split(
            message, media_files,
            max_caption_len=(max_len or _DEFAULT_CAPTION_LIMIT),
        )
        if _sl_caption is not None:
            result = await _slack_entry.standalone_sender_fn(
                pconfig,
                chat_id,
                "",
                thread_id=thread_id,
                media_files=media_files,
                caption=_sl_caption,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            return result
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _slack_entry.standalone_sender_fn(
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
        # MEDIA:<path> caption: a single captionable file + short text rides
        # as the media's native caption instead of a separate message before
        # the bubble (single enforced decision in _media_caption_split). Cap on
        # the platform's own message limit so the caption is always deliverable.
        _wa_caption, _ = _media_caption_split(
            message, media_files,
            max_caption_len=(max_len or _DEFAULT_CAPTION_LIMIT),
        )
        last_result = None
        if _wa_caption is not None:
            # Single-file captioned send: no separate text chunk, caption on
            # the media itself.
            result = await _wa_entry.standalone_sender_fn(
                pconfig,
                chat_id,
                "",
                media_files=media_files,
                thread_id=thread_id,
                force_document=force_document,
                caption=_wa_caption,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            return result
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

    # --- Slack: prefer the live gateway adapter, then the plugin's
    # standalone sender.  The live adapter is multi-workspace aware (it maps
    # channels to the workspace client that owns them) and honors adapter-side
    # gates like ignored_channels; the standalone Web-API path may only have a
    # comma-separated token list.  ``_send_via_adapter`` tries the in-process
    # adapter first and falls back to the registry standalone sender for
    # out-of-process cron runs, preserving MEDIA delivery on the fallback
    # (media-bearing sends were already intercepted by the branch above).
    if platform == Platform.SLACK:
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            result = await _send_via_adapter(
                platform,
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files if is_last else [],
                force_document=force_document,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- WeCom: native media attachment support via live gateway adapter ---
    if platform == Platform.WECOM and media_files:
        last_result = None
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = await _send_via_adapter(
                platform,
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files if is_last else None,
                force_document=force_document,
            )
            if isinstance(result, dict) and result.get("error"):
                return result
            last_result = result
        return last_result

    # --- Non-media platforms ---
    # Buzz is a plugin platform with verified native media delivery through
    # _send_via_adapter below, including valid media-only sends.
    if media_files and not message.strip() and platform.value != "buzz":
        return {
            "error": (
                f"send_message MEDIA delivery is currently only supported for telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack; "
                f"target {platform.value} had only media attachments"
            )
        }
    warning = None
    if media_files and platform.value != "buzz":
        warning = (
            f"MEDIA attachments were omitted for {platform.value}; "
            "native send_message media delivery is currently only supported for telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack"
        )

    last_result = None
    for i, chunk in enumerate(chunks):
        if platform == Platform.WHATSAPP:
            result = await _registry_standalone_send("whatsapp", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.SIGNAL:
            result = await _send_signal(pconfig.extra, chat_id, chunk)
        elif platform == Platform.EMAIL:
            result = await _registry_standalone_send("email", pconfig, chat_id, chunk, thread_id)
        elif platform == Platform.SMS:
            result = await _registry_standalone_send("sms", pconfig, chat_id, chunk, thread_id)
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
            from gateway.platform_registry import platform_registry

            entry = platform_registry.get(platform_name)
            handler = entry.send_message_handler if entry is not None else None
            if handler is not None:
                try:
                    import inspect

                    result = handler(args or {}, chat_id, platform_name, pconfig)
                    if inspect.isawaitable(result):
                        result = await result
                    return result
                except Exception as e:
                    return {"error": f"Plugin send_message handler failed: {e}"}
            # Plugin platform: route through the gateway's live adapter if
            # available, otherwise the plugin's standalone_sender_fn.
            result = await _send_via_adapter(
                platform,
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files if i == len(chunks) - 1 else [],
                force_document=force_document,
            )

        if isinstance(result, dict) and result.get("error"):
            return result
        last_result = result

    if (
        warning
        and isinstance(last_result, dict)
        and last_result.get("success")
        and not last_result.get("media_delivered")
    ):
        warnings = list(last_result.get("warnings", []))
        warnings.append(warning)
        last_result["warnings"] = warnings
    return last_result
