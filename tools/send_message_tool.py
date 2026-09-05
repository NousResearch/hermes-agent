"""Send Message Tool -- cross-channel messaging via platform APIs (send, list targets,
react); works in both CLI and gateway contexts."""

import asyncio
import json
import logging
import os
from functools import partial

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

from tools.send_message_targets import _HOME_CHANNEL_ENV_OVERRIDES, _SLACK_USER_ID_RE, resolve_send_target
from tools.send_message_senders import (
    _AUDIO_EXTS, _DEFAULT_CAPTION_LIMIT, _IMAGE_EXTS, _NO_DELIVERABLE, _VIDEO_EXTS, _VOICE_EXTS,
    _adapter_media_method, _error, _live_adapter, _media_caption_split, _plugin_standalone_sender,
    _registry_standalone_send, _resolve_slack_user_target, _sanitize_error_text, _send_bluebubbles,
    _send_matrix_via_adapter, _send_qqbot, _send_signal, _send_telegram, _send_weixin, _send_yuanbao)
from tools.registry import tool_error

# NOTE: ``send_message`` is intentionally NOT registered as an agent-callable model tool
# (the agent must not fire cross-platform messages on its own); cron delivery, the
# ``hermes send`` CLI, the kanban notifier and the opt-in MCP server import the helpers.


def prepare_send_message_platforms() -> None:
    """Load enabled standalone plugins before tool schemas/cache keys are built."""
    from hermes_cli.plugins import discover_plugins
    discover_plugins()


def send_message_tool(args, **kw):
    """Handle cross-channel send_message tool calls."""
    action = args.get("action", "send")
    if action == "list":
        return _handle_list()
    if action in ("react", "unreact"):
        return _handle_react(args, remove=action == "unreact")
    return _handle_send(args)


def _resolve_tool_target(target: str, *, pass_unresolved_references: bool = False):
    """``(platform_name, chat_id, thread_id, error)``; ``chat_id`` is None when no ref was given
    (caller falls back to the home channel)."""
    platform_name, _, target_ref = target.partition(":")
    platform_name, target_ref = platform_name.strip().lower(), target_ref.strip() or None
    prepare_send_message_platforms()
    if not target_ref:
        return platform_name, None, None, None
    return platform_name, *resolve_send_target(platform_name, target_ref,
                                               pass_unresolved_references=pass_unresolved_references)


def _handle_list():
    try:
        from gateway.channel_directory import format_directory_for_display
        return json.dumps({"targets": format_directory_for_display()})
    except Exception as e:
        return json.dumps(_error(f"Failed to load channel directory: {e}"))


def _handle_react(args, remove=False):
    """Attach (``remove=True``: retract) an emoji reaction via the live gateway adapter; no
    standalone fallback because reacting needs the adapter's live message-id state."""
    target, emoji = args.get("target", ""), (args.get("emoji") or "").strip()
    message_id = (args.get("message_id") or "").strip() or None
    if not target or (not remove and not emoji):
        return tool_error("'target' is required when action='unreact'" if remove
                          else "Both 'target' and 'emoji' are required when action='react'")

    # Platform-native ids (e.g. photon GUIDs) match no parser/directory entry; the adapter validates.
    platform_name, chat_id, _thread_id, resolution_error = _resolve_tool_target(target, pass_unresolved_references=True)
    if resolution_error:
        return tool_error(resolution_error)
    platform, err = _platform_enum(platform_name)
    if err:
        return tool_error(err)
    if not chat_id:
        try:
            from gateway.config import load_gateway_config
            chat_id = load_gateway_config().get_home_channel(platform).chat_id
        except Exception:
            return tool_error(f"No chat specified and no home channel set for {platform_name}. "
                              f"Use '{platform_name}:chat_id'.")
    _, adapter = _live_adapter(platform)
    if adapter is None:
        return tool_error(f"Reactions require a live {platform_name} adapter in the running "
                          "gateway (not available from cron/standalone contexts).")
    react_fn = getattr(adapter, "remove_reaction" if remove else "add_reaction", None)
    if not callable(react_fn):
        return tool_error(f"Platform '{platform_name}' does not support message reactions.")
    try:
        from model_tools import _run_async
        result = _run_async(react_fn(chat_id=chat_id, message_id=message_id, **({} if remove else {"emoji": emoji})))
    except Exception as e:
        return json.dumps(_error(f"Reaction failed: {e}"))
    return json.dumps(result if isinstance(result, dict) else {"success": bool(result)})


def _handle_send(args):
    target, message = args.get("target", ""), args.get("message", "")
    if not target or not message:
        return tool_error("Both 'target' and 'message' are required when action='send'")
    platform_name, chat_id, thread_id, resolution_error = _resolve_tool_target(target)
    if resolution_error:
        return tool_error(resolution_error)
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return tool_error("Interrupted")
    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
    except Exception as e:
        return json.dumps(_error(f"Failed to load gateway config: {e}"))
    platform, pconfig, entry, err = _resolve_platform_config(platform_name, config)
    if err:
        return tool_error(err)
    from gateway.platforms.base import BasePlatformAdapter
    # Capture [[as_document]] before extract_media strips it (images keep original bytes via send_document).
    force_document_attachments = "[[as_document]]" in message
    media_files, cleaned_message = BasePlatformAdapter.extract_media(message)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)
    used_home_channel = not chat_id
    if used_home_channel:
        chat_id, err = _home_chat_id(config, platform, platform_name)
        if err:
            return tool_error(err)
    if duplicate_skip := _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id):
        return json.dumps(duplicate_skip)
    # Slack: resolve user targets to DM channel IDs before sending. _parse_target_ref emits internal
    # ``user:U...`` / ``user_name:@handle`` targets; a bare U... id can also arrive from session metadata or
    # the home-channel config. All are opened via conversations.open (fixes #19236).
    if platform_name == "slack" and chat_id:
        chat_id, resolve_err = _slack_dm_chat_id(pconfig, chat_id)
        if resolve_err:
            return json.dumps(resolve_err)
    try:
        from model_tools import _run_async
        # Only custom plugin handlers receive the complete typed request.
        handler_args = {"args": args} if entry is not None and entry.send_message_handler is not None else {}
        result = _run_async(_send_to_platform(platform, pconfig, chat_id, cleaned_message, thread_id=thread_id,
                                              media_files=media_files, force_document=force_document_attachments,
                                              **handler_args))
        if isinstance(result, dict) and result.get("success"):
            if used_home_channel:
                result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"
            if mirror_text and _mirror_sent_message(platform_name, chat_id, mirror_text, thread_id):
                result["mirrored"] = True
        if isinstance(result, dict) and "error" in result:
            result["error"] = _sanitize_error_text(result["error"])
        return json.dumps(result)
    except Exception as e:
        return json.dumps(_error(f"Send failed: {e}"))


def _platform_enum(platform_name):
    """``(Platform, None)`` or ``(None, error)`` for a platform name."""
    from gateway.config import Platform
    try:
        return Platform(platform_name), None
    except (ValueError, KeyError):
        return None, f"Unknown platform: {platform_name}"


def _resolve_platform_config(platform_name, config):
    """``(platform, pconfig, registry_entry, error)``. Plugin platforms must be registered;
    disabled/missing platforms error, except Weixin, which may be configured purely via .env."""
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    entry = platform_registry.get(platform_name)
    if entry is None and platform_name not in {member.value for member in Platform}:
        return None, None, None, f"Unknown or unregistered plugin platform: {platform_name}"
    platform, err = _platform_enum(platform_name)
    if err:
        return None, None, None, err
    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        pconfig = _weixin_env_pconfig() if platform_name == "weixin" else None
    if pconfig is None:
        return None, None, None, (f"Platform '{platform_name}' is not configured. Set up credentials in "
                                  "~/.hermes/config.yaml or environment variables.")
    return platform, pconfig, entry, None


def _home_chat_id(config, platform, platform_name):
    """``(home chat_id, None)`` or ``(None, actionable error)``; Weixin also honours WEIXIN_HOME_CHANNEL."""
    home = config.get_home_channel(platform)
    if home:
        return home.chat_id, None
    wx_home = os.getenv("WEIXIN_HOME_CHANNEL", "").strip() if platform_name == "weixin" else ""
    if wx_home:
        return wx_home, None
    home_env = _HOME_CHANNEL_ENV_OVERRIDES.get(platform_name, f"{platform_name.upper()}_HOME_CHANNEL")
    return None, (f"No home channel set for {platform_name} to determine where to send the message. "
                  f"Either specify a channel directly with '{platform_name}:CHANNEL_NAME', "
                  f"or set a home channel via: hermes config set {home_env} <channel_id>")


def _slack_dm_chat_id(pconfig, chat_id):
    """Open Slack user targets (``user:``/``user_name:`` from the parser, or a bare U... id from
    session metadata / home-channel config) as DM conversations. ``(chat_id, None)`` or ``(None, error_dict)``."""
    dm_target = f"user:{chat_id}" if chat_id.startswith("U") and _SLACK_USER_ID_RE.fullmatch(chat_id) else chat_id
    if not dm_target.startswith(("user:", "user_name:")):
        return chat_id, None
    from model_tools import _run_async
    return _run_async(_resolve_slack_user_target(pconfig.token, dm_target))


def _mirror_sent_message(platform_name, chat_id, mirror_text, thread_id):
    """Best-effort mirror of the sent message into the target's gateway session."""
    try:
        from gateway.mirror import mirror_to_session
        from gateway.session_context import get_session_env
        return bool(mirror_to_session(
            platform_name, chat_id, mirror_text, thread_id=thread_id,
            source_label=get_session_env("HERMES_SESSION_PLATFORM", "cli"),
            user_id=get_session_env("HERMES_SESSION_USER_ID", "") or None))
    except Exception:
        return False


def _weixin_env_pconfig():
    """Synthesize a Weixin PlatformConfig from .env secrets, or None."""
    wx_token = get_secret("WEIXIN_TOKEN", "").strip()
    wx_account = get_secret("WEIXIN_ACCOUNT_ID", "").strip()
    if not (wx_token and wx_account):
        return None
    from gateway.config import PlatformConfig
    return PlatformConfig(enabled=True, token=wx_token, extra={
        "account_id": wx_account, "base_url": get_secret("WEIXIN_BASE_URL", "").strip(),
        "cdn_base_url": get_secret("WEIXIN_CDN_BASE_URL", "").strip()})


def _describe_media_for_mirror(media_files):
    """Return a human-readable mirror summary when a message only contains media."""
    if not media_files:
        return ""
    if len(media_files) != 1:
        return f"[Sent {len(media_files)} media attachments]"
    media_path, is_voice = media_files[0]
    ext = os.path.splitext(media_path)[1].lower()
    if is_voice and ext in _VOICE_EXTS:
        return "[Sent voice message]"
    kind = next((k for exts, k in ((_IMAGE_EXTS, "image"), (_VIDEO_EXTS, "video"), (_AUDIO_EXTS, "audio"))
                 if ext in exts), "document")
    return f"[Sent {kind} attachment]"


def _maybe_skip_cron_duplicate_send(platform_name: str, chat_id: str, thread_id: str | None):
    """Skip redundant cron send_message calls when the scheduler will auto-deliver there."""
    from gateway.session_context import get_session_env
    auto_platform = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip().lower()
    auto_chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
    if not (auto_platform and auto_chat_id and auto_platform == platform_name and auto_chat_id == str(chat_id)
            and (get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip() or None) == thread_id):
        return None
    target_label = f"{platform_name}:{chat_id}" + (f":{thread_id}" if thread_id is not None else "")
    return {"success": True, "skipped": True, "reason": "cron_auto_delivery_duplicate_target", "target": target_label,
        "note": (f"Skipped send_message to {target_label}. This cron job will already auto-deliver "
                 "its final response to that same target. Put the intended user-facing content in "
                 "your final response instead, or use a different target if you want an additional message.")}


def _bounded_send_error(detail, max_chars=900):
    """Bound untrusted adapter/plugin error detail returned by send_message."""
    text = str(detail or "send failed")
    return text if len(text) <= max_chars else f"{text[: max_chars - 3]}..."


async def _send_live_adapter_media(adapter, chat_id, message, media_files, *, thread_id=None, metadata=None,
                                   force_document=False):
    """Deliver text and every media descriptor through adapter media APIs; adapters that only
    inherit the BasePlatformAdapter stub for a kind are unsupported, not no-op'd."""
    caption, separate_text = _media_caption_split(message, media_files, max_caption_len=_DEFAULT_CAPTION_LIMIT)
    last_result = None
    if separate_text and separate_text.strip():
        last_result = await adapter.send(chat_id=chat_id, content=separate_text, metadata=metadata)
        if not last_result.success:
            return {"error": f"Adapter send failed: {_bounded_send_error(last_result.error)}"}
    from gateway.platforms.base import BasePlatformAdapter
    total = len(media_files)
    for index, descriptor in enumerate(media_files):
        media_path = descriptor[0] if isinstance(descriptor, (list, tuple)) and descriptor else None
        if not isinstance(media_path, str) or not media_path:
            return {"error": f"Adapter media send failed: invalid media descriptor {index + 1}/{total}"}
        is_voice = len(descriptor) > 1 and bool(descriptor[1])
        if not os.path.exists(media_path):
            return {"error": f"Adapter media send failed: media file {index + 1}/{total} was not found"}
        ext = os.path.splitext(media_path)[1].lower()
        method_name, media_kind = _adapter_media_method(ext, is_voice or ext in _AUDIO_EXTS, force_document)
        adapter_method = getattr(type(adapter), method_name, None)
        if adapter_method is None or adapter_method is getattr(BasePlatformAdapter, method_name):
            return {"error": (f"Live adapter does not implement native {media_kind} delivery; "
                              f"media file {index + 1}/{total} was not sent")}
        try:
            last_result = await getattr(adapter, method_name)(
                chat_id, media_path, caption=caption if index == 0 else None, reply_to=thread_id, metadata=metadata)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = _bounded_send_error(exc)
        else:
            if last_result.success:
                continue
            detail = _bounded_send_error(last_result.error or "media send failed")
        return {"error": f"Adapter media send failed after {index}/{total} files: {detail}"}
    if last_result is None:
        return {"error": _NO_DELIVERABLE}
    return {"success": True, "message_id": last_result.message_id, "media_delivered": True}


async def _dispatch_on_gateway_loop(runner, make_coro, log_message):
    """Await ``make_coro()`` on the gateway's loop: adapter.send() uses queues/tasks bound to it,
    so awaiting from another loop (the tool worker thread) deadlocks."""
    gateway_loop = getattr(runner, "_gateway_loop", None)
    if gateway_loop is None or asyncio.get_running_loop() is gateway_loop:
        return await make_coro()  # same loop / no gateway loop (CLI, tests)
    if not gateway_loop.is_running():
        return {"error": "Gateway loop is not running; cannot dispatch adapter send"}
    from agent.async_utils import safe_schedule_threadsafe
    fut = safe_schedule_threadsafe(make_coro(), gateway_loop, logger=logger, log_message=log_message)
    if fut is None:
        return {"error": "Gateway loop unavailable for send dispatch"}
    # shield: a cancelled caller must not cancel the enqueued send (a retry would duplicate it).
    # No timeout: the adapter and outer _run_async bound the wait.
    return await asyncio.shield(asyncio.wrap_future(fut))


async def _send_via_adapter(platform, pconfig, chat_id, chunk, *, thread_id=None, media_files=None,
                            force_document=False, subject=None):
    """Live in-process gateway adapter first, else the plugin's ``standalone_sender_fn`` (cron),
    else an error naming both; media uses the adapter's native media APIs under the same rules."""
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    runner, adapter = _live_adapter(platform)
    if adapter is not None:
        try:
            metadata = {**({"thread_id": thread_id} if thread_id else {}),
                        **({"publish_topic": chat_id} if platform_name == "ntfy" and chat_id else {}),
                        **({"subject": subject} if subject else {})} or None
            if media_files:  # always a dict result, returned as-is below
                make_coro = lambda: _send_live_adapter_media(  # noqa: E731
                    adapter, chat_id, chunk, media_files, thread_id=thread_id, metadata=metadata,
                    force_document=force_document)
            else:
                make_coro = lambda: adapter.send(chat_id=chat_id, content=chunk, metadata=metadata)  # noqa: E731
            result = await _dispatch_on_gateway_loop(
                runner, make_coro, f"send_message: failed to schedule{' media send' if media_files else ''} on gateway loop")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"error": f"Plugin platform send failed: {_bounded_send_error(e)}"}
        if isinstance(result, dict):
            return result
        if result.success:
            return {"success": True, "message_id": result.message_id}
        return {"error": f"Adapter send failed: {_bounded_send_error(result.error)}"}
    try:
        from gateway.platform_registry import platform_registry
        sender = platform_registry.get(platform_name).standalone_sender_fn
    except Exception:
        sender = None
    if sender is None:
        return {"error": (f"No live adapter for platform '{platform_name}'. Is the gateway running with this platform "
                          f"connected? For out-of-process delivery (e.g. cron in a separate process), the platform "
                          f"plugin must register a standalone_sender_fn on its PlatformEntry.")}
    try:
        import inspect as _inspect
        _sender_params = _inspect.signature(sender).parameters
        _sender_extra = {"subject": subject} if subject and "subject" in _sender_params else {}
        result = await sender(pconfig, chat_id, chunk, thread_id=thread_id, media_files=media_files,
                              force_document=force_document, **_sender_extra)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Plugin standalone send for %s raised", platform_name, exc_info=True)
        return {"error": f"Plugin standalone send failed: {_bounded_send_error(e)}"}
    if isinstance(result, dict) and (result.get("success") or result.get("error")):
        return {**result, "error": _bounded_send_error(result["error"])} if result.get("error") else result
    return {"error": (f"Plugin standalone send for '{platform_name}' returned an invalid result: "
                      f"expected a dict with 'success' or 'error' keys, got {type(result).__name__}")}


async def _send_chunks(chunks, send_one):
    """``send_one(chunk, is_last)`` in order; stop at the first error dict, else last result."""
    result = None
    # --- Matrix: route ALL sends through the native adapter so text is encrypted in E2EE rooms too (issue:
    # text-only sends arrived with a red padlock because they took the raw-HTTP standalone path). The
    # adapter reuses the live gateway's E2EE session when available (#46310) and falls back to an
    # encryption-aware ephemeral adapter for standalone/cron. ---
    for i, chunk in enumerate(chunks):
        result = await send_one(chunk, i == len(chunks) - 1)
        if isinstance(result, dict) and result.get("error"):
            break
    return result


def _platform_max_length(platform):
    """Chunking limit: Signal's adapter constant (its raw JSON-RPC path bypasses the adapter's
    chunking), the registry's ``max_message_length`` for plugins, else None (no chunking)."""
    from gateway.config import Platform
    if platform == Platform.SIGNAL:
        try:
            from gateway.platforms.signal import MAX_MESSAGE_LENGTH
            return MAX_MESSAGE_LENGTH
        except ImportError:
            return 8000
    try:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform.value)
        return entry.max_message_length if entry and entry.max_message_length > 0 else None
    except Exception:
        return None


# Plugin platforms whose media (Discord: all) sends deliberately bypass the live adapter for the
# registry ``standalone_sender_fn`` (Discord: forums/threads/multipart; Slack: files_upload_v2;
# WhatsApp: Baileys /send-media). platform -> (error label, run discover_plugins first,
# caption-capable, media_files sentinel for non-final chunks, forward force_document)
_PLUGIN_STANDALONE_MEDIA = {"discord": ("Discord", False, True, [], False), "feishu": ("Feishu", True, False, None, False),
                            "slack": ("Slack", True, True, [], False), "whatsapp": ("WhatsApp", True, True, None, True)}


async def _send_plugin_standalone(platform_name, pconfig, chat_id, message, chunks, media_files, *, thread_id,
                                  max_len, force_document):
    """Chunked send through a plugin's standalone_sender_fn; one captionable file + short text
    rides as the media caption."""
    label, discover, captionable, empty_media, pass_force = _PLUGIN_STANDALONE_MEDIA[platform_name]
    sender, err = _plugin_standalone_sender(platform_name, label=label, discover=discover)
    if err:
        return err
    extra = {"force_document": force_document} if pass_force else {}
    if captionable:
        # Cap on the platform's own message limit so the caption is deliverable.
        caption, _ = _media_caption_split(message, media_files, max_caption_len=(max_len or _DEFAULT_CAPTION_LIMIT))
        if caption is not None:
            return await sender(pconfig, chat_id, "", thread_id=thread_id, media_files=media_files,
                                caption=caption, **extra)
    return await _send_chunks(chunks, lambda chunk, is_last: sender(
        pconfig, chat_id, chunk, thread_id=thread_id, media_files=media_files if is_last else empty_media, **extra))


def _via_adapter_route(p, pc, cid, chunk, media, tid, fd, subject=None):
    return _send_via_adapter(p, pc, cid, chunk, thread_id=tid, media_files=media, force_document=fd,
                             subject=subject)


# Native-media chunked routes for built-in platforms; media rides on the final chunk, non-final
# chunks get the sentinel. platform -> (media required, sentinel, sender(platform, pconfig,
# chat_id, chunk, media, thread_id, force_document)). Matrix: ALL sends use the native adapter
# (E2EE text). Signal: attachments ride the JSON-RPC param. Yuanbao / WeCom: media needs the
# running gateway. Slack text: live adapter (multi-workspace, ignored_channels gates) else the
# plugin's standalone sender. Names resolve at call time so tests can monkeypatch ``_send_signal``.
_CHUNKED_ROUTES = {
    "matrix": (False, [], lambda p, pc, cid, chunk, media, tid, fd: _send_matrix_via_adapter(
        pc, cid, chunk, media_files=media, thread_id=tid)),
    "signal": (True, [], lambda p, pc, cid, chunk, media, tid, fd: _send_signal(
        pc.extra, cid, chunk, media_files=media)),
    "yuanbao": (True, None, lambda p, pc, cid, chunk, media, tid, fd: _send_yuanbao(cid, chunk, media_files=media)),
    "slack": (False, [], _via_adapter_route),
    "wecom": (True, None, _via_adapter_route)}

# Text-only senders for built-in platforms (generic path; media is dropped with a
# warning). Signature: (pconfig, chat_id, chunk, thread_id) -> result.
_TEXT_SENDERS = {
    **{name: partial(_registry_standalone_send, name)
       for name in ("whatsapp", "email", "sms", "dingtalk", "feishu", "wecom")},
    "signal": lambda pc, cid, chunk, tid: _send_signal(pc.extra, cid, chunk),
    "bluebubbles": lambda pc, cid, chunk, tid: _send_bluebubbles(pc.extra, cid, chunk),
    "qqbot": lambda pc, cid, chunk, tid: _send_qqbot(pc, cid, chunk),
    "yuanbao": lambda pc, cid, chunk, tid: _send_yuanbao(cid, chunk)}

_MEDIA_PLATFORMS_NOTE = "telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack"


async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None,
                      subject=None):
    """Route to the platform sender, chunking long text with the adapters' splitter. Order matters:
    Weixin first (its native helper must not be blocked by unrelated optional imports such as
    lark-oapi), Telegram (chunks itself), plugin standalone media, native chunked, generic text."""
    from gateway.config import Platform
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    media_files = media_files or []
    if platform == Platform.WEIXIN:
        return await _send_weixin(pconfig, chat_id, message, media_files=media_files)
    # Telegram chunks internally on the *formatted* text (escaping inflates length).
    if platform == Platform.TELEGRAM:
        return await _send_telegram(
            pconfig.token, chat_id, message, media_files=media_files, thread_id=thread_id, force_document=force_document,
            disable_link_previews=bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews")))
    from gateway.platforms.base import BasePlatformAdapter
    max_len = _platform_max_length(platform)
    chunks = BasePlatformAdapter.truncate_message(message, max_len) if max_len else [message]
    if platform_name == "discord" or (media_files and platform_name in _PLUGIN_STANDALONE_MEDIA):
        return await _send_plugin_standalone(platform_name, pconfig, chat_id, message, chunks, media_files,
                                             thread_id=thread_id, max_len=max_len, force_document=force_document)
    route = _CHUNKED_ROUTES.get(platform_name)
    if route is not None and (media_files or not route[0]):
        _, empty_media, sender = route
        return await _send_chunks(chunks, lambda chunk, is_last: sender(
            platform, pconfig, chat_id, chunk, media_files if is_last else empty_media, thread_id, force_document))

    # Generic path: text only. Buzz delivers media natively via _send_via_adapter, so no warning.
    warning = None
    if media_files and platform_name != "buzz":
        if not message.strip():
            return {"error": (f"send_message MEDIA delivery is currently only supported for {_MEDIA_PLATFORMS_NOTE}; "
                              f"target {platform_name} had only media attachments")}
        warning = (f"MEDIA attachments were omitted for {platform_name}; "
                   f"native send_message media delivery is currently only supported for {_MEDIA_PLATFORMS_NOTE}")
    text_sender = _TEXT_SENDERS.get(platform_name)
    if text_sender is not None:
        send_one = lambda chunk, is_last: text_sender(pconfig, chat_id, chunk, thread_id)  # noqa: E731
    else:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name)
        if entry is not None and entry.send_message_handler is not None:
            # Custom handler receives the full typed request once (not per chunk).
            try:
                import inspect
                result = entry.send_message_handler(args or {}, chat_id, platform_name, pconfig)
                return await result if inspect.isawaitable(result) else result
            except Exception as e:
                return {"error": f"Plugin send_message handler failed: {e}"}
        # Plugin platform: live gateway adapter if available, else standalone_sender_fn.
        send_one = lambda chunk, is_last: _via_adapter_route(  # noqa: E731
            platform, pconfig, chat_id, chunk, media_files if is_last else [], thread_id, force_document,
            subject)
    last_result = await _send_chunks(chunks, send_one)
    if (warning and isinstance(last_result, dict) and last_result.get("success")
            and not last_result.get("media_delivered")):
        last_result["warnings"] = [*last_result.get("warnings", []), warning]
    return last_result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import re  # noqa: F401,E402
import time  # noqa: F401,E402

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


_PLUGIN_COMPAT_LAZY = {
    'redact_sensitive_text': ('agent.redact', 'redact_sensitive_text'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
