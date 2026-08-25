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


_TOKEN_UNSET = object()


def _authorize_relay_target(platform_name: str, chat_id, thread_id=None, *,
                            native_token=_TOKEN_UNSET) -> str | None:
    """Relay egress-authorization guard (P5a); None when the send may proceed.

    Thin delegate to ``gateway.relay.egress`` so the tool keeps working in
    environments where the gateway package can't be imported.

    THE TWO FAILURES ARE NOT THE SAME, and conflating them disabled the
    boundary. A missing gateway module means there is no relay egress to
    authorize, so proceeding is correct. A fault INSIDE the guard means
    authorization did not happen — and returning None there means "authorized",
    so a single runtime bug in the guard silently switched the whole P5(a)
    boundary off. Review found this by making the guard raise and watching the
    send go through.

    So: the import is tolerated, the CALL is not. A guard that cannot answer
    refuses, which is the only safe polarity for an authorization check.
    """
    try:
        from gateway.relay.egress import authorize_relay_target
    except ImportError as exc:
        # ABSENCE ONLY, and absence means the gateway relay module ITSELF is
        # missing — `exc.name` says which module was not found. An ImportError
        # naming a NESTED dependency is a broken installation, i.e. a fault,
        # and returning None here means "authorized". Review probed exactly
        # that (`ImportError.name = "gateway.relay.dependency"`) and got an
        # authorized verdict, so `except ImportError` alone was still fail-open.
        # ABSENCE has one shape and it is checkable: a genuinely missing module
        # raises ModuleNotFoundError with `.name` set to the module that was not
        # found (verified: `import gateway.relay.x` -> ModuleNotFoundError,
        # name="gateway.relay.x"). So a plain ImportError, or a nameless one, is
        # an unattributable FAULT — never proof that there is no relay here.
        # I previously admitted the nameless case to protect the CLI/cron path;
        # that reasoning was wrong, because that path does not produce one.
        _missing = getattr(exc, "name", None)
        if not isinstance(exc, ModuleNotFoundError) or _missing not in (
            "gateway",
            "gateway.relay",
            "gateway.relay.egress",
        ):
            logger.exception(
                "relay egress module failed to import for %s — refusing the send",
                platform_name,
            )
            return (
                f"Refusing to send to relay target '{platform_name}': the egress "
                "authorization module could not be loaded, so this destination "
                "could not be verified."
            )
        logger.debug("relay target authorization unavailable", exc_info=True)
        return None
    except Exception:  # noqa: BLE001 - the module is THERE and broke; FAIL CLOSED
        logger.exception(
            "relay egress module failed to import for %s — refusing the send",
            platform_name,
        )
        return (
            f"Refusing to send to relay target '{platform_name}': the egress "
            "authorization module could not be loaded, so this destination "
            "could not be verified."
        )

    try:
        # ONE SNAPSHOT. `native_token` is the token from the SAME pconfig the
        # dispatch below will actually send with. Letting the guard reload
        # config independently allowed a transition where authorization saw a
        # connector-only setup (exemption granted) while dispatch still held a
        # native token and sent the unattested handle itself.
        if native_token is _TOKEN_UNSET:
            # A caller that forgets the snapshot must NOT silently look like
            # "no native token", which would grant the @handle exemption.
            return authorize_relay_target(platform_name, chat_id, thread_id)
        return authorize_relay_target(
            platform_name, chat_id, thread_id, native_token=native_token
        )
    except Exception:  # noqa: BLE001 - the guard faulted; FAIL CLOSED
        logger.exception(
            "relay target authorization FAILED for %s — refusing the send",
            platform_name,
        )
        return (
            f"Refusing to send to relay target '{platform_name}': the egress "
            "authorization check failed, so this destination could not be "
            "verified. This is a bug — the send was blocked rather than "
            "allowed through unchecked."
        )


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
    # P5(a): same egress-authorization floor as the send path — a reaction is
    # an outbound act against a named destination, so an unattested relay
    # target must be refused here too, not just on `send`.
    # The react path has no pconfig snapshot of its own; it dispatches through
    # the LIVE adapter below, never through a native token, so the guard does
    # its own credential probe here.
    _relay_denial = _authorize_relay_target(platform_name, chat_id, _thread_id)
    if _relay_denial:
        return tool_error(_relay_denial)

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
    # POSITION IS LOAD-BEARING — this must stay BELOW Slack user→DM resolution.
    # `_parse_target_ref` emits internal pseudo-ids (`user_name:ben`,
    # `user:U...`) that no provenance can ever contain, because provenances
    # record RESOLVED conversation ids. Authorizing above the resolver compared
    # a handle against a set of `D...` ids and refused every Slack DM — a fix
    # that caused the outage it was meant to prevent. Pinned by
    # test_slack_user_targets_resolve_then_authorize; moving this call back up
    # turns those cases red.
    # thread_id is part of the DESTINATION: on Discord the thread is the literal
    # REST target, so an attested parent must not vouch for an arbitrary thread.
    _relay_denial = _authorize_relay_target(platform_name, chat_id, thread_id,
                                            native_token=getattr(pconfig, "token", None))
    if _relay_denial:
        return tool_error(_relay_denial)

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
    # Home channel is a per-profile target like the token beside it: a raw environ read would post a
    # multiplexed secondary's message into the default profile's Weixin chat.
    wx_home = (get_secret("WEIXIN_HOME_CHANNEL", "") or "").strip() if platform_name == "weixin" else ""
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
                            force_document=False):
    """Live in-process gateway adapter first, else the plugin's ``standalone_sender_fn`` (cron),
    else an error naming both; media uses the adapter's native media APIs under the same rules."""
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    runner, adapter = _live_adapter(platform)
    if adapter is not None:
        try:
            metadata = {**({"thread_id": thread_id} if thread_id else {}),
                        **({"publish_topic": chat_id} if platform_name == "ntfy" and chat_id else {})} or None
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
        result = await sender(pconfig, chat_id, chunk, thread_id=thread_id, media_files=media_files,
                              force_document=force_document)
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


def _via_adapter_route(p, pc, cid, chunk, media, tid, fd):
    return _send_via_adapter(p, pc, cid, chunk, thread_id=tid, media_files=media, force_document=fd)


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


async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None):
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
            platform, pconfig, chat_id, chunk, media_files if is_last else [], thread_id, force_document)
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

    # --- Non-media platforms ---
    if media_files and not message.strip():
        return {
            "error": (
                f"send_message MEDIA delivery is currently only supported for telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack; "
                f"target {platform.value} had only media attachments"
            )
        }
    warning = None
    if media_files:
        warning = (
            f"MEDIA attachments were omitted for {platform.value}; "
            "native send_message media delivery is currently only supported for telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack"
        )

    last_result = None
    for chunk in chunks:
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

        # MEDIA:<path> caption: when a single captionable file is accompanied
        # by short text, attach the text to the media bubble as its native
        # caption instead of sending it as a separate message beforehand
        # (single enforced decision in _media_caption_split). Caption with the
        # *formatted* text so MarkdownV2/HTML styling is preserved, but guard
        # the formatted length against Telegram's 1024 cap — formatting can
        # inflate a raw-<1024 string past it, in which case fall back to a
        # separate body message.
        _tg_caption = None
        from gateway.platforms.base import utf16_len as _utf16_len
        _cap, _ = _media_caption_split(
            message, media_files, max_caption_len=_TELEGRAM_CAPTION_LIMIT
        )
        if _cap is not None and _utf16_len(formatted) <= _TELEGRAM_CAPTION_LIMIT:
            _tg_caption = formatted
            formatted = ""  # suppress the separate text send below

        if formatted.strip():
            # Chunk *after* formatting: MarkdownV2/HTML escaping inflates the
            # text (each escaped char like `!`/`.`/`-` becomes `\!`/`\.`/`\-`),
            # so a message that fit under 4096 UTF-16 units raw can exceed the
            # Telegram limit once formatted and get rejected as "Message is too
            # long". Sizing on the formatted text in UTF-16 units guarantees
            # every chunk is deliverable. (issue #28557)
            from gateway.platforms.base import BasePlatformAdapter, utf16_len

            text_chunks = BasePlatformAdapter.truncate_message(
                formatted, 4096, len_fn=utf16_len
            )
            for chunk in text_chunks:
                try:
                    last_msg = await _send_telegram_message_with_retry(
                        bot,
                        chat_id=int_chat_id, text=chunk,
                        parse_mode=send_parse_mode, **text_kwargs
                    )
                except Exception as md_error:
                    # Thread not found — retry without message_thread_id so the
                    # message still delivers (matching the gateway adapter's
                    # fallback behaviour, issue #27012).
                    if _is_telegram_thread_not_found(md_error) and text_kwargs.get("message_thread_id") is not None:
                        logger.warning(
                            "Thread %s not found in _send_telegram, retrying without message_thread_id",
                            text_kwargs.get("message_thread_id"),
                        )
                        text_kwargs.pop("message_thread_id", None)
                        last_msg = await _send_telegram_message_with_retry(
                            bot,
                            chat_id=int_chat_id, text=chunk,
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
                                plain = _strip_mdv2(chunk)
                            except Exception:
                                plain = chunk
                        else:
                            plain = chunk
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
                # Caption mode suppressed the separate text send; if the file
                # it was meant to caption is gone, deliver the caption text on
                # its own so the words aren't silently lost.
                if _tg_caption is not None and last_msg is None:
                    try:
                        last_msg = await _send_telegram_message_with_retry(
                            bot, chat_id=int_chat_id, text=_tg_caption,
                            parse_mode=send_parse_mode, **text_kwargs
                        )
                        _tg_caption = None  # delivered — don't re-caption a later file
                    except Exception as _cap_err:
                        logger.warning(
                            "Telegram caption-fallback send failed for missing media: %s",
                            _sanitize_error_text(_cap_err),
                        )
                continue

            ext = os.path.splitext(media_path)[1].lower()
            try:
                with open(media_path, "rb") as f:
                    media_kwargs = dict(thread_kwargs)
                    # Attach the MEDIA:<path> caption to the bubble itself for
                    # captionable kinds (photo/video/document). _tg_caption is
                    # only set for a single captionable file, so this never
                    # double-captions a multi-file send or a voice note.
                    if _tg_caption is not None and not (ext in _VOICE_EXTS and is_voice):
                        media_kwargs["caption"] = _tg_caption
                        media_kwargs["parse_mode"] = send_parse_mode
                    if (ext in _VOICE_EXTS and is_voice) or ext in _TELEGRAM_SEND_AUDIO_EXTS:
                        try:
                            from plugins.platforms.telegram.adapter import _probe_voice_duration_seconds
                            duration = await asyncio.to_thread(_probe_voice_duration_seconds, media_path)
                            if duration is not None:
                                media_kwargs["duration"] = duration
                        except Exception:
                            pass
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
                        elif media_kwargs.get("parse_mode") and (
                            "parse" in str(media_err).lower()
                            or "caption" in str(media_err).lower()
                        ):
                            # Caption failed to parse as MarkdownV2/HTML —
                            # retry with a plain-text caption so the media
                            # (and its caption) still deliver.
                            logger.warning(
                                "Caption parse failed for media send, retrying plain: %s",
                                _sanitize_error_text(media_err),
                            )
                            f.seek(0)
                            media_kwargs.pop("parse_mode", None)
                            if not _has_html and media_kwargs.get("caption"):
                                try:
                                    from plugins.platforms.telegram.adapter import _strip_mdv2
                                    media_kwargs["caption"] = _strip_mdv2(media_kwargs["caption"])
                                except Exception:
                                    pass
                            if ext in _IMAGE_EXTS and not force_document:
                                last_msg = await bot.send_photo(
                                    chat_id=int_chat_id, photo=f, **media_kwargs
                                )
                            elif ext in _VIDEO_EXTS:
                                last_msg = await bot.send_video(
                                    chat_id=int_chat_id, video=f, **media_kwargs
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


async def _resolve_slack_user_target(token, chat_id):
    """Resolve a Slack user target to a D... DM conversation ID.

    ``chat_id`` may be a Slack conversation ID (C/G/D...) — returned unchanged —
    or an internal user target (``user:U...`` / ``user_name:<handle>``). User
    targets are opened as DMs via conversations.open because Slack
    chat.postMessage requires a conversation ID. ``user_name:`` targets are
    first resolved to a user ID through users.list (stable handle match only).

    Returns ``(chat_id, None)`` on success or ``(None, error_dict)`` on failure.
    """
    if not (chat_id.startswith("user:") or chat_id.startswith("user_name:")):
        return chat_id, None
    try:
        import aiohttp
    except ImportError:
        return None, {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url()
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        base_url = "https://slack.com/api"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async def post_api(session, method, payload):
            async with session.post(f"{base_url}/{method}", headers=headers, json=payload, **_req_kw) as resp:
                return await resp.json()

        async def resolve_user_name(session, name):
            query = name.strip().lstrip("@").lower()
            matches = []
            cursor = None
            for _page in range(20):
                payload = {"limit": 200}
                if cursor:
                    payload["cursor"] = cursor
                data = await post_api(session, "users.list", payload)
                if not data.get("ok"):
                    return None, f"Slack users.list error: {data.get('error', 'unknown')}"
                for member in data.get("members", []):
                    if member.get("deleted") or member.get("is_bot"):
                        continue
                    # ``@name`` should match the stable Slack handle only. Display
                    # and real names are mutable/non-unique enough that using them
                    # could DM the wrong person with sensitive content.
                    if str(member.get("name", "")).strip().lower() == query:
                        matches.append(member)
                cursor = (data.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
            if not matches:
                return None, f"Could not resolve Slack user '@{name}'."
            if len(matches) > 1:
                return None, f"Slack user '@{name}' matched multiple Slack users. Use a Slack user ID instead."
            return matches[0].get("id"), None

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            if chat_id.startswith("user_name:"):
                user_id, error = await resolve_user_name(session, chat_id[len("user_name:"):])
                if error:
                    return None, _error(error)
                chat_id = f"user:{user_id}"

            user_id = chat_id[len("user:"):]
            opened = await post_api(session, "conversations.open", {"users": user_id})
            if not opened.get("ok"):
                return None, _error(
                    f"Slack conversations.open error: {opened.get('error', 'unknown')}. "
                    "Check bot permissions (im:write)."
                )
            dm_id = (opened.get("channel") or {}).get("id")
            if not dm_id:
                return None, _error("Slack conversations.open did not return a DM channel ID")
            return dm_id, None
    except Exception as e:
        return None, _error(f"Slack DM resolution failed: {e}")


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

    # Resolve credential fallbacks through the profile secret scope (with the
    # plain-environ fallback for unscoped single-profile runs) so a multiplex
    # profile's direct send never borrows another profile's QQ credentials.
    from gateway.config import _getenv

    extra = pconfig.extra or {}
    appid = extra.get("app_id") or _getenv("QQ_APP_ID", "")
    secret = (pconfig.token or extra.get("client_secret")
              or _getenv("QQ_CLIENT_SECRET", ""))
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
                return _error("QQBot: no access_token in response")

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
