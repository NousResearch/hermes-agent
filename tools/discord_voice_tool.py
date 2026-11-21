#!/usr/bin/env python3
"""``discord_voice``: let the agent join, leave, or inspect a Discord voice channel itself.

It reaches the in-process gateway runner through
``gateway.run._gateway_runner_ref`` (the same pattern ``send_message`` uses) and runs the SAME
code paths as ``/voice join`` and ``/voice leave``
(``GatewayVoiceMixin._voice_channel_join_for_source`` / ``_voice_channel_leave_for_source``) on the
gateway's event loop, so leave also sets voice mode off and clears the voice input callback.

Only Discord sessions whose requester passes the gateway's own allowlist
(``_is_user_authorized``) may use it; cron and other non-chat sessions are refused.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from concurrent.futures import CancelledError
from dataclasses import replace
from typing import Optional

from agent.secret_scope import get_secret
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_ACTIONS = ("join", "leave", "status")
_TIMEOUT_S = 45


def _runner():
    try:
        from gateway.run import _gateway_runner_ref
    except ImportError:  # gateway extras not installed: no runner can exist in this process
        return None
    return _gateway_runner_ref()


def _session_source():
    """The current turn's origin, rebuilt from the session contextvars (None outside a gateway turn)."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_context import get_session_env
    from utils import is_truthy_value

    if get_session_env("HERMES_SESSION_PLATFORM", "") != "discord":
        return None
    if is_truthy_value(get_session_env("HERMES_CRON_SESSION", "")):
        return None
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    user_id = get_session_env("HERMES_SESSION_USER_ID", "")
    if not chat_id or not user_id:
        return None
    return SessionSource(
        platform=Platform.DISCORD, chat_id=chat_id,
        chat_type=get_session_env("HERMES_SESSION_CHAT_TYPE", "") or "group",
        chat_name=get_session_env("HERMES_SESSION_CHAT_NAME", "") or None,
        user_id=user_id, user_name=get_session_env("HERMES_SESSION_USER_NAME", "") or None,
        thread_id=get_session_env("HERMES_SESSION_THREAD_ID", "") or None,
        scope_id=get_session_env("HERMES_SESSION_SCOPE_ID", "") or None,
        parent_chat_id=get_session_env("HERMES_SESSION_PARENT_CHAT_ID", "") or None,
        profile=get_session_env("HERMES_SESSION_PROFILE", "") or None)


def _adapter_for(runner, source):
    resolve = getattr(runner, "_authorization_adapter", None)
    if callable(resolve):
        return resolve(source.platform, source.profile)
    return runner.adapters.get(source.platform)


def _home_channel_source(adapter, source, home: str):
    """DM fallback: bind to DISCORD_HOME_CHANNEL (and its guild) so a DM 'join voice' still works.
    ``home`` is read by the caller on the tool thread, where the turn's profile secret scope is bound
    (``_do`` runs on the gateway loop, whose task context does not carry that scope)."""
    client = getattr(adapter, "_client", None)
    if not home.isdigit() or client is None:
        return None
    channel = client.get_channel(int(home))
    guild = getattr(channel, "guild", None)
    if guild is None:
        return None
    return replace(source, chat_id=home, chat_type="group", thread_id=None,
                   scope_id=str(guild.id), guild_id=str(guild.id), chat_name=None)


def _connected_guilds(adapter) -> list:
    clients = getattr(adapter, "_voice_clients", None) or {}
    out = []
    for gid, vc in list(clients.items()):
        if vc is not None and vc.is_connected():
            out.append(gid)
    return out


async def _do(runner, adapter, source, action: str, home: str = "") -> dict:
    from gateway.session import SessionSource

    # A DM never names a guild, even if a scope id leaked into its session; DM rules always apply.
    scoped = source.chat_type != "dm" and (source.scope_id or "").isdigit()
    guild_id: Optional[int] = int(source.scope_id) if scoped else None
    if action == "join":
        if guild_id is None:
            if source.chat_type != "dm":  # a server channel with no guild id never joins elsewhere
                return {"error": "Could not determine this server; join from a server text channel."}
            fallback = _home_channel_source(adapter, source, home)
            if fallback is None:
                return {"error": "Not in a server channel and DISCORD_HOME_CHANNEL is not resolvable."}
            source, guild_id = fallback, int(fallback.scope_id)
        sources = getattr(adapter, "_voice_sources", None)
        before = sources.get(guild_id) if isinstance(sources, dict) else None
        message = await runner._voice_channel_join_for_source(source, guild_id)
        joined = bool(adapter.is_in_voice_channel(guild_id))
        if isinstance(sources, dict):
            # The shared join path writes a fresh binding only on success, so an older connection
            # that survived a failed join does not read as success.
            after = sources.get(guild_id)
            joined = joined and after is not None and after is not before
        return {"success": joined, "action": "join", "message": message,
                "text_channel_id": source.chat_id}
    if guild_id is None and source.chat_type == "dm":
        # DM: act on the bot's voice connection only when exactly one exists. A request from a
        # server channel (even one whose guild id is missing) never falls back to another server.
        connected = _connected_guilds(adapter)
        if len(connected) > 1:
            return {"error": "Connected to voice in more than one server; ask from a channel in that server."}
        guild_id = connected[0] if connected else None
    if action == "status":
        info = adapter.get_voice_channel_info(guild_id) if guild_id and hasattr(
            adapter, "get_voice_channel_info") else None
        bound = (getattr(adapter, "_voice_text_channels", None) or {}).get(guild_id) if guild_id else None
        mode = runner._voice_mode.get(runner._voice_key_for_source(source), "off")
        connected = bool(guild_id) and bool(adapter.is_in_voice_channel(guild_id))
        return {"connected": connected, "guild_id": str(guild_id) if guild_id else None,
                "channel": (info or {}).get("channel_name"), "member_count": (info or {}).get("member_count"),
                "members": [m.get("display_name") for m in (info or {}).get("members", [])],
                "bound_text_channel_id": str(bound) if bound else None, "voice_mode_this_chat": mode}
    # leave: clear voice mode on the text chat the voice channel is bound to (the join source),
    # falling back to the requester's chat — same helper as /voice leave.
    target = source
    bound_source = (getattr(adapter, "_voice_sources", None) or {}).get(guild_id) if guild_id else None
    if isinstance(bound_source, dict):
        try:
            target = SessionSource.from_dict(bound_source)
        except (KeyError, TypeError, ValueError):  # malformed binding: fall back to the requester's chat
            target = source
    target._transport_adapter_ref = weakref.ref(adapter)
    was_in = bool(guild_id) and adapter.is_in_voice_channel(guild_id)
    message = await runner._voice_channel_leave_for_source(target, guild_id)
    return {"success": was_in, "action": "leave", "message": message}


def discord_voice(action: str) -> str:
    action = (action or "").strip().lower()
    if action not in _ACTIONS:
        return tool_error(f"Unknown action: {action!r}", available_actions=list(_ACTIONS))
    runner = _runner()
    if runner is None:
        return tool_error("No live gateway in this process; voice control only works from a gateway chat turn.")
    source = _session_source()
    if source is None:
        return tool_error("discord_voice only works in a Discord chat turn from an allowed user.")
    adapter = _adapter_for(runner, source)
    if adapter is None or not hasattr(adapter, "join_voice_channel"):
        return tool_error("No live Discord adapter with voice support.")
    source._transport_adapter_ref = weakref.ref(adapter)
    if not runner._is_user_authorized(source):
        return tool_error("Requester is not an allowed user for this gateway.")
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or not loop.is_running():
        return tool_error("Gateway event loop is not running.")
    try:
        if asyncio.get_running_loop() is loop:
            return tool_error("discord_voice cannot block the gateway loop it needs.")
    except RuntimeError:
        pass
    home = (get_secret("DISCORD_HOME_CHANNEL", "") or "").strip() if action == "join" else ""
    from agent.async_utils import safe_schedule_threadsafe
    fut = safe_schedule_threadsafe(_do(runner, adapter, source, action, home), loop, logger=logger,
                                   log_message="discord_voice: failed to schedule on gateway loop")
    if fut is None:
        return tool_error("Could not schedule the voice action on the gateway loop.")
    # Anything else the helper raises propagates to the registry's dispatch boundary, which logs
    # the traceback and returns it as a tool error.
    try:
        result = fut.result(timeout=_TIMEOUT_S)
    except TimeoutError:
        fut.cancel()  # best effort: stops a join/leave still awaiting; a completed step is not undone
        logger.warning("discord_voice %s timed out after %ss", action, _TIMEOUT_S)
        return tool_error(f"Voice {action} timed out after {_TIMEOUT_S}s.")
    except CancelledError:  # the gateway loop shut down underneath the action
        return tool_error(f"Voice {action} was cancelled (gateway shutting down).")
    if "error" in result:
        return tool_error(result["error"])
    return tool_result(result)


def check_discord_voice_requirements() -> bool:
    """Offered when this process has a gateway runner with a Discord adapter; each call re-checks the live turn."""
    runner = _runner()
    if runner is None:
        return False
    from gateway.config import Platform
    return any(Platform.DISCORD in (m or {}) for m in
               [getattr(runner, "adapters", None), *((getattr(runner, "_profile_adapters", None) or {}).values())])


DISCORD_VOICE_SCHEMA = {
    "name": "discord_voice",
    "description": (
        "Join, leave, or check your own Discord voice-channel presence (runs the same join/leave code as "
        "/voice join and /voice leave). 'join' joins the voice channel the requesting user is currently "
        "in and binds it to this text chat, or to the home channel from a DM (spoken replies on). 'leave' "
        "asks the voice client to disconnect, turns voice replies off and stops listening (only when you are "
        "in voice in that server; otherwise it changes nothing). 'status' "
        "reports whether you are connected, where, and who is there. "
        "Use when the user asks you to join/leave voice or hang up."),
    "parameters": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(_ACTIONS),
                                  "description": "join | leave | status"}},
        "required": ["action"],
    },
}

registry.register(
    name="discord_voice", toolset="discord", schema=DISCORD_VOICE_SCHEMA,
    handler=lambda args, **kw: discord_voice(args.get("action", "")),
    check_fn=check_discord_voice_requirements, emoji="🎙️")
