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
import os
import weakref
from dataclasses import replace
from typing import Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_ACTIONS = ("join", "leave", "status")
_TIMEOUT_S = 45


def _runner():
    try:
        from gateway.run import _gateway_runner_ref
        return _gateway_runner_ref()
    except Exception:
        return None


def _session_source():
    """The current turn's origin, rebuilt from the session contextvars (None outside a gateway turn)."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_context import get_session_env

    if get_session_env("HERMES_SESSION_PLATFORM", "") != "discord":
        return None
    if get_session_env("HERMES_CRON_SESSION", "") == "1":
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
    try:
        resolve = getattr(runner, "_authorization_adapter", None)
        if callable(resolve):
            return resolve(source.platform, source.profile)
        return runner.adapters.get(source.platform)
    except Exception:
        return None


def _home_channel_source(adapter, source):
    """DM fallback: bind to DISCORD_HOME_CHANNEL (and its guild) so a DM 'join voice' still works."""
    home = (os.getenv("DISCORD_HOME_CHANNEL") or "").strip()
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
        try:
            if vc is not None and vc.is_connected():
                out.append(gid)
        except Exception:
            continue
    return out


async def _do(runner, adapter, source, action: str) -> dict:
    from gateway.session import SessionSource

    guild_id: Optional[int] = int(source.scope_id) if (source.scope_id or "").isdigit() else None
    if action == "join":
        if guild_id is None:
            fallback = _home_channel_source(adapter, source)
            if fallback is None:
                return {"error": "Not in a server channel and DISCORD_HOME_CHANNEL is not resolvable."}
            source, guild_id = fallback, int(fallback.scope_id)
        message = await runner._voice_channel_join_for_source(source, guild_id)
        return {"success": adapter.is_in_voice_channel(guild_id), "action": "join", "message": message,
                "text_channel_id": source.chat_id}
    if guild_id is None or not adapter.is_in_voice_channel(guild_id):
        connected = _connected_guilds(adapter)
        guild_id = connected[0] if connected else guild_id
    if action == "status":
        info = adapter.get_voice_channel_info(guild_id) if guild_id and hasattr(
            adapter, "get_voice_channel_info") else None
        bound = (getattr(adapter, "_voice_text_channels", None) or {}).get(guild_id) if guild_id else None
        mode = runner._voice_mode.get(runner._voice_key_for_source(source), "off")
        return {"connected": bool(info), "guild_id": str(guild_id) if guild_id else None,
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
        except Exception:
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
    try:
        authorized = bool(runner._is_user_authorized(source))
    except Exception:
        authorized = False
    if not authorized:
        return tool_error("Requester is not an allowed user for this gateway.")
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or not loop.is_running():
        return tool_error("Gateway event loop is not running.")
    try:
        if asyncio.get_running_loop() is loop:
            return tool_error("discord_voice cannot block the gateway loop it needs.")
    except RuntimeError:
        pass
    from agent.async_utils import safe_schedule_threadsafe
    fut = safe_schedule_threadsafe(_do(runner, adapter, source, action), loop, logger=logger,
                                   log_message="discord_voice: failed to schedule on gateway loop")
    if fut is None:
        return tool_error("Could not schedule the voice action on the gateway loop.")
    try:
        result = fut.result(timeout=_TIMEOUT_S)
    except Exception as e:  # timeout or the helper raised
        logger.warning("discord_voice %s failed: %s", action, e)
        return tool_error(f"Voice {action} failed: {e}")
    if "error" in result:
        return tool_error(result["error"])
    return tool_result(result)


def check_discord_voice_requirements() -> bool:
    """Only offered inside a running gateway that has a Discord adapter."""
    runner = _runner()
    if runner is None:
        return False
    try:
        from gateway.config import Platform
        return any(Platform.DISCORD in (m or {}) for m in
                   [getattr(runner, "adapters", None), *((getattr(runner, "_profile_adapters", None) or {}).values())])
    except Exception:
        return False


DISCORD_VOICE_SCHEMA = {
    "name": "discord_voice",
    "description": (
        "Join, leave, or check your own Discord voice-channel presence (same effect as the user typing "
        "/voice join or /voice leave). 'join' joins the voice channel the requesting user is currently "
        "in and binds it to this text chat (spoken replies on). 'leave' disconnects, turns voice replies "
        "off and stops listening. 'status' reports whether you are connected, where, and who is there. "
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
