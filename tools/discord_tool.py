"""Primitive Discord management tool for creating threads and channels."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger(__name__)


DISCORD_MANAGE_SCHEMA = {
    "name": "discord_manage",
    "description": (
        "Create Discord threads or text channels when Discord is connected. "
        "This is a low-level primitive for explicit workspace creation requests."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_thread", "create_channel"],
                "description": "Discord management action to perform."
            },
            "name": {
                "type": "string",
                "description": "Name to create. Required for create_thread and create_channel."
            },
            "target": {
                "type": "string",
                "description": (
                    "Where to create the Discord space. Defaults to the current Discord chat when invoked from Discord, "
                    "otherwise falls back to the configured Discord home channel. Accepts 'origin', 'discord:chat_id', "
                    "or 'discord:#channel-name'. For threads, if the target resolves to an existing thread, Hermes "
                    "creates the new thread under that thread's parent channel. For channels, Hermes creates the new "
                    "channel in the same guild and category context as the target channel when possible."
                )
            },
            "message": {
                "type": "string",
                "description": "Optional starter message to seed the new thread."
            },
            "topic": {
                "type": "string",
                "description": "Optional channel topic. Used for create_channel."
            },
            "channel_type": {
                "type": "string",
                "enum": ["text"],
                "description": "Discord channel type to create. Currently only 'text' is supported."
            },
            "nsfw": {
                "type": "boolean",
                "description": "Whether the new channel should be marked NSFW. Used for create_channel."
            },
            "auto_archive_duration": {
                "type": "integer",
                "enum": [60, 1440, 4320, 10080],
                "description": "Discord thread auto-archive duration in minutes. Default: 1440 (24 hours)."
            },
            "reason": {
                "type": "string",
                "description": "Optional audit-log reason for Discord."
            },
        },
        "required": ["action", "name"],
    },
}


THREAD_CHANNEL_TYPES = {10, 11, 12}
DM_CHANNEL_TYPES = {1, 3}
DEFAULT_AUTO_ARCHIVE_DURATION = 1440


def discord_manage_tool(args, **kwargs):
    action = (args.get("action") or "").strip().lower()
    if action == "create_thread":
        return _handle_create_thread(args)
    if action == "create_channel":
        return _handle_create_channel(args)
    return json.dumps({"error": f"Unsupported discord_manage action: {action}"})


def _run_discord_manage_action(action_name: str, coro) -> str:
    try:
        from model_tools import _run_async

        result = _run_async(coro)
        return json.dumps(result)
    except Exception as e:
        logger.exception("discord_manage %s failed", action_name)
        label = action_name.replace("_", " ")
        return json.dumps({"error": f"Discord {label} failed: {e}"})


def _handle_create_thread(args: Dict[str, Any]) -> str:
    return _run_discord_manage_action("create_thread", _create_thread(args))


def _handle_create_channel(args: Dict[str, Any]) -> str:
    return _run_discord_manage_action("create_channel", _create_channel(args))


async def _create_thread(args: Dict[str, Any]) -> Dict[str, Any]:
    token = _load_discord_token()
    if not token:
        return {"error": "Discord is not configured. Add a Discord bot token in gateway config first."}

    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "'name' is required for action='create_thread'"}

    target_id, target_note = _resolve_target_chat_id(args.get("target"))
    if not target_id:
        return {"error": target_note or "Could not resolve Discord target."}

    auto_archive_duration = int(args.get("auto_archive_duration") or DEFAULT_AUTO_ARCHIVE_DURATION)
    starter_message = (args.get("message") or "").strip()
    reason = (args.get("reason") or "").strip() or None

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    if reason:
        headers["X-Audit-Log-Reason"] = quote(reason, safe="")

    async with aiohttp.ClientSession(headers=headers) as session:
        channel_data = await _discord_api_json(session, "GET", f"/channels/{target_id}")
        if "error" in channel_data:
            return channel_data

        channel_type = channel_data.get("type")
        if channel_type in DM_CHANNEL_TYPES:
            return {"error": "Discord threads can only be created inside server text channels, not DMs."}

        parent_channel_id = channel_data.get("parent_id") if channel_type in THREAD_CHANNEL_TYPES else str(target_id)
        if not parent_channel_id:
            return {"error": "Could not determine a parent text channel for the new thread."}

        direct_payload = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
            "type": 11,
        }
        direct_result = await _discord_api_json(
            session,
            "POST",
            f"/channels/{parent_channel_id}/threads",
            payload=direct_payload,
            allowed_statuses={200, 201},
            include_status=True,
        )

        starter_message_id = None
        thread_data: Optional[Dict[str, Any]] = None

        if "error" not in direct_result:
            thread_data = direct_result
            if starter_message:
                post_result = await _discord_api_json(
                    session,
                    "POST",
                    f"/channels/{thread_data['id']}/messages",
                    payload={"content": starter_message},
                )
                if "error" in post_result:
                    return post_result
                starter_message_id = post_result.get("id")
        else:
            seed_content = starter_message or f"🧵 Thread created by Hermes: **{name}**"
            seed_result = await _discord_api_json(
                session,
                "POST",
                f"/channels/{parent_channel_id}/messages",
                payload={"content": seed_content},
            )
            if "error" in seed_result:
                return {
                    "error": (
                        "Discord rejected direct thread creation and Hermes could not create a starter message either. "
                        f"Direct error: {direct_result['error']}"
                    )
                }
            starter_message_id = seed_result.get("id")
            thread_result = await _discord_api_json(
                session,
                "POST",
                f"/channels/{parent_channel_id}/messages/{starter_message_id}/threads",
                payload={
                    "name": name,
                    "auto_archive_duration": auto_archive_duration,
                },
            )
            if "error" in thread_result:
                return thread_result
            thread_data = thread_result

    return {
        "success": True,
        "platform": "discord",
        "action": "create_thread",
        "thread_id": str(thread_data.get("id")),
        "thread_name": thread_data.get("name") or name,
        "parent_channel_id": str(parent_channel_id),
        "starter_message_id": str(starter_message_id) if starter_message_id else None,
    }


async def _create_channel(args: Dict[str, Any]) -> Dict[str, Any]:
    token = _load_discord_token()
    if not token:
        return {"error": "Discord is not configured. Add a Discord bot token in gateway config first."}

    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "'name' is required for action='create_channel'"}

    channel_type = (args.get("channel_type") or "text").strip().lower()
    if channel_type != "text":
        return {"error": f"Unsupported channel_type '{channel_type}'. Only 'text' is supported right now."}

    target_id, target_note = _resolve_target_chat_id(args.get("target"))
    if not target_id:
        return {"error": target_note or "Could not resolve Discord target."}

    reason = (args.get("reason") or "").strip() or None
    topic = (args.get("topic") or "").strip() or None
    nsfw = bool(args.get("nsfw", False))

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    if reason:
        headers["X-Audit-Log-Reason"] = quote(reason, safe="")

    async with aiohttp.ClientSession(headers=headers) as session:
        channel_data = await _discord_api_json(session, "GET", f"/channels/{target_id}")
        if "error" in channel_data:
            return channel_data

        if channel_data.get("type") in DM_CHANNEL_TYPES:
            return {"error": "Discord channels can only be created inside servers, not DMs."}

        base_channel_data = channel_data
        if channel_data.get("type") in THREAD_CHANNEL_TYPES:
            parent_channel_id = channel_data.get("parent_id")
            if not parent_channel_id:
                return {"error": "Could not determine the parent channel for the current thread."}
            base_channel_data = await _discord_api_json(session, "GET", f"/channels/{parent_channel_id}")
            if "error" in base_channel_data:
                return base_channel_data

        guild_id = base_channel_data.get("guild_id") or channel_data.get("guild_id")
        if not guild_id:
            return {"error": "Could not determine which Discord server should own the new channel."}

        payload = {
            "name": name,
            "type": 0,
            "nsfw": nsfw,
        }
        parent_category_id = base_channel_data.get("parent_id")
        if parent_category_id:
            payload["parent_id"] = parent_category_id
        if topic:
            payload["topic"] = topic

        created = await _discord_api_json(
            session,
            "POST",
            f"/guilds/{guild_id}/channels",
            payload=payload,
        )
        if "error" in created:
            return created

    return {
        "success": True,
        "platform": "discord",
        "action": "create_channel",
        "channel_id": str(created.get("id")),
        "channel_name": created.get("name") or name,
        "guild_id": str(guild_id),
        "parent_category_id": str(parent_category_id) if parent_category_id else None,
        "topic": created.get("topic") if created.get("topic") is not None else topic,
        "nsfw": bool(created.get("nsfw", nsfw)),
    }


async def _discord_api_json(
    session,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    allowed_statuses: Optional[set[int]] = None,
    include_status: bool = False,
) -> Dict[str, Any]:
    allowed_statuses = allowed_statuses or {200, 201}
    url = f"https://discord.com/api/v10{path}"
    async with session.request(method, url, json=payload) as resp:
        text = await resp.text()
        if resp.status not in allowed_statuses:
            body = text.strip() or "<empty body>"
            return {"error": f"Discord API error ({resp.status}): {body}", "status": resp.status}
        if not text.strip():
            data: Dict[str, Any] = {}
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"raw": text}
        if include_status:
            data["_status"] = resp.status
        return data


def _resolve_target_chat_id(target: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    target = (target or "").strip()

    if not target or target.lower() == "origin":
        session_platform = os.getenv("HERMES_SESSION_PLATFORM", "").lower()
        session_chat_id = os.getenv("HERMES_SESSION_CHAT_ID", "").strip()
        if session_platform == "discord" and session_chat_id:
            return session_chat_id, "Used the current Discord chat as the thread target."

        home_channel = _load_discord_home_channel()
        if home_channel:
            return home_channel, "Used the configured Discord home channel as the thread target."

        return None, "No Discord target provided and no current Discord chat/home channel was available."

    if target.isdigit():
        return target, f"Used Discord target {target}."

    if target.lower().startswith("discord:"):
        chat_ref = target.split(":", 1)[1].strip()
        if chat_ref.isdigit():
            return chat_ref, f"Used Discord target {chat_ref}."
        try:
            from gateway.channel_directory import resolve_channel_name

            resolved = resolve_channel_name("discord", chat_ref)
        except Exception:
            resolved = None
        if resolved:
            return resolved, f"Resolved {target} to Discord target {resolved}."
        return None, f"Could not resolve Discord target '{target}'. Use send_message(action='list') to inspect known channels."

    return None, "Discord targets must be 'origin', a numeric channel ID, or 'discord:CHANNEL'."


def _load_discord_token() -> Optional[str]:
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.DISCORD)
        if pconfig and pconfig.enabled and pconfig.token:
            return pconfig.token
    except Exception:
        return None
    return None


def _load_discord_home_channel() -> Optional[str]:
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        home = config.get_home_channel(Platform.DISCORD)
        if home and home.chat_id:
            return str(home.chat_id)
    except Exception:
        return None
    return None


def _check_discord_manage() -> bool:
    session_platform = os.getenv("HERMES_SESSION_PLATFORM", "").lower()
    if session_platform == "discord":
        return True
    return bool(_load_discord_token())


from tools.registry import registry

registry.register(
    name="discord_manage",
    toolset="messaging",
    schema=DISCORD_MANAGE_SCHEMA,
    handler=discord_manage_tool,
    check_fn=_check_discord_manage,
)
