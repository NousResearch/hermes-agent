"""Canonical ``discord_admin`` actions for request-owned channel overwrites."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

from gateway.session_context import get_session_env
from tools.discord_api.action_registry import DiscordAction, register_discord_action
from tools.discord_api.permissions import (
    PermissionOverwriteError,
    TYPE_MEMBER,
    TYPE_ROLE,
    delete_channel_permission_request,
    set_channel_permission_request,
)
from tools.registry import tool_error

SET_ACTION = "set_current_guild_channel_permission"
DELETE_ACTION = "delete_current_guild_channel_permission"
_SNOWFLAKE_SCHEMA = {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"}
_BITFIELD_SCHEMA = {
    "anyOf": [
        {"type": "integer", "minimum": 0},
        {"type": "string", "pattern": "^[0-9]{1,20}$"},
    ]
}


def _request_owner() -> tuple[str, str] | str:
    platform = get_session_env("HERMES_SESSION_PLATFORM").strip().lower()
    requester_id = get_session_env("HERMES_SESSION_USER_ID").strip()
    guild_id = get_session_env("HERMES_SESSION_SCOPE_ID").strip()
    if platform != "discord":
        return tool_error("Discord permission actions require an active Discord request context.")
    if not requester_id:
        return tool_error("Discord permission actions require an authenticated requester.")
    if not guild_id:
        return tool_error(
            "Discord permission actions require an active guild context; DMs are not eligible."
        )
    return guild_id, requester_id


def _channel_id_from_path(path: str) -> str:
    return path.split("/", 3)[2]


def _canonical_id(value: Any) -> str:
    text = str(value or "")
    if not text.isdigit() or int(text) <= 0 or int(text) > (1 << 64) - 1:
        return ""
    return str(int(text))


def _verify_request_owned_channel(
    *,
    token: str,
    guild_id: str,
    channel_id: str,
    _request: Callable[..., Any],
) -> str | None:
    channel = _request("GET", f"/channels/{channel_id}", token)
    if not isinstance(channel, dict):
        return tool_error("Discord returned an invalid channel object; refusing mutation.")
    expected_guild = _canonical_id(guild_id)
    actual_guild = _canonical_id(channel.get("guild_id"))
    if not expected_guild or actual_guild != expected_guild:
        return tool_error(
            "Target channel is not owned by the active Discord request guild; refusing mutation."
        )
    return None


def set_current_guild_channel_permission(
    *,
    token: str,
    channel_id: Any = None,
    overwrite_id: Any = None,
    target_type: Any = None,
    allow: Any = 0,
    deny: Any = 0,
    _request: Callable[..., Any],
    **_kwargs: Any,
) -> str:
    owner = _request_owner()
    if isinstance(owner, str):
        return owner
    guild_id, _requester_id = owner
    type_map = {"role": TYPE_ROLE, "member": TYPE_MEMBER}
    if target_type not in type_map:
        return tool_error("'target_type' must be 'role' or 'member'.")
    try:
        request = set_channel_permission_request(
            channel_id,
            overwrite_id,
            allow=allow,
            deny=deny,
            type_=type_map[target_type],
        )
    except PermissionOverwriteError as exc:
        return tool_error(str(exc))

    canonical_channel = _channel_id_from_path(request["path"])
    ownership_error = _verify_request_owned_channel(
        token=token,
        guild_id=guild_id,
        channel_id=canonical_channel,
        _request=_request,
    )
    if ownership_error:
        return ownership_error

    _request("PUT", request["path"], token, body=request["payload"])
    return json.dumps(
        {
            "success": True,
            "operation": "set",
            "guild_id": _canonical_id(guild_id),
            "channel_id": canonical_channel,
            "overwrite_id": request["path"].rsplit("/", 1)[-1],
            "target_type": target_type,
            "allow": request["payload"]["allow"],
            "deny": request["payload"]["deny"],
        }
    )


def delete_current_guild_channel_permission(
    *,
    token: str,
    channel_id: Any = None,
    overwrite_id: Any = None,
    _request: Callable[..., Any],
    **_kwargs: Any,
) -> str:
    owner = _request_owner()
    if isinstance(owner, str):
        return owner
    guild_id, _requester_id = owner
    try:
        request = delete_channel_permission_request(channel_id, overwrite_id)
    except PermissionOverwriteError as exc:
        return tool_error(str(exc))

    canonical_channel = _channel_id_from_path(request["path"])
    ownership_error = _verify_request_owned_channel(
        token=token,
        guild_id=guild_id,
        channel_id=canonical_channel,
        _request=_request,
    )
    if ownership_error:
        return ownership_error

    _request("DELETE", request["path"], token)
    return json.dumps(
        {
            "success": True,
            "operation": "delete",
            "guild_id": _canonical_id(guild_id),
            "channel_id": canonical_channel,
            "overwrite_id": request["path"].rsplit("/", 1)[-1],
        }
    )


_COMMON_PROPERTIES: Dict[str, Dict[str, Any]] = {
    # Reuse the canonical discord_admin channel schema verbatim. Runtime
    # validation below remains stricter and proves request-guild ownership.
    "channel_id": {
        "type": "string",
        "description": "Discord channel ID.",
    },
    "overwrite_id": {
        **_SNOWFLAKE_SCHEMA,
        "description": "Role or member snowflake whose overwrite is changed.",
    },
}

register_discord_action(
    DiscordAction(
        name=SET_ACTION,
        surface="admin",
        signature="(channel_id, overwrite_id, target_type, allow='0', deny='0')",
        description="set a role/member overwrite in the active request guild",
        handler=set_current_guild_channel_permission,
        properties={
            **_COMMON_PROPERTIES,
            "target_type": {"type": "string", "enum": ["role", "member"]},
            "allow": _BITFIELD_SCHEMA,
            "deny": _BITFIELD_SCHEMA,
        },
        required=("channel_id", "overwrite_id", "target_type"),
        permission_hint=(
            "Bot lacks MANAGE_ROLES for the target channel. Grant the bot a role with "
            "Manage Roles and ensure its role is high enough for the requested overwrite."
        ),
    )
)

register_discord_action(
    DiscordAction(
        name=DELETE_ACTION,
        surface="admin",
        signature="(channel_id, overwrite_id)",
        description="delete a role/member overwrite in the active request guild",
        handler=delete_current_guild_channel_permission,
        properties=_COMMON_PROPERTIES,
        required=("channel_id", "overwrite_id"),
        permission_hint=(
            "Bot lacks MANAGE_ROLES for the target channel. Grant the bot a role with "
            "Manage Roles and ensure its role is high enough for the requested overwrite."
        ),
    )
)
