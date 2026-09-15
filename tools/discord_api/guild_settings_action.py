"""Canonical ``discord_admin`` action for request-owned guild settings."""

from __future__ import annotations

import json
from typing import Any, Callable

from gateway.session_context import get_session_env
from tools.discord_api.action_registry import DiscordAction, register_discord_action
from tools.discord_api.guild_settings import GuildSettingsError, edit_guild_request
from tools.registry import tool_error

ACTION_NAME = "edit_current_guild_settings"

SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Approved scalar settings for the active Discord request guild. "
        "The target guild is request-owned and cannot be supplied by the model."
    ),
    "properties": {
        "name": {"type": "string", "minLength": 2, "maxLength": 100},
        "description": {
            "anyOf": [
                {"type": "string", "maxLength": 1024},
                {"type": "null"},
            ]
        },
        "verification_level": {
            "anyOf": [
                {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                {"type": "null"},
            ]
        },
        "default_message_notifications": {
            "anyOf": [
                {"type": "integer", "enum": [0, 1]},
                {"type": "null"},
            ]
        },
        "explicit_content_filter": {
            "anyOf": [
                {"type": "integer", "enum": [0, 1, 2]},
                {"type": "null"},
            ]
        },
        "premium_progress_bar_enabled": {"type": "boolean"},
        "afk_channel_id": {
            "anyOf": [
                {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"},
                {"type": "null"},
            ]
        },
        "system_channel_id": {
            "anyOf": [
                {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"},
                {"type": "null"},
            ]
        },
        "rules_channel_id": {
            "anyOf": [
                {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"},
                {"type": "null"},
            ]
        },
        "public_updates_channel_id": {
            "anyOf": [
                {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"},
                {"type": "null"},
            ]
        },
        "safety_alerts_channel_id": {
            "anyOf": [
                {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"},
                {"type": "null"},
            ]
        },
        "afk_timeout": {
            "type": "integer",
            "enum": [60, 300, 900, 1800, 3600],
        },
    },
    "additionalProperties": False,
}


def edit_current_guild_settings(
    *,
    token: str,
    settings: Any = None,
    _request: Callable[..., Any],
    **_kwargs: Any,
) -> str:
    """Validate and PATCH the exact guild that owns the active Discord turn."""
    platform = get_session_env("HERMES_SESSION_PLATFORM").strip().lower()
    requester_id = get_session_env("HERMES_SESSION_USER_ID").strip()
    guild_id = get_session_env("HERMES_SESSION_SCOPE_ID").strip()

    if platform != "discord":
        return tool_error(
            "edit_current_guild_settings requires an active Discord request context."
        )
    if not requester_id:
        return tool_error(
            "edit_current_guild_settings requires an authenticated Discord requester."
        )
    if not guild_id:
        return tool_error(
            "edit_current_guild_settings requires an active Discord guild context; "
            "it is unavailable in DMs and unowned cross-platform sessions."
        )
    if not isinstance(settings, dict):
        return tool_error("'settings' must be a JSON object.")

    try:
        request = edit_guild_request(guild_id, **settings)
    except GuildSettingsError as exc:
        return tool_error(str(exc))

    _request(
        request["method"],
        request["path"],
        token,
        body=request["json"],
    )

    canonical_guild_id = request["path"].rsplit("/", 1)[-1]
    return json.dumps(
        {
            "success": True,
            "guild_id": canonical_guild_id,
            "updated_settings": request["json"],
        }
    )


register_discord_action(
    DiscordAction(
        name=ACTION_NAME,
        surface="admin",
        signature="(settings)",
        description="edit approved scalars on the active request guild",
        handler=edit_current_guild_settings,
        properties={"settings": SETTINGS_SCHEMA},
        required=("settings",),
        permission_hint=(
            "Bot lacks MANAGE_GUILD in the active Discord server. "
            "Grant the bot a role with Manage Server permission."
        ),
    )
)
