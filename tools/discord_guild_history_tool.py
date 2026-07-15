"""Service-gated authorized Discord guild history read."""

from __future__ import annotations

import json
from typing import Any

from gateway.discord_connector_protocol import MAX_HISTORY_MESSAGES
from gateway.discord_guild_history_client import (
    DiscordGuildHistoryClientError,
    discord_guild_history_configured,
    privileged_discord_guild_history_client,
)
from tools.registry import registry, tool_error


def _read_guild_history(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        page = privileged_discord_guild_history_client().read(
            channel_id=args.get("channel_id"),
            limit=args.get("limit"),
            before_message_id=args.get("before_message_id"),
            after_message_id=args.get("after_message_id"),
        )
        return json.dumps(page, ensure_ascii=False, sort_keys=True)
    except (DiscordGuildHistoryClientError, TypeError, ValueError) as exc:
        code = getattr(exc, "code", "discord_guild_history_invalid")
        return tool_error(
            code,
            instruction=(
                "Use one exact Discord guild channel or type-10/11 thread ID. "
                "The privileged connector independently checks the authenticated "
                "requester's live View Channel and Read Message History access, "
                "plus its own read/send access. Scheduled reads are available only "
                "to exact deployment-reviewed jobs and targets. Discord DMs, group "
                "DMs, and type-12 private threads are structurally unavailable."
            ),
        )


DISCORD_GUILD_HISTORY_SCHEMA = {
    "name": "discord_guild_history",
    "description": (
        "Read a bounded chronological page from one exact Discord guild text "
        "channel or type-10/11 guild thread that the active authenticated "
        "requester may currently view and read. Choose the evidence target and "
        "interpret message meaning yourself. Permissions are proven mechanically "
        "at execution time; you cannot supply or broaden authority. The isolated "
        "canary remains public-only, while production supports ACL-authorized "
        "private guild lanes. Discord DMs, group DMs, and type-12 private threads "
        "are structurally blocked. Use after_message_id to continue forward from "
        "a known message or before_message_id to page backward; never provide both."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "pattern": "^[1-9][0-9]{0,24}$",
                "description": "Exact Discord guild channel or type-10/11 thread ID.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_HISTORY_MESSAGES,
                "description": "Maximum messages in this bounded page.",
            },
            "before_message_id": {
                "type": "string",
                "pattern": "^[1-9][0-9]{0,24}$",
                "description": "Optional exclusive cursor for an older page.",
            },
            "after_message_id": {
                "type": "string",
                "pattern": "^[1-9][0-9]{0,24}$",
                "description": "Optional exclusive cursor for newer follow-up messages.",
            },
        },
        "required": ["channel_id", "limit"],
        "additionalProperties": False,
    },
}


registry.register(
    name="discord_guild_history",
    toolset="discord_guild_read",
    schema=DISCORD_GUILD_HISTORY_SCHEMA,
    handler=_read_guild_history,
    check_fn=discord_guild_history_configured,
    emoji="🎮",
    max_result_size_chars=128_000,
)


__all__ = ["DISCORD_GUILD_HISTORY_SCHEMA"]
