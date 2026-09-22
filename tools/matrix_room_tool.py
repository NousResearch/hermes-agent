"""Administer Matrix rooms through the client for the current session."""

from __future__ import annotations

import json
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.matrix_tool_runtime import run_matrix_mutation
from tools.registry import registry


_UNKNOWN_OUTCOME_STEPS = {
    "create": "Ask the user whether the room was created before retrying, so that no duplicate room is created",
    "invite": "Ask the user whether the invite arrived before retrying",
    "leave": "Ask the user whether the bot is still in the room before retrying",
    "forget": "Forget changes only the bot's account, so retrying it is harmless",
    "redact": "Read the event with matrix_read kind=event before retrying",
}


async def _matrix_room_admin(args: dict[str, Any]) -> str:
    adapter, owner_loop = get_session_transport()
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    if (get_session_env("HERMES_SESSION_PLATFORM") != "matrix" or adapter is None
            or not room_id or not get_session_env("HERMES_SESSION_USER_ID")):
        return json.dumps({"error": "Matrix administration requires a live Matrix session"})
    if args.get("room_id", room_id) != room_id:
        return json.dumps({"error": "Matrix administration is limited to the current room"})
    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    return await run_matrix_mutation(
        owner_loop,
        lambda interrupted, before_write: adapter.administer_matrix_room(
            args, interrupt_check=interrupted, before_write=before_write,
        ),
        operation_label="Matrix administration",
        next_step=_UNKNOWN_OUTCOME_STEPS.get(args.get("action"), ""),
    )


registry.register(
    name="matrix_room_admin",
    toolset="matrix_admin",
    schema={
        "name": "matrix_room_admin",
        "description": (
            "Create a private Matrix room, invite a user to the current room, leave or forget "
            "the current room, or redact an event. Forget requires the bot to have left. "
            "Requires an explicitly enabled Matrix admin toolset and an authorised live session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "invite", "leave", "forget", "redact"]},
                "name": {"type": "string", "description": "Display name for a new private room."},
                "topic": {"type": "string"},
                "invite": {"type": "array", "items": {"type": "string"},
                           "description": "Full Matrix user IDs to invite to a new room. Only the requester "
                                          "and users authorised to use the gateway can be invited."},
                "is_direct": {"type": "boolean", "default": False,
                              "description": "Create a direct chat. Requires exactly one invitee."},
                "encrypted": {"type": "boolean", "default": False,
                              "description": "Enable Megolm using the owning client's active crypto store."},
                "user_id": {"type": "string", "description": "Full Matrix user ID for invite."},
                "event_id": {"type": "string", "description": "Event ID for redact."},
                "reason": {"type": "string"},
            },
            "required": ["action"],
        },
    },
    handler=_matrix_room_admin,
    is_async=True,
)
