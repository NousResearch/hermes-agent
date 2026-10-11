"""Change pinned events through the Matrix adapter for the current turn."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry


async def _matrix_pin(args: dict[str, Any]) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    if get_session_env("HERMES_SESSION_PLATFORM") != "matrix" or not room_id or not requester or adapter is None:
        return json.dumps({"error": "Matrix pin actions require a live Matrix session"})

    action = args.get("action")
    if action not in {"pin", "unpin"}:
        return json.dumps({"error": "action must be pin or unpin"})
    event_id = args.get("event_id")
    if not isinstance(event_id, str) or not event_id.startswith("$") or len(event_id) < 2:
        return json.dumps({"error": "event_id must be a Matrix event ID"})
    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})

    from tools.matrix_tool_runtime import run_matrix_mutation

    return await run_matrix_mutation(
        owner_loop,
        lambda interrupted, before_write: adapter.change_matrix_pin(
            action, room_id, event_id, requester=requester,
            interrupt_check=interrupted, before_write=before_write,
        ),
        operation_label="Matrix pin update",
        next_step="Read the current pins with matrix_read kind=pins before retrying",
    )


registry.register(
    name="matrix_pin",
    toolset="matrix_admin",
    schema={
        "name": "matrix_pin",
        "description": "Pin or unpin a Matrix event in the current room. The requesting user and the bot both need room permission to change pins.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["pin", "unpin"]},
                "event_id": {"type": "string", "description": "Event ID of the message to pin or unpin."},
            },
            "required": ["action", "event_id"],
        },
    },
    handler=_matrix_pin,
    is_async=True,
)
