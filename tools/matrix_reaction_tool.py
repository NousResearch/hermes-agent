"""React to a Matrix event through the adapter for the current session."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agent.async_utils import safe_schedule_threadsafe
from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry

logger = logging.getLogger(__name__)

_REACTION_TIMEOUT_SECONDS = 30.0


async def _react_in_session(
    adapter: Any,
    room_id: str,
    requester: str,
    action: str,
    message_id: str,
    emoji: Any,
) -> dict[str, Any]:
    access = await adapter.check_session_access(room_id, requester)
    if access.error:
        return {"error": access.error}
    if action == "unreact":
        return await adapter.remove_reaction(chat_id=room_id, message_id=message_id)
    return await adapter.add_reaction(
        chat_id=room_id, emoji=emoji.strip(), message_id=message_id
    )


async def _matrix_reaction(args: dict[str, Any]) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    if (
        get_session_env("HERMES_SESSION_PLATFORM") != "matrix"
        or not room_id
        or not requester
        or adapter is None
    ):
        return json.dumps({"error": "Matrix reactions require a live Matrix session"})

    action = args.get("action")
    if action not in {"react", "unreact"}:
        return json.dumps({"error": "action must be react or unreact"})

    message_id = args.get("message_id") or get_session_env("HERMES_SESSION_MESSAGE_ID")
    if not isinstance(message_id, str) or not message_id.startswith("$"):
        return json.dumps({
            "error": "message_id is required when the session has no current Matrix event"
        })

    emoji = args.get("emoji")
    if action == "react" and (not isinstance(emoji, str) or not emoji.strip()):
        return json.dumps({"error": "emoji is required for react"})

    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})

    future = safe_schedule_threadsafe(
        _react_in_session(adapter, room_id, requester, action, message_id, emoji), owner_loop,
        logger=logger, log_message="matrix_reaction: failed to schedule on the gateway loop",
    )
    if future is None:
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    # Cancelling the send cannot withdraw a reaction that the homeserver
    # has already accepted, and it would stop the adapter recording the
    # reaction for a later unreact.
    try:
        result = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=_REACTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return json.dumps({"error": "Matrix reaction timed out and may still complete"})
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="matrix_reaction",
    toolset="matrix_reaction",
    schema={
        "name": "matrix_reaction",
        "description": "Add or remove the agent's emoji reaction to a message in the current Matrix room.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["react", "unreact"]},
                "emoji": {
                    "type": "string",
                    "description": "Emoji to add when action is react.",
                },
                "message_id": {
                    "type": "string",
                    "description": "Matrix event ID. Defaults to the current inbound message.",
                },
            },
            "required": ["action"],
        },
    },
    handler=_matrix_reaction,
    is_async=True,
)
