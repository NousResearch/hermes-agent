"""React to a Matrix event through the adapter for the current session."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry


async def _matrix_reaction(args: dict[str, Any]) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    adapter, owner_loop = get_session_transport()
    if (
        get_session_env("HERMES_SESSION_PLATFORM") != "matrix"
        or not room_id
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

    if action == "react":
        reaction = adapter.add_reaction(
            chat_id=room_id, emoji=emoji.strip(), message_id=message_id
        )
    else:
        reaction = adapter.remove_reaction(chat_id=room_id, message_id=message_id)

    if owner_loop is not asyncio.get_running_loop():
        try:
            future = asyncio.run_coroutine_threadsafe(reaction, owner_loop)
        except RuntimeError:
            reaction.close()
            return json.dumps({"error": "Matrix gateway loop is unavailable"})
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=30.0)
        except asyncio.TimeoutError:
            future.cancel()
            return json.dumps({"error": "Matrix reaction timed out"})
    else:
        result = await reaction
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
