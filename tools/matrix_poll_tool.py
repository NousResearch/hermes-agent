"""Native poll tools for the owning Matrix session."""

from __future__ import annotations

import asyncio
from functools import partial
import json
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry


async def _matrix_poll(action: str, args: dict[str, Any]) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    if get_session_env("HERMES_SESSION_PLATFORM") != "matrix" or not room_id or not requester or adapter is None:
        return json.dumps({"error": "Matrix polls require a live Matrix session"})
    if args.get("room_id", room_id) != room_id:
        return json.dumps({"error": "Matrix polls are limited to the current room"})
    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    operation = adapter.matrix_poll_action(room_id, requester, action, args)
    if owner_loop is asyncio.get_running_loop():
        return json.dumps(await operation, ensure_ascii=False)
    try:
        future = asyncio.run_coroutine_threadsafe(operation, owner_loop)
    except RuntimeError:
        operation.close()
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    try:
        result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=60.0)
    except asyncio.TimeoutError:
        future.cancel()
        return json.dumps({"error": "Matrix poll operation timed out"})
    return json.dumps(result, ensure_ascii=False)


_POLL_ID = {"type": "string", "description": "Native poll start event ID in the current room."}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 200, "default": 100, "description": "Maximum relations to examine. Incomplete results contain no vote totals."}
_OPERATIONS = {
    "create": (
        "Create a native Matrix poll as the bot in the current room. Poll events do not automatically start a model turn.",
        {"question": {"type": "string", "maxLength": 1200},
         "answers": {"type": "array", "minItems": 2, "maxItems": 20, "items": {"type": "string", "maxLength": 300}},
         "kind": {"type": "string", "enum": ["disclosed", "undisclosed"], "default": "disclosed"},
         "max_selections": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1}},
        ["question", "answers"],
    ),
    "vote": (
        "Vote as the Matrix bot in a native poll. A response replaces the bot's previous vote. Empty answers withdraw the vote.",
        {"poll_id": _POLL_ID, "answers": {"type": "array", "items": {"type": "string"}}, "limit": _LIMIT},
        ["poll_id", "answers"],
    ),
    "results": (
        "Read bounded native Matrix poll results in the current room. Open undisclosed polls hide totals. Truncated or undecryptable relations make results incomplete.",
        {"poll_id": _POLL_ID, "limit": _LIMIT}, ["poll_id"],
    ),
    "close": (
        "Close a native Matrix poll as the bot. Requires poll ownership or permission to redact other users' events. Closing reveals undisclosed results.",
        {"poll_id": _POLL_ID, "limit": _LIMIT}, ["poll_id"],
    ),
}

for _action, (_description, _properties, _required) in _OPERATIONS.items():
    _name = f"matrix_poll_{_action}"
    registry.register(
        name=_name, toolset="matrix_polls",
        schema={"name": _name, "description": _description,
                "parameters": {"type": "object", "properties": _properties, "required": _required}},
        handler=partial(_matrix_poll, _action), is_async=True,
    )
