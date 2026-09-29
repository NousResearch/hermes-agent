"""Read bounded Matrix history through the adapter that owns the current turn."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agent.async_utils import safe_schedule_threadsafe
from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry

logger = logging.getLogger(__name__)


async def _matrix_read(args: dict[str, Any]) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    read_context = getattr(adapter, "read_matrix_context", None)
    if get_session_env("HERMES_SESSION_PLATFORM") != "matrix" or not room_id or not requester or not callable(read_context):
        return json.dumps({"error": "Matrix reads require a live Matrix session"})

    kind = args.get("kind")
    event_id = args.get("event_id")
    if kind == "thread" and not event_id:
        event_id = get_session_env("HERMES_SESSION_THREAD_ID")
    if kind not in {"room", "thread", "event"}:
        return json.dumps({"error": "kind must be room, thread, or event"})
    if kind != "room" and (not isinstance(event_id, str) or not event_id.startswith("$")):
        return json.dumps({"error": "event_id is required for thread and event reads"})

    limit = args.get("limit", 20)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        return json.dumps({"error": "limit must be between 1 and 50"})

    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    future = safe_schedule_threadsafe(
        read_context(kind, room_id, event_id, limit, requester=requester), owner_loop,
        logger=logger, log_message="matrix_read: failed to schedule on the gateway loop",
    )
    if future is None:
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    try:
        result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=60.0)
    except asyncio.TimeoutError:
        future.cancel()
        return json.dumps({"error": "Matrix read timed out"})
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="matrix_read",
    toolset="matrix_read",
    schema={
        "name": "matrix_read",
        "description": "Read recent messages, one thread, or one event in the current Matrix room.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["room", "thread", "event"]},
                "event_id": {"type": "string", "description": "Event ID for an event read, or thread root. A thread read defaults to the current thread."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "required": ["kind"],
        },
    },
    handler=_matrix_read,
    is_async=True,
)
