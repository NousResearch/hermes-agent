"""Read bounded Matrix history through the adapter that owns the current turn."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agent.async_utils import safe_schedule_threadsafe
from gateway.session_context import get_session_env, get_session_transport
from plugins.platforms.matrix.discovery import DiscoveryRequest
from tools.registry import registry

logger = logging.getLogger(__name__)

_READ_DEADLINE_SECONDS = 60.0


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
    history_kinds = {"room", "thread", "event"}
    inspection_kinds = {"state", "members", "permissions", "pins"}
    discovery_kinds = {"joined_rooms", "joined_spaces", "users"}
    if not isinstance(kind, str) or kind not in history_kinds | inspection_kinds | discovery_kinds:
        return json.dumps({"error": "kind must be room, thread, event, state, members, permissions, pins, joined_rooms, joined_spaces, or users"})
    if kind in {"thread", "event"} and (not isinstance(event_id, str) or not event_id.startswith("$")):
        return json.dumps({"error": "event_id is required for thread and event reads"})

    limit = args.get("limit", 20)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        return json.dumps({"error": "limit must be between 1 and 50"})

    if kind in discovery_kinds:
        try:
            request = DiscoveryRequest.parse(kind, limit, args.get("search_term"))
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        if not callable(getattr(adapter, "discover_matrix", None)):
            return json.dumps({"error": "Matrix discovery requires a Matrix adapter"})

    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})

    if kind in discovery_kinds:
        read = adapter.discover_matrix(kind, room_id, limit, requester=requester, search_term=request.search_term)
    elif kind in inspection_kinds:
        read = adapter.inspect_matrix_room(kind, room_id, limit, requester=requester)
    else:
        read = read_context(kind, room_id, event_id, limit, requester=requester)
    future = safe_schedule_threadsafe(
        read, owner_loop,
        logger=logger, log_message="matrix_read: failed to schedule on the gateway loop",
    )
    if future is None:
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    try:
        result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=_READ_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        future.cancel()
        return json.dumps({"error": "Matrix read timed out"})
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="matrix_read",
    toolset="matrix_read",
    schema={
        "name": "matrix_read",
        "description": (
            "Read messages, events, state, joined members, permissions, or pins in the current Matrix room. "
            "Room and thread reads list events oldest first: a room read returns the latest messages, "
            "and a thread read returns the thread root followed by its latest replies. Pins follow the "
            "room's pinned order. `skipped` counts events in the read window that have no readable "
            "message body, such as redacted messages. `errors` lists events that could not be read "
            "or decrypted, with the reason. "
            "List allowed joined rooms and joined Spaces that the requester is also in, or search the "
            "homeserver user directory for users who share a joined room with the requester, through the "
            "live Matrix session. Discovery never joins rooms; results can be truncated and include errors."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["room", "thread", "event", "state", "members", "permissions", "pins", "joined_rooms", "joined_spaces", "users"], "description": "joined_rooms lists non-Space rooms; joined_spaces lists Spaces. Both list only rooms that the requester has joined and inspect at most 50 joined candidates. users searches the homeserver directory and returns only users who share one of those rooms with the requester."},
                "search_term": {"type": "string", "minLength": 1, "maxLength": 200, "description": "Required for users: a user ID or display-name search term."},
                "event_id": {"type": "string", "description": "Event ID for an event read, or thread root. A thread read defaults to the current thread."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20,
                          "description": "Maximum number of events or members to return. A thread read counts the root."},
            },
            "required": ["kind"],
        },
    },
    handler=_matrix_read,
    is_async=True,
)
