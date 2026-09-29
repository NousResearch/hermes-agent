"""Matrix notification reads and deliberate receipts for the current session."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry


async def _matrix_unread_action(args: dict[str, Any], *, mark_read: bool) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    if get_session_env("HERMES_SESSION_PLATFORM") != "matrix" or not room_id or not requester or adapter is None:
        return json.dumps({"error": "Matrix unread actions require a live Matrix session"})
    if args.get("room_id", room_id) != room_id:
        return json.dumps({"error": "Matrix unread actions are limited to the current room"})
    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    thread_id = args.get("thread_id")
    if thread_id is None and not mark_read:
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID") or "main"
    if mark_read:
        operation = adapter.mark_matrix_read(
            room_id, args.get("event_id"), thread_id, args.get("visibility"), requester=requester,
        )
    else:
        operation = adapter.read_matrix_unread(room_id, thread_id, requester=requester)
    if owner_loop is asyncio.get_running_loop():
        return json.dumps(await operation, ensure_ascii=False)
    try:
        future = asyncio.run_coroutine_threadsafe(operation, owner_loop)
    except RuntimeError:
        operation.close()
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    result = await asyncio.shield(asyncio.wrap_future(future))
    return json.dumps(result, ensure_ascii=False)


async def _matrix_unread(args: dict[str, Any]) -> str:
    return await _matrix_unread_action(args, mark_read=False)


async def _matrix_mark_read(args: dict[str, Any]) -> str:
    return await _matrix_unread_action(args, mark_read=True)


registry.register(
    name="matrix_unread", toolset="matrix_unread", is_async=True, handler=_matrix_unread,
    schema={
        "name": "matrix_unread",
        "description": "Read the bot account's observed Matrix notification and highlight counts, based on its push rules. These are not all unread messages or the requesting person's unread state. Unknown counts remain unavailable. Defaults to the current thread, or main timeline.",
        "parameters": {"type": "object", "properties": {
            "thread_id": {"type": "string", "description": "main for the main timeline, or a thread root event ID."},
        }},
    },
)

registry.register(
    name="matrix_mark_read", toolset="matrix_unread", is_async=True, handler=_matrix_mark_read,
    schema={
        "name": "matrix_mark_read",
        "description": "Explicitly acknowledge a Matrix event for the bot account, even when automatic receipts are disabled. Choose main, a thread root, or room (unthreaded, all timelines). Public receipts are visible to others; private receipts are not. Only room scope resets the room's marked-unread flag. Never advances the fully-read marker. Counts update through later sync. Check each operation's result before retrying.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string", "description": "The exact event to acknowledge."},
            "thread_id": {"type": "string", "description": "main, a thread root event ID, or room for an unthreaded receipt."},
            "visibility": {"type": "string", "enum": ["public", "private"]},
        }, "required": ["event_id", "thread_id", "visibility"]},
    },
)
