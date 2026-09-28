"""Explicit thread creation on the Matrix transport for the current session."""

from __future__ import annotations

import asyncio
import json
import threading
from functools import partial
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.interrupt import is_thread_interrupted
from tools.registry import registry


async def _matrix_thread_create(args: dict[str, Any]) -> str:
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    error = None
    root_text = args.get("root_text")
    root_event_id = args.get("root_event_id")
    message = args.get("message")
    if (
        get_session_env("HERMES_SESSION_PLATFORM") != "matrix"
        or not room_id
        or not requester
        or adapter is None
    ):
        error = "Matrix thread creation requires a live Matrix session"
    elif args.get("room_id", room_id) != room_id:
        error = "Matrix thread creation is limited to the current room"
    elif not isinstance(message, str) or not message.strip():
        error = "message must contain non-whitespace text"
    elif (root_text is None) == (root_event_id is None):
        error = "Supply either root_text or root_event_id"
    elif root_event_id is not None and (
        not isinstance(root_event_id, str) or not root_event_id.startswith("$")
    ):
        error = "root_event_id must be a Matrix event ID"
    elif root_text is not None and (
        not isinstance(root_text, str) or not root_text.strip()
    ):
        error = "root_text must contain non-whitespace text"
    elif owner_loop is None or not owner_loop.is_running():
        error = "Matrix gateway loop is unavailable"
    if error:
        return json.dumps({"success": False, "error": error})

    interrupted = partial(is_thread_interrupted, threading.get_ident())
    task = None

    async def create():
        nonlocal task
        task = asyncio.current_task()
        return await adapter.create_matrix_thread(
            room_id,
            message,
            requester=requester,
            root_text=root_text,
            root_event_id=root_event_id,
            interrupted=interrupted,
        )

    operation = create()
    if owner_loop is asyncio.get_running_loop():
        result = await operation
        return json.dumps(result, ensure_ascii=False)
    try:
        future = asyncio.run_coroutine_threadsafe(operation, owner_loop)
    except RuntimeError:
        operation.close()
        return json.dumps({
            "success": False,
            "error": "Matrix gateway loop is unavailable",
        })
    wrapped = asyncio.wrap_future(future)
    try:
        result = await asyncio.shield(wrapped)
    except asyncio.CancelledError:
        if task is None:
            future.cancel()
            return json.dumps({
                "success": False,
                "error": "Matrix thread creation cancelled",
            })
        owner_loop.call_soon_threadsafe(task.cancel)
        result = await asyncio.shield(wrapped)
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="matrix_thread_create",
    toolset="matrix_threads",
    schema={
        "name": "matrix_thread_create",
        "description": "Create a Matrix thread in the current joined room with an initial message. Supply visible root text for a new main-timeline message, or an existing main-timeline root event ID. The current conversation stays in its original session. A partial result lists confirmed deliveries; do not retry a delivered root.",
        "parameters": {
            "type": "object",
            "properties": {
                "root_text": {
                    "type": "string",
                    "description": "Visible root text, including any desired label. Matrix threads have no separate name.",
                },
                "root_event_id": {
                    "type": "string",
                    "description": "Existing eligible main-timeline message in the current room. Thread replies and edits cannot become roots.",
                },
                "message": {
                    "type": "string",
                    "description": "First message to send inside the thread.",
                },
            },
            "required": ["message"],
        },
    },
    handler=_matrix_thread_create,
    is_async=True,
)
