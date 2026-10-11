"""Change pinned events through the Matrix adapter for the current turn."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.interrupt import acting_for_tid, is_thread_interrupted
from tools.registry import registry

_monotonic = time.monotonic


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

    worker_id = threading.get_ident()
    parent_id = acting_for_tid.get()
    cancelled = threading.Event()
    write_started = threading.Event()

    def interrupted() -> bool:
        return cancelled.is_set() or is_thread_interrupted(worker_id) or is_thread_interrupted(parent_id)

    if interrupted():
        return json.dumps({"error": "Matrix pin update interrupted"})

    change = adapter.change_matrix_pin(
        action, room_id, event_id, requester=requester,
        interrupt_check=interrupted, before_write=write_started.set,
    )
    owner_task: asyncio.Task[dict[str, Any]] | None = None

    async def run_change() -> dict[str, Any]:
        nonlocal owner_task
        owner_task = asyncio.current_task()
        if cancelled.is_set():
            change.close()
            return {"error": "Matrix pin update interrupted"}
        return await change

    def cancel_change() -> None:
        if owner_task is not None:
            owner_task.cancel()

    operation = run_change()
    if owner_loop is asyncio.get_running_loop():
        pending = asyncio.create_task(operation)
    else:
        try:
            future = asyncio.run_coroutine_threadsafe(operation, owner_loop)
        except RuntimeError:
            operation.close()
            change.close()
            return json.dumps({"error": "Matrix gateway loop is unavailable"})
        pending = asyncio.wrap_future(future)

    deadline = _monotonic() + 30.0
    failure: str | None = None
    try:
        while not pending.done():
            if interrupted():
                failure = "Matrix pin update interrupted"
                break
            remaining = deadline - _monotonic()
            if remaining <= 0:
                failure = "Matrix pin update timed out"
                break
            await asyncio.wait({pending}, timeout=min(0.1, remaining))
        if failure is None:
            result = await pending
    except asyncio.CancelledError:
        cancelled.set()
        raise
    finally:
        if not pending.done():
            cancelled.set()
            owner_loop.call_soon_threadsafe(cancel_change)
            # Cancelling the result future would finish before the owning task stops.
            completion = asyncio.gather(pending, return_exceptions=True)
            cancellation: asyncio.CancelledError | None = None
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            if cancellation is not None:
                raise cancellation
    if failure is None:
        return json.dumps(result, ensure_ascii=False)
    if not write_started.is_set():
        return json.dumps({"error": failure})
    return json.dumps({
        "error": f"{failure} after the change was sent to the homeserver",
        "outcome": "unknown",
        "next_step": "Read the current pins with matrix_read kind=pins before retrying",
    })


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
