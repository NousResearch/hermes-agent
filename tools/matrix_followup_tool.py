"""Configure one Matrix turn's reaction-triggered follow-up."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gateway.session_context import get_session_env, get_session_transport
from tools.registry import registry


async def _matrix_followup(args: dict[str, Any]) -> str:
    session_key = get_session_env("HERMES_SESSION_KEY")
    room_id = get_session_env("HERMES_SESSION_CHAT_ID")
    requester = get_session_env("HERMES_SESSION_USER_ID")
    adapter, owner_loop = get_session_transport()
    if (
        get_session_env("HERMES_SESSION_PLATFORM") != "matrix"
        or not session_key
        or not room_id
        or not requester
        or adapter is None
    ):
        return json.dumps({"error": "Matrix follow-ups require a live Matrix session"})
    enabled = args.get("enabled")
    if not isinstance(enabled, bool):
        return json.dumps({"error": "enabled must be a boolean"})
    emoji = args.get("emoji", [])
    if "emoji" in args and (
        not isinstance(emoji, list)
        or not emoji
        or any(not isinstance(value, str) or not value.strip() for value in emoji)
    ):
        return json.dumps({
            "error": "emoji must be a non-empty list of non-empty strings"
        })
    if owner_loop is None or not owner_loop.is_running():
        return json.dumps({"error": "Matrix gateway loop is unavailable"})
    selected = tuple(dict.fromkeys(value.strip() for value in emoji)) if enabled else ()
    action = adapter.configure_reaction_followups(
        session_key, enabled, selected, room_id=room_id, requester=requester,
        thread_id=get_session_env("HERMES_SESSION_THREAD_ID"),
        profile=get_session_env("HERMES_SESSION_PROFILE"),
    )
    if owner_loop is not asyncio.get_running_loop():
        try:
            future = asyncio.run_coroutine_threadsafe(action, owner_loop)
        except RuntimeError:
            action.close()
            return json.dumps({"error": "Matrix gateway loop is unavailable"})
        try:
            configured = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=30.0
            )
        except asyncio.TimeoutError:
            future.cancel()
            return json.dumps({"error": "Matrix follow-up action timed out"})
    else:
        configured = await action
    if not configured:
        return json.dumps({"error": "Matrix turn is no longer active"})
    return json.dumps(
        {"success": True, "enabled": enabled, "emoji": list(selected)},
        ensure_ascii=False,
    )


registry.register(
    name="matrix_followup",
    toolset="matrix_followup",
    schema={
        "name": "matrix_followup",
        "description": "Allow one new reaction by the current requester to the final Matrix reply to start a follow-up turn within ten minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "emoji": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Optional emoji filter. Omit to accept any emoji.",
                },
            },
            "required": ["enabled"],
        },
    },
    handler=_matrix_followup,
    is_async=True,
)
