"""LLM-callable Matrix tools.

These tools wrap the live Matrix gateway adapter when Hermes is connected to
Matrix. They are intentionally thin wrappers around the existing
``MatrixAdapter`` public APIs so the tool layer does NOT duplicate Matrix SDK
logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_adapter: Optional[Any] = None
_adapter_lock = threading.Lock()

_MAX_EMOJI_LEN = 32
_MAX_REASON_LEN = 500
_MAX_NAME_LEN = 255
_MAX_TOPIC_LEN = 1000
_MAX_STATUS_LEN = 255
_ALLOWED_PRESETS = frozenset(("private_chat", "public_chat"))


def set_matrix_adapter(adapter: Optional[Any]) -> None:
    """Bind the currently connected Matrix adapter for tool use."""
    global _adapter
    with _adapter_lock:
        _adapter = adapter


def _check_matrix_connected() -> bool:
    """Return True only when a live Matrix adapter instance is available."""
    with _adapter_lock:
        return _adapter is not None


def _ensure_adapter() -> Any:
    with _adapter_lock:
        adapter = _adapter
    if adapter is None:
        raise RuntimeError("Matrix adapter is not connected")
    return adapter


def _run_async(coro):
    """Run an async coroutine from a sync tool handler."""
    adapter = _ensure_adapter()
    adapter_loop = getattr(adapter, "_loop", None)
    if adapter_loop is not None and adapter_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, adapter_loop)
        try:
            return future.result(timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)

    return asyncio.run(coro)


def _json_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _handle_send_reaction(args: dict, **_kw) -> str:
    room_id = str(args.get("room_id", "")).strip()
    event_id = str(args.get("event_id", "")).strip()
    emoji = str(args.get("emoji", "")).strip()

    if not room_id or not room_id.startswith("!"):
        return _json_error("room_id is required and must start with '!'")
    if not event_id or not event_id.startswith("$"):
        return _json_error("event_id is required and must start with '$'")
    if not emoji:
        return _json_error("emoji is required and must be non-empty")
    if len(emoji) > _MAX_EMOJI_LEN:
        return _json_error(f"emoji must be at most {_MAX_EMOJI_LEN} characters")

    try:
        adapter = _ensure_adapter()
        reaction_event_id = _run_async(adapter.send_reaction(room_id, event_id, emoji))
        return json.dumps({"success": bool(reaction_event_id), "reaction_event_id": reaction_event_id}, ensure_ascii=False)
    except Exception as exc:
        logger.error("matrix_send_reaction error: %s", exc)
        return _json_error(f"Failed to send reaction: {exc}")


def _handle_redact_message(args: dict, **_kw) -> str:
    room_id = str(args.get("room_id", "")).strip()
    event_id = str(args.get("event_id", "")).strip()
    reason = str(args.get("reason", ""))[:_MAX_REASON_LEN]

    if not room_id or not room_id.startswith("!"):
        return _json_error("room_id is required and must start with '!'")
    if not event_id or not event_id.startswith("$"):
        return _json_error("event_id is required and must start with '$'")

    try:
        adapter = _ensure_adapter()
        ok = _run_async(adapter.redact_message(room_id, event_id, reason=reason))
        return json.dumps({"success": bool(ok)}, ensure_ascii=False)
    except Exception as exc:
        logger.error("matrix_redact_message error: %s", exc)
        return _json_error(f"Failed to redact message: {exc}")


def _handle_create_room(args: dict, **_kw) -> str:
    name = str(args.get("name", ""))[:_MAX_NAME_LEN]
    topic = str(args.get("topic", ""))[:_MAX_TOPIC_LEN]
    invite = args.get("invite", [])
    is_direct = bool(args.get("is_direct", False))
    preset = str(args.get("preset", "private_chat"))

    if invite and not isinstance(invite, list):
        return _json_error("invite must be a list of user IDs")
    for user_id in invite or []:
        if not isinstance(user_id, str) or not user_id.startswith("@"):
            return _json_error(f"Invalid user ID in invite list: {user_id!r}")
    if preset not in _ALLOWED_PRESETS:
        return _json_error(f"Invalid preset: {preset!r}. Must be one of: {', '.join(sorted(_ALLOWED_PRESETS))}")
    if preset == "public_chat" and os.getenv("MATRIX_ALLOW_PUBLIC_ROOMS", "").lower() not in ("true", "1", "yes"):
        return _json_error("Public room creation is disabled by default. Set MATRIX_ALLOW_PUBLIC_ROOMS=true to allow.")

    try:
        adapter = _ensure_adapter()
        room_id = _run_async(adapter.create_room(name=name, topic=topic, invite=invite, is_direct=is_direct, preset=preset))
        if not room_id:
            return json.dumps({"success": False, "error": "Room creation failed"}, ensure_ascii=False)
        return json.dumps({"success": True, "room_id": room_id}, ensure_ascii=False)
    except Exception as exc:
        logger.error("matrix_create_room error: %s", exc)
        return _json_error(f"Failed to create room: {exc}")


def _handle_invite_user(args: dict, **_kw) -> str:
    room_id = str(args.get("room_id", "")).strip()
    user_id = str(args.get("user_id", "")).strip()

    if not room_id or not room_id.startswith("!"):
        return _json_error("room_id is required and must start with '!'")
    if not user_id or not user_id.startswith("@"):
        return _json_error("user_id is required and must start with '@'")

    try:
        adapter = _ensure_adapter()
        ok = _run_async(adapter.invite_user(room_id, user_id))
        return json.dumps({"success": bool(ok)}, ensure_ascii=False)
    except Exception as exc:
        logger.error("matrix_invite_user error: %s", exc)
        return _json_error(f"Failed to invite user: {exc}")


def _handle_fetch_history(args: dict, **_kw) -> str:
    room_id = str(args.get("room_id", "")).strip()
    limit = args.get("limit", 50)
    start = str(args.get("start", "")).strip()

    if not room_id or not room_id.startswith("!"):
        return _json_error("room_id is required and must start with '!'")
    if not isinstance(limit, int) or limit < 1:
        limit = 50
    limit = min(limit, 200)

    try:
        adapter = _ensure_adapter()
        messages = _run_async(adapter.fetch_room_history(room_id, limit=limit, start=start))
        return json.dumps({"count": len(messages), "messages": messages}, ensure_ascii=False)
    except Exception as exc:
        logger.error("matrix_fetch_history error: %s", exc)
        return _json_error(f"Failed to fetch history: {exc}")


def _handle_set_presence(args: dict, **_kw) -> str:
    state = str(args.get("state", "")).strip().lower()
    status_msg = str(args.get("status_msg", ""))[:_MAX_STATUS_LEN]

    if state not in ("online", "offline", "unavailable"):
        return _json_error(f"Invalid state: {state!r}. Must be 'online', 'offline', or 'unavailable'")

    try:
        adapter = _ensure_adapter()
        ok = _run_async(adapter.set_presence(state=state, status_msg=status_msg))
        return json.dumps({"success": bool(ok)}, ensure_ascii=False)
    except Exception as exc:
        logger.error("matrix_set_presence error: %s", exc)
        return _json_error(f"Failed to set presence: {exc}")


registry.register(
    name="matrix_send_reaction",
    toolset="matrix",
    schema={
        "name": "matrix_send_reaction",
        "description": "Send an emoji reaction to a specific message in a Matrix room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "description": "Matrix room ID starting with '!'."},
                "event_id": {"type": "string", "description": "Target Matrix event ID starting with '$'."},
                "emoji": {"type": "string", "description": "Emoji reaction to apply."},
            },
            "required": ["room_id", "event_id", "emoji"],
        },
    },
    handler=_handle_send_reaction,
    check_fn=_check_matrix_connected,
    emoji="🟣",
)

registry.register(
    name="matrix_redact_message",
    toolset="matrix",
    schema={
        "name": "matrix_redact_message",
        "description": "Redact a Matrix event from a room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string"},
                "event_id": {"type": "string"},
                "reason": {"type": "string", "description": "Optional reason for redaction."},
            },
            "required": ["room_id", "event_id"],
        },
    },
    handler=_handle_redact_message,
    check_fn=_check_matrix_connected,
    emoji="🟣",
)

registry.register(
    name="matrix_create_room",
    toolset="matrix",
    schema={
        "name": "matrix_create_room",
        "description": "Create a Matrix room. Public room creation is gated by MATRIX_ALLOW_PUBLIC_ROOMS.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "topic": {"type": "string"},
                "invite": {"type": "array", "items": {"type": "string"}},
                "is_direct": {"type": "boolean"},
                "preset": {"type": "string", "enum": ["private_chat", "public_chat"]},
            },
            "required": [],
        },
    },
    handler=_handle_create_room,
    check_fn=_check_matrix_connected,
    emoji="🟣",
)

registry.register(
    name="matrix_invite_user",
    toolset="matrix",
    schema={
        "name": "matrix_invite_user",
        "description": "Invite a user to a Matrix room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": ["room_id", "user_id"],
        },
    },
    handler=_handle_invite_user,
    check_fn=_check_matrix_connected,
    emoji="🟣",
)

registry.register(
    name="matrix_fetch_history",
    toolset="matrix",
    schema={
        "name": "matrix_fetch_history",
        "description": "Fetch recent message history from a Matrix room using the adapter's existing history API.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string"},
                "limit": {"type": "integer", "description": "Maximum messages to return (1-200)."},
                "start": {"type": "string", "description": "Optional pagination token."},
            },
            "required": ["room_id"],
        },
    },
    handler=_handle_fetch_history,
    check_fn=_check_matrix_connected,
    emoji="🟣",
)

registry.register(
    name="matrix_set_presence",
    toolset="matrix",
    schema={
        "name": "matrix_set_presence",
        "description": "Set Matrix presence to online, offline, or unavailable.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["online", "offline", "unavailable"]},
                "status_msg": {"type": "string"},
            },
            "required": ["state"],
        },
    },
    handler=_handle_set_presence,
    check_fn=_check_matrix_connected,
    emoji="🟣",
)
