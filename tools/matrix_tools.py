"""Matrix-native gateway tools.

These tools are intentionally context-gated at call time. They are included
only in the Matrix platform toolset, but the runtime guard keeps them safe if a
cached schema is reused outside a Matrix session.
"""

from __future__ import annotations

import os
from typing import Any

from gateway.session_context import get_session_env
from tools.registry import registry, tool_error, tool_result


def _matrix_adapter():
    if get_session_env("HERMES_SESSION_PLATFORM", "").lower() != "matrix":
        return None, "Matrix tools are only available while handling a Matrix conversation."
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if not runner:
            return None, "Matrix gateway is not running."
        adapter = runner.adapters.get(Platform.MATRIX)
        if not adapter:
            return None, "Matrix adapter is not connected."
        return adapter, ""
    except Exception as exc:
        return None, f"Failed to resolve Matrix adapter: {exc}"


def _current_room_id(room_id: str = "") -> str:
    return str(room_id or get_session_env("HERMES_SESSION_CHAT_ID", "") or "")


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false")
    return raw.lower() in ("true", "1", "yes", "on")


def _allowed_rooms() -> set[str]:
    raw = os.getenv("MATRIX_ALLOWED_ROOMS", "")
    return {room.strip() for room in raw.split(",") if room.strip()}


def _admin_gate_enabled(name: str) -> bool:
    return _env_enabled(name, default=False)


def _require_admin_gate(name: str, action: str) -> str:
    if _admin_gate_enabled(name):
        return ""
    return f"{action} requires {name}=true."


def _authorize_room_id(
    requested_room_id: str = "",
    *,
    destructive: bool = False,
) -> tuple[str, str]:
    """Authorize a Matrix tool room target against the current session room.

    Matrix tools are room-local by default. Cross-room operation requires an
    explicit operator opt-in, and destructive/member-changing operations have a
    separate opt-in so read/reaction access cannot silently become redaction or
    invite access.
    """
    current_room_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    room_id = _current_room_id(requested_room_id)
    if not room_id:
        return "", "room_id is required."

    allowed = _allowed_rooms()
    if allowed and room_id not in allowed:
        return "", "Matrix tool target room is not listed in MATRIX_ALLOWED_ROOMS."

    if current_room_id and room_id != current_room_id:
        if not _env_enabled("MATRIX_TOOLS_ALLOW_CROSS_ROOM", default=False):
            return "", (
                "Matrix tools are limited to the current room by default. "
                "Set MATRIX_TOOLS_ALLOW_CROSS_ROOM=true to allow explicit cross-room targets."
            )
        if destructive and not _env_enabled(
            "MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE",
            default=False,
        ):
            return "", (
                "Cross-room Matrix redaction/invite actions require "
                "MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE=true."
            )
    return room_id, ""


def _run(coro):
    from model_tools import _run_async

    return _run_async(coro)


def _ok(success: bool, **extra: Any) -> str:
    return tool_result(success=bool(success), **extra)


def matrix_send_reaction(args, **kw) -> str:
    room_id, auth_error = _authorize_room_id(args.get("room_id", ""))
    if auth_error:
        return tool_error(auth_error)
    event_id = str(args.get("event_id", "") or "")
    emoji = str(args.get("emoji", "") or "")
    if not room_id or not event_id or not emoji:
        return tool_error("room_id, event_id, and emoji are required.")
    adapter, err = _matrix_adapter()
    if err:
        return tool_error(err)
    event = _run(adapter._send_reaction(room_id, event_id, emoji))
    return _ok(bool(event), event_id=str(event) if event else "")


def matrix_redact_message(args, **kw) -> str:
    gate_error = _require_admin_gate("MATRIX_TOOLS_ALLOW_REDACTION", "Matrix redaction")
    if gate_error:
        return tool_error(gate_error)
    room_id, auth_error = _authorize_room_id(
        args.get("room_id", ""),
        destructive=True,
    )
    if auth_error:
        return tool_error(auth_error)
    event_id = str(args.get("event_id", "") or "")
    reason = str(args.get("reason", "") or "")
    if not room_id or not event_id:
        return tool_error("room_id and event_id are required.")
    adapter, err = _matrix_adapter()
    if err:
        return tool_error(err)
    return _ok(_run(adapter.redact_message(room_id, event_id, reason)))


def matrix_create_room(args, **kw) -> str:
    gate_error = _require_admin_gate("MATRIX_TOOLS_ALLOW_ROOM_CREATE", "Matrix room creation")
    if gate_error:
        return tool_error(gate_error)
    preset = str(args.get("preset", "private_chat") or "private_chat")
    if preset == "public_chat" and os.getenv("MATRIX_ALLOW_PUBLIC_ROOMS", "").lower() not in (
        "true",
        "1",
        "yes",
    ):
        return tool_error("Public Matrix room creation requires MATRIX_ALLOW_PUBLIC_ROOMS=true.")
    invite = args.get("invite") or []
    if isinstance(invite, str):
        invite = [u.strip() for u in invite.split(",") if u.strip()]
    adapter, err = _matrix_adapter()
    if err:
        return tool_error(err)
    room_id = _run(
        adapter.create_room(
            name=str(args.get("name", "") or ""),
            topic=str(args.get("topic", "") or ""),
            invite=list(invite),
            is_direct=bool(args.get("is_direct", False)),
            preset=preset,
        )
    )
    if not room_id:
        return tool_error("Matrix room creation failed.")
    return tool_result(success=True, room_id=str(room_id))


def matrix_invite_user(args, **kw) -> str:
    gate_error = _require_admin_gate("MATRIX_TOOLS_ALLOW_INVITES", "Matrix invites")
    if gate_error:
        return tool_error(gate_error)
    room_id, auth_error = _authorize_room_id(
        args.get("room_id", ""),
        destructive=True,
    )
    if auth_error:
        return tool_error(auth_error)
    user_id = str(args.get("user_id", "") or "")
    if not room_id or not user_id:
        return tool_error("room_id and user_id are required.")
    adapter, err = _matrix_adapter()
    if err:
        return tool_error(err)
    return _ok(_run(adapter.invite_user(room_id, user_id)))


def matrix_fetch_history(args, **kw) -> str:
    room_id, auth_error = _authorize_room_id(args.get("room_id", ""))
    if auth_error:
        return tool_error(auth_error)
    try:
        limit = int(args.get("limit", 20) or 20)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer.")
    limit = max(1, min(100, limit))
    adapter, err = _matrix_adapter()
    if err:
        return tool_error(err)
    events = _run(adapter.fetch_history(room_id, limit, str(args.get("from_token", "") or "")))
    return tool_result(success=True, events=events)


def matrix_set_presence(args, **kw) -> str:
    state = str(args.get("state", "online") or "online")
    status_msg = str(args.get("status_msg", "") or "")
    adapter, err = _matrix_adapter()
    if err:
        return tool_error(err)
    return _ok(_run(adapter.set_presence(state, status_msg)))


registry.register(
    name="matrix_send_reaction",
    toolset="matrix",
    schema={
        "name": "matrix_send_reaction",
        "description": "Add a Matrix reaction to a message in the current room or a specified room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "description": "Matrix room ID. Defaults to the current Matrix room."},
                "event_id": {"type": "string", "description": "Target Matrix event ID."},
                "emoji": {"type": "string", "description": "Reaction emoji to add."},
            },
            "required": ["event_id", "emoji"],
        },
    },
    handler=matrix_send_reaction,
)

registry.register(
    name="matrix_redact_message",
    toolset="matrix",
    schema={
        "name": "matrix_redact_message",
        "description": "Redact a Matrix message or event when MATRIX_TOOLS_ALLOW_REDACTION=true and the bot has permission.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "description": "Matrix room ID. Defaults to the current Matrix room."},
                "event_id": {"type": "string", "description": "Matrix event ID to redact."},
                "reason": {"type": "string", "description": "Optional redaction reason."},
            },
            "required": ["event_id"],
        },
    },
    handler=matrix_redact_message,
)

registry.register(
    name="matrix_create_room",
    toolset="matrix",
    schema={
        "name": "matrix_create_room",
        "description": "Create a private Matrix room when MATRIX_TOOLS_ALLOW_ROOM_CREATE=true and optionally invite users. Public rooms require MATRIX_ALLOW_PUBLIC_ROOMS=true.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Room name."},
                "topic": {"type": "string", "description": "Room topic."},
                "invite": {"type": "array", "items": {"type": "string"}, "description": "Matrix user IDs to invite."},
                "is_direct": {"type": "boolean", "description": "Whether to mark the room as a direct chat."},
                "preset": {
                    "type": "string",
                    "enum": ["private_chat", "trusted_private_chat", "public_chat"],
                    "description": "Matrix create-room preset. Defaults to private_chat.",
                },
            },
            "required": [],
        },
    },
    handler=matrix_create_room,
)

registry.register(
    name="matrix_invite_user",
    toolset="matrix",
    schema={
        "name": "matrix_invite_user",
        "description": "Invite a Matrix user when MATRIX_TOOLS_ALLOW_INVITES=true to the current room or a specified room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "description": "Matrix room ID. Defaults to the current Matrix room."},
                "user_id": {"type": "string", "description": "Matrix user ID to invite."},
            },
            "required": ["user_id"],
        },
    },
    handler=matrix_invite_user,
)

registry.register(
    name="matrix_fetch_history",
    toolset="matrix",
    schema={
        "name": "matrix_fetch_history",
        "description": "Fetch recent Matrix room history from the current room or a specified room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "description": "Matrix room ID. Defaults to the current Matrix room."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum events to fetch."},
                "from_token": {"type": "string", "description": "Optional pagination token."},
            },
            "required": [],
        },
    },
    handler=matrix_fetch_history,
)

registry.register(
    name="matrix_set_presence",
    toolset="matrix",
    schema={
        "name": "matrix_set_presence",
        "description": "Set the Matrix bot presence state.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["online", "offline", "unavailable"]},
                "status_msg": {"type": "string", "description": "Optional presence status message."},
            },
            "required": [],
        },
    },
    handler=matrix_set_presence,
)
