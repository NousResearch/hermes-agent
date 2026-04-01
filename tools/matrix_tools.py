"""Matrix room management and interaction tools.

Registers LLM-callable tools for controlling Matrix rooms and interactions:
- ``matrix_send_reaction`` -- react to a message with an emoji
- ``matrix_redact_message`` -- redact (delete) a message
- ``matrix_create_room`` -- create a new Matrix room
- ``matrix_invite_user`` -- invite a user to a room
- ``matrix_fetch_history`` -- fetch recent message history from a room
- ``matrix_set_presence`` -- set the bot's presence status

The adapter instance is injected at runtime via ``set_matrix_adapter()``.
"""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter reference (set at runtime by MatrixAdapter.connect)
# ---------------------------------------------------------------------------

_adapter: Optional[Any] = None


def set_matrix_adapter(adapter: Optional[Any]) -> None:
    """Set (or clear) the running MatrixAdapter instance."""
    global _adapter
    _adapter = adapter


def _check_matrix_available() -> bool:
    """Tool is only available when a MatrixAdapter is connected."""
    return _adapter is not None


# ---------------------------------------------------------------------------
# Async bridge (sync handler → async adapter methods)
# ---------------------------------------------------------------------------

_MAX_EMOJI_LEN = 32
_MAX_REASON_LEN = 500
_MAX_NAME_LEN = 255
_MAX_TOPIC_LEN = 1000
_MAX_STATUS_LEN = 255
_ALLOWED_PRESETS = frozenset(("private_chat", "public_chat"))


def _run_async(coro):
    """Run an async coroutine from a sync handler.

    Schedules the coroutine on the adapter's own event loop via
    ``run_coroutine_threadsafe`` when possible, to avoid creating a
    second event loop that can't share matrix-nio client state.
    Falls back to ``asyncio.run()`` only when no loop is available.
    """
    # Prefer the adapter's event loop for nio client safety.
    adapter_loop = getattr(_adapter, "_loop", None) if _adapter else None
    if adapter_loop is not None and adapter_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, adapter_loop)
        return future.result(timeout=30)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Schedule on the current running loop.
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=30)
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _ensure_adapter():
    """Return the adapter or raise if unavailable."""
    if _adapter is None:
        raise RuntimeError("Matrix adapter is not connected")
    return _adapter


def _handle_send_reaction(args: dict, **kw) -> str:
    """Handler for matrix_send_reaction tool."""
    room_id = args.get("room_id", "").strip()
    event_id = args.get("event_id", "").strip()
    emoji = args.get("emoji", "").strip()

    if not room_id or not room_id.startswith("!"):
        return json.dumps({"error": "room_id is required and must start with '!'"})
    if not event_id or not event_id.startswith("$"):
        return json.dumps({"error": "event_id is required and must start with '$'"})
    if not emoji:
        return json.dumps({"error": "emoji is required and must be non-empty"})
    if len(emoji) > _MAX_EMOJI_LEN:
        return json.dumps({"error": f"emoji must be at most {_MAX_EMOJI_LEN} characters"})

    try:
        adapter = _ensure_adapter()
        result = _run_async(adapter._send_reaction(room_id, event_id, emoji))
        return json.dumps({"success": bool(result)})
    except Exception as e:
        logger.error("matrix_send_reaction error: %s", e)
        return json.dumps({"error": f"Failed to send reaction: {e}"})


def _handle_redact_message(args: dict, **kw) -> str:
    """Handler for matrix_redact_message tool."""
    room_id = args.get("room_id", "").strip()
    event_id = args.get("event_id", "").strip()
    reason = args.get("reason", "")

    if not room_id or not room_id.startswith("!"):
        return json.dumps({"error": "room_id is required and must start with '!'"})
    if not event_id or not event_id.startswith("$"):
        return json.dumps({"error": "event_id is required and must start with '$'"})
    if len(reason) > _MAX_REASON_LEN:
        reason = reason[:_MAX_REASON_LEN]

    try:
        adapter = _ensure_adapter()
        result = _run_async(adapter.redact_message(room_id, event_id, reason=reason))
        return json.dumps({"success": bool(result)})
    except Exception as e:
        logger.error("matrix_redact_message error: %s", e)
        return json.dumps({"error": f"Failed to redact message: {e}"})


def _handle_create_room(args: dict, **kw) -> str:
    """Handler for matrix_create_room tool."""
    name = args.get("name", "")
    topic = args.get("topic", "")
    invite = args.get("invite", [])
    is_direct = args.get("is_direct", False)
    preset = args.get("preset", "private_chat")

    if name and len(name) > _MAX_NAME_LEN:
        name = name[:_MAX_NAME_LEN]
    if topic and len(topic) > _MAX_TOPIC_LEN:
        topic = topic[:_MAX_TOPIC_LEN]
    if invite and not isinstance(invite, list):
        return json.dumps({"error": "invite must be a list of user IDs"})
    for uid in (invite or []):
        if not isinstance(uid, str) or not uid.startswith("@"):
            return json.dumps({"error": f"Invalid user ID in invite list: {uid!r} (must start with '@')"})
    if preset not in _ALLOWED_PRESETS:
        return json.dumps({"error": f"Invalid preset: {preset!r}. Must be one of: {', '.join(sorted(_ALLOWED_PRESETS))}"})

    try:
        adapter = _ensure_adapter()
        room_id = _run_async(adapter.create_room(
            name=name, topic=topic, invite=invite,
            is_direct=is_direct, preset=preset,
        ))
        if room_id:
            return json.dumps({"success": True, "room_id": room_id})
        return json.dumps({"success": False, "error": "Room creation failed"})
    except Exception as e:
        logger.error("matrix_create_room error: %s", e)
        return json.dumps({"error": f"Failed to create room: {e}"})


def _handle_invite_user(args: dict, **kw) -> str:
    """Handler for matrix_invite_user tool."""
    room_id = args.get("room_id", "").strip()
    user_id = args.get("user_id", "").strip()

    if not room_id or not room_id.startswith("!"):
        return json.dumps({"error": "room_id is required and must start with '!'"})
    if not user_id or not user_id.startswith("@"):
        return json.dumps({"error": "user_id is required and must start with '@'"})

    try:
        adapter = _ensure_adapter()
        result = _run_async(adapter.invite_user(room_id, user_id))
        return json.dumps({"success": bool(result)})
    except Exception as e:
        logger.error("matrix_invite_user error: %s", e)
        return json.dumps({"error": f"Failed to invite user: {e}"})


def _handle_fetch_history(args: dict, **kw) -> str:
    """Handler for matrix_fetch_history tool."""
    room_id = args.get("room_id", "").strip()
    limit = args.get("limit", 50)

    if not room_id or not room_id.startswith("!"):
        return json.dumps({"error": "room_id is required and must start with '!'"})
    if not isinstance(limit, int) or limit < 1:
        limit = 50
    if limit > 200:
        limit = 200

    try:
        adapter = _ensure_adapter()
        messages = _run_async(adapter.fetch_room_history(room_id, limit=limit))
        return json.dumps({"count": len(messages), "messages": messages})
    except Exception as e:
        logger.error("matrix_fetch_history error: %s", e)
        return json.dumps({"error": f"Failed to fetch history: {e}"})


def _handle_set_presence(args: dict, **kw) -> str:
    """Handler for matrix_set_presence tool."""
    state = args.get("state", "").strip().lower()
    status_msg = args.get("status_msg", "")

    if state not in ("online", "offline", "unavailable"):
        return json.dumps({"error": f"Invalid state: {state!r}. Must be 'online', 'offline', or 'unavailable'"})
    if len(status_msg) > _MAX_STATUS_LEN:
        status_msg = status_msg[:_MAX_STATUS_LEN]

    try:
        adapter = _ensure_adapter()
        result = _run_async(adapter.set_presence(state=state, status_msg=status_msg))
        return json.dumps({"success": bool(result)})
    except Exception as e:
        logger.error("matrix_set_presence error: %s", e)
        return json.dumps({"error": f"Failed to set presence: {e}"})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MATRIX_SEND_REACTION_SCHEMA = {
    "name": "matrix_send_reaction",
    "description": (
        "Send an emoji reaction to a message in a Matrix room. "
        "Requires the room_id and event_id of the target message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": (
                    "The Matrix room ID (starts with '!', e.g. '!abc123:matrix.org')."
                ),
            },
            "event_id": {
                "type": "string",
                "description": (
                    "The event ID of the message to react to (starts with '$')."
                ),
            },
            "emoji": {
                "type": "string",
                "description": "The emoji to react with (e.g. '👍', '❤️', '🎉').",
            },
        },
        "required": ["room_id", "event_id", "emoji"],
    },
}

MATRIX_REDACT_MESSAGE_SCHEMA = {
    "name": "matrix_redact_message",
    "description": (
        "Redact (delete) a message or event from a Matrix room. "
        "The message content will be removed but a tombstone remains."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "The Matrix room ID (starts with '!').",
            },
            "event_id": {
                "type": "string",
                "description": "The event ID of the message to redact (starts with '$').",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for the redaction.",
            },
        },
        "required": ["room_id", "event_id"],
    },
}

MATRIX_CREATE_ROOM_SCHEMA = {
    "name": "matrix_create_room",
    "description": (
        "Create a new Matrix room. Returns the room_id on success. "
        "Can optionally set a name, topic, invite users, and configure visibility."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Human-readable room name.",
            },
            "topic": {
                "type": "string",
                "description": "Room topic/description.",
            },
            "invite": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of Matrix user IDs to invite (e.g. ['@user:matrix.org'])."
                ),
            },
            "is_direct": {
                "type": "boolean",
                "description": "Mark as a direct message room. Default: false.",
            },
            "preset": {
                "type": "string",
                "description": (
                    "Room preset: 'private_chat' (default), 'public_chat', "
                    "or 'trusted_private_chat'."
                ),
            },
        },
        "required": [],
    },
}

MATRIX_INVITE_USER_SCHEMA = {
    "name": "matrix_invite_user",
    "description": "Invite a user to a Matrix room.",
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "The Matrix room ID (starts with '!').",
            },
            "user_id": {
                "type": "string",
                "description": (
                    "The Matrix user ID to invite (starts with '@', "
                    "e.g. '@user:matrix.org')."
                ),
            },
        },
        "required": ["room_id", "user_id"],
    },
}

MATRIX_FETCH_HISTORY_SCHEMA = {
    "name": "matrix_fetch_history",
    "description": (
        "Fetch recent message history from a Matrix room. "
        "Returns messages in chronological order with sender, body, and timestamp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "The Matrix room ID (starts with '!').",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to fetch (default: 50, max: 200).",
            },
        },
        "required": ["room_id"],
    },
}

MATRIX_SET_PRESENCE_SCHEMA = {
    "name": "matrix_set_presence",
    "description": (
        "Set the bot's presence status on Matrix. "
        "Controls whether the bot appears as online, offline, or unavailable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "Presence state: 'online', 'offline', or 'unavailable'.",
            },
            "status_msg": {
                "type": "string",
                "description": "Optional human-readable status message.",
            },
        },
        "required": ["state"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="matrix_send_reaction",
    toolset="matrix",
    schema=MATRIX_SEND_REACTION_SCHEMA,
    handler=_handle_send_reaction,
    check_fn=_check_matrix_available,
    emoji="💬",
)

registry.register(
    name="matrix_redact_message",
    toolset="matrix",
    schema=MATRIX_REDACT_MESSAGE_SCHEMA,
    handler=_handle_redact_message,
    check_fn=_check_matrix_available,
    emoji="💬",
)

registry.register(
    name="matrix_create_room",
    toolset="matrix",
    schema=MATRIX_CREATE_ROOM_SCHEMA,
    handler=_handle_create_room,
    check_fn=_check_matrix_available,
    emoji="💬",
)

registry.register(
    name="matrix_invite_user",
    toolset="matrix",
    schema=MATRIX_INVITE_USER_SCHEMA,
    handler=_handle_invite_user,
    check_fn=_check_matrix_available,
    emoji="💬",
)

registry.register(
    name="matrix_fetch_history",
    toolset="matrix",
    schema=MATRIX_FETCH_HISTORY_SCHEMA,
    handler=_handle_fetch_history,
    check_fn=_check_matrix_available,
    emoji="💬",
)

registry.register(
    name="matrix_set_presence",
    toolset="matrix",
    schema=MATRIX_SET_PRESENCE_SCHEMA,
    handler=_handle_set_presence,
    check_fn=_check_matrix_available,
    emoji="💬",
)
