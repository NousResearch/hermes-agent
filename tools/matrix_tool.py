"""Matrix-native gateway tools (Discord-parity compact action tools).

Two toolsets:
- ``matrix`` — safe room-local actions (reaction, history, presence)
- ``matrix_admin`` — gated redaction / invite / room create

Non-secret gates prefer ``config.yaml`` under ``matrix.tools.*`` with legacy
env overrides (``MATRIX_TOOLS_*`` / ``MATRIX_ALLOW_PUBLIC_ROOMS``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set, Tuple

from gateway.session_context import get_session_env
from tools.registry import registry, tool_error, tool_result

_CORE_ACTIONS = {
    "send_reaction": "Add a reaction emoji to a Matrix event.",
    "fetch_history": "Fetch recent Matrix room history events.",
    "set_presence": "Set the bot Matrix presence state.",
}

_ADMIN_ACTIONS = {
    "redact_message": "Redact a Matrix message/event (requires tools.allow_redaction).",
    "invite_user": "Invite a user to a Matrix room (requires tools.allow_invites).",
    "create_room": "Create a Matrix room (requires tools.allow_room_create).",
}


def check_matrix_tool_requirements() -> bool:
    """Available when Matrix credentials exist or a Matrix session is active."""
    if get_session_env("HERMES_SESSION_PLATFORM", "").lower() == "matrix":
        return True
    token = (os.getenv("MATRIX_ACCESS_TOKEN") or "").strip()
    homeserver = (os.getenv("MATRIX_HOMESERVER") or "").strip()
    return bool(token and homeserver)


def _matrix_tools_cfg() -> Dict[str, Any]:
    """Return *user-set* matrix.tools keys only (no DEFAULT_CONFIG merge).

    load_config() deep-merges defaults, which would make every gate key appear
    present and kill legacy MATRIX_TOOLS_* env fallback. Use raw config.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config() or {}
        matrix = cfg.get("matrix") if isinstance(cfg, dict) else None
        if not isinstance(matrix, dict):
            return {}
        tools = matrix.get("tools")
        return tools if isinstance(tools, dict) else {}
    except Exception:
        return {}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off", ""}:
            return False
    return default


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _parse_bool(raw, default)


def _gate(cfg_key: str, env_name: str, default: bool = False) -> bool:
    """User YAML wins; else env; else default. Defaults alone do not mask env."""
    tools = _matrix_tools_cfg()
    if cfg_key in tools:
        return _parse_bool(tools[cfg_key], default)
    return _env_enabled(env_name, default=default)


def _matrix_adapter() -> Tuple[Any, str]:
    """Resolve the live Matrix adapter for the *current session profile*.

    Multiplex gateways keep secondary-profile adapters in
    ``runner._profile_adapters[profile]``. Always consulting
    ``runner.adapters`` would operate through the default profile bot.
    Fail closed when a stamped secondary profile has no Matrix adapter.
    """
    if get_session_env("HERMES_SESSION_PLATFORM", "").lower() != "matrix":
        return None, "Matrix tools are only available while handling a Matrix conversation."
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if not runner:
            return None, "Matrix gateway is not running."

        profile = (get_session_env("HERMES_SESSION_PROFILE", "") or "").strip() or None
        resolver = getattr(runner, "_authorization_adapter", None)
        if callable(resolver):
            adapter = resolver(Platform.MATRIX, profile)
        else:
            # Lightweight fallback for unit tests / older runners.
            if profile and profile != "default":
                profile_adapters = getattr(runner, "_profile_adapters", None) or {}
                if profile in profile_adapters:
                    adapter = profile_adapters[profile].get(Platform.MATRIX)
                else:
                    adapter = None
            else:
                adapters = getattr(runner, "adapters", None) or {}
                adapter = adapters.get(Platform.MATRIX)

        if not adapter:
            if profile and profile != "default":
                return None, (
                    f"Matrix adapter is not connected for profile '{profile}'."
                )
            return None, "Matrix adapter is not connected."
        return adapter, ""
    except Exception as exc:
        return None, f"Failed to resolve Matrix adapter: {exc}"


def _current_room_id(room_id: str = "") -> str:
    return str(room_id or get_session_env("HERMES_SESSION_CHAT_ID", "") or "")


def _allowed_rooms() -> Set[str]:
    tools = _matrix_tools_cfg()
    raw = tools.get("allowed_rooms")
    if raw is None:
        raw = os.getenv("MATRIX_ALLOWED_ROOMS", "")
    if isinstance(raw, (list, tuple, set)):
        return {str(r).strip() for r in raw if str(r).strip()}
    return {room.strip() for room in str(raw or "").split(",") if room.strip()}


def _authorize_room_id(
    requested_room_id: str = "",
    *,
    destructive: bool = False,
) -> Tuple[str, str]:
    current_room_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    room_id = _current_room_id(requested_room_id)
    if not room_id:
        return "", "room_id is required."

    allowed = _allowed_rooms()
    if allowed and room_id not in allowed:
        return "", "Matrix tool target room is not listed in matrix.tools.allowed_rooms / MATRIX_ALLOWED_ROOMS."

    if current_room_id and room_id != current_room_id:
        if not _gate("allow_cross_room", "MATRIX_TOOLS_ALLOW_CROSS_ROOM", False):
            return "", (
                "Matrix tools are limited to the current room by default. "
                "Set matrix.tools.allow_cross_room: true (or MATRIX_TOOLS_ALLOW_CROSS_ROOM=true) "
                "to allow explicit cross-room targets."
            )
        if destructive and not _gate(
            "allow_cross_room_destructive",
            "MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE",
            False,
        ):
            return "", (
                "Cross-room Matrix redaction/invite actions require "
                "matrix.tools.allow_cross_room_destructive: true "
                "(or MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE=true)."
            )
    return room_id, ""


def _run(coro):
    """Run Matrix adapter coroutines on the gateway loop that owns the client.

    Gateway agent turns execute tool handlers in an executor thread. Using
    ``model_tools._run_async`` there would schedule work on a disposable
    worker loop, not ``GatewayRunner._gateway_loop`` where the live Matrix
    adapter/mautrix client lives. Prefer ``run_coroutine_threadsafe`` onto
    the gateway loop; fall back to ``_run_async`` only when no live gateway
    loop exists (unit tests / non-gateway callers).
    """
    import asyncio

    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
    except Exception:
        runner = None

    loop = getattr(runner, "_gateway_loop", None) if runner is not None else None
    if loop is not None:
        try:
            running = not loop.is_closed() and loop.is_running()
        except Exception:
            running = False
        if running:
            try:
                from agent.async_utils import safe_schedule_threadsafe

                fut = safe_schedule_threadsafe(
                    coro,
                    loop,
                    log_message="Matrix tool schedule onto gateway loop failed",
                )
            except Exception:
                fut = None
                try:
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception:
                    if asyncio.iscoroutine(coro):
                        coro.close()
                    raise
            if fut is None:
                # Schedule failed; do not also run on a foreign loop for live
                # gateway turns — surface the failure.
                raise RuntimeError("Matrix gateway loop unavailable for tool dispatch")
            return fut.result(timeout=120)

    from model_tools import _run_async

    return _run_async(coro)


def _ok(success: bool, **extra: Any) -> str:
    return tool_result(success=bool(success), **extra)


def _require_admin_gate(cfg_key: str, env_name: str, action: str) -> str:
    if _gate(cfg_key, env_name, False):
        return ""
    return (
        f"{action} requires matrix.tools.{cfg_key}: true "
        f"(legacy env {env_name}=true)."
    )


def _run_matrix_action(action: str, allowed: Dict[str, str], tool_name: str, **kwargs) -> str:
    action = str(action or "").strip()
    if not action:
        return tool_error(f"{tool_name} requires an 'action' parameter.")
    if action not in allowed:
        return tool_error(
            f"Unknown {tool_name} action '{action}'. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )

    if action == "send_reaction":
        room_id, auth_error = _authorize_room_id(kwargs.get("room_id", ""))
        if auth_error:
            return tool_error(auth_error)
        event_id = str(kwargs.get("event_id", "") or "")
        emoji = str(kwargs.get("emoji", "") or "")
        if not event_id or not emoji:
            return tool_error("event_id and emoji are required.")
        adapter, err = _matrix_adapter()
        if err:
            return tool_error(err)
        event = _run(adapter._send_reaction(room_id, event_id, emoji))
        return _ok(bool(event), event_id=str(event) if event else "")

    if action == "fetch_history":
        room_id, auth_error = _authorize_room_id(kwargs.get("room_id", ""))
        if auth_error:
            return tool_error(auth_error)
        try:
            limit = int(kwargs.get("limit", 20) or 20)
        except (TypeError, ValueError):
            return tool_error("limit must be an integer.")
        limit = max(1, min(100, limit))
        adapter, err = _matrix_adapter()
        if err:
            return tool_error(err)
        events = _run(
            adapter.fetch_history(
                room_id,
                limit,
                str(kwargs.get("from_token", "") or ""),
            )
        )
        return tool_result(success=True, events=events)

    if action == "set_presence":
        state = str(kwargs.get("state", "online") or "online")
        status_msg = str(kwargs.get("status_msg", "") or "")
        adapter, err = _matrix_adapter()
        if err:
            return tool_error(err)
        return _ok(_run(adapter.set_presence(state, status_msg)))

    if action == "redact_message":
        gate_error = _require_admin_gate(
            "allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION", "Matrix redaction"
        )
        if gate_error:
            return tool_error(gate_error)
        room_id, auth_error = _authorize_room_id(
            kwargs.get("room_id", ""), destructive=True
        )
        if auth_error:
            return tool_error(auth_error)
        event_id = str(kwargs.get("event_id", "") or "")
        reason = str(kwargs.get("reason", "") or "")
        if not event_id:
            return tool_error("event_id is required.")
        adapter, err = _matrix_adapter()
        if err:
            return tool_error(err)
        return _ok(_run(adapter.redact_message(room_id, event_id, reason)))

    if action == "invite_user":
        gate_error = _require_admin_gate(
            "allow_invites", "MATRIX_TOOLS_ALLOW_INVITES", "Matrix invites"
        )
        if gate_error:
            return tool_error(gate_error)
        room_id, auth_error = _authorize_room_id(
            kwargs.get("room_id", ""), destructive=True
        )
        if auth_error:
            return tool_error(auth_error)
        user_id = str(kwargs.get("user_id", "") or "")
        if not user_id:
            return tool_error("user_id is required.")
        adapter, err = _matrix_adapter()
        if err:
            return tool_error(err)
        return _ok(_run(adapter.invite_user(room_id, user_id)))

    if action == "create_room":
        gate_error = _require_admin_gate(
            "allow_room_create",
            "MATRIX_TOOLS_ALLOW_ROOM_CREATE",
            "Matrix room creation",
        )
        if gate_error:
            return tool_error(gate_error)
        preset = str(kwargs.get("preset", "private_chat") or "private_chat")
        if preset == "public_chat" and not _gate(
            "allow_public_rooms", "MATRIX_ALLOW_PUBLIC_ROOMS", False
        ):
            return tool_error(
                "Public Matrix room creation requires matrix.tools.allow_public_rooms: true "
                "(or MATRIX_ALLOW_PUBLIC_ROOMS=true)."
            )
        invite = kwargs.get("invite") or []
        if isinstance(invite, str):
            invite = [u.strip() for u in invite.split(",") if u.strip()]
        adapter, err = _matrix_adapter()
        if err:
            return tool_error(err)
        room_id = _run(
            adapter.create_room(
                name=str(kwargs.get("name", "") or ""),
                topic=str(kwargs.get("topic", "") or ""),
                invite=list(invite),
                is_direct=bool(kwargs.get("is_direct", False)),
                preset=preset,
            )
        )
        if not room_id:
            return tool_error("Matrix room creation failed.")
        return tool_result(success=True, room_id=str(room_id))

    return tool_error(f"Unhandled action '{action}'.")


def _build_schema(actions: Dict[str, str], tool_name: str) -> Dict[str, Any]:
    action_list = sorted(actions)
    desc_lines = [f"{tool_name} Matrix actions:"] + [
        f"- {name}: {desc}" for name, desc in sorted(actions.items())
    ]
    return {
        "name": tool_name,
        "description": " ".join(desc_lines),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": action_list,
                    "description": "Matrix action to perform.",
                },
                "room_id": {
                    "type": "string",
                    "description": "Matrix room ID. Defaults to the current Matrix room.",
                },
                "event_id": {
                    "type": "string",
                    "description": "Target Matrix event ID (send_reaction, redact_message).",
                },
                "emoji": {
                    "type": "string",
                    "description": "Reaction emoji (send_reaction).",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional redaction reason (redact_message).",
                },
                "user_id": {
                    "type": "string",
                    "description": "Matrix user ID to invite (invite_user).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "History limit (fetch_history, default 20).",
                },
                "from_token": {
                    "type": "string",
                    "description": "Optional pagination token (fetch_history).",
                },
                "state": {
                    "type": "string",
                    "enum": ["online", "offline", "unavailable"],
                    "description": "Presence state (set_presence).",
                },
                "status_msg": {
                    "type": "string",
                    "description": "Optional presence status message (set_presence).",
                },
                "name": {"type": "string", "description": "Room name (create_room)."},
                "topic": {"type": "string", "description": "Room topic (create_room)."},
                "invite": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Matrix user IDs to invite (create_room).",
                },
                "is_direct": {
                    "type": "boolean",
                    "description": "Mark room as DM (create_room).",
                },
                "preset": {
                    "type": "string",
                    "enum": ["private_chat", "trusted_private_chat", "public_chat"],
                    "description": "createRoom preset (create_room).",
                },
            },
            "required": ["action"],
        },
    }


_HANDLER_DEFAULTS = {
    "action": "",
    "room_id": "",
    "event_id": "",
    "emoji": "",
    "reason": "",
    "user_id": "",
    "limit": 20,
    "from_token": "",
    "state": "online",
    "status_msg": "",
    "name": "",
    "topic": "",
    "invite": None,
    "is_direct": False,
    "preset": "private_chat",
}


def _make_handler(actions: Dict[str, str], tool_name: str):
    def _handler(args, **kw):
        payload = {k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()}
        return _run_matrix_action(payload.pop("action"), actions, tool_name, **payload)

    return _handler


registry.register(
    name="matrix",
    toolset="matrix",
    schema=_build_schema(_CORE_ACTIONS, "matrix"),
    handler=_make_handler(_CORE_ACTIONS, "matrix"),
    check_fn=check_matrix_tool_requirements,
    requires_env=[],
    description="Matrix room tools (reaction, history, presence)",
    emoji="🟩",
)

registry.register(
    name="matrix_admin",
    toolset="matrix_admin",
    schema=_build_schema(_ADMIN_ACTIONS, "matrix_admin"),
    handler=_make_handler(_ADMIN_ACTIONS, "matrix_admin"),
    check_fn=check_matrix_tool_requirements,
    requires_env=[],
    description="Matrix admin tools (redact, invite, create room)",
    emoji="🛡️",
)
