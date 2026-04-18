"""Minimal tool wrapper for the process-local Codex app-server bridge."""

from __future__ import annotations

import threading
from typing import Any

from gateway.session_context import get_session_env
from agent.codex_app_server_bridge import (
    DEFAULT_INITIALIZE_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    START_THREAD_METHOD,
    START_TURN_METHOD,
    CodexAppServerBridge,
    CodexAppServerBridgeError,
    CodexAppServerTimeoutError,
)
from tools.registry import registry, tool_error, tool_result


_ACTIONS = ("start_bridge", "stop_bridge", "start_turn", "start_job", "status", "events")
_bridges: dict[str, CodexAppServerBridge] = {}
_bridge_lock = threading.RLock()


def _session_bridge_key() -> str:
    session_key = get_session_env("HERMES_SESSION_KEY", "")
    if session_key:
        return session_key
    platform = get_session_env("HERMES_SESSION_PLATFORM", "") or "local"
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "") or "local"
    thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "")
    return f"{platform}:{chat_id}:{thread_id}"


def _current_route_metadata() -> dict[str, str]:
    metadata = {
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME", ""),
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
        "user_id": get_session_env("HERMES_SESSION_USER_ID", ""),
        "user_name": get_session_env("HERMES_SESSION_USER_NAME", ""),
        "session_key": _session_bridge_key(),
    }
    return {key: value for key, value in metadata.items() if value}


def _get_bridge() -> CodexAppServerBridge:
    key = _session_bridge_key()
    with _bridge_lock:
        bridge = _bridges.get(key)
        if bridge is None:
            bridge = CodexAppServerBridge()
            _bridges[key] = bridge
        if hasattr(bridge, "set_route_metadata"):
            bridge.set_route_metadata(**_current_route_metadata())
        return bridge


def _current_bridge() -> CodexAppServerBridge | None:
    key = _session_bridge_key()
    with _bridge_lock:
        bridge = _bridges.get(key)
        if bridge is not None and hasattr(bridge, "set_route_metadata"):
            bridge.set_route_metadata(**_current_route_metadata())
        return bridge


def _clear_bridge() -> None:
    key = _session_bridge_key()
    with _bridge_lock:
        _bridges.pop(key, None)


def _set_bridge_for_tests(bridge: CodexAppServerBridge | None) -> None:
    key = _session_bridge_key()
    with _bridge_lock:
        if bridge is None:
            _bridges.pop(key, None)
        else:
            _bridges[key] = bridge


def _coerce_timeout(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError("timeout_seconds must be a number")
    if timeout <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    return min(timeout, 300.0)


def _coerce_limit(value: Any) -> int:
    if value is None or value == "":
        return 20
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    return max(0, min(limit, 200))


def codex_app_server_tool(
    *,
    action: str,
    repo_path: str | None = None,
    prompt: str | None = None,
    limit: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Dispatch a small set of Codex app-server bridge actions."""
    action = (action or "").strip()
    if action not in _ACTIONS:
        return tool_error(f"Unknown action: {action}. Use one of: {', '.join(_ACTIONS)}")

    try:
        if action == "start_bridge":
            timeout = _coerce_timeout(timeout_seconds, DEFAULT_INITIALIZE_TIMEOUT_SECONDS)
            bridge = _get_bridge()
            return tool_result(
                success=True,
                action=action,
                status=bridge.start(timeout=timeout),
            )

        if action == "stop_bridge":
            bridge = _current_bridge()
            if bridge is None:
                return tool_result(
                    success=True,
                    action=action,
                    status={"bridge_status": "stopped"},
                )
            status = bridge.stop()
            _clear_bridge()
            return tool_result(success=True, action=action, status=status)

        if action == "status":
            bridge = _current_bridge()
            status = bridge.get_status() if bridge is not None else {"bridge_status": "stopped"}
            return tool_result(
                success=True,
                action=action,
                status=status,
                methods={
                    "start_thread": START_THREAD_METHOD,
                    "start_turn": START_TURN_METHOD,
                },
            )

        if action == "events":
            bridge = _current_bridge()
            event_limit = _coerce_limit(limit)
            events = bridge.get_recent_events(event_limit) if bridge is not None else []
            return tool_result(
                success=True,
                action=action,
                events=events,
                count=len(events),
            )

        if action in ("start_turn", "start_job"):
            if not repo_path:
                return tool_error(f"repo_path is required for {action}")
            if not prompt or not str(prompt).strip():
                return tool_error(f"prompt is required for {action}")
            bridge = _get_bridge() if action == "start_job" else _current_bridge()
            if bridge is None:
                return tool_error("Bridge is not started. Call start_bridge first.")
            timeout = _coerce_timeout(timeout_seconds, DEFAULT_REQUEST_TIMEOUT_SECONDS)
            start_status = None
            if action == "start_job":
                start_status = bridge.start(timeout=timeout)
            result = bridge.start_turn(repo_path=repo_path, prompt=prompt, timeout=timeout)
            response = {
                "thread_id": result.get("thread_id"),
                "turn_id": result.get("turn_id"),
                "status": result.get("status"),
                "result": result,
            }
            if start_status is not None:
                response["bridge_start_status"] = start_status
            return tool_result(success=True, action=action, **response)

    except (ValueError, CodexAppServerTimeoutError, CodexAppServerBridgeError) as exc:
        return tool_error(str(exc))

    return tool_error(f"Unhandled action: {action}")


def _handle_codex_app_server(args: dict[str, Any], **_: Any) -> str:
    return codex_app_server_tool(
        action=args.get("action", ""),
        repo_path=args.get("repo_path"),
        prompt=args.get("prompt"),
        limit=args.get("limit"),
        timeout_seconds=args.get("timeout_seconds"),
    )


CODEX_APP_SERVER_SCHEMA = {
    "name": "codex_app_server",
    "description": (
        "Start and inspect a process-local Codex app-server monitor. Supports "
        "starting/stopping the bridge, starting a turn, starting a job, status, "
        "and recent events. Does not handle approvals, websocket listeners, or dashboards."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Bridge action to perform.",
            },
            "repo_path": {
                "type": "string",
                "description": "Repository path for action='start_turn' or action='start_job'.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt for action='start_turn' or action='start_job'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum recent events to return for action='events'.",
                "minimum": 0,
                "maximum": 200,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Request timeout for bridge startup or turn-start requests.",
                "minimum": 0.001,
                "maximum": 300,
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="codex_app_server",
    toolset="codex_monitor",
    schema=CODEX_APP_SERVER_SCHEMA,
    handler=_handle_codex_app_server,
)
