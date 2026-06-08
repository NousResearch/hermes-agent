"""Client-side terminal execution tool for TUI/WebSocket sessions.

This tool lets an agent running on the Hermes host ask the connected GUI client
(neon-companion / desktop) to execute a command on the user's local machine.
The actual bridge callback is installed by ``tui_gateway.server`` per session.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from tools.registry import registry

_ClientTerminalCallback = Callable[[str, int | None, bool], dict[str, Any]]
_client_terminal_callback: _ClientTerminalCallback | None = None


def set_client_terminal_callback(callback: _ClientTerminalCallback | None) -> None:
    global _client_terminal_callback
    _client_terminal_callback = callback


def check_client_terminal_requirements() -> bool:
    return _client_terminal_callback is not None


def client_terminal_tool(command: str, timeout: int | None = None, persistent: bool = False) -> str:
    if _client_terminal_callback is None:
        return json.dumps(
            {
                "success": False,
                "error": "No connected WebSocket client terminal is registered for this session.",
            },
            ensure_ascii=False,
        )

    command = str(command or "").strip()
    if not command:
        return json.dumps({"success": False, "error": "command is required"}, ensure_ascii=False)

    try:
        result = _client_terminal_callback(command, timeout, bool(persistent))
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)


CLIENT_TERMINAL_SCHEMA = {
    "name": "client_terminal",
    "description": (
        "Execute a shell command on the connected user's local machine via "
        "neon-companion/WebSocket. Use this when the user asks you to inspect "
        "or operate on THEIR computer rather than the Hermes VPS. Requires an "
        "active connected client and may be subject to client-side approval/auto-approve settings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Command to execute on the connected client machine.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds to wait. Defaults to 30, maximum 600.",
                "minimum": 1,
            },
            "persistent": {
                "type": "boolean",
                "description": "Run in the client's persistent shell when supported. Default false.",
                "default": False,
            },
        },
        "required": ["command"],
    },
}


def _handle_client_terminal(args, **kw):
    return client_terminal_tool(
        command=args.get("command"),
        timeout=args.get("timeout"),
        persistent=args.get("persistent", False),
    )


registry.register(
    name="client_terminal",
    toolset="terminal",
    schema=CLIENT_TERMINAL_SCHEMA,
    handler=_handle_client_terminal,
    check_fn=check_client_terminal_requirements,
    emoji="🖥️",
    max_result_size_chars=100_000,
)
