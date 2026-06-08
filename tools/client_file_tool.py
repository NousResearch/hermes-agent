"""Client-side file transfer tools for TUI/WebSocket sessions.

These tools let an agent move small files between the Hermes host and the
connected GUI client (neon-companion / desktop) over the WebSocket bridge.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from tools.registry import registry

_ClientFilePushCallback = Callable[[str, str, str, int | None], dict[str, Any]]
_ClientFilePullCallback = Callable[[str, str, str, int | None], dict[str, Any]]

_client_file_push_callback: _ClientFilePushCallback | None = None
_client_file_pull_callback: _ClientFilePullCallback | None = None


def set_client_file_push_callback(callback: _ClientFilePushCallback | None) -> None:
    global _client_file_push_callback
    _client_file_push_callback = callback


def set_client_file_pull_callback(callback: _ClientFilePullCallback | None) -> None:
    global _client_file_pull_callback
    _client_file_pull_callback = callback


def check_client_file_requirements() -> bool:
    return _client_file_push_callback is not None or _client_file_pull_callback is not None


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def client_file_push_tool(
    source_path: str,
    destination_root: str = "downloads",
    destination_path: str | None = None,
    timeout: int | None = None,
) -> str:
    """Send a Hermes-host file to the connected client."""
    if _client_file_push_callback is None:
        return _json_result({"success": False, "error": "No connected WebSocket client file bridge is registered for this session."})

    source_path = str(source_path or "").strip()
    destination_root = str(destination_root or "downloads").strip() or "downloads"
    destination_path = str(destination_path or "").strip()
    if not source_path:
        return _json_result({"success": False, "error": "source_path is required"})

    try:
        result = _client_file_push_callback(source_path, destination_root, destination_path, timeout)
    except Exception as exc:
        return _json_result({"success": False, "error": str(exc)})
    return _json_result(result)


def client_file_pull_tool(
    source_root: str,
    source_path: str,
    destination_path: str,
    timeout: int | None = None,
) -> str:
    """Request a file from the connected client and save it on the Hermes host."""
    if _client_file_pull_callback is None:
        return _json_result({"success": False, "error": "No connected WebSocket client file bridge is registered for this session."})

    source_root = str(source_root or "").strip()
    source_path = str(source_path or "").strip()
    destination_path = str(destination_path or "").strip()
    if not source_root:
        return _json_result({"success": False, "error": "source_root is required"})
    if not source_path:
        return _json_result({"success": False, "error": "source_path is required"})
    if not destination_path:
        return _json_result({"success": False, "error": "destination_path is required"})

    try:
        result = _client_file_pull_callback(source_root, source_path, destination_path, timeout)
    except Exception as exc:
        return _json_result({"success": False, "error": str(exc)})
    return _json_result(result)


CLIENT_FILE_PUSH_SCHEMA = {
    "name": "client_file_push",
    "description": (
        "Send a file from the Hermes host/VPS to the connected user's local client "
        "via neon-companion/WebSocket. destination_root is one of downloads, workspace, temp, session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_path": {"type": "string", "description": "Absolute or relative path on the Hermes host."},
            "destination_root": {"type": "string", "description": "Client root: downloads, workspace, temp, or session.", "default": "downloads"},
            "destination_path": {"type": "string", "description": "Relative path inside destination_root. Defaults to source basename."},
            "timeout": {"type": "integer", "description": "Maximum seconds to wait. Defaults to 60.", "minimum": 1},
        },
        "required": ["source_path"],
    },
}

CLIENT_FILE_PULL_SCHEMA = {
    "name": "client_file_pull",
    "description": (
        "Request a file from the connected user's local client and save it on the Hermes host/VPS. "
        "source_root is one of downloads, workspace, temp, session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_root": {"type": "string", "description": "Client root: downloads, workspace, temp, or session."},
            "source_path": {"type": "string", "description": "Relative path inside source_root on the client."},
            "destination_path": {"type": "string", "description": "Absolute or relative path to save on the Hermes host."},
            "timeout": {"type": "integer", "description": "Maximum seconds to wait. Defaults to 60.", "minimum": 1},
        },
        "required": ["source_root", "source_path", "destination_path"],
    },
}


def _handle_client_file_push(args, **kw):
    return client_file_push_tool(
        source_path=args.get("source_path"),
        destination_root=args.get("destination_root", "downloads"),
        destination_path=args.get("destination_path"),
        timeout=args.get("timeout"),
    )


def _handle_client_file_pull(args, **kw):
    return client_file_pull_tool(
        source_root=args.get("source_root"),
        source_path=args.get("source_path"),
        destination_path=args.get("destination_path"),
        timeout=args.get("timeout"),
    )


registry.register(
    name="client_file_push",
    toolset="file",
    schema=CLIENT_FILE_PUSH_SCHEMA,
    handler=_handle_client_file_push,
    check_fn=check_client_file_requirements,
    emoji="📤",
    max_result_size_chars=100_000,
)

registry.register(
    name="client_file_pull",
    toolset="file",
    schema=CLIENT_FILE_PULL_SCHEMA,
    handler=_handle_client_file_pull,
    check_fn=check_client_file_requirements,
    emoji="📥",
    max_result_size_chars=100_000,
)
