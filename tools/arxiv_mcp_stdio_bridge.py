#!/usr/bin/env python3
"""JSON-lines MCP bridge for ``arxiv-mcp-server``.

The upstream package's server object is valid in-process on this host, but its
packaged stdio entrypoint does not answer initialize requests over a real pipe.
This bridge keeps the package's tool/prompt handlers and exposes the minimal
MCP JSON-RPC methods Hermes needs for runtime discovery and tool calls.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

_arxiv_server: Any = import_module("arxiv_mcp_server.server")
call_tool: Any = getattr(_arxiv_server, "call_tool")
get_prompt: Any = getattr(_arxiv_server, "get_prompt")
list_prompts: Any = getattr(_arxiv_server, "list_prompts")
list_tools: Any = getattr(_arxiv_server, "list_tools")
settings: Any = getattr(_arxiv_server, "settings")

JSONRPC_VERSION = "2.0"
DEFAULT_PROTOCOL_VERSION = "2025-11-25"
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)((?:api[_-]?key|authorization|auth|cookie|password|passwd|secret|token)\s*[:=]\s*)[^\s\"']+"
)
SECRET_PREFIX_RE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|(?:sk-|ghp_|github_pat_|AKIA|xoxb-|xoxp-|AIza)[A-Za-z0-9_./+=-]+"
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": message_id, "result": result}


def _redact_error_message(message: str) -> str:
    redacted = SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", message)
    return SECRET_PREFIX_RE.sub(
        lambda match: f"{match.group(1)}[REDACTED]" if match.group(1) else "[REDACTED]",
        redacted,
    )


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": message_id,
        "error": {"code": code, "message": _redact_error_message(message)},
    }


def _package_version() -> str:
    try:
        return version("arxiv-mcp-server")
    except PackageNotFoundError:
        return getattr(settings, "APP_VERSION", "0.0.0")


async def _handle_request(payload: dict[str, Any]) -> dict[str, Any] | None:
    method = payload.get("method")
    message_id = payload.get("id")
    raw_params = payload.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}

    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _response(
            message_id,
            {
                "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION,
                "capabilities": {
                    "experimental": {},
                    "prompts": {"listChanged": False},
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": getattr(settings, "APP_NAME", "arxiv-mcp-server"),
                    "version": _package_version(),
                },
            },
        )
    if method == "tools/list":
        return _response(message_id, {"tools": _jsonable(await list_tools())})
    if method == "prompts/list":
        return _response(message_id, {"prompts": _jsonable(await list_prompts())})
    if method == "prompts/get":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else None
        return _response(message_id, _jsonable(await get_prompt(name, arguments)))
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        return _response(
            message_id,
            {
                "content": _jsonable(await call_tool(name, arguments)),
                "isError": False,
            },
        )
    if method == "resources/list":
        return _error(message_id, -32601, "Method not found: resources/list")
    return _error(message_id, -32601, f"Method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                json.dumps(_error(None, -32700, f"Parse error: {exc.msg}")),
                flush=True,
            )
            continue
        try:
            response = asyncio.run(_handle_request(payload))
        except Exception as exc:
            response = _error(payload.get("id"), -32000, str(exc))
        if response is not None:
            print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
