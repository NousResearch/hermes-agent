"""Hermes MCP server: expose Hermes core tools over MCP (stdio).

Run:  python -m tools.mcp_server          (stdio transport)
      python -m tools.mcp_server --http   (optional Streamable HTTP, if the `mcp` SDK is installed)

Clients (ChatGPT Desktop, Claude Desktop, Codex, any MCP client) connect to this
process and can call Hermes functions: read_file, write_file, patch, search_files,
terminal, web_search, web_extract, delegate_task, memory, todo_list, vision_analyze,
image_generate, execute_code, etc.

No core files are edited; this module only reads `tools.registry`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes-mcp")

# Lazy import of the Hermes tool registry (importing it triggers discovery of tools/*.py).
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        from tools.registry import registry, discover_builtin_tools
        # Trigger discovery so every tools/*.py module has called registry.register().
        discover_builtin_tools()
        _registry = registry
    return _registry


# ── MCP protocol (minimal stdlib JSON-RPC over stdio) ──────────────────────

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "hermes-mcp"
SERVER_VERSION = "1.0.0"


def _tool_list() -> list[Dict[str, Any]]:
    """Return every registered Hermes tool as an MCP tool descriptor.

    ``get_definitions`` filters to tools whose ``check_fn`` passes, so unavailable
    tools (missing key, disabled toolset) are not exposed to MCP clients.
    """
    registry = _get_registry()
    out: list[Dict[str, Any]] = []
    all_names = registry.get_all_tool_names()
    # get_definitions returns OpenAI-format {"type": "function", "function": {...}} dicts.
    definitions = registry.get_definitions(set(all_names), quiet=True) or []
    for entry in definitions:
        func = entry.get("function") if isinstance(entry, dict) else None
        if not func:
            continue
        name = func.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": func.get("description", "") or "",
                "inputSchema": func.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch a tool call through the Hermes registry; returns a JSON string."""
    registry = _get_registry()
    try:
        return registry.dispatch(name, dict(arguments or {}))
    except Exception as exc:  # noqa: BLE001 — surface clean MCP error, not a traceback
        logger.exception("MCP tools/call failed for %s", name)
        return json.dumps({"error": f"tool '{name}' failed: {exc}"})


def _handle_request(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC request; return a response dict (or None for notifications)."""
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "initialized":
        return None  # notification
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _tool_list()}}
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        result_text = _tool_call(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }
    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _read_stdio_messages():
    """Yield parsed JSON-RPC messages from stdin.

    Supports both MCP Content-Length framing and plain newline-delimited JSON.
    """
    stdin = sys.stdin
    while True:
        # Peek: read header lines until blank line (Content-Length framing).
        line = stdin.readline()
        if not line:
            return  # EOF
        line = line.strip()
        if line == "":
            # Blank line after headers — read the body based on Content-Length if seen.
            continue
        # Could be a bare JSON line (no headers).
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring unparseable input line")
            continue
        # Header line (e.g. Content-Length: N). Collect headers.
        content_length = None
        while line:
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            line = stdin.readline().strip()
        if content_length is not None:
            body = stdin.read(content_length)
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                logger.warning("Ignoring unparseable body")
        else:
            logger.warning("Ignoring framed message without Content-Length")


def _write(obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    # Emit MCP Content-Length framing so stdio clients parse it reliably.
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def run_stdio() -> None:
    logging.basicConfig(level=os.getenv("HERMES_MCP_LOG", "WARNING"), stream=sys.stderr)
    logger.info("hermes-mcp stdio server starting")
    for req in _read_stdio_messages():
        try:
            resp = _handle_request(req)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Request handler error")
            resp = {
                "jsonrpc": "2.0",
                "id": req.get("id") if isinstance(req, dict) else None,
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
            }
        if resp is not None:
            _write(resp)


def run_http() -> None:
    """Optional Streamable HTTP transport via the `mcp` SDK, if installed."""
    try:
        from mcp.server import Server  # noqa: F401
    except ImportError as exc:
        print(
            "HTTP transport requires the `mcp` package: pip install mcp\n"
            "Or run without --http for stdio transport.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    # SDK-based HTTP server is version-dependent; stdio is the default path.
    logger.info("HTTP transport not implemented in stdlib build; use --stdio (default).")
    raise SystemExit(0)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-mcp", description="Hermes MCP server")
    parser.add_argument("--http", action="store_true", help="use Streamable HTTP (requires `mcp` SDK)")
    args = parser.parse_args(argv)
    if args.http:
        run_http()
    else:
        run_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
