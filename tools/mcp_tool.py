#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client Support

Connects to external MCP servers via stdio transport, discovers their tools,
and registers them into the hermes-agent tool registry so the agent can call
them like any built-in tool.

Configuration is read from ~/.hermes/config.yaml under the ``mcp_servers`` key.
The ``mcp`` Python package is optional -- if not installed, this module is a
no-op and logs a debug message.

Example config::

    mcp_servers:
      filesystem:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env: {}
      github:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."

Architecture:
    A dedicated background event loop (_mcp_loop) runs in a daemon thread.
    All MCP connections live on this loop. Tool handlers schedule coroutines
    onto it via run_coroutine_threadsafe(), so they work from any thread.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import -- MCP SDK is an optional dependency
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from contextlib import AsyncExitStack
    _MCP_AVAILABLE = True
except ImportError:
    logger.debug("mcp package not installed -- MCP tool support disabled")


# ---------------------------------------------------------------------------
# Connection tracking
# ---------------------------------------------------------------------------

class MCPConnection:
    """Holds a live MCP server connection and its async resource stack."""

    __slots__ = ("server_name", "session", "stack")

    def __init__(self, server_name: str, session: Any, stack: Any):
        self.server_name = server_name
        self.session: Optional[Any] = session
        self.stack: Optional[Any] = stack


_connections: Dict[str, MCPConnection] = {}

# Dedicated event loop running in a background daemon thread.
# All MCP async operations (connect, call_tool, shutdown) run here.
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None


def _ensure_mcp_loop():
    """Start the background event loop thread if not already running."""
    global _mcp_loop, _mcp_thread
    if _mcp_loop is not None and _mcp_loop.is_running():
        return
    _mcp_loop = asyncio.new_event_loop()
    _mcp_thread = threading.Thread(
        target=_mcp_loop.run_forever,
        name="mcp-event-loop",
        daemon=True,
    )
    _mcp_thread.start()


def _run_on_mcp_loop(coro, timeout: float = 30):
    """Schedule a coroutine on the MCP event loop and block until done."""
    if _mcp_loop is None or not _mcp_loop.is_running():
        raise RuntimeError("MCP event loop is not running")
    future = asyncio.run_coroutine_threadsafe(coro, _mcp_loop)
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_mcp_config() -> Dict[str, dict]:
    """Read ``mcp_servers`` from the Hermes config file.

    Returns a dict of ``{server_name: {command, args, env}}`` or empty dict.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        servers = config.get("mcp_servers")
        if not servers or not isinstance(servers, dict):
            return {}
        return servers
    except Exception as exc:
        logger.debug("Failed to load MCP config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Server connection
# ---------------------------------------------------------------------------

async def _connect_server(name: str, config: dict) -> MCPConnection:
    """Start an MCP server subprocess and initialize a ClientSession.

    Args:
        name:   Logical server name (e.g. "filesystem").
        config: Dict with ``command``, ``args``, and optional ``env``.

    Returns:
        An ``MCPConnection`` with a live session.

    Raises:
        Exception on connection or initialization failure.
    """
    command = config.get("command")
    args = config.get("args", [])
    env = config.get("env")

    if not command:
        raise ValueError(f"MCP server '{name}' has no 'command' in config")

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env if env else None,
    )

    stack = AsyncExitStack()
    stdio_transport = await stack.enter_async_context(stdio_client(server_params))
    read_stream, write_stream = stdio_transport
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()

    return MCPConnection(server_name=name, session=session, stack=stack)


# ---------------------------------------------------------------------------
# Handler / check-fn factories
# ---------------------------------------------------------------------------

def _make_tool_handler(server_name: str, tool_name: str):
    """Return a sync handler that calls an MCP tool via the background loop.

    The handler conforms to the registry's dispatch interface:
    ``handler(args_dict, **kwargs) -> str``
    """

    def _handler(args: dict, **kwargs) -> str:
        conn = _connections.get(server_name)
        if not conn or not conn.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            })

        async def _call():
            result = await conn.session.call_tool(tool_name, arguments=args)
            # MCP CallToolResult has .content (list of content blocks) and .isError
            if result.isError:
                error_text = ""
                for block in (result.content or []):
                    if hasattr(block, "text"):
                        error_text += block.text
                return json.dumps({"error": error_text or "MCP tool returned an error"})

            # Collect text from content blocks
            parts: List[str] = []
            for block in (result.content or []):
                if hasattr(block, "text"):
                    parts.append(block.text)
            return json.dumps({"result": "\n".join(parts) if parts else ""})

        try:
            return _run_on_mcp_loop(_call(), timeout=120)
        except Exception as exc:
            logger.error("MCP tool %s/%s call failed: %s", server_name, tool_name, exc)
            return json.dumps({"error": f"MCP call failed: {type(exc).__name__}: {exc}"})

    return _handler


def _make_check_fn(server_name: str):
    """Return a check function that verifies the MCP connection is alive."""

    def _check() -> bool:
        conn = _connections.get(server_name)
        return conn is not None and conn.session is not None

    return _check


# ---------------------------------------------------------------------------
# Discovery & registration
# ---------------------------------------------------------------------------

def _convert_mcp_schema(server_name: str, mcp_tool) -> dict:
    """Convert an MCP tool listing to the Hermes registry schema format.

    Args:
        server_name: The logical server name for prefixing.
        mcp_tool:    An MCP ``Tool`` object with ``.name``, ``.description``,
                     and ``.inputSchema``.

    Returns:
        A dict suitable for ``registry.register(schema=...)``.
    """
    # Sanitize: replace hyphens and dots with underscores for LLM API compatibility
    safe_tool_name = mcp_tool.name.replace("-", "_").replace(".", "_")
    safe_server_name = server_name.replace("-", "_").replace(".", "_")
    prefixed_name = f"mcp_{safe_server_name}_{safe_tool_name}"
    return {
        "name": prefixed_name,
        "description": mcp_tool.description or f"MCP tool {mcp_tool.name} from {server_name}",
        "parameters": mcp_tool.inputSchema if mcp_tool.inputSchema else {
            "type": "object",
            "properties": {},
        },
    }


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """Connect to a single MCP server, discover tools, and register them.

    Returns list of registered tool names.
    """
    from tools.registry import registry
    from toolsets import create_custom_toolset

    conn = await _connect_server(name, config)
    _connections[name] = conn

    # Discover tools
    tools_result = await conn.session.list_tools()
    tools = tools_result.tools if hasattr(tools_result, "tools") else []

    registered_names: List[str] = []
    toolset_name = f"mcp-{name}"

    for mcp_tool in tools:
        schema = _convert_mcp_schema(name, mcp_tool)
        tool_name_prefixed = schema["name"]

        registry.register(
            name=tool_name_prefixed,
            toolset=toolset_name,
            schema=schema,
            handler=_make_tool_handler(name, mcp_tool.name),
            check_fn=_make_check_fn(name),
            is_async=False,
            description=schema["description"],
        )
        registered_names.append(tool_name_prefixed)

    # Create a custom toolset so these tools are discoverable
    if registered_names:
        create_custom_toolset(
            name=toolset_name,
            description=f"MCP tools from {name} server",
            tools=registered_names,
        )

    logger.info(
        "MCP server '%s': registered %d tool(s): %s",
        name, len(registered_names), ", ".join(registered_names),
    )
    return registered_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_mcp_tools() -> List[str]:
    """Entry point: load config, connect to MCP servers, register tools.

    Called from ``model_tools._discover_tools()``. Safe to call even when
    the ``mcp`` package is not installed (returns empty list).

    Returns:
        List of all registered MCP tool names.
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK not available -- skipping MCP tool discovery")
        return []

    servers = _load_mcp_config()
    if not servers:
        logger.debug("No MCP servers configured")
        return []

    # Start the background event loop for MCP connections
    _ensure_mcp_loop()

    all_tools: List[str] = []

    async def _discover_all():
        for name, cfg in servers.items():
            try:
                registered = await _discover_and_register_server(name, cfg)
                all_tools.extend(registered)
            except Exception as exc:
                logger.warning("Failed to connect to MCP server '%s': %s", name, exc)

    _run_on_mcp_loop(_discover_all(), timeout=60)

    if all_tools:
        # Add MCP tools to hermes-cli and other platform toolsets
        from toolsets import TOOLSETS
        for ts_name in ("hermes-cli", "hermes-telegram", "hermes-discord",
                        "hermes-whatsapp", "hermes-slack"):
            ts = TOOLSETS.get(ts_name)
            if ts:
                for tool_name in all_tools:
                    if tool_name not in ts["tools"]:
                        ts["tools"].append(tool_name)

    return all_tools


def shutdown_mcp_servers():
    """Close all MCP server connections and stop the background loop."""
    global _mcp_loop, _mcp_thread

    if not _connections:
        _stop_mcp_loop()
        return

    async def _shutdown():
        for name, conn in list(_connections.items()):
            try:
                if conn.stack:
                    await conn.stack.aclose()
            except Exception as exc:
                logger.debug("Error closing MCP server '%s': %s", name, exc)
            finally:
                conn.session = None
                conn.stack = None
        _connections.clear()

    if _mcp_loop is not None and _mcp_loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), _mcp_loop)
            future.result(timeout=10)
        except Exception as exc:
            logger.debug("Error during MCP shutdown: %s", exc)

    _stop_mcp_loop()


def _stop_mcp_loop():
    """Stop the background event loop and join its thread."""
    global _mcp_loop, _mcp_thread
    if _mcp_loop is not None:
        _mcp_loop.call_soon_threadsafe(_mcp_loop.stop)
        if _mcp_thread is not None:
            _mcp_thread.join(timeout=5)
            _mcp_thread = None
        _mcp_loop.close()
        _mcp_loop = None
