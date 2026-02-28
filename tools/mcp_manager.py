"""
MCP Manager -- Singleton lifecycle manager for MCP server connections.

Responsibilities:
  - Parse ``mcp_servers`` config from ``~/.hermes/config.yaml``
  - Manage MCP client connections (connect, reconnect, shutdown)
  - Bridge MCP tools into the hermes tool registry as native tools
  - Provide check_fn per server so disconnected tools are auto-hidden
  - Exponential backoff on reconnection to avoid spin-loops
  - Error message sanitization to prevent credential leakage

Usage:
    from tools.mcp_manager import mcp_manager

    mcp_manager.initialize()          # Load config, connect, register tools
    mcp_manager.shutdown_all()        # Graceful shutdown
    mcp_manager.reconnect("github")   # Reconnect a specific server
"""

import copy
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.mcp_client import (
    MCPClient,
    StdioTransport,
    HttpTransport,
    MCPTransportError,
    MCPProtocolError,
    sanitize_error,
    _MCP_LOG_LEVELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config structures
# ---------------------------------------------------------------------------

class MCPServerConfig:
    """Parsed configuration for a single MCP server."""

    def __init__(
        self,
        name: str,
        transport_type: str,  # "stdio" or "http"
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        enabled: bool = True,
        auto_connect: bool = True,
    ):
        self.name = name
        self.transport_type = transport_type
        self.command = command
        self.args = args or []
        self.url = url
        self.headers = headers or {}
        self.env = env or {}
        self.cwd = cwd
        self.enabled = enabled
        self.auto_connect = auto_connect

    @classmethod
    def from_dict(cls, name: str, cfg: dict) -> "MCPServerConfig":
        """Parse a server config dict from config.yaml with validation."""
        if not isinstance(cfg, dict):
            logger.warning("MCP server '%s' config is not a dict -- skipping", name)
            return cls(name=name, transport_type="unknown", enabled=False)

        enabled = bool(cfg.get("enabled", True))
        auto_connect = bool(cfg.get("auto_connect", True))

        # Validate env
        env = cfg.get("env", {})
        if not isinstance(env, dict):
            logger.warning("MCP server '%s': env must be a dict", name)
            env = {}
        # Ensure all values are strings
        env = {str(k): str(v) for k, v in env.items()}

        if "url" in cfg:
            url = str(cfg["url"])
            headers = cfg.get("headers", {})
            if not isinstance(headers, dict):
                headers = {}
            headers = {str(k): str(v) for k, v in headers.items()}
            return cls(
                name=name,
                transport_type="http",
                url=url,
                headers=headers,
                env=env,
                enabled=enabled,
                auto_connect=auto_connect,
            )
        elif "command" in cfg:
            command = str(cfg["command"])
            args = cfg.get("args", [])
            if not isinstance(args, list):
                args = [str(args)]
            else:
                args = [str(a) for a in args]
            cwd = cfg.get("cwd")
            if cwd is not None:
                cwd = str(cwd)
            return cls(
                name=name,
                transport_type="stdio",
                command=command,
                args=args,
                env=env,
                cwd=cwd,
                enabled=enabled,
                auto_connect=auto_connect,
            )
        else:
            logger.warning(
                "MCP server '%s' has no 'command' or 'url' -- skipping", name
            )
            return cls(
                name=name,
                transport_type="unknown",
                enabled=False,
            )


class MCPServerConnection:
    """A live connection to an MCP server with its discovered tools."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.client: Optional[MCPClient] = None
        self.tools: List[dict] = []
        self.resources: List[dict] = []
        self.prompts: List[dict] = []
        self.subscribed_resources: set = set()
        self.connected = False
        self.error: Optional[str] = None
        # Reconnection backoff tracking
        self.reconnect_attempts = 0
        self.last_reconnect_time: float = 0


# ---------------------------------------------------------------------------
# Tool name helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convert a string to a valid tool name component (lowercase, underscores)."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def make_tool_name(server_name: str, tool_name: str) -> str:
    """Create a hermes tool name: ``mcp_{server}_{tool}``."""
    return f"mcp_{_sanitize_name(server_name)}_{_sanitize_name(tool_name)}"


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def _dereference_schema(schema: dict) -> dict:
    """Recursively inline JSON Schema ``$ref`` definitions.

    Some MCP tools use ``$ref`` for shared type definitions. Many LLMs
    handle inlined schemas better than ``$ref`` pointers.
    """
    defs = schema.get("$defs", schema.get("definitions", {}))
    if not defs:
        return schema

    result = copy.deepcopy(schema)

    def _resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Handle #/$defs/Name and #/definitions/Name
                for prefix in ("#/$defs/", "#/definitions/"):
                    if ref_path.startswith(prefix):
                        key = ref_path[len(prefix):]
                        if key in defs:
                            return _resolve(copy.deepcopy(defs[key]))
                return obj
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    result = _resolve(result)
    result.pop("$defs", None)
    result.pop("definitions", None)
    return result


def _convert_mcp_schema(mcp_tool: dict, server_name: str) -> dict:
    """Convert an MCP tool definition to a hermes-compatible OpenAI schema."""
    tool_name = make_tool_name(server_name, mcp_tool["name"])
    input_schema = mcp_tool.get("inputSchema", {})

    # Dereference $ref for LLM compatibility
    if "$defs" in input_schema or "definitions" in input_schema:
        input_schema = _dereference_schema(input_schema)
    else:
        input_schema = copy.deepcopy(input_schema)

    # Ensure we have a proper JSON Schema object
    if not input_schema.get("type"):
        input_schema["type"] = "object"
    if "properties" not in input_schema:
        input_schema["properties"] = {}

    return {
        "name": tool_name,
        "description": (
            f"[MCP:{server_name}] {mcp_tool.get('description', mcp_tool['name'])}"
        ),
        "parameters": input_schema,
    }


# ---------------------------------------------------------------------------
# MCPManager (singleton)
# ---------------------------------------------------------------------------

class MCPManager:
    """Manages MCP server connections and bridges tools into the hermes registry."""

    def __init__(self):
        self._servers: Dict[str, MCPServerConnection] = {}
        self._initialized = False
        self._reconnect_lock = threading.Lock()

    @property
    def servers(self) -> Dict[str, MCPServerConnection]:
        return self._servers

    def _load_config(self) -> Dict[str, dict]:
        """Load mcp_servers section from ~/.hermes/config.yaml."""
        try:
            config_path = Path(os.path.expanduser("~/.hermes/config.yaml"))
            if not config_path.exists():
                return {}
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return config.get("mcp_servers", {})
        except ImportError:
            # yaml not available -- try json fallback
            try:
                config_path = Path(os.path.expanduser("~/.hermes/config.json"))
                if not config_path.exists():
                    return {}
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return config.get("mcp_servers", {})
            except Exception:
                return {}
        except Exception as e:
            logger.debug("Failed to load MCP config: %s", e)
            return {}

    def _create_transport(self, config: MCPServerConfig):
        """Create the appropriate transport for a server config."""
        if config.transport_type == "stdio":
            return StdioTransport(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
                cwd=config.cwd,
            )
        elif config.transport_type == "http":
            return HttpTransport(
                url=config.url,
                headers=config.headers,
            )
        else:
            raise MCPTransportError(
                f"Unknown transport type: {config.transport_type}"
            )

    def _connect_server(self, conn: MCPServerConnection) -> bool:
        """Connect to a single MCP server with exponential backoff.

        Returns True on success.
        """
        config = conn.config

        # Exponential backoff check
        if conn.reconnect_attempts > 0:
            backoff = min(2 ** conn.reconnect_attempts, 60)  # max 60s
            elapsed = time.time() - conn.last_reconnect_time
            if elapsed < backoff:
                logger.debug(
                    "MCP '%s' backoff: %ds (elapsed %.1fs, attempt %d)",
                    config.name, backoff, elapsed, conn.reconnect_attempts,
                )
                return False

        try:
            transport = self._create_transport(config)
            client = MCPClient(transport)
            caps = client.connect()
            server_caps = caps.get("capabilities", {})

            tools = client.list_tools()
            conn.client = client
            conn.tools = tools
            conn.connected = True
            conn.error = None
            conn.reconnect_attempts = 0  # reset on success

            # Discover resources if server supports them
            if "resources" in server_caps:
                try:
                    conn.resources = client.list_resources()
                except Exception as e:
                    logger.debug("MCP '%s' resource discovery failed: %s", config.name, e)
                    conn.resources = []

                # Register resource update notification handler
                def _on_res_updated(params, srv=config.name):
                    self._on_resource_updated(srv, params)
                client.on_notification("notifications/resources/updated", _on_res_updated)

            # Discover prompts if server supports them
            if "prompts" in server_caps:
                try:
                    conn.prompts = client.list_prompts()
                except Exception as e:
                    logger.debug("MCP '%s' prompt discovery failed: %s", config.name, e)
                    conn.prompts = []

            # Register progress notification handler
            def _on_progress(params, srv=config.name):
                self._on_progress(srv, params)
            client.on_notification("notifications/progress", _on_progress)

            # Register list_changed notification handlers (capability-gated)
            if server_caps.get("tools", {}).get("listChanged"):
                def _on_tools_changed(params, srv=config.name):
                    self._on_tools_list_changed(srv)
                client.on_notification("notifications/tools/list_changed", _on_tools_changed)

            if server_caps.get("resources", {}).get("listChanged"):
                def _on_res_changed(params, srv=config.name):
                    self._on_resources_list_changed(srv)
                client.on_notification("notifications/resources/list_changed", _on_res_changed)

            if server_caps.get("prompts", {}).get("listChanged"):
                def _on_prompts_changed(params, srv=config.name):
                    self._on_prompts_list_changed(srv)
                client.on_notification("notifications/prompts/list_changed", _on_prompts_changed)

            # Register logging notification handler
            if "logging" in server_caps:
                def _on_log(params, srv=config.name):
                    self._on_log_message(srv, params)
                client.on_notification("notifications/message", _on_log)

            logger.info(
                "MCP server '%s' connected (%d tools, %d resources, %d prompts)",
                config.name, len(tools), len(conn.resources), len(conn.prompts),
            )
            return True

        except Exception as e:
            conn.connected = False
            conn.error = sanitize_error(str(e))
            conn.reconnect_attempts += 1
            conn.last_reconnect_time = time.time()
            logger.warning(
                "MCP server '%s' failed to connect (attempt %d): %s",
                config.name, conn.reconnect_attempts, conn.error,
            )
            return False

    def _register_all_tools(self) -> List[str]:
        """Register all discovered MCP tools into the hermes registry.

        Returns list of registered tool names.
        """
        from tools.registry import registry
        from toolsets import create_custom_toolset

        registered_names = []

        for server_name, conn in self._servers.items():
            if not conn.connected or not conn.tools:
                continue

            for mcp_tool in conn.tools:
                schema = _convert_mcp_schema(mcp_tool, server_name)
                tool_name = schema["name"]
                original_tool_name = mcp_tool["name"]

                # Create a closure that captures the correct server + tool name
                def _make_handler(srv_name: str, orig_name: str) -> Callable:
                    def handler(args: dict, **kwargs) -> str:
                        return self._handle_tool_call(srv_name, orig_name, args)
                    return handler

                def _make_check_fn(srv_name: str) -> Callable:
                    def check_fn() -> bool:
                        c = self._servers.get(srv_name)
                        return c is not None and c.connected
                    return check_fn

                registry.register(
                    name=tool_name,
                    toolset="mcp",
                    schema=schema,
                    handler=_make_handler(server_name, original_tool_name),
                    check_fn=_make_check_fn(server_name),
                    description=schema["description"],
                )
                registered_names.append(tool_name)

        # Create/update the mcp toolset with all registered tool names
        all_mcp_tools = ["mcp"] + registered_names
        create_custom_toolset(
            name="mcp",
            description="MCP (Model Context Protocol) server tools -- connect to external tool servers",
            tools=all_mcp_tools,
        )

        return registered_names

    def _handle_tool_call(
        self, server_name: str, tool_name: str, args: dict
    ) -> str:
        """Route a tool call to the correct MCP server."""
        conn = self._servers.get(server_name)
        if not conn:
            return json.dumps({"error": f"MCP server '{server_name}' not found"})

        if not conn.connected or not conn.client:
            # Use lock to prevent parallel reconnect attempts
            with self._reconnect_lock:
                if not conn.connected:
                    logger.info("MCP server '%s' disconnected, attempting reconnect...", server_name)
                    if not self._connect_server(conn):
                        return json.dumps({
                            "error": f"MCP server '{server_name}' is disconnected: {conn.error}",
                        })

        try:
            result = conn.client.call_tool(tool_name, args if args else None)

            # Extract text content from MCP response
            content_blocks = result.get("content", [])
            is_error = result.get("isError", False)

            texts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "resource":
                    texts.append(json.dumps(block.get("resource", {})))
                else:
                    texts.append(json.dumps(block))

            output = "\n".join(texts) if texts else json.dumps(result)

            if is_error:
                return json.dumps({"error": output, "mcp_server": server_name})
            return json.dumps({
                "result": output,
                "mcp_server": server_name,
                "mcp_tool": tool_name,
            }, ensure_ascii=False)

        except MCPProtocolError as e:
            return json.dumps({
                "error": f"MCP tool error: {sanitize_error(e.error_message)}",
                "code": e.code,
                "mcp_server": server_name,
            })
        except MCPTransportError as e:
            conn.connected = False
            return json.dumps({
                "error": f"MCP transport error: {sanitize_error(str(e))}",
                "mcp_server": server_name,
            })
        except Exception as e:
            return json.dumps({
                "error": f"MCP call failed: {sanitize_error(str(e))}",
                "mcp_server": server_name,
            })

    # ------------------------------------------------------------------
    # Notification handlers
    # ------------------------------------------------------------------

    def _on_resource_updated(self, server_name: str, params: dict) -> None:
        """Handle resource update notification -- re-read the resource."""
        uri = params.get("uri", "")
        logger.debug("MCP '%s' resource updated: %s", server_name, uri)

    def _on_progress(self, server_name: str, params: dict) -> None:
        """Handle progress notification from a server."""
        progress = params.get("progress", 0)
        total = params.get("total")
        msg = params.get("message", "")
        token = params.get("progressToken", "?")
        if total is not None:
            logger.debug(
                "MCP '%s' progress [%s]: %s/%s %s",
                server_name, token, progress, total, msg,
            )
        else:
            logger.debug(
                "MCP '%s' progress [%s]: %s %s",
                server_name, token, progress, msg,
            )

    def _on_tools_list_changed(self, server_name: str) -> None:
        """Handle tools/list_changed -- re-discover tools and re-register."""
        conn = self._servers.get(server_name)
        if not conn or not conn.client or not conn.connected:
            return
        try:
            conn.tools = conn.client.list_tools()
            self._register_all_tools()
            logger.info(
                "MCP '%s' tools updated (%d tools)", server_name, len(conn.tools),
            )
        except Exception as e:
            logger.warning("MCP '%s' tools re-discovery failed: %s", server_name, e)

    def _on_resources_list_changed(self, server_name: str) -> None:
        """Handle resources/list_changed -- re-discover resources."""
        conn = self._servers.get(server_name)
        if not conn or not conn.client or not conn.connected:
            return
        try:
            conn.resources = conn.client.list_resources()
            logger.info(
                "MCP '%s' resources updated (%d resources)",
                server_name, len(conn.resources),
            )
        except Exception as e:
            logger.warning("MCP '%s' resources re-discovery failed: %s", server_name, e)

    def _on_prompts_list_changed(self, server_name: str) -> None:
        """Handle prompts/list_changed -- re-discover prompts."""
        conn = self._servers.get(server_name)
        if not conn or not conn.client or not conn.connected:
            return
        try:
            conn.prompts = conn.client.list_prompts()
            logger.info(
                "MCP '%s' prompts updated (%d prompts)",
                server_name, len(conn.prompts),
            )
        except Exception as e:
            logger.warning("MCP '%s' prompts re-discovery failed: %s", server_name, e)

    def _on_log_message(self, server_name: str, params: dict) -> None:
        """Handle logging notification -- map to Python logger."""
        level_str = params.get("level", "info")
        py_level = _MCP_LOG_LEVELS.get(level_str, logging.INFO)
        log_data = params.get("data", "")
        log_logger = params.get("logger", "")
        prefix = f"MCP '{server_name}'"
        if log_logger:
            prefix += f" [{log_logger}]"
        logger.log(py_level, "%s: %s", prefix, log_data)

    def _handle_resource_read(self, server_name: str, uri: str) -> str:
        """Read a resource from a server and return JSON result."""
        conn = self._servers.get(server_name)
        if not conn or not conn.client or not conn.connected:
            return json.dumps({"error": f"MCP server '{server_name}' not connected"})
        try:
            result = conn.client.read_resource(uri)
            return json.dumps(result, ensure_ascii=False)
        except MCPProtocolError as e:
            return json.dumps({"error": sanitize_error(e.error_message)})
        except Exception as e:
            return json.dumps({"error": sanitize_error(str(e))})

    def _handle_prompt_get(
        self, server_name: str, prompt_name: str, arguments: Optional[dict] = None,
    ) -> str:
        """Get a rendered prompt from a server and return JSON result."""
        conn = self._servers.get(server_name)
        if not conn or not conn.client or not conn.connected:
            return json.dumps({"error": f"MCP server '{server_name}' not connected"})
        try:
            result = conn.client.get_prompt(prompt_name, arguments)
            return json.dumps(result, ensure_ascii=False)
        except MCPProtocolError as e:
            return json.dumps({"error": sanitize_error(e.error_message)})
        except Exception as e:
            return json.dumps({"error": sanitize_error(str(e))})

    def _handle_set_log_level(self, server_name: str, level: str) -> str:
        """Set log level on a server."""
        conn = self._servers.get(server_name)
        if not conn or not conn.client or not conn.connected:
            return json.dumps({"error": f"MCP server '{server_name}' not connected"})
        try:
            conn.client.set_log_level(level)
            return json.dumps({"status": "ok", "server": server_name, "level": level})
        except MCPProtocolError as e:
            return json.dumps({"error": sanitize_error(e.error_message)})
        except Exception as e:
            return json.dumps({"error": sanitize_error(str(e))})

    def initialize(self) -> Dict[str, Any]:
        """Load config, connect to servers, and register tools.

        Safe to call multiple times -- will skip if already initialized.
        Returns a summary dict.
        """
        if self._initialized:
            return self.get_status()

        raw_config = self._load_config()
        if not raw_config:
            self._initialized = True
            return {"servers": 0, "tools": 0, "message": "No MCP servers configured"}

        # Parse configs
        for name, cfg in raw_config.items():
            server_cfg = MCPServerConfig.from_dict(name, cfg)
            self._servers[name] = MCPServerConnection(server_cfg)

        # Connect enabled servers
        for name, conn in self._servers.items():
            if conn.config.enabled and conn.config.auto_connect:
                self._connect_server(conn)

        # Register tools
        registered = self._register_all_tools()

        self._initialized = True
        return {
            "servers": len(self._servers),
            "connected": sum(1 for c in self._servers.values() if c.connected),
            "tools_registered": len(registered),
        }

    def reconnect(self, server_name: str) -> Dict[str, Any]:
        """Reconnect a specific server."""
        conn = self._servers.get(server_name)
        if not conn:
            return {"error": f"Unknown MCP server: {server_name}"}

        # Disconnect first
        if conn.client:
            try:
                conn.client.disconnect()
            except Exception:
                pass
            conn.client = None
            conn.connected = False

        # Reset backoff for manual reconnect
        conn.reconnect_attempts = 0

        # Reconnect
        success = self._connect_server(conn)
        if success:
            self._register_all_tools()
            return {
                "status": "connected",
                "server": server_name,
                "tools": len(conn.tools),
            }
        return {
            "status": "failed",
            "server": server_name,
            "error": conn.error,
        }

    def shutdown_all(self) -> None:
        """Disconnect all MCP servers."""
        for name, conn in self._servers.items():
            if conn.client:
                try:
                    conn.client.disconnect()
                except Exception as e:
                    logger.debug("MCP shutdown error for '%s': %s", name, e)
                conn.client = None
                conn.connected = False

    def get_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers."""
        servers = {}
        for name, conn in self._servers.items():
            servers[name] = {
                "connected": conn.connected,
                "transport": conn.config.transport_type,
                "enabled": conn.config.enabled,
                "tools": len(conn.tools),
                "tool_names": [t["name"] for t in conn.tools],
                "resources": len(conn.resources),
                "resource_uris": [r.get("uri", "") for r in conn.resources],
                "prompts": len(conn.prompts),
                "prompt_names": [p.get("name", "") for p in conn.prompts],
                "error": conn.error,
            }
            if conn.client and conn.client._server_info:
                servers[name]["server_info"] = conn.client._server_info

        return {
            "total_servers": len(self._servers),
            "connected": sum(1 for c in self._servers.values() if c.connected),
            "total_tools": sum(len(c.tools) for c in self._servers.values() if c.connected),
            "total_resources": sum(len(c.resources) for c in self._servers.values() if c.connected),
            "total_prompts": sum(len(c.prompts) for c in self._servers.values() if c.connected),
            "servers": servers,
        }

    def get_all_tool_names(self) -> List[str]:
        """Get all registered MCP tool names."""
        names = []
        for server_name, conn in self._servers.items():
            if conn.connected:
                for tool in conn.tools:
                    names.append(make_tool_name(server_name, tool["name"]))
        return names


# Module-level singleton
mcp_manager = MCPManager()
