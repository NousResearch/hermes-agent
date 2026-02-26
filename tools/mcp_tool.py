"""
MCP Meta-tool -- User-facing tool for managing MCP server connections.

Actions:
  - ``status``         : Show all servers, connection state, tool counts
  - ``reconnect``      : Reconnect a specific server by name
  - ``list_tools``     : List all available MCP tools with their server mapping
  - ``list_resources`` : List all available MCP resources across servers
  - ``read_resource``  : Read a specific resource by URI from a server
  - ``list_prompts``   : List all available MCP prompt templates
  - ``get_prompt``     : Get a rendered prompt template with arguments
  - ``set_log_level``  : Set logging level on an MCP server

Import-time side effect:
  Triggers ``mcp_manager.initialize()`` which loads config, connects servers,
  and registers discovered MCP tools as native hermes tools.
"""

import json
import logging

from tools.mcp_manager import mcp_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialize MCP on import (connects servers + registers tools)
# ---------------------------------------------------------------------------

try:
    _init_result = mcp_manager.initialize()
    if _init_result.get("tools_registered", 0) > 0:
        logger.info(
            "MCP initialized: %d servers, %d tools",
            _init_result.get("connected", 0),
            _init_result.get("tools_registered", 0),
        )
except Exception as e:
    logger.debug("MCP initialization failed (non-fatal): %s", e)
    _init_result = {}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _handle_mcp(args: dict, **kwargs) -> str:
    """Handle MCP management actions."""
    action = args.get("action", "status")

    if action == "status":
        status = mcp_manager.get_status()
        return json.dumps(status, indent=2, ensure_ascii=False)

    elif action == "reconnect":
        server_name = args.get("server_name", "")
        if not server_name:
            return json.dumps({
                "error": "server_name is required for reconnect action",
            })
        result = mcp_manager.reconnect(server_name)
        return json.dumps(result, indent=2, ensure_ascii=False)

    elif action == "list_tools":
        status = mcp_manager.get_status()
        tools = []
        for srv_name, srv_info in status.get("servers", {}).items():
            if srv_info.get("connected"):
                for tool_name in srv_info.get("tool_names", []):
                    from tools.mcp_manager import make_tool_name
                    hermes_name = make_tool_name(srv_name, tool_name)
                    tools.append({
                        "hermes_name": hermes_name,
                        "mcp_name": tool_name,
                        "server": srv_name,
                    })
        return json.dumps({
            "total": len(tools),
            "tools": tools,
        }, indent=2, ensure_ascii=False)

    elif action == "list_resources":
        status = mcp_manager.get_status()
        resources = []
        for srv_name, srv_info in status.get("servers", {}).items():
            if srv_info.get("connected"):
                for uri in srv_info.get("resource_uris", []):
                    resources.append({
                        "uri": uri,
                        "server": srv_name,
                    })
        return json.dumps({
            "total": len(resources),
            "resources": resources,
        }, indent=2, ensure_ascii=False)

    elif action == "read_resource":
        server_name = args.get("server_name", "")
        uri = args.get("uri", "")
        if not server_name or not uri:
            return json.dumps({
                "error": "server_name and uri are required for read_resource action",
            })
        return mcp_manager._handle_resource_read(server_name, uri)

    elif action == "list_prompts":
        status = mcp_manager.get_status()
        prompts = []
        for srv_name, srv_info in status.get("servers", {}).items():
            if srv_info.get("connected"):
                for prompt_name in srv_info.get("prompt_names", []):
                    prompts.append({
                        "name": prompt_name,
                        "server": srv_name,
                    })
        return json.dumps({
            "total": len(prompts),
            "prompts": prompts,
        }, indent=2, ensure_ascii=False)

    elif action == "get_prompt":
        server_name = args.get("server_name", "")
        prompt_name = args.get("prompt_name", "")
        if not server_name or not prompt_name:
            return json.dumps({
                "error": "server_name and prompt_name are required for get_prompt action",
            })
        arguments = args.get("arguments")
        return mcp_manager._handle_prompt_get(server_name, prompt_name, arguments)

    elif action == "set_log_level":
        server_name = args.get("server_name", "")
        log_level = args.get("log_level", "")
        if not server_name or not log_level:
            return json.dumps({
                "error": "server_name and log_level are required for set_log_level action",
            })
        return mcp_manager._handle_set_log_level(server_name, log_level)

    else:
        return json.dumps({
            "error": (
                f"Unknown action '{action}'. Available: status, reconnect, "
                f"list_tools, list_resources, read_resource, list_prompts, "
                f"get_prompt, set_log_level"
            ),
        })


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def check_mcp_available() -> bool:
    """MCP meta-tool is always available (shows status even with no servers)."""
    return True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MCP_SCHEMA = {
    "name": "mcp",
    "description": (
        "Manage MCP (Model Context Protocol) server connections. "
        "MCP enables connecting to external tool servers that provide "
        "additional capabilities (GitHub, filesystem, databases, etc.). "
        "Actions: 'status' (show servers), 'reconnect' (reconnect a server), "
        "'list_tools' (list tools), 'list_resources' (list resources), "
        "'read_resource' (read a resource), 'list_prompts' (list prompts), "
        "'get_prompt' (get a prompt), 'set_log_level' (set server log level)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "reconnect", "list_tools",
                    "list_resources", "read_resource",
                    "list_prompts", "get_prompt",
                    "set_log_level",
                ],
                "description": (
                    "Action to perform: "
                    "'status' - Show all MCP servers and their connection state; "
                    "'reconnect' - Reconnect a specific server (requires server_name); "
                    "'list_tools' - List all available MCP tools with server mapping; "
                    "'list_resources' - List all available resources across servers; "
                    "'read_resource' - Read a resource by URI (requires server_name, uri); "
                    "'list_prompts' - List all available prompt templates; "
                    "'get_prompt' - Get a rendered prompt (requires server_name, prompt_name); "
                    "'set_log_level' - Set server log level (requires server_name, log_level)"
                ),
            },
            "server_name": {
                "type": "string",
                "description": "MCP server name (e.g., 'github', 'filesystem')",
            },
            "uri": {
                "type": "string",
                "description": "Resource URI for 'read_resource' action",
            },
            "prompt_name": {
                "type": "string",
                "description": "Prompt template name for 'get_prompt' action",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments for 'get_prompt' action (template variables)",
            },
            "log_level": {
                "type": "string",
                "enum": [
                    "debug", "info", "notice", "warning",
                    "error", "critical", "alert", "emergency",
                ],
                "description": "Log level for 'set_log_level' action",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="mcp",
    toolset="mcp",
    schema=MCP_SCHEMA,
    handler=_handle_mcp,
    check_fn=check_mcp_available,
    description="Manage MCP server connections and tools",
)
