"""An MCP server listing one tool name twice (QwenLM/qwen-code#15757 shape)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_registration import _register_server_tools
from tools.registry import ToolRegistry


def _tool(name, description, prop):
    return SimpleNamespace(name=name, description=description,
                           inputSchema={"type": "object", "properties": {prop: {"type": "string"}}})


def test_same_name_with_different_declarations_is_skipped_and_identical_repeat_kept():
    server = MCPServerTask("unity")
    server.session = MagicMock()
    server._tools = [
        _tool("create", "Create a GameObject", "gameobject_name"),
        _tool("create", "Create a C# script", "script_path"),
        _tool("ping", "Ping", "x"),
        _tool("ping", "Ping", "x"),
        _tool("other", "Unique", "y"),
    ]
    registry = ToolRegistry()
    with patch("tools.registry.registry", registry), patch("tools.mcp_tool_registration._track_mcp_tool_server"):
        registered = _register_server_tools("unity", server, {"tools": {"resources": False, "prompts": False}})

    assert registered == ["mcp__unity__ping", "mcp__unity__other"]
    assert registry.get_entry("mcp__unity__create") is None
