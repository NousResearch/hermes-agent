"""Seam regressions for the extracted MCP provenance helpers."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools import mcp_provenance, mcp_tool


def test_provenance_helpers_are_exact_legacy_reexports():
    assert mcp_tool._track_mcp_tool_server is mcp_provenance._track_mcp_tool_server
    assert mcp_tool._forget_mcp_tool_server is mcp_provenance._forget_mcp_tool_server


def test_provenance_helpers_resolve_authoritative_state_at_call_time(monkeypatch):
    state = {"mcp__srv__existing": "old-server"}
    lock = threading.Lock()
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", state)
    monkeypatch.setattr(mcp_tool, "_lock", lock)

    mcp_provenance._track_mcp_tool_server("mcp__srv__existing", "new-server")
    mcp_tool._track_mcp_tool_server("mcp__srv__new", "raw-server")
    assert state == {
        "mcp__srv__existing": "new-server",
        "mcp__srv__new": "raw-server",
    }

    mcp_provenance._forget_mcp_tool_server("mcp__srv__existing")
    mcp_tool._forget_mcp_tool_server("mcp__srv__missing")
    assert state == {"mcp__srv__new": "raw-server"}


def test_provenance_helpers_handle_overwrite_and_idempotent_forget(monkeypatch):
    state = {}
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", state)
    monkeypatch.setattr(mcp_tool, "_lock", threading.Lock())

    mcp_tool._track_mcp_tool_server("tool", "server-a")
    mcp_tool._track_mcp_tool_server("tool", "server-b")
    mcp_tool._track_mcp_tool_server("other", "server-c")
    assert state == {"tool": "server-b", "other": "server-c"}

    mcp_provenance._forget_mcp_tool_server("tool")
    mcp_provenance._forget_mcp_tool_server("tool")
    assert state == {"other": "server-c"}


def test_original_namespace_monkeypatch_intercepts_registration():
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    server = mcp_tool.MCPServerTask("srv")
    server.session = MagicMock()
    server._tools = [
        SimpleNamespace(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {}},
        )
    ]

    with patch("tools.registry.registry", registry), patch(
        "tools.mcp_tool._track_mcp_tool_server"
    ) as track:
        registered = mcp_tool._register_server_tools(
            "srv", server, {"tools": {"resources": False, "prompts": False}}
        )

    assert registered == ["mcp__srv__read_file"]
    track.assert_called_once_with("mcp__srv__read_file", "srv")
