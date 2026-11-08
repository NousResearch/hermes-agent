"""MCP + UTCP coexistence: both providers live in one agent tool snapshot.

Scenario C from the acceptance criteria: an MCP server and a UTCP provider are
configured at the same time, both tool sets are discovered and registered, and
both execute inside the same process.  The MCP side runs the *real* stdio client
(``tools/mcp_tool.py`` + the official ``mcp`` SDK) against a local FastMCP
server, so this is not a registry-level stub: it proves the two providers do not
collide in the registry, the toolset map, or the dispatch path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools import utcp_tool
from tools.registry import registry
from tests.tools.utcp_mock_provider import MockUtcpProvider, utcp_tool as mock_tool

pytest.importorskip("mcp", reason="coexistence test needs the optional mcp SDK")

MCP_SERVER_SCRIPT = str(Path(__file__).with_name("mcp_mock_server.py"))


def _mcp_api():
    """Locate the MCP registration/shutdown API (module layout moved over time)."""
    try:  # current layout: discovery/lifecycle split out of mcp_tool
        from tools.mcp_tool_discovery import register_mcp_servers
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers

        return register_mcp_servers, shutdown_mcp_servers
    except ImportError:  # older single-module layout
        from tools.mcp_tool import register_mcp_servers, shutdown_mcp_servers

        return register_mcp_servers, shutdown_mcp_servers


@pytest.fixture
def mcp_server():
    """Connect Hermes' MCP client to the local FastMCP server."""
    pytest.importorskip("mcp")
    register_mcp_servers, shutdown_mcp_servers = _mcp_api()

    config = {
        "mockmcp": {
            "command": sys.executable,
            "args": [MCP_SERVER_SCRIPT],
            "timeout": 30,
            "connect_timeout": 30,
        }
    }
    registered = register_mcp_servers(config)
    try:
        yield registered
    finally:
        try:
            shutdown_mcp_servers()
        except Exception:
            pass


@pytest.fixture
def utcp_provider(monkeypatch):
    utcp_tool.shutdown_utcp_providers()
    utcp_tool.reset_utcp_state_for_tests()
    server = MockUtcpProvider([mock_tool("echo", "/echo")]).start()
    section = {
        "enabled": True,
        "providers": [{"name": "demo", "call_template_type": "http", "url": server.manual_url}],
    }
    monkeypatch.setattr(utcp_tool, "_utcp_section", lambda: section)
    utcp_tool.discover_utcp_tools()
    try:
        yield server
    finally:
        utcp_tool.shutdown_utcp_providers()
        utcp_tool.reset_utcp_state_for_tests()
        server.stop()


def test_mcp_and_utcp_tools_coexist(mcp_server, utcp_provider):
    mcp_tools = [name for name in mcp_server if name.startswith("mcp__")]
    assert "mcp__mockmcp__mcp_echo" in mcp_tools, f"unexpected MCP tools: {registered_names()}"
    assert registry.get_toolset_for_tool("utcp__demo__echo") == "utcp-demo"
    assert (registry.get_toolset_for_tool("mcp__mockmcp__mcp_echo") or "").startswith("mcp-")

    # One agent snapshot contains both providers' tools.
    definitions = registry.get_definitions(
        {"utcp__demo__echo", "mcp__mockmcp__mcp_echo", "mcp__mockmcp__mcp_add"}
    )
    names = {d["function"]["name"] for d in definitions}
    assert names == {"utcp__demo__echo", "mcp__mockmcp__mcp_echo", "mcp__mockmcp__mcp_add"}

    # Both execute through the same dispatch path, in the same process.
    mcp_result = json.loads(registry.dispatch("mcp__mockmcp__mcp_echo", {"message": "hi"}))
    assert mcp_result["result"] == "mcp:hi"
    assert json.loads(registry.dispatch("mcp__mockmcp__mcp_add", {"a": 2, "b": 3}))["result"] == "5"

    utcp_result = json.loads(registry.dispatch("utcp__demo__echo", {"message": "hi"}))
    assert utcp_result["result"]["query"] == {"message": "hi"}

    # And again in the opposite order — no cross-provider state leaks.
    assert json.loads(registry.dispatch("utcp__demo__echo", {"message": "again"}))["result"]["query"] == {
        "message": "again"
    }
    assert json.loads(registry.dispatch("mcp__mockmcp__mcp_echo", {"message": "again"}))["result"] == "mcp:again"


def registered_names():
    return sorted(registry.get_all_tool_names())
