"""A parked MCP server's tools must not be reported as a missing capability.

Parking deregisters a server's tools, but the conversation's schema is byte-stable (prompt
caching), so the model keeps calling them. ``Unknown tool: mcp__x__y`` reads as proof the
capability does not exist; the verdict must say the server is configured but not connected.
"""
from __future__ import annotations

import json
import os
import signal
import sys

import pytest

import hermes_yaml as yaml
from hermes_constants import get_hermes_home
from tests.waiting import wait_until

pytest.importorskip("mcp.server")

# Serves one tool; once the flag file exists every (re)spawn exits before the handshake.
_SERVER = """
import os, sys
flag, pid_file = sys.argv[1], sys.argv[2]
if os.path.exists(flag):
    sys.exit(0)
open(pid_file, "w").write(str(os.getpid()))
from mcp.server import MCPServer
server = MCPServer("flaky")

@server.tool()
def ping() -> str:
    return "pong"

server.run("stdio")
"""


@pytest.fixture
def flaky_server(tmp_path, monkeypatch):
    from tools import mcp_tool
    from tools.mcp_tool_discovery import discover_mcp_tools
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)  # park after one failed reconnect
    script, flag, pid_file = tmp_path / "flaky.py", tmp_path / "down", tmp_path / "pid"
    script.write_text(_SERVER, encoding="utf-8")
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"flaky": {
        "command": sys.executable, "args": [str(script), str(flag), str(pid_file)]}}}), encoding="utf-8")
    assert "mcp__flaky__ping" in discover_mcp_tools()
    yield flag, pid_file
    shutdown_mcp_servers()


def test_a_parked_servers_tool_says_not_connected_instead_of_unknown(flaky_server):
    from tools import mcp_tool
    from tools.registry import registry

    flag, pid_file = flaky_server
    assert json.loads(registry.dispatch("mcp__flaky__ping", {})).get("result") == "pong"
    flag.touch()
    os.kill(int(pid_file.read_text()), signal.SIGKILL)
    # The dead child is noticed by the next call; its respawn exits at once, so the server parks.
    assert "stdio subprocess" in json.loads(registry.dispatch("mcp__flaky__ping", {}))["error"]
    server = next(s for s in mcp_tool._servers.values() if s.name == "flaky")
    wait_until(lambda: server._was_parked and registry.get_entry("mcp__flaky__ping") is None,
               timeout=30, what="flaky MCP server to park")

    error = json.loads(registry.dispatch("mcp__flaky__ping", {}))["error"]

    assert "Unknown tool" not in error
    assert "'flaky'" in error and "not connected" in error and "NOT a missing capability" in error
    assert json.loads(registry.dispatch("mcp__nope__ping", {}))["error"] == "Unknown tool: mcp__nope__ping"


def test_another_profiles_tool_name_is_not_explained_to_this_one(monkeypatch):
    """Two profiles' servers can normalize to one tool name: provenance is per registry scope."""
    from tools import mcp_tool
    from tools.mcp_tool_registration import _track_mcp_tool_server

    monkeypatch.setattr(mcp_tool, "_known_mcp_tool_owners", type(mcp_tool._known_mcp_tool_owners)())
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: "profile-b")
    _track_mcp_tool_server("mcp__foo_bar__t", "foo-bar", "profile-a")

    assert mcp_tool._describe_unknown_mcp_tool("mcp__foo_bar__t") is None
    assert "foo-bar" in mcp_tool._describe_unknown_mcp_tool("mcp__foo_bar__t", "profile-a")
