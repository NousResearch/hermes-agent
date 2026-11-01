"""Real stdio MCP acceptance for live tool-filter edits (regression #145265).

The child runs an offline MCPServer, and the *same* Hermes connection must
serve the newly selected tools while revoked tools cease being dispatchable.
No provider credentials, gateway process, or external network is needed.
"""

from __future__ import annotations

import sys

import pytest

_mcp_server = pytest.importorskip("mcp.server")
if not hasattr(_mcp_server, "MCPServer"):
    pytest.skip("Real stdio fixture requires MCPServer (mcp >= 2.0)", allow_module_level=True)


def test_live_stdio_filter_expansion_and_revocation_without_reconnect(tmp_path, monkeypatch):
    from tools import mcp_tool as core
    from tools import mcp_tool_config as config_module
    from tools import mcp_tool_discovery as discovery
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    fixture = tmp_path / "filter_fixture.py"
    fixture.write_text(
        "from mcp.server import MCPServer\n"
        "server = MCPServer('livefilters')\n"
        "@server.tool()\n"
        "def alpha(value: str = '') -> str:\n"
        "    return 'alpha:' + value\n"
        "@server.tool()\n"
        "def beta(value: str = '') -> str:\n"
        "    return 'beta:' + value\n"
        "if __name__ == '__main__':\n"
        "    server.run(transport='stdio')\n",
        encoding="utf-8",
    )
    initial = {
        "command": sys.executable, "args": [str(fixture)],
        "tools": {"include": ["alpha"]},
    }
    disk = {"livefilters": initial}
    monkeypatch.setattr(
        config_module, "_load_mcp_config",
        lambda: {name: dict(config) for name, config in disk.items()},
    )
    alpha, beta = "mcp__livefilters__alpha", "mcp__livefilters__beta"

    try:
        names = discovery.register_mcp_servers(dict(disk))
        assert alpha in names
        assert beta not in registry.get_tool_names_for_toolset("mcp-livefilters")
        with core._lock:
            running = core._servers.get("livefilters")
            session = getattr(running, "session", None)
        assert session is not None, "real stdio transport did not connect"
        assert "alpha:before" in str(registry.dispatch(alpha, {"value": "before"}))

        disk["livefilters"] = {**initial, "tools": {"include": ["alpha", "beta"]}}
        result = discovery.reconcile_mcp_servers_with_config()
        assert result == {"removed": [], "added": [], "pending": []}
        assert {alpha, beta} <= set(registry.get_tool_names_for_toolset("mcp-livefilters"))
        assert "beta:added" in str(registry.dispatch(beta, {"value": "added"}))
        assert core._servers["livefilters"] is running
        assert running.session is session, "filter edit must not reconnect"

        disk["livefilters"] = {**initial, "tools": {"include": ["beta"]}}
        discovery.reconcile_mcp_servers_with_config()
        assert alpha not in registry.get_tool_names_for_toolset("mcp-livefilters")
        assert "Unknown tool" in str(registry.dispatch(alpha, {"value": "revoked"}))
        assert "beta:still-live" in str(registry.dispatch(beta, {"value": "still-live"}))
        assert running.session is session

        disk["livefilters"] = {**initial, "tools": {"include": []}}
        discovery.reconcile_mcp_servers_with_config()
        assert beta not in registry.get_tool_names_for_toolset("mcp-livefilters")
        assert "Unknown tool" in str(registry.dispatch(beta, {"value": "revoked"}))

        disk["livefilters"] = initial
        discovery.reconcile_mcp_servers_with_config()
        assert "alpha:restored" in str(registry.dispatch(alpha, {"value": "restored"}))
        assert core._servers["livefilters"] is running
        assert running.session is session
    finally:
        shutdown_mcp_servers()
