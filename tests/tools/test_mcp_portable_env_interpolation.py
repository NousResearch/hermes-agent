"""Portable (plugin) HTTP MCP servers get ``${VAR}`` rendered like native ones, so a reconnect keeps working."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PORTABLE = {"srv": {"type": "streamable-http", "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer ${EXAMPLE_MCP_TOKEN}"}}}


def _portable(monkeypatch, portable=PORTABLE):
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: {k: dict(v) for k, v in portable.items()})
    return (patch("hermes_cli.plugins.discover_plugins"),
            patch("hermes_cli.plugins.get_plugin_manager", return_value=manager))


def test_portable_http_server_header_is_rendered(monkeypatch):
    from tools.mcp_tool_config import _portable_mcp_servers, _require_rendered_remote
    monkeypatch.setenv("EXAMPLE_MCP_TOKEN", "tk_secret")
    p1, p2 = _portable(monkeypatch)
    with p1, p2:
        servers = {}
        _portable_mcp_servers(servers)
    assert servers["srv"]["headers"]["Authorization"] == "Bearer tk_secret"
    assert _require_rendered_remote("srv", servers["srv"]) is servers["srv"]


@pytest.mark.no_isolate
def test_reconnect_refresh_sees_no_definition_change(monkeypatch, tmp_path):
    """_refresh_remote_config re-reads _load_mcp_config(); the rendered running config must compare equal."""
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_tool import MCPServerTask
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("EXAMPLE_MCP_TOKEN", "tk_secret")
    p1, p2 = _portable(monkeypatch)
    with p1, p2, patch("hermes_cli.config.load_config", return_value={"mcp_servers": {}}):
        # What the server was started with: the same definition with its ${VAR} rendered.
        running = dict(PORTABLE["srv"], headers={"Authorization": "Bearer tk_secret"})
        task = MCPServerTask("srv")
        task._config = running
        assert _load_mcp_config()["srv"] == running
        assert task._refresh_remote_config(running) is running  # same dict back: no "definition changed" rebuild
