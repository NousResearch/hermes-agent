import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tools.mcp_tool import (
    register_mcp_servers,
    _get_active_server,
    _get_mcp_config_fingerprint,
    _servers,
    MCPServerTask,
)

@pytest.fixture(autouse=True)
def clean_servers():
    saved = dict(_servers)
    _servers.clear()
    yield
    _servers.clear()
    _servers.update(saved)

def test_mcp_config_fingerprint():
    """Verify that fingerprint is stable and ignores non-connection keys like enabled."""
    cfg1 = {"command": "node", "args": ["app.js"], "enabled": True}
    cfg2 = {"command": "node", "args": ["app.js"], "enabled": False}
    cfg3 = {"command": "node", "args": ["app.js", "--flag"]}

    fp1 = _get_mcp_config_fingerprint("github", cfg1)
    fp2 = _get_mcp_config_fingerprint("github", cfg2)
    fp3 = _get_mcp_config_fingerprint("github", cfg3)

    assert fp1 == fp2
    assert fp1 != fp3
    assert fp1.startswith("github:")

def test_mcp_servers_connection_isolation():
    """Verify that servers with same name but different config have isolated connections,
    while identical configs share a connection.
    """
    mock_session_a = MagicMock()
    mock_session_b = MagicMock()

    async def fake_connect(name, config):
        server = MCPServerTask(name)
        if config.get("env", {}).get("TOKEN") == "A":
            server.session = mock_session_a
        else:
            server.session = mock_session_b
        server._tools = []
        return server

    cfg_a = {"command": "node", "env": {"TOKEN": "A"}}
    cfg_b = {"command": "node", "env": {"TOKEN": "B"}}
    cfg_a_dup = {"command": "node", "env": {"TOKEN": "A"}, "enabled": True}

    def fake_run(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro)

    with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop"), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run):

        # Register config A
        register_mcp_servers({"github": cfg_a})
        # Determine fingerprint for A
        fp_a = _get_mcp_config_fingerprint("github", cfg_a)
        assert fp_a in _servers
        srv_a = _servers[fp_a]
        assert srv_a.session == mock_session_a

        # Register config B (same name, different config)
        register_mcp_servers({"github": cfg_b})
        fp_b = _get_mcp_config_fingerprint("github", cfg_b)
        assert fp_b in _servers
        srv_b = _servers[fp_b]
        assert srv_b.session == mock_session_b

        # They are isolated connections
        assert srv_a != srv_b
        assert fp_a != fp_b

        # Register identical config A (should be idempotent, no new connection)
        register_mcp_servers({"github": cfg_a_dup})
        assert _servers[fp_a] == srv_a

def test_get_active_server_routing():
    """Verify that _get_active_server routes dynamically based on active profile config."""
    srv_a = MCPServerTask("github")
    srv_b = MCPServerTask("github")

    cfg_a = {"command": "node", "env": {"TOKEN": "A"}}
    cfg_b = {"command": "node", "env": {"TOKEN": "B"}}

    fp_a = _get_mcp_config_fingerprint("github", cfg_a)
    fp_b = _get_mcp_config_fingerprint("github", cfg_b)

    _servers[fp_a] = srv_a
    _servers[fp_b] = srv_b

    # Mock _load_mcp_config to return config A
    with patch("tools.mcp_tool._load_mcp_config", return_value={"github": cfg_a}):
        assert _get_active_server("github") == srv_a

    # Mock _load_mcp_config to return config B
    with patch("tools.mcp_tool._load_mcp_config", return_value={"github": cfg_b}):
        assert _get_active_server("github") == srv_b
