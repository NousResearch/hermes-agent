"""Live MCP tasks stop when another process removes their native config."""

import pytest

import tools.mcp_tool as mcp_tool
from tools import mcp_tool_config


@pytest.mark.asyncio
async def test_removed_native_server_is_not_reconnected(monkeypatch):
    """A transport teardown after config removal must not spawn a successor."""
    configured = {"enabled": True}
    calls = 0
    server = mcp_tool.MCPServerTask("removed")

    monkeypatch.setattr(
        mcp_tool_config,
        "_native_mcp_server_enabled",
        lambda _name: configured["enabled"],
    )

    async def transport(_server, _config):
        nonlocal calls
        calls += 1
        configured["enabled"] = False
        return "reconnect"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", transport)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_deregister_tools", lambda _server: None)
    with mcp_tool._lock:
        mcp_tool._servers[server.name] = server
        mcp_tool._server_scope_keys[server.name] = None

    try:
        await server.run({"command": "fake"})

        assert calls == 1
        assert server._shutdown_event.is_set()
        assert server.name not in mcp_tool._servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)