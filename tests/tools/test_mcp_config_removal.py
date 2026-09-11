"""Live MCP tasks stop when another process removes their native config."""

import time

import pytest

import tools.mcp_tool as mcp_tool
from tools import mcp_tool_config, mcp_tool_discovery


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
        await server.run(mcp_tool_config._MCPServerConfig(
            {"command": "fake"}, native_config_managed=True))

        assert calls == 1
        assert server._shutdown_event.is_set()
        assert server.name not in mcp_tool._servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)


@pytest.mark.asyncio
async def test_native_snapshot_removed_before_run_never_spawns(monkeypatch):
    """Discovery provenance survives removal before task initialization."""
    calls = 0
    server = mcp_tool.MCPServerTask("removed-before-run")
    snapshot = mcp_tool_config._MCPServerConfig(
        {"command": "fake"}, native_config_managed=True)

    monkeypatch.setattr(
        mcp_tool_config,
        "_native_mcp_server_enabled",
        lambda _name: False,
    )

    async def transport(_server, _config):
        nonlocal calls
        calls += 1
        return "shutdown"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", transport)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_deregister_tools", lambda _server: None)
    with mcp_tool._lock:
        mcp_tool._servers[server.name] = server
        mcp_tool._server_scope_keys[server.name] = None

    try:
        await server.run(snapshot)

        assert calls == 0
        assert server._shutdown_event.is_set()
        assert server.name not in mcp_tool._servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)


def test_discovery_completes_when_native_snapshot_was_removed(monkeypatch):
    """Pre-spawn retirement is a clean discovery outcome, not a timeout."""
    name = "removed-during-discovery"
    calls = 0
    snapshot = mcp_tool_config._MCPServerConfig(
        {"command": "fake", "supports_parallel_tool_calls": True},
        native_config_managed=True,
    )

    monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(
        mcp_tool_config,
        "_native_mcp_server_enabled",
        lambda _name: False,
    )
    monkeypatch.setattr(
        mcp_tool_config,
        "_filter_suspicious_mcp_servers",
        lambda servers: servers,
    )

    async def transport(_server, _config):
        nonlocal calls
        calls += 1
        return "shutdown"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", transport)
    with mcp_tool._lock:
        mcp_tool._server_tool_scopes[name] = {"stale-scope"}
        mcp_tool._server_connect_errors[name] = "stale error"
        mcp_tool._server_connect_failures[name] = 2
        mcp_tool._server_connect_retry_after[name] = time.monotonic() - 1

    started = time.monotonic()
    try:
        assert mcp_tool_discovery.register_mcp_servers({name: snapshot}) == []
        assert time.monotonic() - started < 2
        assert calls == 0
        with mcp_tool._lock:
            assert name not in mcp_tool._servers
            assert name not in mcp_tool._server_scope_keys
            assert name not in mcp_tool._server_tool_scopes
            assert name not in mcp_tool._server_connecting
            assert name not in mcp_tool._server_connect_errors
            assert name not in mcp_tool._server_connect_failures
            assert name not in mcp_tool._server_connect_retry_after
            assert name not in mcp_tool._parallel_safe_servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(name, None)
            mcp_tool._server_scope_keys.pop(name, None)
            mcp_tool._server_tool_scopes.pop(name, None)
            mcp_tool._server_connecting.discard(name)
            mcp_tool._server_connect_errors.pop(name, None)
            mcp_tool._server_connect_failures.pop(name, None)
            mcp_tool._server_connect_retry_after.pop(name, None)
            mcp_tool._parallel_safe_servers.discard(name)
