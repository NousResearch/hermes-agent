"""Regression tests for MCP stdio startup stderr diagnostics."""

import asyncio
import logging
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Keep this test from leaking MCP connection state across the suite."""
    import tools.mcp_tool as mcp

    old_servers = dict(mcp._servers)
    old_loop = mcp._mcp_loop
    old_thread = mcp._mcp_thread
    mcp._servers.clear()
    try:
        yield
    finally:
        mcp._servers.clear()
        mcp._servers.update(old_servers)
        mcp._mcp_loop = old_loop
        mcp._mcp_thread = old_thread


def _run_coro(coro, timeout=120):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_register_mcp_servers_logs_stderr_path_for_stdio_startup_failure(caplog):
    """When stdio startup fails, the warning points operators to mcp-stderr.log."""
    import tools.mcp_tool as mcp

    async def fake_discover_and_register(_name, _config):
        raise RuntimeError("synthetic startup failure")

    servers = {
        "broken_stdio": {
            "command": "python",
            "args": ["server.py"],
            "connect_timeout": 1,
        }
    }

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop"), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_run_coro), \
         patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_discover_and_register):
        assert mcp.register_mcp_servers(servers) == []

    assert "Failed to connect to MCP server 'broken_stdio'" in caplog.text
    assert "synthetic startup failure" in caplog.text
    assert "logs/mcp-stderr.log" in caplog.text
    assert "stderr" in caplog.text.lower()


def test_register_mcp_servers_does_not_add_stderr_path_for_http_failure(caplog):
    """HTTP MCP failures do not point users at the stdio subprocess stderr log."""
    import tools.mcp_tool as mcp

    async def fake_discover_and_register(_name, _config):
        raise RuntimeError("synthetic http failure")

    servers = {
        "broken_http": {
            "url": "http://127.0.0.1:65535/mcp",
            "connect_timeout": 1,
        }
    }

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop"), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_run_coro), \
         patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_discover_and_register):
        assert mcp.register_mcp_servers(servers) == []

    assert "Failed to connect to MCP server 'broken_http'" in caplog.text
    assert "synthetic http failure" in caplog.text
    assert "logs/mcp-stderr.log" not in caplog.text
