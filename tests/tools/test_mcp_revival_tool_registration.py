"""Regression test for revived MCP server tool registration (#108087).

Ensures that _register_discovered_tools_if_needed() publishes discovered tools
during revival even when _ready is not set or ownership check is bypassed.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools.mcp_tool import MCPServerTask
from tools import mcp_tool_registration as _mcp_registration


def test_revival_registers_tools_when_ready_cleared_and_not_owned(monkeypatch):
    """A revived server must publish discovered tools when _registered_tool_names is empty."""
    server = MCPServerTask("test_mcp_server")
    server._config = {"url": "https://example.test/mcp"}
    server.session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[SimpleNamespace(name="query_data")],
            )
        )
    )
    server._ready.clear()
    server._registered_tool_names = []

    register_mock = MagicMock(return_value=["test_mcp_server__query_data"])
    monkeypatch.setattr(_mcp_registration, "_register_server_tools", register_mock)

    asyncio.run(server._discover_tools())

    register_mock.assert_called_once_with(server.name, server, server._config)
    assert server._registered_tool_names == ["test_mcp_server__query_data"]
