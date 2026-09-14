"""Behavior-only reproduction for MCP quick-reconnect tool staleness on current main.

This test intentionally models two successful sessions on the same MCPServerTask:
initial discovery registers contract A, then a non-parked reconnect discovers
contract B under the same tool name.  No production code is changed here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools.mcp_tool import MCPServerTask
from tools.registry import ToolRegistry


def _tool(name, schema, description=""):
    return SimpleNamespace(name=name, inputSchema=schema, description=description)


@pytest.mark.asyncio
async def test_quick_reconnect_refreshes_same_name_changed_schema():
    """The registry must expose contract B after a quick reconnect."""
    contract_a = {"type": "object", "properties": {"old": {"type": "string"}}}
    contract_b = {"type": "object", "properties": {"new": {"type": "integer"}}}
    server_tool_name = "mcp__repro_srv__changing_tool"

    task = MCPServerTask("repro_srv")
    # A set ready state represents a live, non-parked reconnect path.  No
    # parked-state teardown/deregistration is performed in this test.
    task._ready.set()
    session = SimpleNamespace(list_tools=AsyncMock())
    task.session = session

    registry = ToolRegistry()
    with patch("tools.registry.registry", registry):
        session.list_tools.return_value = SimpleNamespace(
            tools=[_tool("changing_tool", contract_a)]
        )
        await task._discover_tools()
        first = registry._tools[server_tool_name].schema["parameters"]

        # Simulate the quick reconnect: same task/name, new MCP contract, no
        # parked-state reset and no explicit refresh call.
        session.list_tools.return_value = SimpleNamespace(
            tools=[_tool("changing_tool", contract_b)]
        )
        task._was_parked = False
        await task._discover_tools()
        actual = registry._tools[server_tool_name].schema["parameters"]

    assert first == contract_a
    assert actual == contract_b
