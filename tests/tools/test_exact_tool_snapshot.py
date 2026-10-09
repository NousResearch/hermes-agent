"""MCP rebuilds and saved client schema pins cannot widen exact grants."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.tool_permissions import ToolPermissionPolicy
from tools.mcp_tool_agent import refresh_agent_mcp_tools, restore_agent_tool_prefix
from tools.registry import registry


def schema(name):
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}},
    }}


@pytest.mark.parametrize("operation", ["refresh", "restore"])
def test_schema_merge_filters_after_injection_and_saved_pin(operation):
    allowed = "exact_snapshot_read"
    denied = "exact_snapshot_client"
    for name in [allowed, denied]:
        registry.register(name=name, toolset="desktop_ui", schema=schema(name)["function"],
                          handler=lambda args, **kw: "{}", check_fn=lambda: True)
    agent = SimpleNamespace(
        _tool_policy=ToolPermissionPolicy.from_config({"agent": {"allowed_tools": [allowed]}}),
        tools=[schema(allowed)], valid_tool_names={allowed},
        enabled_toolsets=None, disabled_toolsets=None, _context_engine_tool_names=set(),
        context_compressor=SimpleNamespace(get_tool_schemas=lambda: [schema(denied)["function"]]),
        _memory_manager=None, _session_db=None, session_id="snapshot",
    )
    try:
        with (
            patch("model_tools.get_tool_definitions", return_value=[schema(allowed), schema(denied)]),
            patch("tools.mcp_tool_agent._reinject_authorized_dynamic_tools"),
        ):
            if operation == "refresh":
                refresh_agent_mcp_tools(agent, preserve_prefix=True)
            else:
                restore_agent_tool_prefix(agent, [schema(denied), schema(allowed)])
        assert agent.valid_tool_names == {allowed}
        assert [d["function"]["name"] for d in agent.tools] == [allowed]
    finally:
        for name in [allowed, denied]:
            registry._tools.pop(name, None)
