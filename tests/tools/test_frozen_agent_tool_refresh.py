"""Frozen restricted agents must remain frozen through every refresh caller."""
from types import SimpleNamespace

import pytest

from agent.conversation_compression import _refresh_agent_tool_definitions
from tools.mcp_tool_agent import refresh_agent_mcp_tools


def tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


@pytest.mark.parametrize("catalog", [[], ["tool_call", "tool_search", "tool_describe"], ["read_file", "terminal"]])
@pytest.mark.parametrize("via_compaction", [False, True])
def test_frozen_snapshot_survives_catalog_and_compaction_changes(monkeypatch, catalog, via_compaction):
    schemas = [tool("read_file")]
    names = {"read_file"}
    agent = SimpleNamespace(tools=schemas, valid_tool_names=names, enabled_toolsets=["voice_safe"], disabled_toolsets=None, _skip_mcp_refresh=True)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [tool(n) for n in catalog])
    monkeypatch.setattr("tools.mcp_tool_agent._reinject_post_build_tools", lambda *args: set())
    if via_compaction:
        assert _refresh_agent_tool_definitions(agent) is False
    else:
        assert refresh_agent_mcp_tools(agent, enabled_override=["all"], disabled_override=["web"], content_aware=True) == set()
    assert agent.tools is schemas
    assert agent.valid_tool_names is names
    assert agent.tools == [tool("read_file")]
    assert agent.valid_tool_names == {"read_file"}
    assert agent.enabled_toolsets == ["voice_safe"]
    assert agent.disabled_toolsets is None


def test_frozen_snapshot_does_not_query_catalog_or_reinject(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Frozen agent entered a mutable refresh path")

    agent = SimpleNamespace(_skip_mcp_refresh=True)
    monkeypatch.setattr("model_tools.get_tool_definitions", forbidden)
    monkeypatch.setattr("tools.mcp_tool_agent._reinject_post_build_tools", forbidden)
    assert refresh_agent_mcp_tools(agent, content_aware=True) == set()


def test_unfrozen_agent_can_refresh_and_change_toolset(monkeypatch):
    agent = SimpleNamespace(tools=[tool("read_file")], valid_tool_names={"read_file"}, enabled_toolsets=["voice_safe"], disabled_toolsets=None, _skip_mcp_refresh=False)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [tool("web_search")])
    monkeypatch.setattr("tools.mcp_tool_agent._reinject_post_build_tools", lambda *args: set())
    assert refresh_agent_mcp_tools(agent, enabled_override=["web"], content_aware=True) == {"web_search"}
    assert agent.valid_tool_names == {"web_search"}
    assert agent.enabled_toolsets == ["web"]
