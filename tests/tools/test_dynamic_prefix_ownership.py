"""Prefix carry-forward cannot change the advertised dispatch owner."""
from types import SimpleNamespace

import pytest

from tools.mcp_tool_agent import refresh_agent_mcp_tools
from tools.registry import ToolRegistry


def _schema(description):
    return {"type": "function", "function": {"name": "shared_tool", "description": description, "parameters": {}}}


def _agent(ownership):
    agent = SimpleNamespace(
        tools=[_schema("old contract")], valid_tool_names={"shared_tool"},
        enabled_toolsets=None, disabled_toolsets=None,
        _memory_manager=SimpleNamespace(
            get_all_tool_schemas=lambda: [_schema("provider")],
            has_tool=lambda name: name == "shared_tool",
        ),
        context_compressor=None, _context_engine_tool_names=set(),
    )
    if ownership != "legacy":
        agent._memory_provider_tool_names = {"shared_tool"} if ownership == "provider" else set()
    if ownership == "engine":
        agent._memory_manager = None
        agent._context_engine_tool_names = {"shared_tool"}
        agent.context_compressor = SimpleNamespace(get_tool_schemas=lambda: [_schema("engine")])
    return agent


def _registry(monkeypatch):
    registry = ToolRegistry()
    registry.register(name="shared_tool", toolset="prefix-test", schema={"name": "shared_tool"}, handler=lambda _args, **_kw: "registry")
    monkeypatch.setattr("tools.registry.registry", registry)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
    return registry


@pytest.mark.parametrize("ownership", ["provider", "legacy", "engine"])
def test_denied_dynamic_prefix_cannot_transfer_to_new_registry_owner(monkeypatch, ownership):
    _registry(monkeypatch)
    agent = _agent(ownership)
    refresh_agent_mcp_tools(agent, disabled_override=["all"], preserve_prefix=True)
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._memory_provider_tool_names == set()
    assert agent._context_engine_tool_names == set()
    assert agent._tool_registry_routes == {}


def test_explicit_empty_provider_ownership_preserves_registered_prefix(monkeypatch):
    _registry(monkeypatch)
    agent = _agent("empty")
    old = agent.tools
    refresh_agent_mcp_tools(agent, disabled_override=["all"], preserve_prefix=True)
    assert agent.tools == old
    assert agent._memory_provider_tool_names == set()
    assert set(agent._tool_registry_routes) == {"shared_tool"}


def test_unchanged_legacy_snapshot_publishes_registry_ownership(monkeypatch):
    agent = _agent("legacy")
    # Legacy direct extensions may have no optional registry-route metadata.
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [_schema("old contract")])
    refresh_agent_mcp_tools(agent)
    assert agent.tools == [_schema("old contract")]
    assert agent._memory_provider_tool_names == set()


def test_refresh_preserves_posture_core_memory(monkeypatch):
    import toolsets
    agent = _agent("provider")
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr(toolsets, "validate_toolset", lambda name: name == "custom-posture")
    monkeypatch.setattr(toolsets, "get_toolset", lambda _name: {"posture": True})
    monkeypatch.setattr(toolsets, "resolve_toolset", lambda _name: {"memory", "read_file"})
    monkeypatch.setattr(toolsets, "bundle_non_core_tools", lambda _name: set())
    refresh_agent_mcp_tools(agent, disabled_override=["custom-posture"])
    assert agent.valid_tool_names == {"shared_tool"}
    assert agent._memory_provider_tool_names == {"shared_tool"}
