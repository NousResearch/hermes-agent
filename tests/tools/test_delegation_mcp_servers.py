"""delegation.mcp_servers reserves MCP servers for delegate_task children.

Regression for #15528: the main agent must never carry a reserved server's tools, whatever toolset selection
its surface hands in, while every delegated child gets them. Exercised end to end against a real config.yaml in
the isolated HERMES_HOME, the real registry, and the real tool-definition builder.
"""

from types import SimpleNamespace

import pytest

import model_tools
from agent.delegation_context import delegated_child_context
from hermes_cli.config import save_config
from tools.delegate_tool_toolsets import _resolve_child_toolsets
from tools.registry import registry

_SHARED = "mcp__shared__ping"
_RESERVED = "mcp__childonly__ping"


@pytest.fixture
def mcp_tools():
    save_config({
        "mcp_servers": {"shared": {"command": "true"}, "childonly": {"command": "true"}},
        # A ``mcp-`` prefixed spelling must reserve the same server as the bare name.
        "delegation": {"mcp_servers": ["mcp-childonly"]},
    })
    for name, toolset in ((_SHARED, "mcp-shared"), (_RESERVED, "mcp-childonly")):
        registry.register(name=name, toolset=toolset, handler=lambda a, **kw: "{}",
                          schema={"name": name, "description": "x", "parameters": {"type": "object", "properties": {}}})
        registry.register_toolset_alias(toolset.removeprefix("mcp-"), toolset)  # as MCP registration does
    model_tools._clear_tool_defs_cache()
    yield
    for name in (_SHARED, _RESERVED):
        registry.deregister(name)
    model_tools._clear_tool_defs_cache()


def _names(enabled):
    return {t["function"]["name"] for t in model_tools.get_tool_definitions(
        enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True)}


@pytest.mark.parametrize("enabled", [None, ["all"], ["terminal", "shared", "childonly"], ["mcp-shared", "mcp-childonly"]])
def test_reserved_server_reaches_children_only(mcp_tools, enabled):
    main = _names(enabled)
    assert _SHARED in main and _RESERVED not in main

    with delegated_child_context():
        child = _names(enabled)
    assert {_SHARED, _RESERVED} <= child


def test_child_is_granted_reserved_server_its_parent_never_had(mcp_tools):
    parent = SimpleNamespace(enabled_toolsets=["terminal", "shared"], disabled_toolsets=[])
    for requested in (None, ["terminal"]):
        enabled, _ = _resolve_child_toolsets(parent, requested, "leaf")
        with delegated_child_context():
            assert _RESERVED in _names(enabled)
    assert parent.enabled_toolsets == ["terminal", "shared"]
