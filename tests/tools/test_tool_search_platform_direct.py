"""``PlatformEntry.direct_toolsets``: a platform's own toolsets skip tool search in that platform's
sessions only; every other session keeps deferring them and can still reach them via ``tool_call``."""

import json

import pytest

import model_tools
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session_context import reset_session_vars, set_session_vars
from tools import tool_search as ts
from tools.registry import registry

PLATFORM = "pdt_device"
TOOL, TOOLSET = "pdt_device_action", "pdt_device_tools"
MCP_TOOL, MCP_TOOLSET = "mcp__pdt__op", "mcp-pdt"


def _td(name):
    return {"type": "function", "function": {
        "name": name, "description": "Act on the device.", "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture(autouse=True)
def device_platform():
    for name, toolset in ((TOOL, TOOLSET), (MCP_TOOL, MCP_TOOLSET)):
        registry.register(name=name, toolset=toolset, schema=_td(name)["function"],
                          handler=lambda args, **kw: json.dumps({"ok": True}))
    platform_registry.register(PlatformEntry(
        name=PLATFORM, label="Device", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        source="builtin", direct_toolsets=(TOOLSET, MCP_TOOLSET)))
    try:
        yield
    finally:
        platform_registry.unregister(PLATFORM)
        for name in (TOOL, MCP_TOOL):
            registry.deregister(name)
        reset_session_vars()


def _session(platform):
    reset_session_vars()
    if platform:
        set_session_vars(platform=platform)


def test_sent_directly_in_the_declaring_platforms_sessions():
    _session(PLATFORM)
    visible, deferrable = ts.classify_tools([_td(TOOL)])
    _, _, err = ts.resolve_underlying_call({"name": TOOL, "arguments": {}})

    assert [td["function"]["name"] for td in visible] == [TOOL] and deferrable == []
    assert TOOL not in ts.scoped_deferrable_names([_td(TOOL)])
    assert "directly-listed" in err


@pytest.mark.parametrize("platform", ["telegram", None])
def test_other_sessions_keep_deferring_and_can_reach_it_through_tool_call(platform):
    _session(platform)
    visible, deferrable = ts.classify_tools([_td(TOOL)])

    assert visible == [] and [td["function"]["name"] for td in deferrable] == [TOOL]
    assert TOOL in ts.scoped_deferrable_names([_td(TOOL)])
    assert ts.resolve_underlying_call({"name": TOOL, "arguments": {}}) == (TOOL, {}, None)
    assert ts.out_of_scope_reason(TOOL) is None  # never the desktop-only refusal


def test_the_user_defer_list_and_mcp_tools_still_defer():
    _session(PLATFORM)

    assert ts.is_deferrable_tool_name(TOOL, frozenset({TOOL})) is True
    assert ts.is_deferrable_tool_name(MCP_TOOL) is True


def test_tool_definition_cache_never_shares_lists_across_classifications():
    """``get_tool_definitions`` memoizes process-wide: the declaring platform's sessions get their own entry."""
    _session(PLATFORM)
    declaring = model_tools._tool_defs_cache_key(None, None, False)
    _session("telegram")
    other = model_tools._tool_defs_cache_key(None, None, False)
    _session(None)
    unbound = model_tools._tool_defs_cache_key(None, None, False)

    assert declaring != other
    assert other == unbound
