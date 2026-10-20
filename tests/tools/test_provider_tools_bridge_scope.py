"""Provider tools reach the tool_call bridge only when this session was granted them.

Memory-provider tools (fact_store) and context-engine tools (lcm_*) are appended
onto agent.tools and never registered, so a registry-only bridge scope rejects
them (#34520). The union follows the injection gate (#5544): a disabled or
unselected memory toolset must not leak, and a name the registry now owns is
not forced back in.
"""

from types import SimpleNamespace

import pytest

from agent.tool_executor import _tool_search_scoped_names, _unwrap_tool_search_call
from tools.tool_search import resolve_underlying_call


class _Names:
    def __init__(self, names):
        self._names = set(names)

    def get_all_tool_names(self):
        return set(self._names)


def _tool(name):
    return {"type": "function", "function": {"name": name}}


def _agent(*, enabled, disabled=None, tools=None, memory=(), context=()):
    return SimpleNamespace(
        enabled_toolsets=enabled,
        disabled_toolsets=disabled,
        tools=[] if tools is None else tools,
        _memory_manager=_Names(memory) if memory is not None else None,
        _context_engine_tool_names=set(context),
    )


@pytest.fixture
def registry_scope(monkeypatch):
    """Keep the registry half of the scope to one deferrable name."""
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kwargs: [_tool("session_search")],
    )


def _scoped(agent):
    return _tool_search_scoped_names(agent)


@pytest.mark.usefixtures("registry_scope")
class TestProviderToolsBridgeScope:
    def test_unfiltered_session_includes_memory_provider_tools(self):
        names = _scoped(_agent(enabled=None, memory={"fact_store", "fact_feedback"}))
        assert {"fact_store", "fact_feedback", "session_search"} <= names

    def test_memory_toolset_includes_provider_tools(self):
        names = _scoped(_agent(enabled=["terminal", "memory"], memory={"fact_store"}))
        assert "fact_store" in names

    def test_composite_toolset_that_includes_memory_includes_provider_tools(self):
        names = _scoped(_agent(enabled=["hermes-acp"], memory={"fact_store"}))
        assert "fact_store" in names

    def test_memory_tool_already_on_the_surface_includes_provider_tools(self):
        # Same bypass as injection: the built-in memory tool present means the
        # provider tools were exposed even if the toolset list omits "memory".
        agent = _agent(enabled=["terminal"], tools=[_tool("memory")], memory={"fact_store"})
        assert "fact_store" in _scoped(agent)

    @pytest.mark.parametrize(
        "enabled,disabled",
        [
            ([], None),
            (["terminal", "web"], None),
            (None, ["memory"]),
            (["memory"], ["memory"]),
            (["hermes-acp"], ["memory"]),
        ],
    )
    def test_gated_off_memory_toolset_excludes_provider_tools(self, enabled, disabled):
        names = _scoped(_agent(enabled=enabled, disabled=disabled, memory={"fact_store"}))
        assert "fact_store" not in names
        assert "session_search" in names

    def test_no_tool_surface_does_not_invent_provider_scope(self):
        agent = _agent(enabled=None, memory={"fact_store"})
        agent.tools = None
        assert "fact_store" not in _scoped(agent)

    def test_context_engine_tools_are_in_scope_when_granted(self):
        names = _scoped(_agent(enabled=[], memory={"fact_store"}, context={"lcm_grep"}))
        assert "lcm_grep" in names
        assert "fact_store" not in names

    def test_no_provider_is_a_no_op(self):
        agent = _agent(enabled=None, memory=None, context=())
        names = _scoped(agent)
        assert "fact_store" not in names and "lcm_grep" not in names

    def test_cache_key_separates_provider_scopes(self):
        agent = _agent(enabled=None, memory={"fact_store"})
        assert "fact_store" in _scoped(agent)
        agent._memory_manager = _Names({"other_fact"})
        names = _scoped(agent)
        assert "fact_store" not in names
        assert "other_fact" in names

    def test_registry_owned_name_is_not_readded_by_the_provider_union(self):
        from tools.registry import registry

        registry.register(
            name="fact_store",
            toolset="mcp-removed-provider",
            handler=lambda args, **kwargs: "{}",
            schema={"name": "fact_store", "description": "moved", "parameters": {"type": "object"}},
        )
        try:
            names = _scoped(_agent(enabled=None, memory={"fact_store"}))
        finally:
            registry.deregister("fact_store")
        assert "fact_store" not in names


def _bridge(name, arguments=None):
    return {"calls": [{"name": name, "arguments": arguments or {}}]}


class TestResolveVouchesProviderNames:
    def test_unregistered_name_in_scope_resolves(self):
        name, args, err = resolve_underlying_call(
            _bridge("fact_store", {"text": "remember"}),
            allowed_names=frozenset({"fact_store"}),
        )
        assert (name, args, err) == ("fact_store", {"text": "remember"}, None)

    def test_omitted_scope_still_rejects_unregistered_names(self):
        name, _args, err = resolve_underlying_call(_bridge("fact_store"))
        assert name is None and err and "not a known tool" in err

    def test_core_and_bridge_names_stay_rejected_even_if_listed(self):
        for blocked in ("terminal", "tool_search"):
            name, _args, err = resolve_underlying_call(
                _bridge(blocked), allowed_names=frozenset({blocked, "fact_store"}),
            )
            assert name is None and err, blocked

    def test_registered_non_deferrable_name_is_not_vouched(self):
        from tools.registry import registry

        registry.register(
            name="surface_only_widget",
            toolset="desktop_ui",
            handler=lambda args, **kwargs: "{}",
            schema={"name": "surface_only_widget", "description": "gui", "parameters": {"type": "object"}},
        )
        try:
            name, _args, err = resolve_underlying_call(
                _bridge("surface_only_widget"),
                allowed_names=frozenset({"surface_only_widget"}),
            )
        finally:
            registry.deregister("surface_only_widget")
        assert name is None and err


@pytest.mark.usefixtures("registry_scope")
class TestUnwrapAdmitsGrantedProviderTools:
    def test_unwraps_granted_memory_tool(self):
        agent = _agent(enabled=None, memory={"fact_store"})
        name, args, block = _unwrap_tool_search_call(
            agent, "tool_call", _bridge("fact_store", {"text": "x"}),
        )
        assert (name, args, block) == ("fact_store", {"text": "x"}, None)

    def test_does_not_unwrap_gated_off_memory_tool(self):
        agent = _agent(enabled=[], memory={"fact_store"})
        name, _args, block = _unwrap_tool_search_call(agent, "tool_call", _bridge("fact_store"))
        assert name == "tool_call" and block is None

    def test_unwraps_granted_context_engine_tool(self):
        agent = _agent(enabled=[], context={"lcm_grep"})
        name, args, block = _unwrap_tool_search_call(
            agent, "tool_call", _bridge("lcm_grep", {"pattern": "x"}),
        )
        assert (name, args, block) == ("lcm_grep", {"pattern": "x"}, None)

    def test_listed_core_tool_is_not_unwrapped(self):
        agent = _agent(enabled=None, memory={"terminal"})
        name, _args, block = _unwrap_tool_search_call(agent, "tool_call", _bridge("terminal"))
        assert name == "tool_call" and block is None
