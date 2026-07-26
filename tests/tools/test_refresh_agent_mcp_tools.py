"""Tests for the shared MCP agent-tool refresh helper and discovery-wait bound.

``refresh_agent_mcp_tools`` is the single rebuild path used by the TUI
``reload.mcp`` RPC, the gateway reload, and the late-binding refresh thread —
so a slow MCP server that connects after the agent's one-time tool snapshot is
picked up everywhere identically.  These assert the *contracts* those callers
rely on (name-based diff, in-place mutation, agent-scoped filtering) rather than
freezing any particular tool list.
"""

import threading
import types
import uuid

from tools import mcp_tool
from tools.registry import registry


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _agent(tool_names, *, enabled=None, disabled=None):
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a._tool_registry_routes = {
        name: entry
        for name in tool_names
        if (entry := registry.get_entry(name)) is not None
    }
    a.enabled_toolsets = enabled
    a.disabled_toolsets = disabled
    return a


def test_capture_execution_route_rejects_non_callable_dynamic_handlers():
    """Published dynamic ownership is stale without an invokable handler."""
    agent = _agent(["fact_store", "lcm_grep"])
    agent._tool_snapshot_epoch = 7
    agent._memory_provider_tool_names = {"fact_store"}
    agent._memory_manager = types.SimpleNamespace(handle_tool_call=None)
    agent._context_engine_tool_names = {"lcm_grep"}
    agent.context_compressor = types.SimpleNamespace(handle_tool_call="not callable")

    assert (
        mcp_tool.capture_agent_tool_execution_route(agent, 7, "fact_store")
        is None
    )
    assert (
        mcp_tool.capture_agent_tool_execution_route(agent, 7, "lcm_grep")
        is None
    )


def test_refresh_adds_late_landing_tools(monkeypatch):
    """A server that registers after build → its tools land in the snapshot."""
    agent = _agent(["read_file", "terminal"])

    new_defs = [_tool(n) for n in ("read_file", "terminal", "mcp_granola_get_account_info")]
    monkeypatch.setattr(mcp_tool, "get_tool_definitions", lambda **kw: new_defs, raising=False)
    # get_tool_definitions is imported inside the helper from model_tools, so patch there too.
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: new_defs)

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == {"mcp_granola_get_account_info"}
    assert "mcp_granola_get_account_info" in agent.valid_tool_names
    assert len(agent.tools) == 3


def test_refresh_no_change_returns_empty_and_leaves_agent_untouched(monkeypatch):
    """No new tools → empty set, and the snapshot object is not swapped."""
    agent = _agent(["read_file", "terminal"])
    original_tools = agent.tools

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("terminal")],
    )

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == set()
    assert agent.tools is original_tools  # not replaced → no churn / no cache thrash


def test_refresh_publishes_same_schema_registry_replacement(monkeypatch):
    """A handler identity change advances the snapshot even if schemas match."""
    import model_tools

    name = f"mcp__route_refresh__{uuid.uuid4().hex}"
    schema = _tool(name)["function"]
    registry.register(
        name=name,
        toolset="mcp-origin",
        schema=schema,
        handler=lambda _args, **_kwargs: "old",
    )
    try:
        old_entry = registry.get_entry(name)
        agent = _agent([name])
        agent._tool_snapshot_generation = registry._generation
        agent._tool_snapshot_epoch = 3

        registry.register(
            name=name,
            toolset="mcp-origin",
            schema=schema,
            handler=lambda _args, **_kwargs: "new",
        )
        new_entry = registry.get_entry(name)
        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda **_kw: [_tool(name)],
        )

        added = mcp_tool.refresh_agent_mcp_tools(agent)

        assert added == set()
        assert agent._tool_snapshot_epoch == 4
        assert agent._tool_registry_routes == {name: new_entry}
        assert agent._tool_registry_routes[name] is not old_entry
    finally:
        registry.deregister(name)


def test_refresh_hidden_registry_route_keeps_dynamic_owner_precedence(monkeypatch):
    """Deferred registry names remain reserved during post-build injection."""
    import model_tools

    name = f"mcp__hidden_owner__{uuid.uuid4().hex}"
    registry.register(
        name=name,
        toolset="mcp-hidden-owner",
        schema=_tool(name)["function"],
        handler=lambda _args, **_kwargs: "registry",
    )
    try:
        entry = registry.get_entry(name)
        bridge_names = ["tool_search", "tool_describe", "tool_call"]
        agent = _agent(bridge_names)
        agent._tool_registry_routes = {name: entry}
        agent._memory_manager = types.SimpleNamespace(
            get_all_tool_schemas=lambda: [_tool(name)["function"]],
        )
        agent.context_compressor = types.SimpleNamespace(
            get_tool_schemas=lambda: [_tool(name)["function"]],
        )
        agent._memory_provider_tool_names = set()
        agent._context_engine_tool_names = set()

        def _definitions(*, skip_tool_search_assembly=False, **_kw):
            if skip_tool_search_assembly:
                return [_tool(name)]
            return [_tool(bridge_name) for bridge_name in bridge_names]

        monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)

        mcp_tool.refresh_agent_mcp_tools(agent)

        assert agent.valid_tool_names == set(bridge_names)
        assert agent._memory_provider_tool_names == set()
        assert agent._context_engine_tool_names == set()
        assert agent._tool_registry_routes == {name: entry}
    finally:
        registry.deregister(name)


def test_refresh_detects_equal_size_swap(monkeypatch):
    """Name-based diff catches an add+remove of equal count (count-compare can't)."""
    agent = _agent(["a", "old_mcp_tool"])  # 2 tools

    import model_tools
    # Same COUNT (2) but a different membership: old_mcp_tool removed, new added.
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("a"), _tool("new_mcp_tool")],
    )

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == {"new_mcp_tool"}
    assert agent.valid_tool_names == {"a", "new_mcp_tool"}
    assert "old_mcp_tool" not in agent.valid_tool_names


def test_refresh_passes_agent_toolset_filters(monkeypatch):
    """The rebuild re-derives with the agent's OWN enabled/disabled toolsets."""
    agent = _agent(["a"], enabled=["coding", "granola"], disabled=["messaging"])
    seen = {}

    import model_tools

    def _capture(**kw):
        seen.update(kw)
        return [_tool("a"), _tool("b")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _capture)

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert seen["enabled_toolsets"] == ["coding", "granola"]
    assert seen["disabled_toolsets"] == ["messaging"]


def test_failed_tightening_remains_pending_and_automatic_refresh_recovers(
    monkeypatch,
):
    """Assembly failure must not consume a policy epoch or lose tightening."""
    agent = _agent(
        ["read_file", "fact_store"],
    )
    agent._memory_provider_tool_names = {"fact_store"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "fact_store", "description": "", "parameters": {}}
        ]
    )
    original_tools = agent.tools
    original_names = agent.valid_tool_names
    definitions_available = False

    import model_tools

    def _definitions(*, enabled_toolsets, **_kw):
        if not definitions_available:
            raise RuntimeError("definition failure")
        return [_tool(enabled_toolsets[0])]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)

    import pytest

    with pytest.raises(RuntimeError, match="definition failure"):
        mcp_tool.refresh_agent_mcp_tools(
            agent,
            enabled_override=["terminal"],
        )

    assert agent.enabled_toolsets is None
    assert agent.tools is original_tools
    assert agent.valid_tool_names is original_names
    assert agent._memory_provider_tool_names == {"fact_store"}
    assert agent._tool_policy_epoch == 1
    assert getattr(agent, "_tool_published_policy_epoch", 0) == 0

    definitions_available = True
    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.enabled_toolsets == ["terminal"]
    assert agent.valid_tool_names == {"terminal"}
    assert agent._memory_provider_tool_names == set()
    assert agent._tool_published_policy_epoch == agent._tool_policy_epoch == 1


def test_pending_policy_recovery_cannot_overwrite_newer_policy(monkeypatch):
    """An older recovery finishing last cannot replace the latest policy."""
    agent = _agent(["old-tool"], enabled=["old-policy"])
    fail_first = True
    recovery_entered = threading.Event()
    release_recovery = threading.Event()

    import model_tools

    def _definitions(*, enabled_toolsets, **_kw):
        nonlocal fail_first
        policy = enabled_toolsets[0]
        if policy == "pending-policy":
            if fail_first:
                fail_first = False
                raise RuntimeError("definition failure")
            recovery_entered.set()
            assert release_recovery.wait(5)
        return [_tool(policy)]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)

    import pytest

    with pytest.raises(RuntimeError, match="definition failure"):
        mcp_tool.refresh_agent_mcp_tools(
            agent,
            enabled_override=["pending-policy"],
        )

    recovery_errors = []

    def _recover():
        try:
            mcp_tool.refresh_agent_mcp_tools(agent)
        except Exception as exc:  # pragma: no cover - failure diagnostic
            recovery_errors.append(exc)

    recovery = threading.Thread(
        target=_recover,
    )
    recovery.start()
    assert recovery_entered.wait(5)

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        enabled_override=["latest-policy"],
    )
    release_recovery.set()
    recovery.join(5)

    assert not recovery.is_alive()
    assert recovery_errors == []
    assert agent.enabled_toolsets == ["latest-policy"]
    assert agent.valid_tool_names == {"latest-policy"}
    assert agent._tool_published_policy_epoch == agent._tool_policy_epoch == 2


def test_refresh_preserves_memory_provider_and_context_engine_tools(monkeypatch):
    """B1 regression: a rebuild must NOT drop post-build-injected tools.

    get_tool_definitions() returns only the registry-derived tools. agent_init
    appends memory-provider tools (mem0/honcho/…) and context-engine tools
    (lcm_*) directly onto agent.tools AFTER that. A naive
    `agent.tools = get_tool_definitions()` would silently delete them on every
    refresh. The helper must re-inject them.
    """
    # Agent already carries: a built-in, a memory-provider tool, a context tool.
    agent = _agent(["read_file", "memory_search", "lcm_grep"])

    # Provider exposes its schemas; context compressor exposes lcm_*.
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )

    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [
            {"name": "lcm_grep", "description": "", "parameters": {}}
        ]
    )
    agent._context_engine_tool_names = {"lcm_grep"}

    import model_tools
    # The registry now ALSO has a newly-connected MCP tool, but does NOT contain
    # the memory/context tools (they're never in get_tool_definitions output).
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_server_tool")],
    )

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    # The new MCP tool landed AND the injected families survived.
    assert "mcp_new_server_tool" in agent.valid_tool_names
    assert "memory_search" in agent.valid_tool_names   # not clobbered
    assert "lcm_grep" in agent.valid_tool_names         # not clobbered
    assert added == {"mcp_new_server_tool"}


def test_refresh_does_not_reinject_disabled_memory_provider_tools(monkeypatch):
    """An MCP rebuild must preserve the session's final memory denial."""
    agent = _agent(
        ["read_file", "memory_search"],
        enabled=["all"],
        disabled=["memory"],
    )
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )

    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_server_tool")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert "mcp_new_server_tool" in agent.valid_tool_names
    assert "memory_search" not in agent.valid_tool_names
    assert all(
        tool["function"]["name"] != "memory_search" for tool in agent.tools
    )


def test_refresh_subtracts_only_exact_provider_tool_name(monkeypatch):
    """A custom denial removes its provider schema without hiding siblings."""
    agent = _agent(["read_file", "fact_store", "fact_search"])
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "fact_store", "description": "", "parameters": {}},
            {"name": "fact_search", "description": "", "parameters": {}},
        ]
    )

    import model_tools
    import toolsets

    monkeypatch.setitem(
        toolsets.TOOLSETS,
        "deny-fact-store",
        {"description": "test", "tools": ["fact_store"], "includes": []},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file"), _tool("mcp_new_tool")],
    )

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        disabled_override=["deny-fact-store"],
    )

    assert "fact_store" not in agent.valid_tool_names
    assert "fact_search" in agent.valid_tool_names
    assert "mcp_new_tool" in agent.valid_tool_names


def test_refresh_provider_change_invalidates_prompt_cache(monkeypatch):
    agent = _agent(["read_file", "fact_store"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "fact_store", "description": "", "parameters": {}}
        ]
    )
    agent._cached_system_prompt = "Use fact_store."

    import model_tools
    import toolsets

    monkeypatch.setitem(
        toolsets.TOOLSETS,
        "deny-fact-store",
        {"description": "test", "tools": ["fact_store"], "includes": []},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        disabled_override=["deny-fact-store"],
    )

    assert agent._cached_system_prompt is None


def test_refresh_provider_collision_invalidates_prompt_cache(monkeypatch):
    """A registry collision transfers ownership away from the provider."""
    agent = _agent(["read_file", "fact_store"])
    provider_calls = 0
    provider_lock_was_free = []

    def _provider_schemas():
        nonlocal provider_calls
        provider_calls += 1
        lock_was_free = mcp_tool._agent_tools_lock.acquire(blocking=False)
        provider_lock_was_free.append(lock_was_free)
        if lock_was_free:
            mcp_tool._agent_tools_lock.release()
        return [{"name": "fact_store", "description": "", "parameters": {}}]

    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=_provider_schemas,
    )
    agent._memory_provider_tool_names = {"fact_store"}
    agent._cached_system_prompt = "Use fact_store as provider memory."
    original_tools = agent.tools

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file"), _tool("fact_store")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools is not original_tools
    assert agent.tools == original_tools
    assert agent._cached_system_prompt is None
    assert agent._memory_provider_tool_names == set()
    assert provider_calls == 1
    assert provider_lock_was_free == [True]


def test_refresh_equal_names_publishes_registry_schema_on_provider_transfer(
    monkeypatch,
):
    """Provider -> registry transfer must atomically replace the contract."""
    provider_tool = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "Provider-owned lookup by memory query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    registry_tool = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "Registry-owned lookup by numeric record id.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "integer"}},
                "required": ["record_id"],
            },
        },
    }
    agent = _agent([])
    agent.tools = [provider_tool]
    agent.valid_tool_names = {"shared_tool"}
    agent._memory_provider_tool_names = {"shared_tool"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [provider_tool["function"]],
        handle_tool_call=lambda _name, _args: "provider-dispatch",
    )
    agent.session_id = "session"

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda **_kw: [registry_tool]
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [registry_tool]
    assert agent._memory_provider_tool_names == set()


def test_refresh_equal_names_publishes_provider_schema_on_registry_transfer(
    monkeypatch,
):
    """Registry -> provider transfer must atomically replace the contract."""
    registry_tool = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "Registry-owned lookup by numeric record id.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "integer"}},
                "required": ["record_id"],
            },
        },
    }
    provider_schema = {
        "name": "shared_tool",
        "description": "Provider-owned lookup by memory query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    agent = _agent([])
    agent.tools = [registry_tool]
    agent.valid_tool_names = {"shared_tool"}
    agent._memory_provider_tool_names = set()
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [provider_schema],
        handle_tool_call=lambda name, args: "provider-dispatch",
    )
    agent.session_id = "session"

    import model_tools

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kw: [])

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [{"type": "function", "function": provider_schema}]
    assert agent._memory_provider_tool_names == {"shared_tool"}


def test_refresh_records_genuinely_injected_provider_ownership(monkeypatch):
    agent = _agent(["read_file"])
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "fact_store", "description": "", "parameters": {}}
        ]
    )
    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent._memory_provider_tool_names == {"fact_store"}
    assert "fact_store" in agent.valid_tool_names


def test_schema_callback_failure_accepts_safe_equal_provider_fallback(monkeypatch):
    agent = _agent(["read_file", "fact_store"], enabled=["memory"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._context_engine_tool_names = {"lcm_grep"}
    agent._cached_system_prompt = "provider prompt"
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: (_ for _ in ()).throw(
            RuntimeError("schema callback failed")
        )
    )
    original_tools = list(agent.tools)
    original_names = set(agent.valid_tool_names)
    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )
    added = mcp_tool.refresh_agent_mcp_tools(
        agent,
        enabled_override=["coding"],
    )

    assert added == set()
    # The tool schemas remain unchanged, but clearing stale context-engine
    # ownership is itself a routing-snapshot change and must be published.
    assert agent.tools == original_tools
    assert agent.valid_tool_names == original_names
    assert agent.enabled_toolsets == ["coding"]
    assert agent._memory_provider_tool_names == {"fact_store"}
    assert agent._context_engine_tool_names == set()
    assert agent._cached_system_prompt == "provider prompt"
    assert agent._tool_published_policy_epoch == agent._tool_policy_epoch


def test_schema_failure_cannot_veto_full_provider_family_revocation(monkeypatch):
    """A denied provider family is removable without enumerating schemas."""
    agent = _agent(["read_file", "fact_store"], enabled=["memory"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._cached_system_prompt = "Use fact_store."
    provider_calls = []
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: (_ for _ in ()).throw(
            RuntimeError("schema callback failed")
        ),
        handle_tool_call=lambda name, args: provider_calls.append((name, args)),
    )

    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent, disabled_override=["memory"])

    assert agent.disabled_toolsets == ["memory"]
    assert agent.valid_tool_names == {"read_file"}
    assert agent._memory_provider_tool_names == set()
    assert agent._cached_system_prompt is None
    assert provider_calls == []


def test_schema_failure_still_applies_exact_provider_name_revocation(
    monkeypatch,
):
    """Tightening may retain allowed published schemas without adding any."""
    agent = _agent(
        ["read_file", "fact_store", "fact_search"],
        enabled=["memory"],
    )
    agent._memory_provider_tool_names = {"fact_store", "fact_search"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: (_ for _ in ()).throw(
            RuntimeError("schema callback failed")
        )
    )

    import model_tools
    import toolsets
    monkeypatch.setitem(
        toolsets.TOOLSETS,
        "deny-fact-store",
        {"description": "test", "tools": ["fact_store"], "includes": []},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        disabled_override=["deny-fact-store"],
    )

    assert agent.valid_tool_names == {"read_file", "fact_search"}
    assert agent._memory_provider_tool_names == {"fact_search"}


def test_schema_failure_accepts_equal_provider_set_for_unrelated_tightening(
    monkeypatch,
):
    """A no-op provider fallback must not veto an unrelated policy change."""
    agent = _agent(["read_file", "fact_store"], enabled=["memory"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: (_ for _ in ()).throw(
            RuntimeError("schema callback failed")
        )
    )

    import model_tools
    import toolsets

    monkeypatch.setitem(
        toolsets.TOOLSETS,
        "deny-unrelated",
        {"description": "test", "tools": ["other_tool"], "includes": []},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        disabled_override=["deny-unrelated"],
    )

    assert agent.disabled_toolsets == ["deny-unrelated"]
    assert agent.valid_tool_names == {"read_file", "fact_store"}
    assert agent._memory_provider_tool_names == {"fact_store"}


def test_schema_failure_rejects_invented_provider_names(monkeypatch):
    """Fallback may retain published provider names but never invent one."""
    agent = _agent(["read_file", "fact_store"], enabled=["memory"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: (_ for _ in ()).throw(
            RuntimeError("schema callback failed")
        )
    )
    original_tools = agent.tools
    original_names = agent.valid_tool_names

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )
    monkeypatch.setattr(
        mcp_tool,
        "_effective_memory_provider_schemas_for_refresh",
        lambda *_args, **_kw: [
            {"name": "invented_tool", "description": "", "parameters": {}}
        ],
    )

    import pytest

    with pytest.raises(RuntimeError, match="schema callback failed"):
        mcp_tool.refresh_agent_mcp_tools(
            agent,
            disabled_override=["deny-unrelated"],
        )

    assert agent.tools is original_tools
    assert agent.valid_tool_names is original_names
    assert agent.enabled_toolsets == ["memory"]
    assert agent._memory_provider_tool_names == {"fact_store"}


def test_schema_fallback_snapshots_provider_names_and_schemas_atomically(
    monkeypatch,
):
    """Fallback must never combine ownership from one publish with another."""
    old_schema = {
        "type": "function",
        "function": {
            "name": "old_provider_tool",
            "description": "old provider",
            "parameters": {},
        },
    }
    new_schema = {
        "type": "function",
        "function": {
            "name": "new_provider_tool",
            "description": "new provider",
            "parameters": {},
        },
    }
    ownership_read = threading.Event()
    concurrent_publish_done = threading.Event()

    class RacingAgent:
        def __init__(self):
            self.tools = [_tool("read_file"), old_schema]
            self.valid_tool_names = {"read_file", "old_provider_tool"}
            self.enabled_toolsets = ["memory"]
            self.disabled_toolsets = None
            self.session_id = "session"
            self._tool_snapshot_epoch = 0
            self._provider_names = {"old_provider_tool"}
            self._memory_manager = types.SimpleNamespace(
                get_all_tool_schemas=lambda: (_ for _ in ()).throw(
                    RuntimeError("schema callback failed")
                ),
                handle_tool_call=lambda name, _args: f"provider:{name}",
            )

        @property
        def _memory_provider_tool_names(self):
            names = set(self._provider_names)
            ownership_read.set()
            if not mcp_tool._agent_tools_lock.locked():
                assert concurrent_publish_done.wait(5)
            return names

        @_memory_provider_tool_names.setter
        def _memory_provider_tool_names(self, names):
            self._provider_names = set(names)

    agent = RacingAgent()

    def _concurrent_publish():
        assert ownership_read.wait(5)
        with mcp_tool._agent_tools_lock:
            agent.tools = [_tool("read_file"), new_schema]
            agent.valid_tool_names = {"read_file", "new_provider_tool"}
            agent._provider_names = {"new_provider_tool"}
            agent._tool_snapshot_epoch += 1
        concurrent_publish_done.set()

    publisher_errors = []

    def _publish():
        try:
            _concurrent_publish()
        except Exception as exc:  # pragma: no cover - failure diagnostic
            publisher_errors.append(exc)

    publisher = threading.Thread(target=_publish)
    publisher.start()

    import model_tools
    import toolsets

    monkeypatch.setitem(
        toolsets.TOOLSETS,
        "deny-unrelated",
        {"description": "test", "tools": ["other_tool"], "includes": []},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        disabled_override=["deny-unrelated"],
    )
    publisher.join(5)

    assert not publisher.is_alive()
    assert publisher_errors == []
    provider_names = agent._memory_provider_tool_names
    assert provider_names == {"new_provider_tool"}
    assert agent.valid_tool_names == {"read_file", "new_provider_tool"}
    provider_schema = next(
        tool["function"]
        for tool in agent.tools
        if tool["function"]["name"] in provider_names
    )
    assert provider_schema["description"] == "new provider"


def test_delayed_provider_fallback_cannot_overwrite_newer_same_generation(
    monkeypatch,
):
    """A later same-generation publication must defeat an older fallback."""
    old_schema = {
        "name": "old_provider",
        "description": "old provider",
        "parameters": {},
    }
    new_schema = {
        "name": "new_provider",
        "description": "new provider",
        "parameters": {},
    }
    agent = _agent(["read_file", "old_provider"], enabled=["memory"])
    agent.tools[1] = {"type": "function", "function": old_schema}
    agent._memory_provider_tool_names = {"old_provider"}
    first_schema_call_entered = threading.Event()
    release_first_schema_call = threading.Event()
    schema_call_lock = threading.Lock()
    schema_calls = 0

    def _provider_schemas():
        nonlocal schema_calls
        with schema_call_lock:
            schema_calls += 1
            call = schema_calls
        if call == 1:
            first_schema_call_entered.set()
            assert release_first_schema_call.wait(5)
            raise RuntimeError("transient schema failure")
        return [new_schema]

    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=_provider_schemas,
    )

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )
    older_errors = []

    def _older_refresh():
        try:
            mcp_tool.refresh_agent_mcp_tools(agent)
        except Exception as exc:  # pragma: no cover - failure diagnostic
            older_errors.append(exc)

    older = threading.Thread(target=_older_refresh)
    older.start()
    assert first_schema_call_entered.wait(5)

    mcp_tool.refresh_agent_mcp_tools(agent)
    assert agent._memory_provider_tool_names == {"new_provider"}

    release_first_schema_call.set()
    older.join(5)

    assert not older.is_alive()
    assert older_errors == []
    assert schema_calls == 2
    assert agent._memory_provider_tool_names == {"new_provider"}
    assert agent.valid_tool_names == {"read_file", "new_provider"}
    assert {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in agent.tools
    } == {
        "read_file": "",
        "new_provider": "new provider",
    }


def test_stale_generation_explicit_policy_retries_before_publishing_epoch(
    monkeypatch,
):
    """A winning registry generation cannot consume a tightening policy."""
    from tools import registry as registry_module

    agent = _agent(["read_file", "fact_store"], enabled=["memory"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "fact_store", "description": "", "parameters": {}}
        ]
    )
    explicit_entered = threading.Event()
    release_explicit = threading.Event()
    explicit_calls = 0

    import model_tools

    def _definitions(*, disabled_toolsets, **_kw):
        nonlocal explicit_calls
        if disabled_toolsets == ["memory"]:
            explicit_calls += 1
            if explicit_calls == 1:
                explicit_entered.set()
                assert release_explicit.wait(5)
        return [_tool("read_file")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)
    refresh_error = []

    def _refresh():
        try:
            mcp_tool.refresh_agent_mcp_tools(
                agent,
                disabled_override=["memory"],
            )
        except Exception as exc:  # pragma: no cover - failure diagnostic
            refresh_error.append(exc)

    refresh = threading.Thread(target=_refresh)
    refresh.start()
    assert explicit_entered.wait(5)

    sentinel = "test_policy_generation_sentinel"
    registry_module.registry.register(
        name=sentinel,
        toolset="test",
        schema={"name": sentinel, "description": "", "parameters": {}},
        handler=lambda _args, **_kw: "{}",
    )
    try:
        # Model a newer registry snapshot publishing while the explicit policy
        # rebuild is still staging its older generation.
        agent._tool_snapshot_generation = registry_module.registry._generation
        release_explicit.set()
        refresh.join(5)

        assert not refresh.is_alive()
        assert refresh_error == []
        assert explicit_calls == 2
        assert agent.disabled_toolsets == ["memory"]
        assert agent.valid_tool_names == {"read_file"}
        assert agent._memory_provider_tool_names == set()
        assert agent._tool_published_policy_epoch == agent._tool_policy_epoch
    finally:
        registry_module.registry.deregister(sentinel)


def test_same_generation_automatic_refresh_cannot_beat_newer_policy(monkeypatch):
    agent = _agent(["old_tool"], enabled=["old-policy"])
    automatic_entered = threading.Event()
    release_automatic = threading.Event()

    import model_tools

    def _definitions(*, enabled_toolsets, **_kw):
        if enabled_toolsets == ["old-policy"]:
            automatic_entered.set()
            assert release_automatic.wait(5)
            return [_tool("old_tool")]
        return [_tool("new_tool")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)
    automatic_errors = []

    def _automatic_refresh():
        try:
            mcp_tool.refresh_agent_mcp_tools(agent)
        except Exception as exc:  # pragma: no cover - failure diagnostic
            automatic_errors.append(exc)

    automatic = threading.Thread(
        target=_automatic_refresh,
    )
    automatic.start()
    assert automatic_entered.wait(5)

    mcp_tool.refresh_agent_mcp_tools(
        agent,
        enabled_override=["new-policy"],
    )
    release_automatic.set()
    automatic.join(5)

    assert not automatic.is_alive()
    assert automatic_errors == []
    assert agent.enabled_toolsets == ["new-policy"]
    assert agent.valid_tool_names == {"new_tool"}


def test_continuous_generation_loss_is_bounded_and_preserves_pending_policy(
    monkeypatch,
):
    """A never-stable registry must fail boundedly without consuming policy."""
    from tools import registry as registry_module

    agent = _agent(["old_tool"], enabled=["old-policy"])
    original_tools = agent.tools
    original_names = agent.valid_tool_names
    definition_calls = 0

    import model_tools

    def _definitions(**_kw):
        nonlocal definition_calls
        definition_calls += 1
        agent._tool_snapshot_generation = registry_module.registry._generation + 1
        return [_tool("new_tool")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)

    import pytest

    with pytest.raises(RuntimeError, match="generation.*stabilize"):
        mcp_tool.refresh_agent_mcp_tools(
            agent,
            enabled_override=["new-policy"],
        )

    assert definition_calls <= 4
    assert agent.tools is original_tools
    assert agent.valid_tool_names is original_names
    assert agent.enabled_toolsets == ["old-policy"]
    assert getattr(agent, "_tool_published_policy_epoch", 0) == 0
    assert agent._tool_pending_policy == (1, ["new-policy"], None)


def test_failed_refresh_preserves_prompt_cache(monkeypatch):
    agent = _agent(["read_file", "fact_store"])
    agent._cached_system_prompt = "Use fact_store."

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("definition failure")),
    )

    import pytest

    with pytest.raises(RuntimeError, match="definition failure"):
        mcp_tool.refresh_agent_mcp_tools(
            agent,
            disabled_override=["memory"],
        )

    assert agent._cached_system_prompt == "Use fact_store."


def test_refresh_resolution_failure_does_not_reinject_memory_provider_tools(
    monkeypatch,
):
    """A disabled-toolset resolver failure must keep provider tools denied."""
    agent = _agent(["read_file", "memory_search"], disabled=["coding"])
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )

    import model_tools
    import toolsets

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_server_tool")],
    )
    monkeypatch.setattr(
        toolsets,
        "resolve_toolset",
        lambda _name: (_ for _ in ()).throw(RuntimeError("resolution failed")),
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert "mcp_new_server_tool" in agent.valid_tool_names
    assert "memory_search" not in agent.valid_tool_names
    assert all(
        tool["function"]["name"] != "memory_search" for tool in agent.tools
    )


def test_refresh_respects_context_engine_toolset_gate(monkeypatch):
    """#5544: context-engine tools must NOT be re-injected on a restricted
    toolset. A platform with enabled_toolsets that excludes context_engine
    must not get lcm_* leaked back in by a refresh."""
    agent = _agent(["read_file"], enabled=["coding"])  # context_engine NOT enabled
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [{"name": "lcm_grep", "description": "", "parameters": {}}]
    )
    agent._context_engine_tool_names = set()

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_tool")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert "mcp_new_tool" in agent.valid_tool_names  # MCP tool still lands
    assert "lcm_grep" not in agent.valid_tool_names   # gated out (#5544)


def test_refresh_context_engine_family_denial_does_not_enumerate_schemas(
    monkeypatch,
):
    """Refresh must short-circuit disabled provider code before enumeration."""
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    policies = [
        ([], None),
        (["coding"], None),
        (None, ["all"]),
        (None, ["*"]),
        (None, ["context_engine"]),
    ]
    for enabled, disabled in policies:
        calls = []

        def _disabled_schema_callback():
            calls.append("called")
            raise RuntimeError("disabled engine touched")

        agent = _agent(["read_file"], enabled=enabled, disabled=disabled)
        agent.context_compressor = types.SimpleNamespace(
            get_tool_schemas=_disabled_schema_callback,
        )
        agent._context_engine_tool_names = set()

        mcp_tool.refresh_agent_mcp_tools(agent)

        assert calls == []
        assert agent.valid_tool_names == {"read_file"}
        assert agent._context_engine_tool_names == set()


def test_refresh_context_engine_respects_final_disabled_subtraction(monkeypatch):
    """Dynamic engine schemas must honor global and family-wide denials."""
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    for disabled in (["all"], ["*"], ["context_engine"]):
        agent = _agent(["read_file"], disabled=disabled)
        agent.context_compressor = types.SimpleNamespace(
            get_tool_schemas=lambda: [
                {"name": "lcm_grep", "description": "", "parameters": {}}
            ]
        )
        agent._context_engine_tool_names = set()

        mcp_tool.refresh_agent_mcp_tools(agent)

        assert agent.valid_tool_names == {"read_file"}
        assert agent._context_engine_tool_names == set()


def test_refresh_context_engine_respects_exact_dynamic_name_subtraction(
    monkeypatch,
):
    """A custom disabled toolset may subtract one dynamic engine schema."""
    agent = _agent(["read_file"], disabled=["deny-lcm-grep"])
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [
            {"name": "lcm_grep", "description": "", "parameters": {}},
            {"name": "lcm_describe", "description": "", "parameters": {}},
        ]
    )
    agent._context_engine_tool_names = set()

    import model_tools
    import toolsets

    monkeypatch.setitem(
        toolsets.TOOLSETS,
        "deny-lcm-grep",
        {"description": "test", "tools": ["lcm_grep"], "includes": []},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.valid_tool_names == {"read_file", "lcm_describe"}
    assert agent._context_engine_tool_names == {"lcm_describe"}


def test_refresh_equal_name_context_engine_to_registry_publishes_schema(
    monkeypatch,
):
    """Engine -> registry transfer must replace schema and routing together."""
    context_schema = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "context contract",
            "parameters": {
                "type": "object",
                "properties": {"context_arg": {"type": "string"}},
                "required": ["context_arg"],
            },
        },
    }
    registry_schema = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "registry contract",
            "parameters": {
                "type": "object",
                "properties": {"registry_arg": {"type": "integer"}},
                "required": ["registry_arg"],
            },
        },
    }
    agent = _agent([])
    agent.tools = [context_schema]
    agent.valid_tool_names = {"shared_tool"}
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [context_schema["function"]],
        handle_tool_call=lambda _name, _args: "context-dispatch",
    )
    agent._context_engine_tool_names = {"shared_tool"}
    agent._memory_manager = None
    agent.session_id = "session"

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [registry_schema],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [registry_schema]
    assert agent._context_engine_tool_names == set()
    assert agent.valid_tool_names == {"shared_tool"}

    import run_agent
    from agent.agent_runtime_helpers import invoke_tool
    from hermes_cli import middleware

    monkeypatch.setattr(
        middleware,
        "run_tool_execution_middleware",
        lambda _name, args, execute, **_kw: execute(args),
    )
    monkeypatch.setattr(
        run_agent,
        "handle_function_call",
        lambda *_args, **_kw: "registry-dispatch",
    )
    assert invoke_tool(
        agent,
        "shared_tool",
        {"registry_arg": 7},
        "task",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
    ) == "registry-dispatch"


def test_refresh_initializes_missing_context_engine_owner_set(monkeypatch):
    """Published engine schemas must always have a matching dispatch owner."""
    agent = _agent(["read_file"])
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [
            {"name": "lcm_describe", "description": "", "parameters": {}},
        ],
        handle_tool_call=lambda *_args, **_kwargs: "engine-dispatch",
    )

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.valid_tool_names == {"read_file", "lcm_describe"}
    assert agent._context_engine_tool_names == {"lcm_describe"}


def test_refresh_equal_name_registry_to_context_engine_publishes_schema(
    monkeypatch,
):
    """Registry -> engine transfer must replace schema and routing together."""
    registry_schema = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "registry contract",
            "parameters": {
                "type": "object",
                "properties": {"registry_arg": {"type": "integer"}},
                "required": ["registry_arg"],
            },
        },
    }
    context_schema = {
        "name": "shared_tool",
        "description": "context contract",
        "parameters": {
            "type": "object",
            "properties": {"context_arg": {"type": "string"}},
            "required": ["context_arg"],
        },
    }
    agent = _agent([])
    agent.tools = [registry_schema]
    agent.valid_tool_names = {"shared_tool"}
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [context_schema],
        handle_tool_call=lambda _name, _args: "context-dispatch",
    )
    agent._context_engine_tool_names = set()
    agent._memory_manager = None
    agent.session_id = "session"

    import model_tools

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kw: [])

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [{"type": "function", "function": context_schema}]
    assert agent._context_engine_tool_names == {"shared_tool"}
    assert agent.valid_tool_names == {"shared_tool"}

    from agent.agent_runtime_helpers import agent_runtime_owns_post_tool_hook

    assert agent_runtime_owns_post_tool_hook(agent, "shared_tool")


def test_refresh_equal_names_publishes_same_owner_schema_update(monkeypatch):
    """Equal names with changed contracts must publish the new schema."""
    old_schema = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "old contract",
            "parameters": {
                "type": "object",
                "properties": {"unsafe": {"type": "string"}},
            },
        },
    }
    new_schema = {
        "type": "function",
        "function": {
            "name": "shared_tool",
            "description": "new contract",
            "parameters": {
                "type": "object",
                "properties": {"safe": {"type": "string"}},
                "required": ["safe"],
            },
        },
    }
    agent = _agent([])
    agent.tools = [old_schema]
    agent.valid_tool_names = {"shared_tool"}

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [new_schema],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [new_schema]
    assert agent.valid_tool_names == {"shared_tool"}


def test_refreshed_tool_is_callable_through_valid_tool_names_guard(monkeypatch):
    """The whole point: a late tool, once refreshed, passes the name guard the
    run loop uses to accept/reject tool calls (agent.valid_tool_names)."""
    agent = _agent(["read_file"])

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_granola_list_meetings")],
    )

    # Before refresh the run loop would reject the call ("Tool does not exist").
    assert "mcp_granola_list_meetings" not in agent.valid_tool_names

    mcp_tool.refresh_agent_mcp_tools(agent)

    # After refresh the same guard accepts it AND it's in the tools= payload.
    assert "mcp_granola_list_meetings" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_granola_list_meetings" for t in agent.tools)


def test_refresh_is_thread_safe_under_concurrent_calls(monkeypatch):
    """Concurrent refreshes keep tools / valid_tool_names coherent.

    The registry alternates between two DIFFERENT tool sets every call, so the
    write path (publish) runs repeatedly rather than short-circuiting on the
    no-change early return — this actually exercises the lock. The invariant:
    a reader of ``valid_tool_names`` must always match ``agent.tools``, and the
    final published pair must be one of the two valid sets (never a mix).
    """
    agent = _agent(["a"])

    import itertools
    set_a = [_tool("a"), _tool("b")]
    set_b = [_tool("a"), _tool("c")]
    flip = itertools.cycle([set_a, set_b])
    flip_lock = threading.Lock()

    def _gtd(**kw):
        with flip_lock:
            return list(next(flip))

    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", _gtd)

    errors = []

    def _worker():
        try:
            for _ in range(50):
                mcp_tool.refresh_agent_mcp_tools(agent)
                # Coherence invariant: the name set must match the tool list
                # at every observation, never a torn cross-attribute state.
                names = {t["function"]["name"] for t in agent.tools}
                assert agent.valid_tool_names == names
                assert names in ({"a", "b"}, {"a", "c"})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert agent.valid_tool_names in ({"a", "b"}, {"a", "c"})


# ── discovery-wait bound (mcp_discovery_timeout config) ──────────────────────


def test_resolve_discovery_timeout_explicit_wins(monkeypatch):
    from hermes_cli import mcp_startup

    assert mcp_startup._resolve_discovery_timeout(2.5) == 2.5


def test_wait_returns_instantly_when_no_discovery_thread(monkeypatch):
    """The common case (no MCP / discovery done) pays ~0s regardless of bound."""
    import time
    from hermes_cli import mcp_startup

    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 999.0})

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery()
    assert time.time() - t0 < 0.2  # never blocks on the bound when nothing's pending
