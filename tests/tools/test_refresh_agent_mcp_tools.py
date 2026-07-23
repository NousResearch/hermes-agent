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

from tools import mcp_tool


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _agent(tool_names, *, enabled=None, disabled=None):
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a.enabled_toolsets = enabled
    a.disabled_toolsets = disabled
    return a


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
    agent = _agent(["read_file", "memory_search"], disabled=["memory"])
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
    from agent.agent_runtime_helpers import memory_provider_owns_tool
    assert not memory_provider_owns_tool(agent, "fact_store")
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
    import run_agent
    from agent.agent_runtime_helpers import invoke_tool
    from hermes_cli import middleware

    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda **_kw: [registry_tool]
    )
    monkeypatch.setattr(
        middleware,
        "run_tool_execution_middleware",
        lambda _name, args, execute, **_kw: execute(args),
    )
    registry_calls = []
    monkeypatch.setattr(
        run_agent,
        "handle_function_call",
        lambda name, args, *_a, **_kw: (
            registry_calls.append((name, args)) or "registry-dispatch"
        ),
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [registry_tool]
    assert agent._memory_provider_tool_names == set()
    assert (
        invoke_tool(
            agent,
            "shared_tool",
            {"record_id": 7},
            "task",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
        )
        == "registry-dispatch"
    )
    assert registry_calls == [("shared_tool", {"record_id": 7})]


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
    provider_calls = []
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [provider_schema],
        handle_tool_call=lambda name, args: (
            provider_calls.append((name, args)) or "provider-dispatch"
        ),
    )
    agent.session_id = "session"

    import model_tools
    import run_agent
    from agent.agent_runtime_helpers import invoke_tool
    from hermes_cli import middleware

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr(
        middleware,
        "run_tool_execution_middleware",
        lambda _name, args, execute, **_kw: execute(args),
    )
    registry_calls = []
    monkeypatch.setattr(
        run_agent,
        "handle_function_call",
        lambda name, args, *_a, **_kw: (
            registry_calls.append((name, args)) or "registry-dispatch"
        ),
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == [{"type": "function", "function": provider_schema}]
    assert agent._memory_provider_tool_names == {"shared_tool"}
    assert (
        invoke_tool(
            agent,
            "shared_tool",
            {"query": "needle"},
            "task",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
        )
        == "provider-dispatch"
    )
    assert provider_calls == [("shared_tool", {"query": "needle"})]
    assert registry_calls == []


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
    from agent.agent_runtime_helpers import memory_provider_owns_tool
    assert memory_provider_owns_tool(agent, "fact_store")


def test_schema_callback_failure_preserves_all_refresh_state(monkeypatch):
    agent = _agent(["read_file", "fact_store"], enabled=["memory"])
    agent._memory_provider_tool_names = {"fact_store"}
    agent._context_engine_tool_names = {"lcm_grep"}
    agent._cached_system_prompt = "provider prompt"
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
    import pytest

    with pytest.raises(RuntimeError, match="schema callback failed"):
        mcp_tool.refresh_agent_mcp_tools(
            agent,
            enabled_override=["coding"],
        )

    assert agent.tools is original_tools
    assert agent.valid_tool_names is original_names
    assert agent.enabled_toolsets == ["memory"]
    assert agent._memory_provider_tool_names == {"fact_store"}
    assert agent._context_engine_tool_names == {"lcm_grep"}
    assert agent._cached_system_prompt == "provider prompt"


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
    from agent.agent_runtime_helpers import memory_provider_owns_tool
    assert not memory_provider_owns_tool(agent, "fact_store")
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
    automatic = threading.Thread(
        target=mcp_tool.refresh_agent_mcp_tools,
        args=(agent,),
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
    assert agent.enabled_toolsets == ["new-policy"]
    assert agent.valid_tool_names == {"new_tool"}


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


def test_refresh_equal_names_transfers_context_engine_ownership(monkeypatch):
    """A registry replacement with the same name must stop engine routing."""
    agent = _agent(["read_file", "lcm_grep"])
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [
            {"name": "lcm_grep", "description": "", "parameters": {}}
        ]
    )
    agent._context_engine_tool_names = {"lcm_grep"}
    original_tools = agent.tools

    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kw: [_tool("read_file"), _tool("lcm_grep")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools is original_tools
    assert agent._context_engine_tool_names == set()


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


def test_resolve_discovery_timeout_reads_config(monkeypatch):
    from hermes_cli import mcp_startup
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 8.0})

    assert mcp_startup._resolve_discovery_timeout(None) == 8.0


def test_resolve_discovery_timeout_falls_back_on_bad_value(monkeypatch):
    from hermes_cli import mcp_startup
    import hermes_cli.config as cfg

    # Non-positive / unparsable → DEFAULT_CONFIG value, never hang.
    default = float(cfg.DEFAULT_CONFIG.get("mcp_discovery_timeout", 1.5))
    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 0})
    assert mcp_startup._resolve_discovery_timeout(None) == default

    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": "oops"})
    assert mcp_startup._resolve_discovery_timeout(None) == default


def test_stale_generation_refresh_does_not_clobber_newer(monkeypatch):
    """A slower refresh that computed an OLDER registry generation must not
    overwrite a snapshot a newer-generation refresh already published."""
    from tools import registry as _reg_mod

    agent = _agent(["read_file"])
    # A newer refresh already published generation = current+5, with two tools.
    agent._tool_snapshot_generation = _reg_mod.registry._generation + 5
    agent.tools = [_tool("read_file"), _tool("mcp_new_tool")]
    agent.valid_tool_names = {"read_file", "mcp_new_tool"}

    import model_tools
    # This (stale) refresh computes only the old single-tool set.
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [_tool("read_file")])

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    # Stale write rejected: the newer tool survives.
    assert added == set()
    assert "mcp_new_tool" in agent.valid_tool_names


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
