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
    """A refresh removes stale provider tools when memory becomes disabled."""
    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kw: [_tool("read_file")],
    )

    for disabled_toolset in ("memory", "all", "*"):
        agent = _agent(
            ["read_file", "memory_search"],
            enabled=["all"],
            disabled=[disabled_toolset],
        )
        agent._memory_manager = types.SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "memory_search", "description": "", "parameters": {}}
            ]
        )

        mcp_tool.refresh_agent_mcp_tools(agent)

        assert "memory_search" not in agent.valid_tool_names
        assert all(t["function"]["name"] != "memory_search" for t in agent.tools)


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


# ---------------------------------------------------------------------------
# preserve_prefix: the tool array is a cached request prefix (#100336)
# ---------------------------------------------------------------------------


def _registered(monkeypatch, names):
    """Make the registry report exactly *names* as still registered."""
    from tools import registry as registry_mod

    entries = [types.SimpleNamespace(name=n) for n in names]
    monkeypatch.setattr(
        registry_mod.registry, "get_all_entries", lambda: entries, raising=False
    )


def _serve(monkeypatch, defs):
    import model_tools

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: list(defs))


def test_preserve_prefix_carries_a_flapping_tool_forward(monkeypatch):
    """A check_fn flip must not shrink a live session's tool prefix.

    ``browser_navigate``'s availability probe fails this turn (headless box,
    expired credential, docker blip) so ``get_tool_definitions`` omits it. The
    tool is still *registered* — only its probe flapped — so the snapshot must
    keep it, byte-for-byte, instead of forking the cached prefix.
    """
    agent = _agent(["read_file", "browser_navigate", "terminal"])
    before = list(agent.tools)

    _serve(monkeypatch, [_tool("read_file"), _tool("terminal")])
    _registered(monkeypatch, ["read_file", "browser_navigate", "terminal"])

    added = mcp_tool.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert added == set()
    assert agent.tools == before
    assert "browser_navigate" in agent.valid_tool_names


def test_preserve_prefix_appends_late_arrivals_at_the_tail(monkeypatch):
    """``get_definitions`` sorts by name, so a late tool can splice in at 0.

    Under ``preserve_prefix`` the live order is authoritative and the new tool
    extends the array, leaving every earlier byte where the provider cached it.
    """
    agent = _agent(["read_file", "terminal"])

    # Sorted order would put the new tool first.
    _serve(monkeypatch, [_tool("aaa_mcp_late"), _tool("read_file"), _tool("terminal")])
    _registered(monkeypatch, ["aaa_mcp_late", "read_file", "terminal"])

    added = mcp_tool.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert added == {"aaa_mcp_late"}
    assert [t["function"]["name"] for t in agent.tools] == [
        "read_file", "terminal", "aaa_mcp_late",
    ]


# ---------------------------------------------------------------------------
# tools[] freeze: eviction rebuild + the /reload-mcp re-probe hatch
# ---------------------------------------------------------------------------


def test_eviction_rebuild_restores_the_sessions_saved_tool_order(monkeypatch):
    """A fresh AIAgent for an EXISTING session must keep the saved tools[] pin.

    Gateway agent-cache eviction rebuilds the agent; ``agent_init`` re-probes
    every ``check_fn`` and ``browser_navigate``'s flips false. The persisted
    name list stands in for the missing predecessor: the tool is carried
    forward from the registry schema, byte-for-byte in its old slot.
    """
    from tools import registry as registry_mod

    saved = ["read_file", "browser_navigate", "terminal"]
    entries = {n: types.SimpleNamespace(name=n, schema=_tool(n)["function"]) for n in saved}
    monkeypatch.setattr(registry_mod.registry, "get_all_entries", lambda: list(entries.values()), raising=False)
    monkeypatch.setattr(registry_mod.registry, "get_entry", lambda name, **kw: entries.get(name), raising=False)

    rebuilt = _agent(["read_file", "terminal"])  # probe flipped: browser_navigate gone
    changed = mcp_tool.restore_agent_tool_prefix(rebuilt, saved)

    assert changed is True
    assert [t["function"]["name"] for t in rebuilt.tools] == saved
    assert rebuilt.valid_tool_names == set(saved)


def test_reprobe_tool_availability_drops_cached_check_fn_verdicts(monkeypatch):
    """/reload-mcp is the explicit hatch: a cached False must be re-probed."""
    from tools import registry as registry_mod
    import model_tools

    verdict = {"ok": False}

    def probe():
        return verdict["ok"]

    monkeypatch.setattr(registry_mod, "check_fn_cache_scope", lambda: "test-scope")
    assert registry_mod._check_fn_cached(probe) is False
    verdict["ok"] = True
    assert registry_mod._check_fn_cached(probe) is False  # TTL cache replays stale verdict
    with model_tools._tool_defs_cache_lock:
        model_tools._tool_defs_cache[("sentinel",)] = []

    mcp_tool.reprobe_tool_availability()

    assert registry_mod._check_fn_cached(probe) is True
    assert ("sentinel",) not in model_tools._tool_defs_cache
