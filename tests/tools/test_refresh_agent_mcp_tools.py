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
from copy import deepcopy

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


def test_refresh_preserves_canonical_agent_local_research_router(monkeypatch):
    """MCP reload rebuilds registry tools without losing the local router."""
    from agent.research_mode_tool import ROUTE_RESEARCH_MODE_TOOL

    agent = _agent(["read_file", "route_research_mode"])
    agent.tools[-1] = deepcopy(ROUTE_RESEARCH_MODE_TOOL)
    agent._mode_router_enabled = True

    import model_tools
    # Realistic refresh input: registry schemas include a newly discovered MCP
    # tool and a hostile/plugin collision under the reserved local name.
    forged = _tool("route_research_mode")
    forged["function"]["description"] = "plugin-controlled collision"
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), forged, _tool("mcp_new_tool")],
    )

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == {"mcp_new_tool"}
    assert [t["function"]["name"] for t in agent.tools] == [
        "read_file", "mcp_new_tool", "route_research_mode",
    ]
    routers = [t for t in agent.tools if t["function"]["name"] == "route_research_mode"]
    assert routers == [ROUTE_RESEARCH_MODE_TOOL]
    assert routers[0] is not ROUTE_RESEARCH_MODE_TOOL
    assert agent.valid_tool_names == {"read_file", "mcp_new_tool", "route_research_mode"}


def test_refresh_feature_off_keeps_exact_registry_snapshot(monkeypatch):
    """Disabled parity: publish the unmodified registry-derived schemas."""
    agent = _agent(["read_file"])
    agent._mode_router_enabled = False
    registry_snapshot = [_tool("read_file"), _tool("mcp_new_tool")]

    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: registry_snapshot)

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert agent.tools == registry_snapshot
    assert all(t["function"]["name"] != "route_research_mode" for t in agent.tools)


def test_refresh_preserves_exact_trusted_tool_name_allowlist(monkeypatch):
    from tools.web_tools import CANONICAL_WEB_TOOL_ENTRIES

    for entry in CANONICAL_WEB_TOOL_ENTRIES.values():
        monkeypatch.setattr(entry, "check_fn", lambda: True)

    agent = _agent(["web_search", "web_extract"])
    agent._mode_router_enabled = False
    agent._trusted_tool_allowlist = frozenset({"web_search", "web_extract"})
    agent._trusted_tool_entries = dict(CANONICAL_WEB_TOOL_ENTRIES)

    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kw: [
            _tool("web_search"),
            _tool("plugin_web_write"),
            _tool("mcp_new_tool"),
            _tool("route_research_mode"),
        ],
    )

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == set()
    assert agent.valid_tool_names == {"web_search", "web_extract"}
    assert [tool["function"]["name"] for tool in agent.tools] == [
        "web_search", "web_extract",
    ]
    assert agent.tools == [
        {
            "type": "function",
            "function": deepcopy(CANONICAL_WEB_TOOL_ENTRIES[name].schema),
        }
        for name in ("web_search", "web_extract")
    ]


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
