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


def test_refresh_is_thread_safe_under_concurrent_calls(monkeypatch):
    """Concurrent refreshes never leave tools / valid_tool_names inconsistent."""
    agent = _agent(["a"])

    import model_tools
    defs = [_tool("a"), _tool("b"), _tool("c")]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: defs)

    errors = []

    def _worker():
        try:
            for _ in range(50):
                mcp_tool.refresh_agent_mcp_tools(agent)
                # Invariant: valid_tool_names must always match agent.tools.
                names = {t["function"]["name"] for t in agent.tools}
                assert agent.valid_tool_names == names
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert agent.valid_tool_names == {"a", "b", "c"}


# ── discovery-wait bound (mcp_discovery_timeout config) ──────────────────────


def test_resolve_discovery_timeout_explicit_wins(monkeypatch):
    from hermes_cli import mcp_startup

    assert mcp_startup._resolve_discovery_timeout(2.5) == 2.5


def test_resolve_discovery_timeout_reads_config(monkeypatch):
    from hermes_cli import mcp_startup

    monkeypatch.setattr(mcp_startup, "load_config", None, raising=False)
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 8.0})

    assert mcp_startup._resolve_discovery_timeout(None) == 8.0


def test_resolve_discovery_timeout_falls_back_on_bad_value(monkeypatch):
    from hermes_cli import mcp_startup
    import hermes_cli.config as cfg

    # Non-positive / unparsable → historical safe default, never hang.
    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 0})
    assert mcp_startup._resolve_discovery_timeout(None) == 0.75

    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": "oops"})
    assert mcp_startup._resolve_discovery_timeout(None) == 0.75


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
