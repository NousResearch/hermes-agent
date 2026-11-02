"""A server that revives mid-turn lands in the turn's next request.

The MCP server was parked (tools deregistered) when the turn's tool snapshot was assembled,
revived seconds later and re-registered its tools, and nothing refreshed the snapshot inside
the turn. ``_refresh_mcp_tools_between_turns`` runs only at turn start, and
``_merge_preserving_prefix`` keeps the bridge's sent bytes, so the ``tool_search`` listing the
model read could not name the server's tools until the next turn, and the model guessed names
for the rest of the turn. Pinned here:

- ``refresh_agent_tools_after_registry_move`` folds the live registry into the snapshot
  once the registry generation moved past it, re-rendering only the bridge listing
  (existing tools keep their bytes, new tools append), and is a no-op otherwise
- ``prepare_iteration`` runs it before each request is built
- the frozen cache-parity fork and an agent with zero registered MCP tools never rebuild
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from tools.tool_search import ToolSearchConfig, assemble_tool_defs
from tools.tool_search_catalog import TOOL_SEARCH_NAME

CORE = ["read_file", "terminal"]
DESKTOP = [f"desktop_tool_{i}" for i in range(13)]
SHOP = ["mcp__shop__connection_setup", "mcp__shop__catalog_search", "mcp__shop__order_propose"]
FROZEN = 2_147_483_647  # agent.background_review._FROZEN_TOOL_SNAPSHOT_GENERATION


def _td(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": f"{name} does one thing",
                                             "parameters": {"type": "object", "properties": {}}}}


def _assembled(names: list[str], deferred: list[str]) -> list[dict]:
    config = ToolSearchConfig(enabled="on", threshold_pct=5, search_default_limit=5,
                              max_search_limit=25, listing="on", defer_tools=frozenset(deferred))
    return assemble_tool_defs([_td(n) for n in names], context_length=256_000, config=config).tool_defs


def _bridge_description(tool_defs: list[dict]) -> str:
    return next(t["function"]["description"] for t in tool_defs if t["function"]["name"] == TOOL_SEARCH_NAME)


def _bytes_of(tool_defs: list[dict], name: str) -> str:
    return json.dumps(next(t for t in tool_defs if t["function"]["name"] == name), sort_keys=True)


def _parked_turn_agent(generation: int) -> SimpleNamespace:
    """The snapshot a turn assembled while the shop server was parked: core tools visible, the
    desktop surface deferred behind the bridge, no shop tool anywhere."""
    agent = SimpleNamespace()
    agent.tools = _assembled(CORE + DESKTOP, DESKTOP)
    agent.valid_tool_names = {t["function"]["name"] for t in agent.tools}
    agent.enabled_toolsets = None
    agent.disabled_toolsets = None
    agent._tool_snapshot_generation = generation
    return agent


@pytest.fixture
def revived_registry(monkeypatch):
    """The live registry after the server revived: one generation past the snapshot, the shop
    tools registered and deferred behind the bridge next to the desktop surface."""
    import tools.mcp_tool  # the refresh helpers' import-cost gate
    import model_tools
    from tools import mcp_tool_discovery
    from tools.registry import registry

    base = registry._generation
    calls: list[dict] = []

    def _live(**kw):
        calls.append(kw)
        return _assembled(CORE + DESKTOP + SHOP, DESKTOP + SHOP)

    monkeypatch.setattr(model_tools, "get_tool_definitions", _live)
    monkeypatch.setattr(mcp_tool_discovery, "has_registered_mcp_tools", lambda: True)
    monkeypatch.setattr(registry, "_generation", base + 1)
    return SimpleNamespace(base=base, calls=calls)


def test_revived_server_lands_in_the_next_request(revived_registry):
    from tools.mcp_tool_agent import refresh_agent_tools_after_registry_move

    agent = _parked_turn_agent(revived_registry.base)
    before = _bridge_description(agent.tools)
    assert "connection_setup" not in before
    core_bytes = {n: _bytes_of(agent.tools, n) for n in CORE}

    refresh_agent_tools_after_registry_move(agent)

    after = _bridge_description(agent.tools)
    assert "connection_setup" in after and "catalog_search" in after
    # Only the bridge listing moved: the tools already sent keep their bytes and their slots.
    assert {n: _bytes_of(agent.tools, n) for n in CORE} == core_bytes
    assert [t["function"]["name"] for t in agent.tools][:len(CORE)] == CORE
    assert agent._tool_snapshot_generation == revived_registry.base + 1
    assert len(revived_registry.calls) == 1

    # The registry has not moved again: the next iteration is one integer compare, no rebuild.
    refresh_agent_tools_after_registry_move(agent)
    assert len(revived_registry.calls) == 1


def test_unmoved_registry_never_rebuilds(revived_registry, monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_tools_after_registry_move
    from tools.registry import registry

    monkeypatch.setattr(registry, "_generation", revived_registry.base)
    agent = _parked_turn_agent(revived_registry.base)
    snapshot = json.dumps(agent.tools, sort_keys=True)

    assert refresh_agent_tools_after_registry_move(agent) == set()
    assert json.dumps(agent.tools, sort_keys=True) == snapshot
    assert revived_registry.calls == []


def test_frozen_fork_snapshot_never_rebuilds(revived_registry):
    """The cache-parity review fork freezes its snapshot at the sentinel generation; a registry
    move must not touch its tools[] (the parent's exact outbound payload)."""
    from tools.mcp_tool_agent import refresh_agent_tools_after_registry_move

    agent = _parked_turn_agent(FROZEN)
    snapshot = json.dumps(agent.tools, sort_keys=True)

    assert refresh_agent_tools_after_registry_move(agent) == set()
    assert json.dumps(agent.tools, sort_keys=True) == snapshot
    assert agent._tool_snapshot_generation == FROZEN
    assert revived_registry.calls == []


def test_zero_registered_mcp_tools_skips(revived_registry, monkeypatch):
    """Same gate as the between-turns refresh: a registry move with no MCP tool registered
    (the only server parked again) leaves the snapshot alone."""
    from tools import mcp_tool_discovery
    from tools.mcp_tool_agent import refresh_agent_tools_after_registry_move

    monkeypatch.setattr(mcp_tool_discovery, "has_registered_mcp_tools", lambda: False)
    agent = _parked_turn_agent(revived_registry.base)

    assert refresh_agent_tools_after_registry_move(agent) == set()
    assert revived_registry.calls == []


def test_prefix_merge_rerenders_only_the_bridge_on_request():
    """The between-turns contract (bridge keeps its sent bytes) stays the default; the mid-turn
    path asks for the fresh listing explicitly. Either way non-bridge tools keep their bytes."""
    from tools.mcp_tool_agent import _merge_preserving_prefix

    current = _assembled(CORE + DESKTOP, DESKTOP)
    fresh = _assembled(CORE + DESKTOP + SHOP, DESKTOP + SHOP)
    registered = {t["function"]["name"] for t in current}

    kept, _ = _merge_preserving_prefix(current, fresh, registered)
    assert _bridge_description(kept) == _bridge_description(current)

    rerendered, names = _merge_preserving_prefix(current, fresh, registered, refresh_bridge=True)
    assert _bridge_description(rerendered) == _bridge_description(fresh)
    assert names == {t["function"]["name"] for t in current}
    assert [_bytes_of(rerendered, n) for n in CORE] == [_bytes_of(current, n) for n in CORE]


def test_prepare_iteration_refreshes_before_the_request(tmp_path, monkeypatch):
    """The hook runs once per iteration, before the request is assembled from ``agent.tools``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool  # the call site's import-cost gate
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration
    from tools import mcp_tool_agent

    seen: list = []
    monkeypatch.setattr(mcp_tool_agent, "refresh_agent_tools_after_registry_move", seen.append)
    agent = AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"), model="test-model",
                    provider="openai-compat", api_key="test", base_url="http://127.0.0.1:1/v1",
                    max_iterations=4, quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        _reset_per_turn_agent_state(agent)
        messages = [{"role": "user", "content": "find the item and order it"}]
        prepare_iteration(agent, messages=messages, api_call_count=1)
        prepare_iteration(agent, messages=messages, api_call_count=2)
    finally:
        agent._session_db.close()
    assert seen == [agent, agent]
