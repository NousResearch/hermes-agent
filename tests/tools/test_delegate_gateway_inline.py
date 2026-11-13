"""A session's ``delegate_task`` schema decides whether its children join in the call.

The gateway builds a session's agent inside ``gateway_delegation_inline_scope`` when that session resolved
``delegation.gateway_mode`` to inline (``auto`` + interim assistant messages off, or ``inline``). The tool build
then carries the joined ``delegate_task`` variant (results return in the call, no live control), and dispatch
follows the variant the model actually sees, so schema and behaviour cannot disagree. The variant rides the
session's tools[] pin like every other schema byte: a config edit applies from the next session (``/new``),
the rule ``delegation.backend`` already follows, and a cache eviction or restart never flips a live chat.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.session_context import set_session_vars
from tools.process_registry import process_registry

LIVE_CONTROL = {"action", "subagent_id", "message"}


@pytest.fixture(autouse=True)
def _clean_queue_and_context(monkeypatch):
    import gateway.session_context as sc

    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    yield
    for var in sc._VAR_MAP.values():
        var.set(sc._UNSET)
    sc._SESSION_ASYNC_DELIVERY.set(sc._UNSET)
    sc._SESSION_HISTORY_DELIVERY.set(sc._UNSET)
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _built(inline: bool):
    from agent.agent_init import _load_tools
    from tools.delegate_tool_config import gateway_delegation_inline_scope

    agent = SimpleNamespace(platform="whatsapp", quiet_mode=True)
    with gateway_delegation_inline_scope(inline):
        _load_tools(agent, ["hermes-cli"], None)
    return agent


def _delegate(agent) -> dict:
    return next(t["function"] for t in agent.tools if t["function"]["name"] == "delegate_task")


def test_the_build_scope_picks_the_schema_and_the_agent_keeps_it():
    import model_tools
    from tools.delegate_tool import oneshot_delegate_function
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    background = _built(inline=False)
    registry = next(t["function"] for t in model_tools.get_tool_definitions(
        enabled_toolsets=["hermes-cli"], disabled_toolsets=background.disabled_toolsets, quiet_mode=True)
        if t["function"]["name"] == "delegate_task")
    assert _delegate(background) == registry  # byte-identical to main

    inline = _built(inline=True)
    assert _delegate(inline) == oneshot_delegate_function(registry)
    assert "END YOUR TURN" not in _delegate(inline)["description"]
    assert not LIVE_CONTROL & set(_delegate(inline)["parameters"]["properties"])
    assert "clarify" in {t["function"]["name"] for t in inline.tools}  # nothing else of the one-shot footprint

    # A later rebuild outside any gateway scope (explicit reload, compaction) keeps the session's variant.
    refresh_agent_mcp_tools(inline, quiet_mode=True, content_aware=True)
    assert _delegate(inline) == oneshot_delegate_function(registry)


def test_a_session_keeps_its_mode_until_a_new_session():
    """Operator flips interim back on mid-session, then the cached agent is evicted: the fresh agent is built
    for background, but the session pin restores the joined schema the model has been seeing, and dispatch
    follows the pin. A new session (no pin) takes the new setting."""
    from tools.delegate_tool import session_joins_delegation
    from tools.mcp_tool_agent import restore_agent_tool_prefix, tool_pin_version

    first = _built(inline=True)
    fresh = _built(inline=False)
    restore_agent_tool_prefix(fresh, {"version": tool_pin_version(), "tools": first.tools})
    assert _delegate(fresh) == _delegate(first)
    assert session_joins_delegation(fresh) is True
    assert session_joins_delegation(_built(inline=False)) is False

    # And the other way round: a background session stays background after interim is turned off.
    background = _built(inline=False)
    rebuilt = _built(inline=True)
    restore_agent_tool_prefix(rebuilt, {"version": tool_pin_version(), "tools": background.tools})
    assert session_joins_delegation(rebuilt) is False


@pytest.mark.parametrize("refresh", [
    {"content_aware": True},  # the compaction boundary (agent/conversation_compression.py)
    {},  # /reload-mcp over the cached agents (gateway/run_turn.py)
], ids=["compaction-refresh", "reload-mcp"])
@pytest.mark.parametrize("pinned_inline", [True, False], ids=["pinned-inline", "pinned-background"])
def test_a_restored_mode_survives_a_later_tools_refresh(refresh, pinned_inline):
    """The pin is authoritative: a fresh agent built for the other mode restores the session's variant, and a later
    rebuild from fresh definitions (which reads the agent's frozen decision) keeps it instead of flipping the chat
    mid-conversation to whatever the current config says."""
    from tools.delegate_tool import session_joins_delegation
    from tools.mcp_tool_agent import refresh_agent_mcp_tools, restore_agent_tool_prefix, tool_pin_version

    first = _built(inline=pinned_inline)
    fresh = _built(inline=not pinned_inline)
    restore_agent_tool_prefix(fresh, {"version": tool_pin_version(), "tools": first.tools})
    assert session_joins_delegation(fresh) is pinned_inline

    refresh_agent_mcp_tools(fresh, quiet_mode=True, **refresh)
    assert session_joins_delegation(fresh) is pinned_inline
    assert _delegate(fresh) == _delegate(first)


def _parent(tools, depth=0):
    parent = MagicMock()
    parent._delegate_depth = depth
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    parent._oneshot_session = False
    parent.tools = tools
    return parent


@pytest.fixture
def fast_children(monkeypatch):
    import tools.delegate_tool as dt

    def build_child(**kw):
        child = MagicMock()
        child._delegate_role, child._subagent_id = "leaf", "s1"
        return child

    def run_child(task_index, goal, child=None, parent_agent=None, **kw):
        return {"task_index": task_index, "status": "completed", "summary": f"done: {goal}",
                "api_calls": 1, "duration_seconds": 0.1, "model": "m", "exit_reason": "completed"}

    creds = dict.fromkeys(("provider", "base_url", "api_key", "api_mode", "command", "args"), None)
    monkeypatch.setattr(dt, "_build_child_agent", build_child)
    monkeypatch.setattr(dt, "_run_single_child", run_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: {**creds, "model": "m"})
    set_session_vars(platform="whatsapp", chat_id="c1", session_key="agent:main:whatsapp:dm:c1", session_id="sess")
    return dt


def test_dispatch_follows_the_schema_the_session_carries(fast_children, monkeypatch):
    from tools import async_delegation

    dispatched = []
    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch",
                        lambda **kw: dispatched.append(kw) or {"status": "dispatched", "delegation_id": "d1"})

    joined = json.loads(fast_children.delegate_task(
        goal="audit", background=True, parent_agent=_parent(_built(inline=True).tools)))
    assert joined["results"][0]["summary"] == "done: audit"
    assert "note" not in joined and dispatched == []  # no "end your turn", no async registry

    detached = json.loads(fast_children.delegate_task(
        goal="audit", background=True, parent_agent=_parent(_built(inline=False).tools)))
    assert detached["status"] == "dispatched" and "END YOUR TURN" in detached["note"]
    assert len(dispatched) == 1


def test_nested_and_one_shot_delegation_keep_their_own_paths(fast_children, monkeypatch):
    # An orchestrator child (depth > 0) was always synchronous; a one-shot session keeps its explanatory note.
    monkeypatch.setattr(fast_children, "_get_max_spawn_depth", lambda: 2)
    nested = json.loads(fast_children.delegate_task(
        goal="audit", background=False, parent_agent=_parent(_built(inline=False).tools, depth=1)))
    assert nested["results"][0]["status"] == "completed" and "note" not in nested

    oneshot = _parent(_built(inline=True).tools)
    oneshot._oneshot_session = True
    result = json.loads(fast_children.delegate_task(goal="audit", background=True, parent_agent=oneshot))
    assert "SYNCHRONOUSLY" in result["note"]


def test_live_control_degrades_to_plain_answers_in_an_inline_session(fast_children):
    parent = _parent(_built(inline=True).tools)
    listed = json.loads(fast_children.delegate_task(action="list", parent_agent=parent))
    assert isinstance(listed, dict)
    stopped = json.loads(fast_children.delegate_task(action="stop", subagent_id="nope", parent_agent=parent))
    assert "error" in stopped
