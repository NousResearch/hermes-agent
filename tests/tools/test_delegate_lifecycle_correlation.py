"""Delegation lifecycle correlation across parallel children."""

from __future__ import annotations

from types import SimpleNamespace

from tools import delegate_tool, delegate_tool_results


def test_child_lifecycle_correlation_carries_parent_call_and_task_index(monkeypatch):
    start_events = []
    stop_events = []
    child = SimpleNamespace(
        session_id="child-session",
        _delegate_role="researcher",
        _subagent_id="sa-0-test",
        _delegate_parent_tool_call_id="call-parent-1",
    )
    parent = SimpleNamespace(session_id="parent-session", _current_turn_id="turn-1")

    monkeypatch.setattr(
        delegate_tool_results,
        "_subagent_stop_tool_call_history",
        lambda _trace: [],
    )
    import hermes_cli.plugins as plugin_module
    monkeypatch.setattr(plugin_module, "invoke_hook", lambda name, **kwargs: stop_events.append((name, kwargs)))

    entry = {
        "task_index": 3,
        "status": "completed",
        "summary": "done",
        "duration_seconds": 1.2,
        "tool_trace": [],
        "_child_role": "researcher",
        "_child_cost_usd": 0.0,
    }
    delegate_tool_results._fire_subagent_stop_hooks([entry], {3: child}, parent)

    assert stop_events == [("subagent_stop", {
        "parent_session_id": "parent-session",
        "parent_turn_id": "turn-1",
        "child_session_id": "child-session",
        "child_role": "researcher",
        "child_subagent_id": "sa-0-test",
        "parent_tool_call_id": "call-parent-1",
        "task_index": 3,
        "delegation_purpose": None,
        "child_summary": "done",
        "child_status": "completed",
        "tool_call_history": [],
        "duration_ms": 1200,
    })]


def test_sequential_dispatch_scopes_parent_tool_call_id(monkeypatch):
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch

    seen = []
    agent = SimpleNamespace(
        _delegate_spinner=None,
        _context_engine_tool_names=set(),
        _memory_manager=None,
        _dispatch_delegate_task=lambda args: seen.append((args, getattr(agent, "_delegate_parent_tool_call_id", None))) or "ok",
    )
    monkeypatch.setattr("agent.tool_executor._start_quiet_tool_spinner", lambda *args, **kwargs: None)
    ref = _ToolCallRef(name="delegate_task", args={"tasks": [{"goal": "inspect bounded target"}]}, task_id="task", call_id="call-parent-1", trace=[])

    dispatch = _resolve_sequential_dispatch(agent, ref, [])
    assert dispatch.execute(ref.args) == "ok"
    assert seen == [(ref.args, "call-parent-1")]
    assert not hasattr(agent, "_delegate_parent_tool_call_id")


def test_agent_dispatch_forwards_parent_tool_call_id(monkeypatch):
    from run_agent import AIAgent
    import tools.delegate_tool as delegate_module

    captured = {}
    monkeypatch.setattr(delegate_module, "delegate_task", lambda **kwargs: captured.update(kwargs) or "ok")

    agent = SimpleNamespace(_delegate_depth=0, _delegate_parent_tool_call_id="call-parent-1")
    result = AIAgent._dispatch_delegate_task(
        agent,
        {"tasks": [{"goal": "inspect bounded target"}]},
    )

    assert result == "ok"
    assert captured["tool_call_id"] == "call-parent-1"
    assert captured["parent_agent"] is agent
