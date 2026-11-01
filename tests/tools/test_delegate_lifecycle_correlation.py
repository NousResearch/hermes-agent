"""Delegation hook correlation: child lifecycle and child tool hooks name the parent delegate_task call."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest


def test_delegate_tool_call_id_reaches_delegate_task_on_both_dispatch_paths(monkeypatch):
    """Sequential and concurrent (inline-executor) dispatch both forward the parent's
    ``delegate_task`` tool_call_id; a dropped id leaves parallel same-goal children uncorrelatable."""
    import tools.delegate_tool as delegate_module
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch
    from run_agent import AIAgent

    captured: list = []
    monkeypatch.setattr(delegate_module, "delegate_task", lambda **kw: captured.append(kw["tool_call_id"]) or "ok")
    monkeypatch.setattr("agent.tool_executor._start_quiet_tool_spinner", lambda *a, **k: None)
    agent = SimpleNamespace(_delegate_depth=0, _delegate_spinner=None, _context_engine_tool_names=set(),
                            _memory_manager=None)
    agent._dispatch_delegate_task = functools.partial(AIAgent._dispatch_delegate_task, agent)
    args = {"tasks": [{"goal": "same goal"}, {"goal": "same goal"}]}

    ref = _ToolCallRef(name="delegate_task", args=args, task_id="t", call_id="call-seq", trace=[])
    assert _resolve_sequential_dispatch(agent, ref, []).execute(args) == "ok"
    ctx = InlineToolContext(effective_task_id="t", tool_call_id="call-par")
    assert INLINE_TOOL_EXECUTORS["delegate_task"](agent, args, ctx) == "ok"

    assert captured == ["call-seq", "call-par"]


@pytest.mark.parametrize("hook", ["pre_tool_call", "post_tool_call"])
def test_child_tool_hooks_carry_parent_tool_call_id_only_inside_a_child(monkeypatch, hook):
    """A tool hook fired by a delegate child carries ``parent_tool_call_id``; the same hook fired by the
    parent (outside the child context) has no such key, so top-level payloads keep their shape."""
    from hermes_cli import plugins
    from agent.delegation_context import delegated_child_context
    from model_tools import _emit_post_tool_call_hook

    seen: list[dict] = []
    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks[hook] = [lambda **kw: seen.append(kw)]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    def fire():
        if hook == "pre_tool_call":
            plugins._get_pre_tool_call_directive_details("read_file", {"path": "x"}, tool_call_id="child-call")
        else:
            _emit_post_tool_call_hook(function_name="read_file", function_args={"path": "x"}, result="{}",
                                      tool_call_id="child-call")

    fire()
    with delegated_child_context("child-session", "call-delegate-7"):
        fire()

    assert len(seen) == 2
    assert "parent_tool_call_id" not in seen[0]
    assert seen[1]["parent_tool_call_id"] == "call-delegate-7"
    assert seen[1]["tool_call_id"] == "child-call"


def test_subagent_stop_names_the_spawning_call_and_carries_each_childs_spend(monkeypatch):
    """Two same-goal children of one ``delegate_task`` call: each ``subagent_stop`` names that call, its own
    task index and subagent id, and the child's own model, tokens and cost (observers had to read state.db)."""
    import tools.delegate_tool as dt
    from hermes_cli import plugins
    from unittest.mock import MagicMock

    stops: list[dict] = []
    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["subagent_stop"] = [lambda **kw: stops.append(kw)]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)

    def build(task_index, **kw):
        child = MagicMock(_delegate_saved_tool_names=[], _credential_pool=None, _subagent_id=f"sa-{task_index}",
                          _delegate_parent_tool_call_id=kw.get("parent_tool_call_id"), session_id=f"c{task_index}")
        return child

    def run(task_index, goal, child=None, parent_agent=None, **kw):
        return {"task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1 + task_index,
                "duration_seconds": 0.2, "model": "tiny-model", "tokens": {"input": 7, "output": task_index},
                "cost_usd": 0.001 * (task_index + 1), "cost_status": "estimated"}

    monkeypatch.setattr(dt, "_build_child_agent", build)
    monkeypatch.setattr(dt, "_run_single_child", run)
    parent = MagicMock(_delegate_depth=0, session_id="parent-1", _memory_manager=None, _active_children=[],
                       _print_fn=None, tool_progress_callback=None, _session_db=None)
    dt.delegate_task(tasks=[{"goal": "identical goal text"}, {"goal": "identical goal text"}], background=False, parent_agent=parent,
                     tool_call_id="call-delegate-9")

    by_index = {s["task_index"]: s for s in stops}
    assert sorted(by_index) == [0, 1]
    for i, stop in by_index.items():
        assert (stop["parent_tool_call_id"], stop["child_subagent_id"]) == ("call-delegate-9", f"sa-{i}")
        assert (stop["model"], stop["api_calls"], stop["tokens"]) == ("tiny-model", 1 + i, {"input": 7, "output": i})
        assert (stop["cost_usd"], stop["cost_status"]) == (0.001 * (i + 1), "estimated")
