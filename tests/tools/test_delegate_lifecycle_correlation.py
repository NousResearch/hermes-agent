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
