"""A session pinned to the joined ``delegate_task`` keeps joining its children whatever its description says.

The tools[] pin is keyed by code version, and a pin from the same version keeps its schema bytes. A session pinned
by a build whose joined variant worded the delivery sentence differently therefore restores that wording: the model
reads that results return in the call and has no live-control parameters, so dispatch must join the children.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.session_context import set_session_vars
from tools.process_registry import process_registry


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


def _with_delivery_sentence(tools: list, sentence: str) -> list:
    from tools.delegate_tool import _ONESHOT_DESCRIPTION_DELIVERY

    def reworded(tool):
        function = tool["function"]
        if function["name"] != "delegate_task":
            return tool
        assert _ONESHOT_DESCRIPTION_DELIVERY in function["description"]
        return {**tool, "function": {**function, "description": function["description"].replace(
            _ONESHOT_DESCRIPTION_DELIVERY, sentence)}}

    return [reworded(tool) for tool in tools]


def _parent(tools):
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    parent._oneshot_session = False
    parent.tools = tools
    return parent


def test_a_joined_session_pinned_with_other_delivery_wording_still_joins(monkeypatch):
    import tools.delegate_tool as dt
    from tools import async_delegation
    from tools.mcp_tool_agent import restore_agent_tool_prefix, tool_pin_version

    earlier_wording = "Parallel children are joined; their results come back in this tool call.\n\n"
    pin = {"version": tool_pin_version(), "tools": _with_delivery_sentence(_built(inline=True).tools, earlier_wording)}
    fresh = _built(inline=False)
    restore_agent_tool_prefix(fresh, pin)
    delegate = next(t["function"] for t in fresh.tools if t["function"]["name"] == "delegate_task")
    assert earlier_wording in delegate["description"] and "action" not in delegate["parameters"]["properties"]

    def run_child(task_index, goal, child=None, parent_agent=None, **kw):
        return {"task_index": task_index, "status": "completed", "summary": f"done: {goal}",
                "api_calls": 1, "duration_seconds": 0.1, "model": "m", "exit_reason": "completed"}

    creds = dict.fromkeys(("provider", "base_url", "api_key", "api_mode", "command", "args"), None)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: MagicMock(_delegate_role="leaf", _subagent_id="s1"))
    monkeypatch.setattr(dt, "_run_single_child", run_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: {**creds, "model": "m"})
    dispatched = []
    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch",
                        lambda **kw: dispatched.append(kw) or {"status": "dispatched", "delegation_id": "d1"})
    set_session_vars(platform="whatsapp", chat_id="c1", session_key="agent:main:whatsapp:dm:c1", session_id="sess")

    result = json.loads(dt.delegate_task(goal="audit", background=True, parent_agent=_parent(fresh.tools)))
    assert dispatched == [] and result.get("results", [{}])[0].get("summary") == "done: audit", result
