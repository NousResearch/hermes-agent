"""A tool this session cannot reach must never be answered with "call it directly".

``_HERMES_CORE_TOOLS`` is static, so ``tool_call`` matched any core tool by name, including
``clarify`` inside a subagent, where no direct call can ever succeed and the model retries. Absence
is reported as unavailability. In a managed worker (every Kanban card) ``delegate_task`` works
through owner-reserved children (#141793), so a turn whose toolsets leave delegation off gets the
same ordinary answer, never a claim that the worker boundary itself forbids delegation. The tests
drive the real ``tool_call`` bridge in ``model_tools``.
"""
import json

import pytest

import model_tools
import tools.delegate_tool  # registers delegate_task in the global registry
from agent import runtime_session_store
from agent.conversation_loop import _invalid_tool_name_error_content
from tools import tool_search
from tools.tool_search import ToolSearchConfig, dispatch_tool_describe


def _bridge(name: str, toolsets: list[str]) -> str:
    result = model_tools.handle_function_call(
        "tool_call", {"calls": [{"name": name, "arguments": {}}]}, enabled_toolsets=toolsets)
    return json.loads(result).get("error", "")


@pytest.mark.parametrize("worker", [True, False])
def test_delegate_task_absent_from_a_turn_is_reported_as_unavailable(monkeypatch, worker):
    monkeypatch.setattr(runtime_session_store, "_worker_process", worker)
    turn_tools = [{"type": "function", "function": {"name": "kanban_create", "parameters": {}}}]
    described = json.loads(dispatch_tool_describe({"names": ["delegate_task"]}, current_tool_defs=turn_tools,
                                                  config=ToolSearchConfig.from_raw({})))
    for message in (_bridge("delegate_task", ["file"]), described["errors"]["delegate_task"]):
        assert "not available in this execution context" in message, message
        assert "Call it directly" not in message and "managed worker" not in message, message
    assert "managed worker" not in _invalid_tool_name_error_content("delegate_task", ["kanban_create"])


@pytest.mark.parametrize("name, toolsets, expected", [
    ("clarify", ["file"], "not available in this execution context"),  # stripped: never "call it directly"
    ("clarify", ["file", "clarify"], "Call it directly"),               # reachable: the door guidance holds
    ("delegate_task", ["file", "delegation"], "Call it directly"),      # a worker turn that has it
])
def test_core_tool_guidance_follows_the_session(monkeypatch, name, toolsets, expected):
    monkeypatch.setattr(runtime_session_store, "_worker_process", name == "delegate_task")
    message = _bridge(name, toolsets)
    assert expected in message and "managed worker" not in message, message


def test_a_deferred_tool_absent_from_the_session_is_not_sent_to_tool_search(monkeypatch):
    """Deferral is a property of the tool, availability one of the session (``defer: [delegate_task]``)."""
    monkeypatch.setattr(runtime_session_store, "_worker_process", False)
    deferred = ToolSearchConfig.from_raw({"defer": ["delegate_task"]})
    monkeypatch.setattr(tool_search, "load_config_readonly", lambda: deferred)
    described = json.loads(model_tools.handle_function_call(
        "tool_describe", {"names": ["delegate_task"]}, enabled_toolsets=["file"]))
    for message in (_bridge("delegate_task", ["file"]), described["errors"]["delegate_task"]):
        assert "not available in this execution context" in message and "tool_search" not in message, message
