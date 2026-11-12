"""The parent's side of ask_parent: delegate_task(action="reply") answers its own child's open
question; listings show what a child is waiting on; the schema only grows when the feature is on."""
import json
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup(monkeypatch):
    from tools import ask_parent, delegate_tool_config
    from tools import delegate_tool_registry as registry

    settings = {"enabled": True}
    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {"ask_parent": settings})
    monkeypatch.setattr(ask_parent, "_questions", {})
    monkeypatch.setattr(ask_parent, "_asked", {})
    monkeypatch.setattr(registry, "_active_subagents", {})
    parent = SimpleNamespace(session_id="master-session", _session_db=None)
    stranger = SimpleNamespace(session_id="another-conversation", _session_db=None)
    registry._register_subagent({"subagent_id": "sa-1", "goal": "Port the parser", "status": "awaiting_parent",
                                 "owner_agent_session_id": "master-session", "agent": SimpleNamespace()})
    question, _ = ask_parent.open_question("sa-1", question="Keep the old API?", max_questions=3)
    return SimpleNamespace(parent=parent, stranger=stranger, question=question, settings=settings)


def _reply(agent, **kwargs):
    from tools.delegate_tool import delegate_task

    return json.loads(delegate_task(action="reply", parent_agent=agent, **kwargs))


def test_the_parent_reply_answers_its_child_question(setup):
    from tools.ask_parent import wait_for_answer

    out = {}
    worker = threading.Thread(target=lambda: out.update(wait_for_answer(setup.question, timeout=5)))
    worker.start()

    result = _reply(setup.parent, subagent_id="sa-1", request_id=setup.question.request_id, message="Keep it.")
    worker.join(3)

    assert result["status"] == "answered" and result["subagent_id"] == "sa-1"
    assert out == {**out, "outcome": "answered", "answer": "Keep it."}


def test_another_conversation_cannot_answer(setup):
    from tools.ask_parent import pending_question

    result = _reply(setup.stranger, subagent_id="sa-1", request_id=setup.question.request_id, message="Break it.")

    assert "error" in result
    assert pending_question("sa-1") is not None


@pytest.mark.parametrize("missing", ["request_id", "message", "subagent_id"])
def test_a_reply_needs_the_child_the_question_and_the_answer(setup, missing):
    args = {"subagent_id": "sa-1", "request_id": setup.question.request_id, "message": "Keep it."}
    args.pop(missing)

    assert "error" in _reply(setup.parent, **args)


def test_a_reply_to_a_question_that_ended_is_refused(setup):
    from tools.ask_parent import wait_for_answer

    wait_for_answer(setup.question, timeout=0)  # timed out

    result = _reply(setup.parent, subagent_id="sa-1", request_id=setup.question.request_id, message="Keep it.")

    assert "error" in result and "no open question" in result["error"]


def test_list_shows_what_a_child_is_waiting_on(setup):
    from tools.delegate_tool import delegate_task

    listing = json.loads(delegate_task(action="list", parent_agent=setup.parent))
    (entry,) = listing["subagents"]

    assert entry["status"] == "awaiting_parent"
    assert entry["pending_question"]["request_id"] == setup.question.request_id
    assert entry["pending_question"]["question"] == "Keep the old API?"


def _schema(monkeypatch, enabled):
    from tools import delegate_tool, delegate_tool_config

    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {"ask_parent": {"enabled": enabled}})
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"ask_parent": {"enabled": enabled}})
    return delegate_tool._build_dynamic_schema_overrides()["parameters"]["properties"]


def test_the_schema_is_unchanged_while_the_feature_is_off(monkeypatch):
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    props = _schema(monkeypatch, enabled=False)

    assert props["action"] == DELEGATE_TASK_SCHEMA["parameters"]["properties"]["action"]
    assert props["message"] == DELEGATE_TASK_SCHEMA["parameters"]["properties"]["message"]
    assert "request_id" not in props


def test_the_schema_offers_reply_while_the_feature_is_on(monkeypatch):
    props = _schema(monkeypatch, enabled=True)

    assert "reply" in props["action"]["enum"]
    assert "reply" in props["action"]["description"]
    assert "request_id" in props
    assert "reply" in props["message"]["description"]


def test_a_one_shot_session_drops_request_id_with_the_other_live_controls(monkeypatch):
    from tools.delegate_tool import oneshot_delegate_function

    props = _schema(monkeypatch, enabled=True)
    function = oneshot_delegate_function({"name": "delegate_task", "description": "",
                                          "parameters": {"type": "object", "properties": props}})

    assert "request_id" not in function["parameters"]["properties"]


def test_the_live_dispatch_path_passes_the_request_id(monkeypatch):
    from run_agent import AIAgent
    from tools import delegate_tool

    seen = {}
    monkeypatch.setattr(delegate_tool, "delegate_task", lambda **kwargs: seen.update(kwargs) or "{}")
    agent = SimpleNamespace(_delegate_depth=0)

    AIAgent._dispatch_delegate_task(agent, {"action": "reply", "subagent_id": "sa-1", "request_id": "req-1",
                                            "message": "Keep it."})

    assert seen["request_id"] == "req-1" and seen["action"] == "reply"
