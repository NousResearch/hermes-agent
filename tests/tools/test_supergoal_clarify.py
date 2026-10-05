"""Supergoal policy at the actual inline and registry clarification boundaries."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from hermes_cli import goals
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.clarify_tool import CLARIFY_SCHEMA
from tools.registry import registry


@pytest.fixture
def goal_db(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(str(home))
    try:
        yield goals._get_session_db()
    finally:
        reset_hermes_home_override(token)


def store_goal(db, session_id, *, mode="supergoal", status="active"):
    # Persist the wire representation, including compatibility with old rows without mode.
    payload = {"goal": "Finish independently", "status": status}
    if mode is not None:
        payload["mode"] = mode
    db.set_meta(f"goal:{session_id}", json.dumps(payload))


def invoke(path, agent, args):
    if path == "inline":
        return json.loads(INLINE_TOOL_EXECUTORS["clarify"](
            agent, args, InlineToolContext(effective_task_id="different-resource-id")))
    result = registry.dispatch(
        "clarify", args, session_id=agent.session_id, callback=agent.clarify_callback)
    return json.loads(result) if isinstance(result, str) else result


@pytest.mark.parametrize("path", ["inline", "registry"])
@pytest.mark.parametrize("args", [
    {"question": "Which route?"},
    {"questions": [{"question": "Which route?"}, {"question": "Which file?"}]},
])
def test_active_supergoal_rejects_before_any_ui_callback(goal_db, path, args):
    calls = []
    agent = SimpleNamespace(session_id="autonomous", clarify_callback=lambda *a, **k: calls.append(a))
    store_goal(goal_db, agent.session_id)
    schema_before = deepcopy(CLARIFY_SCHEMA)
    result = invoke(path, agent, args)
    assert calls == []
    assert "supergoal" in result["error"].lower()
    assert "user_response" not in result
    assert "timed_out" not in result
    assert CLARIFY_SCHEMA == schema_before


@pytest.mark.parametrize("path", ["inline", "registry"])
def test_cached_agent_observes_current_mode_and_lifecycle(goal_db, path):
    calls = []

    def callback(*args, **kwargs):
        calls.append(args)
        return "chosen"

    agent = SimpleNamespace(session_id="reused", clarify_callback=callback)
    args = {"question": "Choose?", "session_id": "spoofed-other-session"}
    for mode, status, blocked in [
        (None, "active", False),
        ("goal", "active", False),
        ("supergoal", "active", True),
        ("supergoal", "paused", False),
        ("supergoal", "active", True),
        ("supergoal", "done", False),
        ("supergoal", "cleared", False),
        ("goal", "active", False),
    ]:
        store_goal(goal_db, agent.session_id, mode=mode, status=status)
        before = len(calls)
        result = invoke(path, agent, args)
        assert len(calls) == before + (not blocked)
        assert ("error" in result) is blocked
    agent.session_id = "no-goal"
    assert invoke(path, agent, args)["user_response"] == "chosen"


def test_policy_is_scoped_to_profile_and_conversation(goal_db, tmp_path, monkeypatch):
    from agent import secret_scope

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    store_goal(goal_db, "same-id")
    store_goal(goal_db, "other-id", mode="goal")
    a = SimpleNamespace(session_id="same-id", clarify_callback=lambda *a, **k: "answer")
    b = SimpleNamespace(session_id="other-id", clarify_callback=a.clarify_callback)
    args = {"question": "Choose?"}
    assert "error" in invoke("inline", a, args)
    assert invoke("inline", b, args)["user_response"] == "answer"
    second_home = tmp_path / "second-home"
    second_home.mkdir()
    token = set_hermes_home_override(str(second_home))
    try:
        second_db = goals._get_session_db()
        store_goal(second_db, "same-id", mode="goal")
        assert invoke("inline", a, args)["user_response"] == "answer"
    finally:
        reset_hermes_home_override(token)
    assert "error" in invoke("inline", a, args)


@pytest.mark.parametrize("bad_row", ["{invalid-json", "[]"])
def test_unreadable_goal_policy_does_not_open_a_question(goal_db, bad_row):
    calls = []
    agent = SimpleNamespace(session_id="broken", clarify_callback=lambda *a, **k: calls.append(a))
    goal_db.set_meta("goal:broken", bad_row)
    result = invoke("inline", agent, {"question": "Choose?"})
    assert calls == []
    assert "error" in result


def test_unavailable_database_does_not_open_a_question(goal_db, monkeypatch):
    calls = []
    agent = SimpleNamespace(session_id="unavailable", clarify_callback=lambda *a, **k: calls.append(a))
    monkeypatch.setattr(goals, "_get_session_db", lambda: None)
    result = invoke("registry", agent, {"question": "Choose?"})
    assert calls == []
    assert "error" in result
