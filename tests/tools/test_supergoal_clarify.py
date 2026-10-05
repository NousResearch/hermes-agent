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
    # Use the production writer so restriction policy follows the goal lifecycle.
    payload = {"goal": "Finish independently", "status": status}
    if mode is not None:
        payload["mode"] = mode
    goals.save_goal(session_id, goals.GoalState.from_json(json.dumps(payload)))


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
    store_goal(goal_db, agent.session_id)
    goal_db.set_meta("goal:broken", bad_row)
    result = invoke("inline", agent, {"question": "Choose?"})
    assert calls == []
    assert "error" in result


@pytest.mark.parametrize("path", ["inline", "registry"])
@pytest.mark.parametrize("mode", [None, "goal", "supergoal"])
def test_unavailable_sessiondb_does_not_disable_ordinary_clarify(goal_db, monkeypatch, path, mode):
    calls = []
    agent = SimpleNamespace(session_id="unavailable", clarify_callback=lambda *a, **k: calls.append(a) or "answer")
    if mode:
        store_goal(goal_db, agent.session_id, mode=mode)
    monkeypatch.setattr(goals, "_get_session_db", lambda: None)
    result = invoke(path, agent, {"question": "Choose?"})
    if mode == "supergoal":
        assert calls == [] and "error" in result
    else:
        assert len(calls) == 1 and result["user_response"] == "answer"


@pytest.mark.parametrize("path", ["inline", "registry"])
def test_cold_ordinary_session_needs_no_database_bootstrap(tmp_path, monkeypatch, path):
    from hermes_constants import get_hermes_home

    home = tmp_path / "never-started"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    bootstraps = []
    monkeypatch.setattr(goals, "_get_session_db", lambda: bootstraps.append(True))
    try:
        agent = SimpleNamespace(session_id="new", clarify_callback=lambda *a, **k: "answer")
        assert invoke(path, agent, {"question": "Choose?"})["user_response"] == "answer"
        assert not bootstraps
        assert not (get_hermes_home() / "state.db").exists()
    finally:
        reset_hermes_home_override(token)


def test_compressed_supergoal_keeps_guard_on_cached_agent(goal_db, monkeypatch):
    mgr = goals.GoalManager("before-compression")
    mgr.set("Finish independently", mode="supergoal", max_turns=13)
    agent = SimpleNamespace(session_id=mgr.session_id, clarify_callback=lambda *a, **k: "answer")
    assert goals.migrate_goal_to_session(agent.session_id, "after-compression", reason="compression")
    agent.session_id = "after-compression"
    cold = goals.GoalManager(agent.session_id)
    assert cold.state.mode == "supergoal" and cold.state.max_turns == 13
    assert goals.load_goal(mgr.session_id).status == "cleared"
    args = {"question": "Choose?"}
    for action, blocked in [(None, True), (cold.pause, False), (cold.resume, True), (cold.clear, False)]:
        if action:
            action()
        with monkeypatch.context() as unavailable:
            unavailable.setattr(goals, "_get_session_db", lambda: None)
            for path in ("inline", "registry"):
                result = invoke(path, agent, args)
                assert ("error" in result) is blocked
                if not blocked:
                    assert result["user_response"] == "answer"


@pytest.mark.parametrize("mode", [None, "goal", "supergoal"])
def test_fresh_process_policy_read_is_authoritative_and_never_mutates_schema(goal_db, mode):
    import os
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path
    from hermes_constants import get_hermes_home

    if mode:
        store_goal(goal_db, "cold", mode=mode)
    db_path = get_hermes_home() / "state.db"
    with sqlite3.connect(db_path) as conn:
        before = conn.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
        version = conn.execute("PRAGMA user_version").fetchone()
        rows = conn.execute("SELECT * FROM state_meta ORDER BY key").fetchall()
    script = '''
import json
from hermes_cli import goals
from tools.clarify_tool import clarify_tool
# An unavailable writer must not be needed even in a fresh interpreter.
goals._get_session_db = lambda: None
calls = []
result = json.loads(clarify_tool("Choose?", callback=lambda *a, **k: calls.append(a) or "answer", session_id="cold"))
print(json.dumps({"result": result, "calls": len(calls)}))
'''
    probe = subprocess.run([sys.executable, "-c", script],
                           cwd=Path(__file__).resolve().parents[2],
                           env={**os.environ, "HERMES_HOME": str(get_hermes_home())},
                           capture_output=True, text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr
    output = json.loads(probe.stdout)
    assert output["calls"] == (0 if mode == "supergoal" else 1)
    assert ("error" in output["result"]) is (mode == "supergoal")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall() == before
        assert conn.execute("PRAGMA user_version").fetchone() == version
        assert conn.execute("SELECT * FROM state_meta ORDER BY key").fetchall() == rows


@pytest.mark.parametrize("raw", [b"not a SQLite database", b""])
def test_unmarked_session_ignores_broken_database(tmp_path, raw):
    home = tmp_path / "unreadable"
    home.mkdir()
    (home / "state.db").write_bytes(raw)
    token = set_hermes_home_override(str(home))
    try:
        agent = SimpleNamespace(session_id="unknown", clarify_callback=lambda *a, **k: "answer")
        assert invoke("inline", agent, {"question": "Choose?"})["user_response"] == "answer"
        assert (home / "state.db").read_bytes() == raw
    finally:
        reset_hermes_home_override(token)
