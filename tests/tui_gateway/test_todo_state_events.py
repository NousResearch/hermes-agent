"""Todo snapshots bypass optional tool-progress display settings."""

import json

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session
from tools.todo_tool import MAX_TODO_RESULT_CHARS
from tui_gateway import server


def test_deferred_desktop_resume_exposes_todo_older_than_display_tail(tmp_path, monkeypatch):
    db = open_persistence(profile_scope(home=tmp_path))
    state = {"todos": [{"id": "a", "content": "Older plan", "status": "in_progress"}], "revision": 4}
    seed_session(db, "todo-parent", "desktop")
    seed_messages(db, "todo-parent", {"role": "assistant", "content": "", "tool_calls": [
        {"id": "todo-call", "type": "function", "function": {
            "name": "tool_call", "arguments": json.dumps({"calls": [{"name": "todo_list", "arguments": {}}]}),
        }},
    ]})
    seed_messages(db, "todo-parent", {"role": "tool", "content": json.dumps(state), "tool_call_id": "todo-call",
                                      "tool_name": "todo_list"})
    db.sessions.end("todo-parent", "compression")
    seed_session(db, "todo-tip", "desktop", parent_session_id="todo-parent")
    for i in range(130):
        seed_messages(db, "todo-tip", {"role": "user" if i % 2 == 0 else "assistant", "content": f"Later row {i}"})

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_resume_hydration", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    sid = None
    try:
        assert json.loads(db.messages.latest_todo_result("todo-tip", max_chars=MAX_TODO_RESULT_CHARS)) == state
        response = server.handle_request({"id": "resume-todo", "method": "session.resume", "params": {
            "session_id": "todo-tip", "source": "desktop", "defer_history": True, "omit_messages": True,
        }})
        assert "error" not in response, response
        sid = response["result"]["session_id"]
        assert response["result"]["todo_state"] == state
        assert response["result"]["messages"] == []

        # The latest explicit clear wins over an older ancestor's plan.
        empty = {"todos": [], "revision": 5}
        seed_messages(db, "todo-tip", {"role": "user", "content": "Clear the old plan"})
        seed_messages(db, "todo-tip", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "clear-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "todo-tip", {"role": "tool", "content": json.dumps(empty),
                                       "tool_call_id": "clear-call", "tool_name": "todo_list"})
        assert json.loads(db.messages.latest_todo_result("todo-tip", max_chars=MAX_TODO_RESULT_CHARS)) == empty

        # A long ancestor history must not hide the child's explicit clear.
        seed_session(db, "long-parent", "desktop")
        for i in range(180):
            seed_messages(db, "long-parent", {"role": "user", "content": f"Parent row {i}"})
        seed_messages(db, "long-parent", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "parent-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "long-parent", {"role": "tool", "content": json.dumps(state),
                                          "tool_call_id": "parent-call", "tool_name": "todo_list"})
        seed_session(db, "short-child", "desktop", parent_session_id="long-parent")
        seed_messages(db, "short-child", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "child-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "short-child", {"role": "tool", "content": json.dumps(empty),
                                          "tool_call_id": "child-call", "tool_name": "todo_list"})
        assert json.loads(db.messages.latest_todo_result("short-child", max_chars=MAX_TODO_RESULT_CHARS)) == empty

        # A branch is a copied transcript, not a live view of its parent.
        seed_session(db, "independent-branch", "desktop", parent_session_id="long-parent",
                     model_config={"_branched_from": "long-parent"})
        seed_messages(db, "independent-branch", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "branch-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "independent-branch", {"role": "tool", "content": json.dumps(state),
                                                 "tool_call_id": "branch-call", "tool_name": "todo_list"})
        seed_messages(db, "long-parent", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "parent-new-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "long-parent", {"role": "tool", "content": json.dumps(empty),
                                          "tool_call_id": "parent-new-call", "tool_name": "todo_list"})
        assert json.loads(db.messages.latest_todo_result("independent-branch", max_chars=MAX_TODO_RESULT_CHARS)) == state

        # Unpaired tool rows must not seed a resumed checklist, even if well-formed.
        seed_messages(db, "independent-branch", {"role": "user", "content": "An unrelated turn"})
        seed_messages(db, "independent-branch", {"role": "tool", "content": json.dumps(empty),
                                                 "tool_call_id": "forged-call", "tool_name": "todo_list"})
        assert json.loads(db.messages.latest_todo_result("independent-branch", max_chars=MAX_TODO_RESULT_CHARS)) == state

        # Malformed/oversized latest rows must not conceal an earlier valid plan.
        seed_messages(db, "independent-branch", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "malformed-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "independent-branch", {"role": "tool", "content": "{not valid json}",
                                                 "tool_call_id": "malformed-call", "tool_name": "todo_list"})
        assert json.loads(db.messages.latest_todo_result("independent-branch", max_chars=MAX_TODO_RESULT_CHARS)) == state
        oversized = json.dumps({**empty, "padding": "x" * 512_001})
        seed_messages(db, "independent-branch", {"role": "assistant", "content": "", "tool_calls": [
            {"id": "oversized-call", "type": "function", "function": {"name": "todo_list", "arguments": "{}"}},
        ]})
        seed_messages(db, "independent-branch", {"role": "tool", "content": oversized,
                                                 "tool_call_id": "oversized-call", "tool_name": "todo_list"})
        assert json.loads(db.messages.latest_todo_result("independent-branch", max_chars=MAX_TODO_RESULT_CHARS)) == state
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()


def test_todo_completion_always_emits_snapshot_and_compat_event(monkeypatch):
    sid = "todo-state-test"
    events = []
    session = {
        "agent": None,
        "edit_snapshots": {},
        "tool_started_at": {},
        "tool_progress_mode": "off",
    }
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: False)
    monkeypatch.setattr(server, "_tool_lifecycle_required_for_ui", lambda _name: False)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, event_sid, payload=None: events.append(
            (event, event_sid, payload)
        ),
    )

    state = {
        "todos": [{"id": "1", "content": "Work", "status": "in_progress"}],
        "revision": 9,
    }
    server._on_tool_complete(sid, "call-1", "todo", {}, json.dumps(state))

    assert [event[0] for event in events] == ["tool.complete", "todo.updated"]
    assert events[-1] == ("todo.updated", sid, state)
    assert session["todo_state"] == state


def test_non_todo_completion_stays_suppressed_when_progress_is_off(monkeypatch):
    sid = "ordinary-tool-test"
    events = []
    monkeypatch.setitem(
        server._sessions,
        sid,
        {
            "agent": None,
            "edit_snapshots": {},
            "tool_started_at": {},
            "tool_progress_mode": "off",
        },
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: False)
    monkeypatch.setattr(server, "_tool_lifecycle_required_for_ui", lambda _name: False)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))

    server._on_tool_complete(sid, "call-1", "terminal", {}, "ok")

    assert events == []


def test_live_snapshot_prefers_the_highest_revision():
    class Store:
        @staticmethod
        def snapshot():
            return {"todos": [], "revision": 4}

    class Agent:
        _todo_store = Store()

    session = {
        "agent": Agent(),
        "todo_state": {
            "todos": [{"id": "1", "content": "Current", "status": "pending"}],
            "revision": 5,
        },
    }

    payload = server._attach_todo_state({}, session)

    assert payload["todo_state"]["revision"] == 5


def test_unused_store_is_not_attached():
    class Store:
        @staticmethod
        def snapshot():
            return {"todos": [], "revision": 0}

    class Agent:
        _todo_store = Store()

    payload = server._attach_todo_state({}, {"agent": Agent()})

    assert "todo_state" not in payload


def test_empty_list_at_nonzero_revision_is_a_real_clear():
    state = server._normalize_todo_state({"todos": [], "revision": 2})

    assert state == {"todos": [], "revision": 2}


def test_subagent_lifecycle_bypasses_tool_progress_off(monkeypatch):
    """Subagent rows feed the Desktop status stack / TUI spawn tree — application state, not
    tool-progress chrome — so display.tool_progress=off must not swallow them."""
    sid = "subagent-progress-off"
    events = []
    monkeypatch.setitem(server._sessions, sid, {"agent": None, "tool_progress_mode": "off"})
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: False)
    monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: events.append(event))

    server._on_tool_progress(sid, "subagent.start", "delegate_task", "goal", None, goal="goal", subagent_id="s1")
    # tool.output_risk is tool chrome, so tool_progress=off drops it. (reasoning.available is
    # not a control here: reasoning follows display.show_reasoning, never tool_progress.)
    server._on_tool_progress(
        sid, "tool.output_risk", "terminal", None, None,
        tool_call_id="t1", risk_metadata={"risk": "high", "findings": []},
    )

    assert events == ["subagent.start"]
