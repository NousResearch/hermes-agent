"""Todo snapshots bypass optional tool-progress display settings."""

import json

import tui_gateway.server as server


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
