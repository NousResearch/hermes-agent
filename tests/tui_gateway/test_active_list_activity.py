"""``session.active_list`` carries what a working turn is doing right now.

Desktop paints that as the working row's subtitle. The session list carries only the durable
stamp, and no ``sessions.changed`` fires on an activity heartbeat, so without the live snapshot
the subtitle froze on the turn's first label ("starting new turn") for the whole tool call.
"""
import threading
import types

from tui_gateway import server


def _live(running, desc):
    agent = types.SimpleNamespace(
        model="m", session_id="key",
        get_activity_summary=lambda: {"last_activity_description": desc})
    return {"agent": agent, "session_key": "key", "history": [], "history_lock": threading.Lock(),
            "running": running, "created_at": 1.0, "last_active": 2.0}


def _rows(monkeypatch, session):
    monkeypatch.setattr(server, "_get_db", lambda: types.SimpleNamespace(get_session_title=lambda key: ""))
    previous = dict(server._sessions)
    server._sessions.clear()
    server._sessions["sid"] = session
    try:
        resp = server.handle_request({"id": "1", "method": "session.active_list", "params": {}})
    finally:
        server._sessions.clear()
        server._sessions.update(previous)
    return resp["result"]["sessions"]


def test_a_working_row_carries_the_agents_current_activity(monkeypatch):
    [row] = _rows(monkeypatch, _live(True, "executing tool: terminal"))
    assert (row["status"], row["activity"]) == ("working", "executing tool: terminal")


def test_an_isolated_turn_carries_the_label_its_compute_host_relayed(monkeypatch):
    # turn_isolation: the parent holds no agent; the child relays its label on compute_host.activity.
    session = {**_live(True, ""), "agent": None, "_compute_host_activity_desc": "tool running: terminal"}
    [row] = _rows(monkeypatch, session)
    assert row["activity"] == "tool running: terminal"


def test_an_idle_row_carries_no_activity(monkeypatch):
    [row] = _rows(monkeypatch, _live(False, "tool completed: terminal (3.0s)"))
    assert row["status"] == "idle" and "activity" not in row
