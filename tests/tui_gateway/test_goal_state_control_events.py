"""Goal criteria/gates land mid-turn, so the tool-completion choke point republishes the session-control
snapshot. ``session.control.update`` is otherwise published only on slash dispatch and after a turn
completes, so without this a client painting the goal card from that event keeps the pre-tool criteria
count for the whole rest of the turn."""

import tui_gateway.server as server


def _session():
    return {"agent": None, "edit_snapshots": {}, "tool_started_at": {}, "tool_progress_mode": "off"}


def _capture(monkeypatch, published):
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: False)
    monkeypatch.setattr(server, "_tool_lifecycle_required_for_ui", lambda _name: False)
    monkeypatch.setattr(server, "_emit", lambda *args: None)
    monkeypatch.setattr(
        server,
        "_publish_session_control_snapshot",
        lambda sid, session, *, only_if_present=False: published.append((sid, session, only_if_present)),
    )


def test_goal_state_tool_completion_republishes_control_snapshot(monkeypatch):
    """INVARIANT: a tool that can mutate the active goal mid-turn republishes the control snapshot."""
    sid = "goal-state-tool"
    session = _session()
    published = []
    monkeypatch.setitem(server._sessions, sid, session)
    _capture(monkeypatch, published)

    server._on_tool_complete(sid, "call-1", "goal_criteria", {"action": "add_criteria"}, "Added 6 criterion/criteria to the active goal.")

    assert published == [(sid, session, True)]


def test_tool_completions_that_cannot_touch_goal_state_stay_silent(monkeypatch):
    """INVARIANT: no control frame for an ordinary tool, and none when the session is already gone."""
    published = []
    _capture(monkeypatch, published)
    monkeypatch.setitem(server._sessions, "ordinary-tool", _session())

    server._on_tool_complete("ordinary-tool", "call-1", "terminal", {"command": "ls"}, "ok")
    server._on_tool_complete("missing-session", "call-2", "goal_criteria", {"action": "show"}, "No active goal.")

    assert published == []
