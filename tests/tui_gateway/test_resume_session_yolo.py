"""A session /yolo (TUI Shift+Tab, the Desktop zap) survives a backend restart: persisted on the session
row and re-armed by ``session.resume`` in a process whose in-memory approval set starts empty."""

import types

from tui_gateway import server


def test_session_yolo_toggle_survives_a_backend_restart(monkeypatch, tmp_path):
    """A session /yolo (Shift+Tab / the Desktop zap) is persisted on the row and re-armed when a fresh
    backend resumes it, including a toggle made before the row existed; switching it off persists too."""
    from hermes_state import SessionDB
    from tools.approval import clear_session, is_session_yolo_enabled

    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    try:
        # Toggled before the first message: the lazily created row must still record it.
        server._sessions["sid"] = {"agent": types.SimpleNamespace(), "session_key": "row-a"}
        server.handle_request({"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "yolo"}})
        assert server._ensure_session_db_row(server._sessions["sid"]) is not False
        assert SessionDB.session_yolo_enabled(db.get_session("row-a"))

        server._sessions.clear()
        clear_session("row-a")  # a new backend process starts with an empty in-memory set
        resp = server.handle_request({"id": "2", "method": "session.resume", "params": {"session_id": "row-a"}})
        assert resp["result"]["session_key"] == "row-a"
        assert is_session_yolo_enabled("row-a") is True

        server.handle_request(
            {"id": "3", "method": "config.set", "params": {"session_id": resp["result"]["session_id"], "key": "yolo"}})
        assert is_session_yolo_enabled("row-a") is False
        assert not SessionDB.session_yolo_enabled(db.get_session("row-a"))
    finally:
        server._sessions.clear()
        clear_session("row-a")
        db.close()
