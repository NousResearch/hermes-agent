"""Deferred builds retain diagnostics when resumed history has already failed."""

import logging
import threading
from unittest.mock import Mock

from tui_gateway import server
from tui_gateway.user_messages import agent_init_failed_message


def test_history_failure_is_logged_before_agent_construction(monkeypatch, tmp_path):
    sid = "history-failure"
    message = "transcript load failed"
    history_ready = threading.Event()
    history_ready.set()
    session = {
        "session_key": "stored-session",
        "agent": None,
        "agent_ready": threading.Event(),
        "resume_history_ready": history_ready,
        "resume_history_error": message,
    }
    monkeypatch.setitem(server._sessions, sid, session)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    make_agent = Mock(side_effect=AssertionError("failed history must prevent construction"))
    monkeypatch.setattr(server, "_make_agent", make_agent)
    log_path = tmp_path / "history.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    server.logger.addHandler(handler)
    try:
        # Exercise the real history-wait helper and the real build worker.
        server._start_agent_build(sid, session)
        worker = session["_agent_build_thread"]
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        server.logger.removeHandler(handler)
        handler.close()

    make_agent.assert_not_called()
    assert session["agent_ready"].is_set()
    assert session["agent"] is None
    assert session["agent_error"] == message
    assert emitted == [("error", sid, {"message": agent_init_failed_message(RuntimeError(message))})]
    diagnostic = log_path.read_text(encoding="utf-8")
    assert sid in diagnostic
    assert "Traceback (most recent call last):" in diagnostic
    assert "in _await_resume_history" in diagnostic
    assert "RuntimeError: " + message in diagnostic
