"""A cold resume that fails in its background thread leaves its traceback in the log.

``_schedule_resume_hydration`` loads the transcript off the JSON-RPC path and hands the client only
``resume_failed_message(exc)``. Like the deferred agent build (#134890), it logged nothing, so a
failed resume left no frame in errors.log to say why.
"""

import logging
import sqlite3
import threading

from tui_gateway import server


def _error_records(caplog):
    return [r for r in caplog.records if r.name == server.logger.name and r.levelno >= logging.ERROR]


def test_a_failed_resume_hydration_logs_its_traceback(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=server.logger.name)
    session = {"history": [], "history_lock": threading.RLock(), "resume_hydrating": True,
               "resume_history_ready": threading.Event(), "agent_ready": threading.Event()}
    monkeypatch.setitem(server._sessions, "sid-resume", session)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    workers = []
    real_thread = threading.Thread

    def start_thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(server.threading, "Thread", start_thread)

    def _malformed(*_a, **_kw):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(server, "_load_resume_transcript", _malformed)

    server._schedule_resume_hydration("sid-resume", "stored-id", object())

    workers[0].join(timeout=10)
    assert not workers[0].is_alive()
    assert session["agent_ready"].is_set()
    assert session["resume_history_ready"].is_set()
    assert session["resume_hydrating"] is False
    assert "sid-resume" not in server._sessions
    assert "malformed" in session["resume_history_error"]
    assert emitted[-1] == ("error", "sid-resume", {"message": session["resume_history_error"]})
    [record] = _error_records(caplog)
    assert "sid-resume" in record.getMessage() and "stored-id" in record.getMessage()
    assert record.exc_info and record.exc_info[0] is sqlite3.DatabaseError
