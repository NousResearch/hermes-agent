"""start_flow() must surface the worker's real failure cause, not a blanket timeout.

The polling loop distinguishes a worker error (RuntimeError with the flow's
error) from a genuine deadline miss (TimeoutError); the error handlers around
it must preserve that distinction in what poll_flow() later reports.
"""

import threading

import pytest

from tui_gateway import mcp_oauth_sessions as sessions


def _start(home, server, **kwargs):
    return sessions.start_flow(home, server, {"url": "https://mcp.example"}, **kwargs)


def test_worker_failure_surfaces_real_error(tmp_path, monkeypatch):
    finished = threading.Event()

    def worker(session_id, *_args):
        rec = sessions._sessions[session_id]
        rec["flow"].mark_error("registration failed: invalid client metadata")
        rec["flow"].mark_worker_done()
        finished.set()

    monkeypatch.setattr(sessions, "_worker", worker)
    monkeypatch.setattr(sessions, "_sessions", {})
    home = str(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", home)

    with pytest.raises(RuntimeError, match="registration failed"):
        _start(home, "reports")

    sid = next(iter(sessions._sessions))
    rec = sessions._sessions[sid]
    assert finished.wait(5)
    out = sessions.poll_flow(sid, "reports")
    assert out["status"] == "error"
    assert out["error_message"] == "registration failed: invalid client metadata"
    assert "Timed out" not in out["error_message"]
    assert rec["httpd"] is None


def test_genuine_timeout_keeps_timeout_message(tmp_path, monkeypatch):
    finished = threading.Event()
    worker_exited = threading.Event()

    def worker(session_id, *_args):
        rec = sessions._sessions[session_id]
        finished.wait(10)
        rec["flow"].mark_worker_done()
        worker_exited.set()

    monkeypatch.setattr(sessions, "_worker", worker)
    monkeypatch.setattr(sessions, "_sessions", {})
    home = str(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", home)

    try:
        with pytest.raises(TimeoutError):
            _start(home, "reports", url_timeout=0.3)

        sid = next(iter(sessions._sessions))
        rec = sessions._sessions[sid]
        out = sessions.poll_flow(sid, "reports")
        assert out["status"] == "error"
        assert out["error_message"] == "Timed out waiting for MCP authorization URL"
        assert rec["httpd"] is None
    finally:
        finished.set()
        worker_exited.wait(5)


def test_empty_worker_error_uses_fallback_message(tmp_path, monkeypatch):
    finished = threading.Event()

    def worker(session_id, *_args):
        rec = sessions._sessions[session_id]
        rec["flow"].mark_error("")
        rec["flow"].mark_worker_done()
        finished.set()

    monkeypatch.setattr(sessions, "_worker", worker)
    monkeypatch.setattr(sessions, "_sessions", {})
    home = str(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", home)

    with pytest.raises(RuntimeError, match="MCP OAuth flow failed"):
        _start(home, "reports")

    assert finished.wait(5)
    sid = next(iter(sessions._sessions))
    out = sessions.poll_flow(sid, "reports")
    assert out["error_message"] == "MCP OAuth flow failed before authorization"
