"""Compute-host turns retain failures for reconnect/resume.

Regression coverage for issue #110106: a DeletedWalGenerationError that escapes
through the compute-host boundary must not discard the only in-flight failure
snapshot before the terminal frame is delivered.
"""

from __future__ import annotations

import io
import json
import threading
import types

from hermes_state_errors import DeletedWalGenerationError
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


class _FrameOutput(io.StringIO):
    def __init__(self):
        super().__init__()
        self.error_written = threading.Event()

    def write(self, text: str) -> int:
        written = super().write(text)
        if '"type":"turn.error"' in text:
            self.error_written.set()
        return written


def _frames(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]


def _wait_for_error(output: io.StringIO, timeout: float = 5.0) -> dict:
    assert isinstance(output, _FrameOutput)
    if output.error_written.wait(timeout):
        for frame in _frames(output):
            if frame.get("type") == "turn.error":
                return frame
    raise AssertionError(f"timed out waiting for turn.error; saw={_frames(output)}")


def _session() -> dict:
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "compute-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
    }


def _raise_deleted_wal(*_args, **_kwargs):
    raise DeletedWalGenerationError("deleted WAL generation")


def test_deleted_wal_failure_remains_replayable_after_compute_host_error(monkeypatch):
    """The child keeps the failed prompt/error snapshot after emitting turn.error."""
    output = _FrameOutput()
    host = ComputeHost(stdout=output, heartbeat_secs=0)
    sid = "compute-wal-failure"
    session = _session()
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        _raise_deleted_wal,
    )

    try:
        host.handle_frame({
            "type": "turn.start",
            "sid": sid,
            "request_id": "request-1",
            "prompt": "continue the task",
        })
        error_frame = _wait_for_error(output)

        with session["history_lock"]:
            retained = session["inflight_turn"]
            running = session["running"]

        assert error_frame["request_id"] == "request-1"
        assert error_frame["message"] == "deleted WAL generation"
        assert running is False
        assert retained is not None
        assert retained["assistant"] == ""
        assert retained["streaming"] is False
        assert retained["user"] == "continue the task"
        assert retained["error"] == "deleted WAL generation"
        assert retained["status"] == "error"
        assert retained["recoverable"] is True
    finally:
        server._sessions.pop(sid, None)
        host.close()


def test_compute_host_error_remains_replayable_in_parent_session_mirror(monkeypatch):
    """The parent keeps the failure when the child reports turn.error."""
    emitted = []
    sid = "compute-wal-parent"
    session = _session()
    server._start_inflight_turn(session, "continue the task")
    monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: emitted.append(
        (event, event_sid, payload)
    ))
    monkeypatch.setattr(server, "_compute_host_session_info", lambda _session: {})
    monkeypatch.setattr(server, "_apply_compute_host_metadata_mirror", lambda *_args: None)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: None)

    server._on_compute_host_turn_done(
        "request-1", sid, session,
        {"type": "turn.error", "request_id": "request-1", "message": "deleted WAL generation"},
    )

    with session["history_lock"]:
        retained = session["inflight_turn"]

    assert retained is not None
    assert retained["user"] == "continue the task"
    assert retained["error"] == "deleted WAL generation"
    assert retained["status"] == "error"
    assert retained["recoverable"] is True
    complete = [payload for event, event_sid, payload in emitted
                if event == "message.complete" and event_sid == sid]
    assert complete == [{
        "text": "Error: deleted WAL generation",
        "status": "error",
        "error": "deleted WAL generation",
        "recoverable": True,
    }]
