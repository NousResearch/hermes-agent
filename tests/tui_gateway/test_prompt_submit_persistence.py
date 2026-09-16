import contextlib
import threading

from agent.context_compressor import _DB_PERSISTED_MARKER
from tui_gateway import server


class _RecordingDb:
    def __init__(self):
        self.calls = []

    def append_messages_batch(self, session_id, messages):
        self.calls.append((session_id, messages))


def test_prompt_submit_persists_user_before_deferred_agent_build(monkeypatch):
    """Regression for #111868: an accepted first send survives process loss while the agent is still building."""
    db = _RecordingDb()
    session = {
        "attached_images": [],
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-first-turn",
    }
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_session_db", lambda _session: contextlib.nullcontext(db))

    assert server._persist_session_row_for_submit(1, session, "keep this message") is None
    assert db.calls == [
        (
            "stored-first-turn",
            [
                {
                    "role": "user",
                    "content": "keep this message",
                    "timestamp": session["_prepersisted_user_message"]["timestamp"],
                }
            ],
        )
    ]
    assert session["_prepersisted_user_message"][_DB_PERSISTED_MARKER] is True


def test_compute_host_frame_carries_the_durable_user_marker():
    marker = {
        "role": "user",
        "content": "keep this message",
        "timestamp": 123.0,
        _DB_PERSISTED_MARKER: True,
    }
    session = {
        "_prepersisted_user_message": marker,
        "attached_images": [],
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": "stored-first-turn",
    }
    frame = server._compute_host_turn_frame("rid", "sid", session, "keep this message")

    assert frame["prepersisted_user_message"] == marker


def test_compute_host_dispatch_transfers_the_durable_user_marker(monkeypatch):
    marker = {
        "role": "user",
        "content": "first turn",
        "timestamp": 123.0,
        _DB_PERSISTED_MARKER: True,
    }
    session = {
        "_prepersisted_user_message": marker,
        "attached_images": [],
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": "stored-first-turn",
    }
    sent = []

    class _Supervisor:
        def submit_turn(self, frame, *, on_complete):
            sent.append(frame)

    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg: _Supervisor())

    response = server._submit_prompt_to_compute_host("rid", "sid", session, "first turn")

    assert response["result"]["status"] == "streaming"
    assert sent[0]["prepersisted_user_message"] == marker
    assert "_prepersisted_user_message" not in session
    assert "prepersisted_user_message" not in server._compute_host_turn_frame(
        "next", "sid", session, "queued second turn"
    )


def test_compute_host_dispatch_failure_restores_the_durable_user_marker(monkeypatch):
    marker = {
        "role": "user",
        "content": "fall back inline",
        "timestamp": 123.0,
        _DB_PERSISTED_MARKER: True,
    }
    session = {
        "_prepersisted_user_message": marker,
        "attached_images": [],
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": "stored-first-turn",
    }

    class _FailingSupervisor:
        def submit_turn(self, frame, *, on_complete):
            raise OSError("pipe closed")

    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg: _FailingSupervisor())

    response = server._submit_prompt_to_compute_host("rid", "sid", session, "fall back inline")

    assert response["error"]["code"] == 5019
    assert session["_prepersisted_user_message"] is marker
