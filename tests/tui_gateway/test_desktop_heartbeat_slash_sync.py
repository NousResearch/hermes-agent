"""Desktop must receive Heartbeat state immediately after slash setup."""

from __future__ import annotations


class _Worker:
    def __init__(self, output: str) -> None:
        self.output = output
        self.commands: list[str] = []

    def run(self, command: str) -> str:
        self.commands.append(command)
        return self.output


def test_worker_backed_heartbeat_slash_publishes_control_snapshot(monkeypatch):
    """A slash-created Heartbeat appears without leaving and re-entering its chat."""
    from tui_gateway import server

    sid = "heartbeat-session"
    session_key = "persistent-heartbeat-session"
    worker = _Worker("♥ Heartbeat set (every 1m): Reply HEARTBEAT_OK")
    snapshot = {
        "goal": None,
        "heartbeat": {
            "created_at": 1_700_000_000,
            "fire_count": 0,
            "interval_seconds": 60,
            "last_fired_at": 1_700_000_000,
            "prompt": "Reply HEARTBEAT_OK",
            "status": "active",
        },
        "loop": None,
        "revision": "heartbeat-created",
        "updated_at": 1_700_000_000,
    }
    emitted: list[tuple[str, str, dict]] = []
    session = {
        "agent": None,
        "running": False,
        "session_key": session_key,
        "slash_worker": worker,
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_snapshot_control", lambda key: snapshot if key == session_key else None)
    monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: emitted.append((event, event_sid, payload)))

    try:
        response = server._methods["slash.exec"](
            "request-1", {"command": "/heartbeat every 1m Reply HEARTBEAT_OK", "session_id": sid}
        )
    finally:
        server._sessions.pop(sid, None)

    assert response["result"]["output"] == worker.output
    assert worker.commands == ["/heartbeat every 1m Reply HEARTBEAT_OK"]
    assert emitted == [("session.control.update", sid, {"control": snapshot})]
