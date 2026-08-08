"""Regression tests for sessions viewed by multiple live transports (#81286)."""

from __future__ import annotations

from tui_gateway import server


class _Transport:
    def __init__(self, *, result: bool = True) -> None:
        self.frames: list[dict] = []
        self.result = result

    def write(self, frame: dict) -> bool:
        self.frames.append(frame)
        return self.result

    def close(self) -> None:
        return None


def test_session_events_are_fanned_out_to_all_attached_transports(monkeypatch):
    first = _Transport()
    second = _Transport()
    session = {"transport": first, "transports": {first, second}}
    previous = dict(server._sessions)
    try:
        server._sessions.clear()
        server._sessions["sid"] = session
        frame = server._event_frame("agent.delta", "sid", {"text": "hello"})

        assert server.write_json(frame) is True
        assert first.frames == [frame]
        assert second.frames == [frame]
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_disconnect_detaches_one_peer_without_tearing_down_shared_session(monkeypatch):
    first = _Transport()
    second = _Transport()
    session = {
        "transport": second,
        "transports": {first, second},
        "close_on_disconnect": True,
    }
    previous = dict(server._sessions)
    closed: list[str] = []
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, **kwargs: closed.append(sid),
    )
    try:
        server._sessions.clear()
        server._sessions["sid"] = session

        assert server._close_sessions_for_transport(first) == (0, 0)
        assert session["transports"] == {second}
        assert session["transport"] is second
        assert closed == []
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_detached_session_events_are_dropped_instead_of_falling_back_to_stdio():
    session = {"transport": server._detached_ws_transport, "transports": set()}
    previous = dict(server._sessions)
    try:
        server._sessions.clear()
        server._sessions["sid"] = session
        frame = server._event_frame("agent.delta", "sid")

        assert server.write_json(frame) is False
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_failed_secondary_does_not_hide_event_from_healthy_peer():
    failed = _Transport(result=False)
    healthy = _Transport()
    session = {"transport": failed, "transports": {failed, healthy}}
    previous = dict(server._sessions)
    try:
        server._sessions.clear()
        server._sessions["sid"] = session
        frame = server._event_frame("agent.delta", "sid")

        assert server.write_json(frame) is True
        assert failed.frames == [frame]
        assert healthy.frames == [frame]
    finally:
        server._sessions.clear()
        server._sessions.update(previous)
