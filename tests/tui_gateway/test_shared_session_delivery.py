"""Session observers retain output across attachment and disconnect."""
import threading

from tui_gateway import server


class Peer:
    def __init__(self):
        self.frames = []
        self._closed = False

    def write(self, frame):
        self.frames.append(frame)
        return not self._closed

    def close(self):
        self._closed = True


def test_reattach_preserves_terminal_delivery(monkeypatch):
    first, second = Peer(), Peer()
    session = {"transport": first, "history_lock": threading.Lock(), "running": True}
    monkeypatch.setitem(server._sessions, "shared", session)
    with session["history_lock"]:
        server._rebind_live_transport("shared", session, second)
    server._emit("message.complete", "shared", {"text": "finished"})
    assert first.frames == second.frames
    assert len(first.frames) == 1
    second.close()
    assert server._close_sessions_for_transport(second) == (0, 0)
    assert second not in session.get("viewers", {})
    server._emit("message.complete", "shared", {"text": "still attached"})
    assert len(first.frames) == 2
