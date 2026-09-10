"""Session observers retain output across attachment and disconnect."""
import threading

from tui_gateway import server


class Peer:
    def __init__(self):
        self.frames = []
        self.received = threading.Event()
        self._closed = False

    def write(self, frame):
        self.frames.append(frame)
        self.received.set()
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
    assert first.received.wait(timeout=5)
    assert second.received.wait(timeout=5)
    assert first.frames == second.frames
    assert len(first.frames) == 1
    second.close()
    assert server._close_sessions_for_transport(second) == (0, 0)
    assert second not in session.get("viewers", {})
    first.received.clear()
    server._emit("message.complete", "shared", {"text": "still attached"})
    assert first.received.wait(timeout=5)
    assert len(first.frames) == 2
