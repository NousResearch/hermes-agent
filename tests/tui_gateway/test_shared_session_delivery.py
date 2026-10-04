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


def test_rpc_stdio_remains_attached_when_queue_peer_leaves(monkeypatch):
    """The real TUI channel must survive an ephemeral queue client, while serve's log sink stays excluded."""
    import io
    import json
    from tui_gateway.ws import WSTransport
    from tui_gateway.transport import FanoutTransport

    monkeypatch.setattr(server, "_stdio_is_rpc_channel", True)
    peer = WSTransport.__new__(WSTransport)
    peer._closed = False
    session = {"transport": server._stdio_transport, "history_lock": threading.Lock()}
    monkeypatch.setitem(server._sessions, "stdio-owner", session)
    server._rebind_live_transport("stdio-owner", session, peer)
    assert isinstance(session["transport"], FanoutTransport)
    assert session["transport"].contains(server._stdio_transport)
    assert server._session_client_answers_requests("stdio-owner")
    output, frames = io.StringIO(), []
    monkeypatch.setattr(server, "_real_stdout", output)
    monkeypatch.setattr(server, "_live_transports", {peer})
    peer.write = lambda frame: frames.append(frame) or True
    server._broadcast_global_event("skin.changed", {"name": "default"})
    assert json.loads(output.getvalue()) == frames[0]
    peer._closed = True
    assert server._detach_session_transport(session, peer)
    assert server._session_has_live_transport(session)
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", False)
    assert not server._transport_is_live_peer(server._stdio_transport)
