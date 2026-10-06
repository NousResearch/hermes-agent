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


def test_standalone_stdio_owner_survives_reapers_but_explicit_close_releases_it(tmp_path, monkeypatch):
    """Real stdio is already exempt from TTL/LRU; an ingress peer must not orphan it."""
    import io
    import time
    from hermes_state import SessionDB

    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", True)
    monkeypatch.setattr(server, "_real_stdout", io.StringIO())
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *args: None)
    monkeypatch.setattr(server, "_max_live_sessions", lambda: 1)
    with SessionDB(db_path=tmp_path / "state.db") as db:
        monkeypatch.setattr(server, "_get_db", lambda: db)

        def create():
            result = server.handle_request({"id": "create", "method": "session.create", "params": {
                "source": "tui"}})
            assert "error" not in result, result
            sid = result["result"]["session_id"]
            session = server._sessions[sid]
            session["agent_ready"].set()
            session["created_at"] = session["last_active"] = time.time() - server._SESSION_TTL_S - 10
            assert server._ensure_session_db_row(session)
            assert server._ensure_active_session_slot(sid, session) is None
            return sid, session

        sid, owner = create()
        lease = owner["active_session_lease"]
        assert owner["transport"] is server._stdio_transport
        assert not server._transport_is_dead(owner["transport"])
        server._reap_idle_sessions()
        assert server._sessions[sid] is owner and not lease.released

        peer = Peer()
        server._rebind_live_transport(sid, owner, peer)
        peer.close()
        assert server._close_sessions_for_transport(peer) == (0, 0)
        assert server._session_live_transports(owner) == [server._stdio_transport]
        assert sid not in server._pending_ws_reaps

        for sweep, reason in ((server._enforce_session_cap, "lru_evict"),
                              (server._reap_idle_sessions, "idle_timeout")):
            detached_id, detached = create()
            detached["transport"] = server._detached_ws_transport
            detached_lease = detached["active_session_lease"]
            sweep()
            assert detached_id not in server._sessions and detached_lease.released
            assert db.get_session(detached["session_key"])["end_reason"] == reason
            assert server._sessions[sid] is owner and not lease.released
        assert db.get_session(owner["session_key"])["ended_at"] is None
        response = server.handle_request({"id": "close", "method": "session.close", "params": {
            "session_id": sid}})
        assert "error" not in response, response
        assert sid not in server._sessions and lease.released
        assert db.get_session(owner["session_key"])["end_reason"] == "tui_close"
