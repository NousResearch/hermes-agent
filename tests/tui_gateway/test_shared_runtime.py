"""The local owner handshake grants attachment only for its own live lease."""

import asyncio
import threading
from urllib.parse import urlsplit

import httpx
import pytest
from websockets.exceptions import InvalidStatus
from websockets.asyncio.client import connect

from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_cli.shared_session_attach import discover_attach_url
from hermes_state_registry import acquire, release
from tui_gateway import server, shared_runtime


async def _refuse_connection(url, **options):
    with pytest.raises(InvalidStatus):
        async with connect(url, proxy=None, open_timeout=3, close_timeout=1, **options):
            pytest.fail("unauthorized socket was accepted")


@pytest.fixture(autouse=True)
def _no_background_services(monkeypatch):
    # This harness owns only attachment and RPCs; watcher/probe/ticker lifetimes
    # belong to the production process and must not outlive its per-test homes.
    for name in ("_ensure_skin_watcher", "_ensure_lease_watcher",
                 "_start_backend_heartbeat_refresher", "_schedule_startup_orphan_sweep"):
        monkeypatch.setattr(server, name, lambda: None)


def test_owner_handshake_profile_and_lease_fences(tmp_path, monkeypatch, caplog):
    from tests.tui_gateway.test_queued_prompt_persistence import _desktop_session, _busy
    from agent import secret_scope
    from tests.tui_gateway.test_shared_session_delivery import Peer
    from gateway.browser_control_broker import get_browser_control_broker
    from hermes_cli.session_ingress import send_live_message

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)

    first_disconnect, release_disconnect = threading.Event(), threading.Event()
    departed = []
    broker = get_browser_control_broker()
    disconnect_owner = broker.disconnect_owner

    def hold_first_disconnect(transport):
        departed.append(transport)
        if len(departed) == 1:
            first_disconnect.set()
            assert release_disconnect.wait(10)
        return disconnect_owner(transport)

    monkeypatch.setattr(broker, "disconnect_owner", hold_first_disconnect)
    shared_runtime.start_shared_runtime()
    runtime_thread = shared_runtime._thread
    records = []
    try:
        for name in ("a", "b"):
            home = tmp_path / name
            monkeypatch.setenv("HERMES_HOME", str(home))
            db = acquire(home / "state.db")
            sid, key = _desktop_session(monkeypatch, db)
            session = server._sessions[sid]
            session["profile_home"] = str(home)
            with server._session_profile_runtime_scope(session):
                server._ensure_session_db_row(session)
                assert server._ensure_active_session_slot(sid, session) is None
            session["transport"] = Peer()  # persistent owner; CLI peers are temporary subscribers
            _busy(session)
            records.append((home, db, sid, key, session["active_session_lease"]))
        for index, (home, db, sid, key, lease) in enumerate((records[0], records[1], records[0])):
            monkeypatch.setenv("HERMES_HOME", str(home))
            owners = active_session_registry_snapshot(home)
            assert [o["session_id"] for o in owners] == [key]
            assert owners[0]["metadata"]["shared_runtime_url"] == shared_runtime.attach_origin()
            url = discover_attach_url(key)
            assert urlsplit(url).path == "/api/ws"
            asyncio.run(_refuse_connection(url, origin="https://example.com"))
            assert discover_attach_url(records[1 if home == records[0][0] else 0][3]) is None
            params = {"session_id": key, "lease_id": lease.lease_id, "profile_home": str(home.resolve())}
            with httpx.Client(trust_env=False) as client:
                for field in params:
                    response = client.get(shared_runtime.attach_origin() + "/api/session-attach",
                                          params={**params, field: "wrong"})
                    assert response.status_code == 409
                    assert "websocket_url" not in response.text
                for headers in ({"Origin": "https://example.com"}, {"Host": "example.com"}):
                    response = client.get(shared_runtime.attach_origin() + "/api/session-attach",
                                          params=params, headers=headers)
                    assert response.status_code == 403
                server._sessions[sid]["_closing"] = True
                assert client.get(shared_runtime.attach_origin() + "/api/session-attach",
                                  params=params).status_code == 409
                server._sessions[sid].pop("_closing")
            assert send_live_message(key, "profile-" + home.name, "queue") == 0
            rows = db.get_messages(key)
            assert rows and all("profile-" + home.name in r["content"] for r in rows)
            if index == 0:
                assert first_disconnect.wait(5)
                # Disconnect may still be cleaning up while B connects. The
                # departed socket must already be excluded from live fanout.
                assert departed[0].closed
            elif index == 1:
                release_disconnect.set()
            assert active_session_registry_snapshot(home)[0]["lease_id"] == lease.lease_id
    finally:
        release_disconnect.set()
        shared_runtime.stop_shared_runtime()
        assert not runtime_thread.is_alive()
        for home, db, sid, key, lease in records:
            server._sessions.pop(sid, None)
            lease.release()
            release(db)
    assert shared_runtime.attach_origin() is None
    assert "ws send failed" not in caplog.text
    assert "ws response send failed" not in caplog.text
    assert "timeout graceful shutdown exceeded" not in caplog.text


@pytest.mark.parametrize("options", [{}, {"origin": "https://example.com"}])
def test_local_websocket_requires_credential_and_native_origin(options):
    shared_runtime.start_shared_runtime()
    try:
        url = shared_runtime.attach_origin().replace("http:", "ws:") + "/api/ws?token=wrong"
        asyncio.run(_refuse_connection(url, **options))
    finally:
        shared_runtime.stop_shared_runtime()


def test_stop_keeps_runtime_identity_until_server_thread_has_exited(monkeypatch):
    from types import SimpleNamespace

    class ServerThread:
        alive = True
        joined = False

        def join(self, timeout):
            self.joined = True

        def is_alive(self):
            return self.alive

    owner, thread = SimpleNamespace(should_exit=False), ServerThread()
    origin = "http://127.0.0.1:1234"
    monkeypatch.setattr(shared_runtime, "_server", owner)
    monkeypatch.setattr(shared_runtime, "_thread", thread)
    monkeypatch.setattr(shared_runtime, "_origin", origin)
    with pytest.raises(RuntimeError, match="server thread did not stop"):
        shared_runtime.stop_shared_runtime()
    assert owner.should_exit and thread.joined
    assert (shared_runtime._server, shared_runtime._thread, shared_runtime._origin) == (owner, thread, origin)
    shared_runtime.start_shared_runtime()
    assert shared_runtime._thread is thread  # a timed-out stop must not permit a second server
    thread.alive = False
    shared_runtime.stop_shared_runtime()
    assert (shared_runtime._server, shared_runtime._thread, shared_runtime._origin) == (None, None, None)
