"""hermes-ws-recovery-v1 loopback scenarios for transport-ownership expiry.

These drive the real ``handle_ws`` over a FastAPI loopback to prove the
server-side ownership contract end to end:

* a renderer that released its socket no longer blocks a replacement from
  reclaiming the live stream, and
* a renderer still actively driving the stream is protected from a second
  client stealing it (JSON-RPC 4091).

The stale-busy-state reconciliation is a separate slice, so nothing here
asserts a reconciled ``running`` flag.
"""

import threading
import time
import types

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.ws import handle_ws


def _rpc(ws, request_id: str, method: str, params: dict) -> dict:
    ws.send_json(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
    )
    return ws.receive_json()


def _idle_live_session(tmp_path, stored_id: str, prompt: str, final: str):
    session = {
        "agent": types.SimpleNamespace(model="loopback-model"),
        "cols": 80,
        "created_at": time.time(),
        "cwd": str(tmp_path),
        "display_history_prefix": [],
        "history": [],
        "history_lock": threading.Lock(),
        "inflight_turn": None,
        "last_active": time.time(),
        "running": False,
        "session_key": stored_id,
        "source": "desktop",
    }
    return session


def _app():
    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await handle_ws(ws)

    return app


def test_reconnect_reclaims_released_stream(monkeypatch, tmp_path):
    """A replacement socket reclaims a stream whose owner already disconnected."""
    db = SessionDB(tmp_path / "state.db")
    stored_id = "20260101_010101_a1b2c3"
    runtime_id = "runtime1"
    prompt = "return the persisted result"
    final = "the persisted result"

    db.create_session(stored_id, source="desktop")
    db.append_message(stored_id, role="user", content=prompt)
    db.append_message(stored_id, role="assistant", content=final, finish_reason="stop")

    session = _idle_live_session(tmp_path, stored_id, prompt, final)

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent: {"model": "loopback-model", "desktop_contract": 1},
    )
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    server._sessions.clear()
    server._live_transports.clear()
    server._sessions[runtime_id] = session

    try:
        with TestClient(_app()) as client:
            # First renderer owns the live stream, then drops its socket.
            with client.websocket_connect("/ws") as first:
                ready = first.receive_json()
                assert ready["params"]["payload"]["heartbeat"] is True
                activated = _rpc(
                    first,
                    "activate-first",
                    "session.activate",
                    {"session_id": runtime_id},
                )
                assert "error" not in activated

            # A replacement socket opens and reclaims the stream without a 4091 —
            # the released owner no longer blocks the rebind.
            with client.websocket_connect("/ws") as replacement:
                replacement.receive_json()  # gateway.ready
                recovered = _rpc(
                    replacement,
                    "activate-replacement",
                    "session.activate",
                    {"session_id": runtime_id},
                )
                assert "error" not in recovered
                result = recovered["result"]

        roles = [message["role"] for message in result["messages"]]
        assert roles == ["user", "assistant"]
        assert [message["text"] for message in result["messages"]] == [prompt, final]
    finally:
        server._sessions.clear()
        server._live_transports.clear()
        db.close()


def test_recently_live_owner_is_protected_from_steal(monkeypatch, tmp_path):
    """A second renderer cannot steal a stream the first is actively driving."""
    db = SessionDB(tmp_path / "state.db")
    stored_id = "20260101_020202_d4e5f6"
    runtime_id = "runtime2"
    prompt = "hold the stream"

    db.create_session(stored_id, source="desktop")
    db.append_message(stored_id, role="user", content=prompt)

    session = _idle_live_session(tmp_path, stored_id, prompt, prompt)

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent: {"model": "loopback-model", "desktop_contract": 1},
    )
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    server._sessions.clear()
    server._live_transports.clear()
    server._sessions[runtime_id] = session

    try:
        with TestClient(_app()) as client:
            with client.websocket_connect("/ws") as first:
                first.receive_json()  # gateway.ready
                owned = _rpc(
                    first,
                    "activate-first",
                    "session.activate",
                    {"session_id": runtime_id},
                )
                assert "error" not in owned

                # A concurrent renderer opens while the first is still recently
                # live and tries to take over the same session.
                with client.websocket_connect("/ws") as contender:
                    contender.receive_json()  # gateway.ready
                    stolen = _rpc(
                        contender,
                        "activate-contender",
                        "session.activate",
                        {"session_id": runtime_id},
                    )
                    assert stolen["error"]["code"] == 4091
    finally:
        server._sessions.clear()
        server._live_transports.clear()
        db.close()
