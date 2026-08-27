"""Tests for tui_gateway.entry_ws — the slim WS-only gateway server.

The auth/path handling MUST be exercised against the real, repo-pinned
``websockets`` package (15.x): its asyncio server passes a single
``ServerConnection`` to the handler (no second ``path`` argument), and the
request path lives on ``connection.request.path``. A fake adapter would have
masked exactly the signature mismatch that review F2 on PR #94245 flagged.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest

websockets = pytest.importorskip("websockets")

from tui_gateway import entry_ws


@pytest.fixture()
def ws_server():
    """Real websockets server running entry_ws._handle_connection on loopback.

    Runs in a dedicated event-loop thread so tests can drive real client
    connections with the synchronous websockets client API.
    """
    loop = asyncio.new_event_loop()
    started = threading.Event()
    state: dict = {}

    async def _serve():
        server = await websockets.serve(
            entry_ws._handle_connection, "127.0.0.1", 0
        )
        state["port"] = server.sockets[0].getsockname()[1]
        state["server"] = server
        started.set()
        await asyncio.Event().wait()

    def _run():
        asyncio.set_event_loop(loop)
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(_serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert started.wait(timeout=10), "ws server failed to start"
    yield state
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def _connect(port: int, query: str):
    """Open a real client connection; returns (ok, close_code).

    Deterministic accept/reject probe: an ACCEPTED connection is handed to
    handle_ws, which emits the ``gateway.ready`` event; a REJECTED one is
    closed with 4401. Waiting on recv() observes whichever happens — no race
    against the close frame (a ping can win that race and false-pass).
    """
    from websockets.sync.client import connect as sync_connect

    uri = f"ws://127.0.0.1:{port}/{query}"
    try:
        with sync_connect(uri, open_timeout=5, close_timeout=5) as client:
            first = client.recv(timeout=10)
            return "gateway.ready" in str(first), None
    except Exception as exc:  # ConnectionClosed carries the close code
        code = getattr(getattr(exc, "rcvd", None), "code", None)
        return False, code


def test_handshake_rejects_missing_token(ws_server):
    ok, code = _connect(ws_server["port"], "")
    assert not ok
    assert code == 4401


def test_handshake_rejects_wrong_token(ws_server):
    ok, code = _connect(ws_server["port"], "?token=definitely-wrong")
    assert not ok
    assert code == 4401


def test_handshake_accepts_valid_token_reading_path_from_v15_connection(ws_server):
    """The regression F2 guards against: under websockets 15.x the handler
    receives only a ServerConnection — the ?token= query MUST still be read
    (from ws.request.path). If the handler fell back to path='/', this valid
    token would be rejected exactly like the missing-token case above."""
    token = entry_ws.get_session_token()
    ok, code = _connect(ws_server["port"], f"?token={token}")
    assert ok, f"valid token rejected (close code {code}) — path not read from v15 connection"


def test_run_refuses_non_loopback_host():
    """F3: --ws-only uses static-token auth and must fail closed off-loopback."""
    with pytest.raises(SystemExit, match="loopback"):
        entry_ws.run(host="0.0.0.0", port=0)
    with pytest.raises(SystemExit, match="loopback"):
        entry_ws.run(host="192.168.1.10", port=0)


def test_run_allows_loopback_hosts_past_the_guard(monkeypatch):
    """Loopback names/addresses pass the guard (asyncio.run is stubbed so the
    server itself never starts)."""
    sentinel = RuntimeError("guard passed — reached server start")

    def _boom(_coro):
        _coro.close()
        raise sentinel

    monkeypatch.setattr(entry_ws.asyncio, "run", _boom)
    for host in ("127.0.0.1", "::1", "localhost"):
        with pytest.raises(RuntimeError, match="guard passed"):
            entry_ws.run(host=host, port=0)
