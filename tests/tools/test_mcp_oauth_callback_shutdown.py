"""An idle OAuth callback connection must not freeze the MCP event loop (#149771).

The carrier PR bounds socket reads, but HTTPServer.shutdown() still waits for the
handler. Hold real listener cleanup open to prove unrelated loop work can proceed.
"""
import asyncio
import io
import socket
import threading
from http.client import HTTPConnection

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

import tools.mcp_oauth as oauth


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_callback_shutdown_does_not_block_mcp_event_loop(monkeypatch):
    monkeypatch.setattr(oauth.sys, "stdin", io.StringIO())
    port = _free_port()
    listening = threading.Event()
    shutdown_entered = threading.Event()
    loop_pulsed = threading.Event()
    release_shutdown = threading.Event()
    loop_ref = {}
    outcome = {}
    original_start = oauth._start_callback_server

    def instrumented_start(callback_port, handler):
        server = original_start(callback_port, handler)
        original_shutdown = server.shutdown

        def delayed_shutdown():
            shutdown_entered.set()
            # This callback cannot run while shutdown() monopolizes the loop.
            loop_ref["loop"].call_soon_threadsafe(loop_pulsed.set)
            assert release_shutdown.wait(8), "test failed to release shutdown"
            original_shutdown()

        server.shutdown = delayed_shutdown
        listening.set()
        return server

    monkeypatch.setattr(oauth, "_start_callback_server", instrumented_start)

    def run_waiter():
        async def run():
            loop_ref["loop"] = asyncio.get_running_loop()
            with oauth.force_interactive_oauth():
                return await oauth._make_callback_waiter(port, timeout=10)()

        try:
            outcome["result"] = asyncio.run(run())
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run_waiter, daemon=True)
    worker.start()
    try:
        assert listening.wait(5), "OAuth callback server never bound"
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/callback?code=synthetic&state=s1")
            response = conn.getresponse()
            assert response.status == 200
            response.read()
        finally:
            conn.close()

        assert shutdown_entered.wait(5), "OAuth waiter never reached shutdown"
        assert loop_pulsed.wait(2), "HTTPServer.shutdown() blocked the MCP event loop"
    finally:
        release_shutdown.set()
        worker.join(timeout=8)

    assert not worker.is_alive(), "OAuth waiter did not finish"
    assert "error" not in outcome, f"OAuth waiter raised: {outcome.get('error')!r}"
    assert (outcome["result"].code, outcome["result"].state) == ("synthetic", "s1")
