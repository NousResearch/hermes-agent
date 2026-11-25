"""An idle OAuth callback connection must not freeze the MCP event loop (#149771).

Listener cleanup (shutdown, then server_close) runs on a worker thread. Hold real listener
cleanup open to prove unrelated loop work can proceed, and that a cancellation during
cleanup still releases the port.
"""
import asyncio
import io
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_cancelled_cleanup_still_closes_the_listener(monkeypatch):
    """Cancelling the waiter twice while its listener cleanup runs must not skip ``server_close()``: the
    port has to be bindable as soon as the task ends, or the retry dies with EADDRINUSE (#113771)."""
    monkeypatch.setattr(oauth.sys, "stdin", io.StringIO())
    listening = threading.Event()
    shutdown_entered = threading.Event()
    release_shutdown = threading.Event()
    servers = []
    original_start = oauth._start_callback_server

    def instrumented_start(callback_port, handler):
        server = original_start(callback_port, handler)  # kept referenced: a GC'd socket would free the port
        original_shutdown = server.shutdown

        def held_shutdown():
            shutdown_entered.set()
            release_shutdown.wait(8)
            original_shutdown()

        server.shutdown = held_shutdown
        servers.append(server)
        listening.set()
        return server

    monkeypatch.setattr(oauth, "_start_callback_server", instrumented_start)

    def send_callback(port):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/callback?code=synthetic&state=s1")
            assert conn.getresponse().status == 200
        finally:
            conn.close()

    async def drive():
        with oauth.force_interactive_oauth():
            task = asyncio.create_task(oauth._make_callback_waiter(0, timeout=10)())
        try:
            assert await asyncio.to_thread(listening.wait, 5), "OAuth callback server never bound"
            await asyncio.to_thread(send_callback, servers[0].server_address[1])
            assert await asyncio.to_thread(shutdown_entered.wait, 5), "OAuth waiter never reached cleanup"
            for _ in range(2):
                task.cancel()
                for _ in range(5):
                    await asyncio.sleep(0)  # let the cancellation reach the task
        finally:
            release_shutdown.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with socket.socket() as again:  # what the retry's _start_callback_server does
            again.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            again.bind(("127.0.0.1", servers[0].server_address[1]))
            again.listen()

    try:
        asyncio.run(drive())
    finally:
        release_shutdown.set()
        for server in servers:
            server.server_close()


def test_loop_teardown_during_cleanup_still_closes_the_listener(monkeypatch):
    """``asyncio.run`` cancels every task still pending at teardown. Cleanup must not be one of them: a
    cleanup queued behind a busy default executor was cancelled there and left the port bound."""
    monkeypatch.setattr(oauth.sys, "stdin", io.StringIO())
    servers = []
    original_start = oauth._start_callback_server

    def recording_start(callback_port, handler):
        servers.append(original_start(callback_port, handler))  # kept referenced: a GC'd socket would free the port
        return servers[-1]

    monkeypatch.setattr(oauth, "_start_callback_server", recording_start)
    executor_busy = threading.Event()

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        loop.run_in_executor(None, executor_busy.wait, 5)  # every executor slot is taken
        with oauth.force_interactive_oauth():
            asyncio.create_task(oauth._make_callback_waiter(0, timeout=0)())  # timeout=0: straight to cleanup
        for _ in range(20):
            await asyncio.sleep(0)  # let the waiter bind and enter cleanup
        threading.Timer(0.2, executor_busy.set).start()
        # Returning here leaves the waiter pending: asyncio.run's teardown cancels it.

    try:
        asyncio.run(main())
        assert servers, "OAuth callback server never bound"
        with socket.socket() as again:  # what the retry's _start_callback_server does
            again.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            again.bind(("127.0.0.1", servers[0].server_address[1]))
            again.listen()
    finally:
        executor_busy.set()
        for server in servers:
            server.server_close()


def test_cleanup_failure_still_ends_the_waiter(monkeypatch):
    """A listener whose ``server_close()`` raises must not leave the waiter waiting forever."""
    monkeypatch.setattr(oauth.sys, "stdin", io.StringIO())
    servers = []
    original_start = oauth._start_callback_server

    def failing_close_start(callback_port, handler):
        server = original_start(callback_port, handler)
        real_close = server.server_close

        def failing_close():
            real_close()
            raise OSError("synthetic close failure")

        server.server_close = failing_close
        servers.append(server)
        return server

    monkeypatch.setattr(oauth, "_start_callback_server", failing_close_start)

    outcome = {}

    def run():  # own thread: a waiter that never ends swallows cancellation, so bound it with join()
        async def main():
            with oauth.force_interactive_oauth():
                return await oauth._make_callback_waiter(0, timeout=0)()  # timeout=0: straight to cleanup

        try:
            outcome["result"] = asyncio.run(main())
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "waiter never finished after cleanup failed"
    assert servers, "OAuth callback server never bound"
    # No code arrived, so the waiter reports its own timeout.
    assert isinstance(outcome.get("error"), oauth.OAuthNonInteractiveError), outcome
    assert "callback timed out" in str(outcome["error"])
