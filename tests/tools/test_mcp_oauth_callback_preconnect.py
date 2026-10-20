"""An idle browser preconnect must not hold the authorization code past its lifetime.

Chrome opens a second, speculative connection to the loopback callback alongside the one that
carries ``GET /callback?code=...`` and leaves it idle for about a minute. The single-threaded
callback server accepted that socket after serving the callback and blocked reading a request line
that never came; the waiter's ``server.shutdown()`` then waited on that serve loop, so the code reached
the token endpoint ~60 s after the redirect. Authorization servers with a 60 s code lifetime
answered ``invalid_grant: Invalid or expired authorization code``. A short read timeout on the handler
bounds the stall whichever socket the server accepts first.
"""
import asyncio
import io
import socket
import threading
import time
from http.client import HTTPConnection

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

import tools.mcp_oauth as mo


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listening(port: int) -> None:
    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.02)
    raise AssertionError("callback listener never bound")


@pytest.mark.parametrize("preconnect_first", [True, False])
def test_idle_preconnect_does_not_delay_the_code(monkeypatch, preconnect_first):
    monkeypatch.setattr(mo.sys, "stdin", io.StringIO())  # paste reader sees EOF; the HTTP listener is under test
    port = _free_port()
    out: dict = {}

    def run():
        async def main():
            with mo.force_interactive_oauth():
                return await mo._make_callback_waiter(port, timeout=30)()
        out["result"] = asyncio.run(main())
        out["returned_at"] = time.monotonic()

    waiter = threading.Thread(target=run, daemon=True)
    waiter.start()
    _wait_listening(port)

    idle = []

    def preconnect():  # the browser's speculative socket: connects, sends nothing, stays open
        idle.append(socket.create_connection(("127.0.0.1", port), timeout=5))

    if preconnect_first:
        preconnect()
        time.sleep(0.2)  # let the server accept it before the real request arrives
    sent_at = time.monotonic()
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/callback?code=abc&state=xyz")
    assert conn.getresponse().status == 200
    conn.close()
    if not preconnect_first:
        preconnect()  # accepted right after the callback was served

    try:
        waiter.join(timeout=8)
        assert not waiter.is_alive(), "waiter still blocked on the idle preconnect"
        assert out["result"].code == "abc"
        assert out["returned_at"] - sent_at < 5  # bounded by the handler's read timeout, far inside a 60 s code lifetime
    finally:
        for s in idle:
            s.close()
