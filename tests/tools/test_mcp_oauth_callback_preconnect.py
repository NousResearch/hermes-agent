"""An idle browser preconnect must not hold the authorization code past its lifetime.

Chrome opens a second, speculative connection to the loopback callback alongside the one that
carries ``GET /callback?code=...`` and leaves it idle for about a minute. The single-threaded
callback server accepted that socket after serving the callback and blocked reading a request line
that never came; the waiter's ``server.shutdown()`` then waited on that serve loop, so the code reached
the token endpoint ~60 s after the redirect. Authorization servers with a 60 s code lifetime
answered ``invalid_grant: Invalid or expired authorization code``. The listener now serves each
connection on its own thread, so neither an idle socket nor a slow client holds up another.
"""
import asyncio
import io
import socket
import threading
import time
from contextlib import contextmanager
from http.client import HTTPConnection

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

import tools.mcp_oauth as mo


@contextmanager
def _running_waiter(monkeypatch, timeout=10, hold_cleanup=None):
    """Run the production waiter on its own loop thread; yields ``(server, out)`` once it is listening.

    ``out`` gets ``result`` or ``error`` and ``returned_at``. Port 0: the OS picks a free port.
    ``hold_cleanup``: an Event the listener's real ``shutdown()`` waits for, so a test can connect after
    the callback without racing the listener closing."""
    monkeypatch.setattr(mo.sys, "stdin", io.StringIO())  # paste reader sees EOF; the HTTP listener is under test
    started, done, servers, out = threading.Event(), threading.Event(), [], {}
    original_start = mo._start_callback_server

    def recording_start(port, handler_cls):
        server = original_start(port, handler_cls)
        if hold_cleanup is not None:
            real_shutdown = server.shutdown

            def held_shutdown():
                hold_cleanup.wait(5)
                real_shutdown()

            server.shutdown = held_shutdown
        servers.append(server)
        started.set()
        return server

    monkeypatch.setattr(mo, "_start_callback_server", recording_start)

    def run():
        async def main():
            with mo.force_interactive_oauth():
                return await mo._make_callback_waiter(0, timeout=timeout)()
        try:
            out["result"] = asyncio.run(main())
        except Exception as exc:
            out["error"] = exc
        out["returned_at"] = time.monotonic()
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert started.wait(5), "callback listener never bound"
    try:
        yield servers[0], out
    finally:
        assert done.wait(timeout + 5), "waiter never returned"


def _raw_get(sock: socket.socket, query: str) -> bytes:
    sock.sendall(f"GET /callback?{query} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
    chunks = []
    while chunk := sock.recv(8192):
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.parametrize("preconnect_first", [True, False])
def test_idle_preconnect_does_not_delay_the_code(monkeypatch, preconnect_first):
    opened = []  # every socket the test opens, closed whatever fails
    preconnected = threading.Event()  # cleanup waits for it, so the late preconnect never meets a closed port
    try:
        with _running_waiter(monkeypatch, hold_cleanup=preconnected) as (server, out):
            def preconnect():  # the browser's speculative socket: connects, sends nothing, stays open
                opened.append(socket.create_connection(server.server_address, timeout=5))
                preconnected.set()

            if preconnect_first:
                preconnect()  # the accept queue is FIFO: the server takes this socket ahead of the real request
            sent_at = time.monotonic()
            conn = HTTPConnection(*server.server_address, timeout=5)
            opened.append(conn)
            conn.request("GET", "/callback?code=abc&state=xyz")
            assert conn.getresponse().status == 200
            if not preconnect_first:
                preconnect()  # accepted right after the callback was served
        assert out["result"].code == "abc"
        assert out["returned_at"] - sent_at < 5  # far inside a 60 s code lifetime
    finally:
        preconnected.set()
        for sock in opened:
            sock.close()


def test_queued_idle_preconnects_do_not_delay_the_redirect(monkeypatch):
    """Two silent sockets accepted ahead of the redirect: a per-socket read timeout cost 2 s each in turn."""
    opened = []  # every socket the test opens, closed whatever fails
    try:
        with _running_waiter(monkeypatch) as (server, out):
            opened += [socket.create_connection(server.server_address, timeout=5) for _ in range(2)]
            sent_at = time.monotonic()
            conn = HTTPConnection(*server.server_address, timeout=10)
            opened.append(conn)
            conn.request("GET", "/callback?code=abc&state=xyz")
            assert conn.getresponse().status == 200
            answered = time.monotonic() - sent_at
        assert out["result"].code == "abc"
        assert answered < 3.0, f"redirect answered after {answered:.2f}s behind idle preconnects"
        assert out["returned_at"] - sent_at < 3.0  # two idle preconnects cost ~4 s on a serial listener
    finally:
        for sock in opened:
            sock.close()


def test_slow_client_still_delivers_the_code(monkeypatch):
    """A client that sends its request line after 2.5 s (an SSH-tunnelled callback) is served, not dropped."""
    with _running_waiter(monkeypatch) as (server, out):
        with socket.create_connection(server.server_address, timeout=10) as sock:
            time.sleep(2.5)  # health: allow HX013 -- the late request line IS the behaviour under test
            reply = _raw_get(sock, "code=slow&state=s1")
    assert reply.startswith(b"HTTP/1.0 200"), reply[:80]
    assert (out["result"].code, out["result"].state) == ("slow", "s1")


def test_concurrent_callbacks_one_consistent_winner(monkeypatch):
    """Two valid callbacks in flight at once: exactly one wins, and its code and state stay together."""
    both_in_handler = threading.Barrier(2, timeout=5)
    real_parse = mo._parse_redirect_query

    def rendezvous_parse(query):
        both_in_handler.wait()  # a serial server never has two handlers running: it breaks here
        return real_parse(query)

    monkeypatch.setattr(mo, "_parse_redirect_query", rendezvous_parse)
    replies = {}
    with _running_waiter(monkeypatch) as (server, out):
        def send(code):
            with socket.create_connection(server.server_address, timeout=10) as sock:
                replies[code] = _raw_get(sock, f"code={code}&state=state-{code}")

        senders = [threading.Thread(target=send, args=(code,)) for code in ("one", "two")]
        for sender in senders:
            sender.start()
        for sender in senders:
            sender.join(15)
    winners = [code for code, reply in replies.items() if b"Authorization Successful" in reply]
    losers = [code for code, reply in replies.items() if b"Authorization already received" in reply]
    assert len(winners) == 1 and len(losers) == 1, replies
    assert (out["result"].code, out["result"].state) == (winners[0], f"state-{winners[0]}")
