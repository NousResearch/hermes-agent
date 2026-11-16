"""C1 (t_c1f387b2): ``initiated=`` on the ``ws closed`` line, driven through the REAL stack.

Production runs uvicorn's ``ws=auto``, which resolves to the ``websockets_sansio_impl`` protocol
(``hermes_cli/web_server_lifecycle.py`` builds ``uvicorn.Config`` without ``ws=``). Which close a
connection produces in the ASGI app is that implementation's business, not the WebSocket spec's —
so every label C2a reads is produced here by real uvicorn serving the real ``handle_ws``, not by
injecting the disconnect a test hopes for. Measured on uvicorn 0.54 / websockets 15:

* a keepalive ping nobody answers  → code 1005, no close frame,
* a graceful shutdown (deploy)     → code 1012,
* this server closing the socket   → code 1011 (the ASGI close it sent),
* a client whose TCP died (RST)    → code 1005, no close frame,
* a client's own goodbye frame     → its own code (1000).

A keepalive kill and a vanished client are the SAME report (1005), which is why the line names
both readings and carries ``silent_s`` for C2a to split them against the surface's ping budget.
"""

import contextlib
import logging
import socket
import struct
import threading

import uvicorn
from fastapi import FastAPI, WebSocket
from websockets.sync.client import connect

from tests.waiting import wait_until
from tui_gateway import server as gateway_server
from tui_gateway.ws import handle_ws

#: The keepalive budget the scenarios wait out. Small enough to stay a fast test, real enough that
#: the sansio protocol genuinely pings and genuinely gives up.
PING_INTERVAL_S = 0.4
PING_TIMEOUT_S = 0.4


class _CloseLines(logging.Handler):
    """The ``ws closed`` lines the served gateway logged (append-only, safe to poll)."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("ws closed "):
            self.lines.append(message)


@contextlib.contextmanager
def served_gateway(ping=(PING_INTERVAL_S, PING_TIMEOUT_S)):
    """Real uvicorn on a loopback port serving the real ``handle_ws``; yields it, its port and the
    close lines it logged."""
    handler = _CloseLines()
    logger = logging.getLogger("tui_gateway.ws")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    app = FastAPI()

    async def endpoint(ws: WebSocket):
        await handle_ws(ws)

    app.add_api_websocket_route("/api/ws", endpoint)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    uvicorn_server = uvicorn.Server(uvicorn.Config(
        app, log_level="error", lifespan="off", ws_ping_interval=ping[0], ws_ping_timeout=ping[1],
        timeout_graceful_shutdown=3))
    thread = threading.Thread(target=uvicorn_server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        wait_until(lambda: uvicorn_server.started, what="uvicorn to serve the gateway route", timeout=15)
        yield uvicorn_server, sock.getsockname()[1], handler.lines
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        logger.removeHandler(handler)
        gateway_server._sessions.clear()
        gateway_server._live_transports.clear()
        assert not thread.is_alive(), "the served gateway kept running after should_exit"


def _close_line(lines, what):
    line = wait_until(lambda: lines[0] if lines else None, what=what, timeout=15)
    assert line, "the gateway logged no ws closed line"
    return line


def _field(line, name):
    token = [part for part in line.split() if part.startswith(f"{name}=")]
    assert len(token) == 1, f"{name} missing from: {line}"
    return token[0][len(name) + 1:]


def _upgraded_raw_socket(port):
    """Complete the WebSocket upgrade by hand and then never read: no pong ever goes back."""
    import base64
    import os

    client = socket.create_connection(("127.0.0.1", port))
    key = base64.b64encode(os.urandom(16)).decode()
    client.sendall(
        f"GET /api/ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
    assert b" 101 " in client.recv(200), "the upgrade did not complete"
    return client


def test_keepalive_kill_is_not_claimed_as_the_peer_leaving():
    """A client that upgrades and then stops answering pings is killed by the server's own
    keepalive machinery. That reaches the app as 1005 with no close frame — the same report a
    vanished peer produces — so the line must not name either one."""
    with served_gateway() as (_uvicorn, port, lines):
        client = _upgraded_raw_socket(port)
        try:
            line = _close_line(lines, what="the keepalive kill to reach the teardown line")
        finally:
            client.close()

    assert _field(line, "initiated") == "peer_or_keepalive"
    assert "code=1005" in line
    # The discriminator C2a gets instead of a guess: the socket sat silent for at least the full
    # ping budget, because the protocol only gives up that long after the last unanswered ping.
    assert float(_field(line, "silent_s")) >= PING_INTERVAL_S + PING_TIMEOUT_S - 0.2


def test_a_vanished_client_is_not_claimed_as_the_peer_leaving():
    """A client whose TCP connection is reset with no Close frame is indistinguishable from a
    failed keepalive ping on this stack, so it must land in the same honest bucket."""
    with served_gateway() as (_uvicorn, port, lines):
        client = connect(f"ws://127.0.0.1:{port}/api/ws")
        client.recv(timeout=10)  # gateway.ready
        raw = client.socket
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        raw.close()  # RST: the connection dies with no Close frame
        line = _close_line(lines, what="the reset connection to reach the teardown line")

    assert _field(line, "initiated") == "peer_or_keepalive"
    assert "code=1005" in line


def test_graceful_shutdown_is_shutdown():
    """A deploy or restart tears every socket down through uvicorn's shutdown path; it must not
    read as a keepalive kill (nor as the client leaving)."""
    with served_gateway() as (uvicorn_server, port, lines):
        client = connect(f"ws://127.0.0.1:{port}/api/ws")
        try:
            client.recv(timeout=10)  # gateway.ready
            uvicorn_server.should_exit = True
            line = _close_line(lines, what="the graceful shutdown to reach the teardown line")
        finally:
            with contextlib.suppress(Exception):
                client.close()

    assert _field(line, "initiated") == "shutdown"
    assert "code=1012" in line


def test_the_servers_own_close_is_the_server():
    """Send deadline and fanout overflow end a socket from this side. The label comes from the
    transport's own latch, so it cannot drift with a library's close-code choice."""
    with served_gateway() as (_uvicorn, port, lines):
        client = connect(f"ws://127.0.0.1:{port}/api/ws")
        try:
            client.recv(timeout=10)  # gateway.ready
            transports = wait_until(lambda: list(gateway_server._live_transports),
                                    what="the gateway to register the transport")
            for transport in transports:
                transport.abort()
            line = _close_line(lines, what="the server-initiated close to reach the teardown line")
        finally:
            with contextlib.suppress(Exception):
                client.close()

    assert _field(line, "initiated") == "server"


def test_a_client_goodbye_is_peer():
    """A client that closes the connection properly says so in a Close frame, and the label
    attributes it to the client."""
    with served_gateway() as (_uvicorn, port, lines):
        client = connect(f"ws://127.0.0.1:{port}/api/ws")
        client.recv(timeout=10)  # gateway.ready
        client.close()  # a 1000 Close frame
        line = _close_line(lines, what="the client's own close to reach the teardown line")

    assert _field(line, "initiated") == "peer"
    assert "code=1000" in line
