"""Egress relay for cloud sandboxes (opt-in via ``proxy.cloud_relay``).

A Modal or Daytona sandbox runs in the provider's cloud and has no route back to iron-proxy on
the Hermes host. Opening one would need a public endpoint (a tunnel or an exposed port), which
turns the proxy into something anyone holding a token can reach. Instead the relay rides the one
channel the provider already authenticates: a long-lived exec stream into the sandbox.

Inside the sandbox a stdlib Python agent listens on ``127.0.0.1:<tunnel_port>`` (CONNECT) and
``<tunnel_port+1>`` (plain HTTP), exactly where the SSH reverse forward binds. Each accepted
connection becomes a stream of base64 frames on the agent's stdout; the host side (``RelayHub``)
dials iron-proxy's real listener and pumps bytes both ways. Frames are text lines so the same
protocol survives Modal's exec streams and a raw-mode Daytona PTY.

Frame: ``<op> <conn_id> <base64>`` with op ``R`` (agent ready), ``O`` (open; payload = listener
index), ``D`` (data) or ``C`` (close). Anything else on the stream (a PTY echo, a banner) is ignored.

Fail-closed: when the relay is down the sandbox's ``HTTPS_PROXY`` points at a dead loopback port,
so requests fail; nothing falls back to a direct route with real keys.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import queue
import re
import socket
import threading
from typing import Callable

from agent.memory_provider import spawn_context_thread

logger = logging.getLogger("tools.environments.egress")

REMOTE_AGENT_NAME = ".hermes-egress-relay.py"
READY_TIMEOUT_S = 30.0
_FRAME_RE = re.compile(r"^([RODC]) (\d+) ?([A-Za-z0-9+/=]*)$")
_CHUNK = 32 * 1024

# Runs inside the sandbox (stdlib only, python3.8+). argv[1] = tunnel_port.
AGENT_SOURCE = r'''
import base64, socket, sys, threading
PORT = int(sys.argv[1])
out = sys.stdout.buffer
lock = threading.Lock()
conns = {}
seq = [0]

def send(op, cid, data=b""):
    with lock:
        out.write(b"%s %d %s\n" % (op, cid, base64.b64encode(data)))
        out.flush()

def pump(cid, s):
    try:
        while True:
            data = s.recv(32768)
            if not data:
                break
            send(b"D", cid, data)
    except OSError:
        pass
    if conns.pop(cid, None) is not None:
        send(b"C", cid)
    s.close()

def serve(idx, ls):
    while True:
        s, _ = ls.accept()
        with lock:
            seq[0] += 1
            cid = seq[0]
        conns[cid] = s
        send(b"O", cid, str(idx).encode())
        threading.Thread(target=pump, args=(cid, s), daemon=True).start()

for idx in (0, 1):
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", PORT + idx))
    ls.listen(64)
    threading.Thread(target=serve, args=(idx, ls), daemon=True).start()
send(b"R", 0)
for line in sys.stdin.buffer:
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        continue
    op, cid = parts[0], int(parts[1])
    s = conns.get(cid)
    if s is None:
        continue
    try:
        if op == b"D":
            s.sendall(base64.b64decode(parts[2]) if len(parts) > 2 else b"")
        elif op == b"C":
            conns.pop(cid, None)
            s.shutdown(socket.SHUT_RDWR)
    except OSError:
        conns.pop(cid, None)
'''


class RelayHub:
    """Host side of the relay: one per sandbox. ``send_line`` writes one frame line to the agent's
    stdin; the transport calls ``feed_line`` for every line the agent prints. Each connection gets
    an ordered inbox so data that arrives while iron-proxy is still being dialled is not lost."""

    def __init__(self, send_line: Callable[[str], None], targets: tuple[tuple[str, int], tuple[str, int]]):
        self._send_line, self._targets = send_line, targets
        self._send_lock = threading.Lock()
        self._inboxes: dict[int, queue.SimpleQueue] = {}
        self.ready = threading.Event()

    def _send(self, op: str, cid: int, data: bytes = b"") -> None:
        with self._send_lock:
            self._send_line(f"{op} {cid} {base64.b64encode(data).decode('ascii')}\n")

    def feed_line(self, line: str) -> None:
        # A PTY can prefix the first frame with terminal control output ending in a bare CR.
        match = _FRAME_RE.match(line.rstrip("\r\n").rsplit("\r", 1)[-1])
        if match is None:
            return
        op, cid, payload = match.group(1), int(match.group(2)), base64.b64decode(match.group(3) or "")
        if op == "R":
            self.ready.set()
        elif op == "O":
            self._inboxes[cid] = inbox = queue.SimpleQueue()
            spawn_context_thread(self._serve, name="hermes-egress-relay-conn", args=(cid, payload, inbox)).start()
        elif (inbox := self._inboxes.get(cid)) is not None:
            inbox.put(payload if op == "D" else None)

    def _serve(self, cid: int, payload: bytes, inbox: queue.SimpleQueue) -> None:
        host, port = self._targets[1 if payload == b"1" else 0]
        try:
            sock = socket.create_connection((host, port), timeout=10)
        except OSError as exc:
            logger.warning("egress relay: could not reach iron-proxy at %s:%d: %s", host, port, exc)
            self._inboxes.pop(cid, None)
            self._send("C", cid)
            return
        sock.settimeout(None)
        spawn_context_thread(self._pump_out, name="hermes-egress-relay-out", args=(cid, sock)).start()
        while (data := inbox.get()) is not None:
            with contextlib.suppress(OSError):
                sock.sendall(data)
        self._inboxes.pop(cid, None)
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)

    def _pump_out(self, cid: int, sock: socket.socket) -> None:
        try:
            while data := sock.recv(_CHUNK):
                self._send("D", cid, data)
        except OSError:
            pass
        if (inbox := self._inboxes.get(cid)) is not None:
            self._send("C", cid)
            inbox.put(None)
        sock.close()

    def close(self) -> None:
        for inbox in list(self._inboxes.values()):
            inbox.put(None)

    def wait_ready(self, backend: str) -> None:
        if not self.ready.wait(READY_TIMEOUT_S):
            raise RuntimeError(
                f"{backend}: the egress relay agent did not come up inside the sandbox within {READY_TIMEOUT_S:.0f}s "
                "(it needs python3 in the image).  proxy.cloud_relay is on, so the sandbox refuses to run unprotected; "
                "set proxy.cloud_relay: false to opt out.")


class LineSplitter:
    """Reassemble lines from arbitrary byte/str chunks (PTY reads split anywhere)."""

    def __init__(self, sink: Callable[[str], None]):
        self._sink, self._buf = sink, ""

    def feed(self, chunk: bytes | str) -> None:
        self._buf += chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._sink(line)


def agent_command(remote_home: str, tunnel_port: int) -> list[str]:
    return ["python3", "-u", f"{remote_home}/{REMOTE_AGENT_NAME}", str(tunnel_port)]
