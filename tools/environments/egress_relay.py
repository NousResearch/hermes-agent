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
# Host-side budget per sandbox: connections beyond this are refused, and a connection whose
# unsent backlog exceeds the queue bound (iron-proxy stalled) is dropped instead of buffering.
MAX_CONNECTIONS = 64
_INBOX_FRAMES = 256

# Runs inside the sandbox (stdlib only, python3.8+). argv[1] = tunnel_port.
AGENT_SOURCE = r'''
import base64, queue, socket, sys, threading
PORT = int(sys.argv[1])
out = sys.stdout.buffer
lock = threading.Lock()
conns = {}
seq = [0]

def send(op, cid, data=b""):
    with lock:
        out.write(b"%s %d %s\n" % (op, cid, base64.b64encode(data)))
        out.flush()

def close(cid):
    entry = conns.pop(cid, None)
    if entry is not None:
        entry[1].put(None)
        send(b"C", cid)

def pump(cid, s):
    try:
        while True:
            data = s.recv(32768)
            if not data:
                break
            send(b"D", cid, data)
    except OSError:
        pass
    close(cid)
    s.close()

def writer(cid, s, q):
    # One writer per connection: a client that stops reading stalls only itself.
    try:
        while (data := q.get()) is not None:
            s.sendall(data)
        s.shutdown(socket.SHUT_WR)  # pump() sees the client's EOF next and closes the socket
    except OSError:
        close(cid)

def serve(idx, ls):
    while True:
        s, _ = ls.accept()
        with lock:
            seq[0] += 1
            cid = seq[0]
        q = queue.Queue()
        conns[cid] = (s, q)
        threading.Thread(target=writer, args=(cid, s, q), daemon=True).start()
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
    entry = conns.get(int(parts[1]))
    if entry is None:
        continue
    if parts[0] == b"D":
        entry[1].put(base64.b64decode(parts[2]) if len(parts) > 2 else b"")
    elif parts[0] == b"C":
        conns.pop(int(parts[1]), None)
        entry[1].put(None)
'''


class RelayHub:
    """Host side of the relay: one per sandbox. ``send_line`` writes one frame line to the agent's
    stdin; the transport calls ``feed_line`` for every line the agent prints. Each connection gets
    an ordered, bounded inbox so data that arrives while iron-proxy is still being dialled is kept."""

    def __init__(self, send_line: Callable[[str], None], targets: tuple[tuple[str, int], tuple[str, int]]):
        self._send_line, self._targets = send_line, targets
        self._send_lock = threading.Lock()
        self._inboxes: dict[int, queue.Queue] = {}
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
            if len(self._inboxes) >= MAX_CONNECTIONS:
                logger.warning("egress relay: %d connections open, refusing another", MAX_CONNECTIONS)
                self._send_quietly("C", cid)
                return
            self._inboxes[cid] = inbox = queue.Queue(_INBOX_FRAMES)
            spawn_context_thread(self._serve, name="hermes-egress-relay-conn", args=(cid, payload, inbox)).start()
        elif (inbox := self._inboxes.get(cid)) is not None:
            try:
                inbox.put_nowait(payload if op == "D" else None)
            except queue.Full:
                logger.warning("egress relay: connection %d backlog over %d frames, dropping it", cid, _INBOX_FRAMES)
                self._end(cid)

    def _send_quietly(self, op: str, cid: int) -> None:
        with contextlib.suppress(Exception):  # health: allow BLE001 -- the stream may already be gone; cleanup must go on
            self._send(op, cid)

    def _end(self, cid: int) -> None:
        """Forget a connection and wake its writer; tell the agent, best effort."""
        if (inbox := self._inboxes.pop(cid, None)) is not None:
            with contextlib.suppress(queue.Full):
                inbox.put_nowait(None)
            self._send_quietly("C", cid)

    def _serve(self, cid: int, payload: bytes, inbox: queue.Queue) -> None:
        host, port = self._targets[1 if payload == b"1" else 0]
        try:
            sock = socket.create_connection((host, port), timeout=10)
        except OSError as exc:
            logger.warning("egress relay: could not reach iron-proxy at %s:%d: %s", host, port, exc)
            self._end(cid)
            return
        sock.settimeout(None)
        spawn_context_thread(self._pump_out, name="hermes-egress-relay-out", args=(cid, sock)).start()
        try:
            while (data := inbox.get()) is not None and cid in self._inboxes:
                sock.sendall(data)
        except OSError:
            pass
        finally:
            self._end(cid)
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    def _pump_out(self, cid: int, sock: socket.socket) -> None:
        try:
            while cid in self._inboxes and (data := sock.recv(_CHUNK)):
                self._send("D", cid, data)
        except Exception:  # health: allow BLE001 -- socket or stream (SDK) failure; either way the connection ends
            logger.debug("egress relay: connection %d ended", cid, exc_info=True)
        finally:
            self._end(cid)
            sock.close()

    def close(self) -> None:
        """End every connection (the stream is gone or the sandbox is closing)."""
        for cid in list(self._inboxes):
            self._end(cid)

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
