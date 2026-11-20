"""C2a (t_e15288d4): the keepalive strike policy, pinned on the REAL transport.

Step-1 measured (harness, production ``build_listener_server`` on a non-loopback bind): a
client whose process the OS suspends answers no pong, and uvicorn's sansio keepalive kills it
at exactly ``ws_ping_interval + ws_ping_timeout`` — the 20/20 default is the measured 40 s
churn grid. One missed cycle is inconclusive for a backgrounded renderer (the kernel keeps
ACKing; a waking client pongs within milliseconds), so
``hermes_cli/web_server_keepalive.strike_keepalive_protocol`` requires TWO consecutive missed
pongs before the transport's own failure path — and only when that whole budget still fits
under the ~100 s proxy idle cap the non-loopback bind sits behind.

Every scenario is driven through real uvicorn + real ``handle_ws`` (same fixture shape as
``test_ws_close_initiated.py``; that file owns what the kill REPORTS — 1005, no close frame —
this file owns WHEN it fires). ``tests/tui_gateway/test_ws_close_initiated.py`` runs against
plain ``ws="auto"`` and stays the single-strike upstream baseline.
"""

import base64
import contextlib
import logging
import os
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket

from hermes_cli.web_server_keepalive import (
    _sansio_strike_base,
    strike_keepalive_protocol,
    ws_ping_strikes,
)
from tests.waiting import wait_until
from tui_gateway import server as gateway_server
from tui_gateway.ws import handle_ws

#: Small enough that the whole file runs in seconds, real enough that the sansio protocol
#: genuinely pings, genuinely times out, and genuinely strikes.
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
def served_gateway(ws: "type | str" = "auto", ping=(PING_INTERVAL_S, PING_TIMEOUT_S)):
    """Real uvicorn on a loopback port serving the real ``handle_ws`` with an explicit transport.

    Unlike the production listeners this fixture passes ``ws=`` directly: the strike protocol is
    what is under test, and pinning it here keeps the cadence policy (``ws_ping_strikes``) a
    separate, unit-tested decision.
    """
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
        app, log_level="error", lifespan="off", ws=ws,
        ws_ping_interval=ping[0], ws_ping_timeout=ping[1], timeout_graceful_shutdown=3))
    thread = threading.Thread(target=uvicorn_server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        wait_until(lambda: uvicorn_server.started, what="uvicorn to serve the gateway route", timeout=15)
        # Warm it: the first connection of a fresh process pays boot costs that inflate life_s
        # and skew cadence (measured on the Step-1 harness: 2.4 s vs a warm 2.0 s at 1/1).
        warm = _upgraded_raw_socket(sock.getsockname()[1])
        warm.close()
        wait_until(lambda: len(handler.lines) >= 1, what="the warm connection's teardown", timeout=10)
        handler.lines.clear()
        yield uvicorn_server, sock.getsockname()[1], handler.lines
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        logger.removeHandler(handler)
        gateway_server._sessions.clear()
        gateway_server._live_transports.clear()
        assert not thread.is_alive(), "the served gateway kept running after should_exit"


def _upgraded_raw_socket(port):
    """Complete the WebSocket upgrade by hand and then read nothing: no pong ever goes back."""
    client = socket.create_connection(("127.0.0.1", port))
    key = base64.b64encode(os.urandom(16)).decode()
    client.sendall(
        f"GET /api/ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
    assert b" 101 " in client.recv(200), "the upgrade did not complete"
    return client


def _masked_pong(payload: bytes) -> bytes:
    mask = b"\x01\x02\x03\x04"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes([0x8A, 0x80 | len(payload)]) + mask + masked


class _ScriptedWsClient:
    """Reads the server's frames and answers its keepalive pings per script.

    ``answer`` is consulted once per received PING with that ping's zero-based index; True
    returns the required pong, False silently drops it — the frame-level shape of a renderer
    the OS froze (the kernel still ACKs; the protocol stack answers nothing).
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""
        self.ping_times: list[float] = []
        self.t0 = time.monotonic()
        self.died_at: float | None = None
        self._dropped = 0

    def pump(self, until: float) -> None:
        """Pump frames until wall time *until* (monotonic) or the wire ends."""
        self.sock.settimeout(0.1)
        while time.monotonic() < until and self.died_at is None:
            try:
                data = self.sock.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                data = b""
            if not data:
                self.died_at = time.monotonic() - self.t0
                return
            self.buf += data
            self._drain()

    def _drain(self) -> None:
        while len(self.buf) >= 2:
            op = self.buf[0] & 0x0F
            length = self.buf[1] & 0x7F
            offset = 2
            if length == 126:
                length = int.from_bytes(self.buf[offset:offset + 2], "big")
                offset += 2
            elif length == 127:
                length = int.from_bytes(self.buf[offset:offset + 8], "big")
                offset += 8
            if len(self.buf) < offset + length:
                return
            payload = self.buf[offset:offset + length]
            self.buf = self.buf[offset + length:]
            if op == 0x9:  # PING
                self.ping_times.append(time.monotonic() - self.t0)
                if self.answer(len(self.ping_times) - 1, payload):
                    self.sock.sendall(_masked_pong(payload))
                else:
                    self._dropped += 1
            elif op == 0x8:  # the transport failed us: a Close reached the wire
                self.died_at = time.monotonic() - self.t0
                return

    def answer(self, ping_index: int, payload: bytes) -> bool:
        raise NotImplementedError


class _NeverAnswers(_ScriptedWsClient):
    def answer(self, ping_index, payload):
        return False


class _DropsFirstThenAnswers(_ScriptedWsClient):
    """Misses ping #0 (the strike), answers every ping after — a waking client."""

    def answer(self, ping_index, payload):
        return ping_index >= 1


class _AnswersFirstThenMisses(_ScriptedWsClient):
    """Answers ping #0, then misses every one after: two consecutive strikes, then death."""

    def answer(self, ping_index, payload):
        return ping_index < 1


def _field(line, name):
    token = [part for part in line.split() if part.startswith(f"{name}=")]
    assert len(token) == 1, f"{name} missing from: {line}"
    return token[0][len(name) + 1:]


def _budget(strikes: int) -> float:
    return PING_INTERVAL_S + strikes * PING_TIMEOUT_S


def test_one_missed_pong_is_not_evidence_the_client_is_gone():
    """The kill must NOT fire on the first miss: a suspended renderer pongs the moment it wakes.

    Pre-C2a this client was killed at interval+timeout (0.8 s). The strike protocol re-pings and
    the answering client lives — long past BOTH the old kill point and a second miss's budget —
    and the server keeps its chain alive well past ``interval + 2*timeout``.
    """
    protocol = strike_keepalive_protocol((PING_INTERVAL_S, PING_TIMEOUT_S))
    from uvicorn.protocols.websockets.websockets_sansio_impl import WebSocketsSansIOProtocol
    assert issubclass(protocol, WebSocketsSansIOProtocol)
    with served_gateway(ws=protocol) as (_uvicorn, port, lines):
        client = _DropsFirstThenAnswers(_upgraded_raw_socket(port))
        try:
            client.pump(until=client.t0 + _budget(2) + 1.4)  # past where ANY single-strike kill lands
            assert not lines, f"the strike policy killed a recovering client: {lines[0]}"
            assert len(client.ping_times) >= 3, f"the ping chain broke after the miss: {client.ping_times}"
            assert client._dropped == 1
        finally:
            client.sock.close()


def test_two_consecutive_misses_kill_at_interval_plus_two_timeouts():
    """A client that answers nothing dies — the policy delays the verdict, never waives it —
    at ``interval + 2*timeout`` (the default 20/20 cadence's 60 s), and still reports exactly
    the C1 fingerprint (1005, ``initiated=peer_or_keepalive``)."""
    protocol = strike_keepalive_protocol((PING_INTERVAL_S, PING_TIMEOUT_S))
    with served_gateway(ws=protocol) as (_uvicorn, port, lines):
        started = time.monotonic()
        client = _NeverAnswers(_upgraded_raw_socket(port))
        try:
            line = wait_until(lambda: lines[0] if lines else None,
                              what="the second strike's teardown line", timeout=_budget(2) + 3.0)
            assert line
            life = float(_field(line, "life_s"))
            assert life >= _budget(2) - 0.15, (
                f"killed at {life:.2f}s, before the two-strike budget — a single miss was treated as evidence")
            assert life <= _budget(2) + 1.5, f"kept a fully dead peer for {life:.2f}s"
            assert _field(line, "initiated") == "peer_or_keepalive", line
            assert "code=1005" in line, line  # C1's fingerprint: the strike policy must not change the REPORT
            assert client.died_at is None or client.died_at >= _budget(2) - 0.15
        finally:
            client.sock.close()
        assert time.monotonic() - started < _budget(2) + 3.0


def test_a_miss_reset_by_a_pong_does_not_count_toward_the_second_strike():
    """pong(reset) → miss → miss kills at the SECOND miss's deadline, not at the third cycle's —
    the counter is consecutive, and a mid-life pong buys a full fresh budget, not extra grace."""
    protocol = strike_keepalive_protocol((PING_INTERVAL_S, PING_TIMEOUT_S))
    with served_gateway(ws=protocol) as (_uvicorn, port, lines):
        client = _AnswersFirstThenMisses(_upgraded_raw_socket(port))
        try:
            line = wait_until(lambda: lines[0] if lines else None,
                              what="the post-reset second consecutive miss's teardown",
                              timeout=_budget(3) + 3.0)  # pong@cycle1 then two misses = 3 cycles of deadline
            assert line
            life = float(_field(line, "life_s"))
            assert "code=1005" in line and _field(line, "initiated") == "peer_or_keepalive", line
            assert _budget(2) - 0.15 <= life <= _budget(3) + 1.5, (
                f"kill at {life:.2f}s sits outside the consecutive-strike arithmetic")
        finally:
            client.sock.close()


def test_single_cycle_cadences_keep_the_transports_own_arithmetic():
    """The cap rule: a cadence whose second strike would land past the ~100 s proxy idle cap
    gets no strike protocol at all — half-open detection still has to beat the proxy, and the
    SSH-tunnel 60/600 watchdog budget (detect inside the 900 s idle grace) stays intact."""
    assert ws_ping_strikes(20.0, 20.0) == 2          # production default: kill at 60 s, under the cap
    assert ws_ping_strikes(10.0, 10.0) == 2
    assert ws_ping_strikes(60.0, 40.0) == 1          # the documented remote value: ONE 100 s cycle
    assert ws_ping_strikes(60.0, 600.0) == 1         # SSH-tunnel path: its 660 s detection owns the budget
    assert ws_ping_strikes(None, None) == 1          # loopback: no pings at all
    assert strike_keepalive_protocol((None, None)) == "auto"
    assert strike_keepalive_protocol((60.0, 600.0)) == "auto"
    extended = strike_keepalive_protocol((20.0, 20.0))
    assert strike_keepalive_protocol((20.0, 20.0)) is extended  # one class per strike count
    base = _sansio_strike_base()
    if base is None:  # the pyproject floor's uvicorn has no sansio keepalive to extend
        assert extended == "auto"  # the fallback tests below pin WHY, and that detection survives
    else:
        assert issubclass(extended, base)


def test_the_strike_class_extends_the_transport_uvicorn_actually_runs():
    """The policy is an override of the sansio keepalive hooks, so it is only sound while
    ``ws="auto"`` resolves to that same class. Subclassing anything else would swap the
    listener's transport rather than tune it."""
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    protocol = strike_keepalive_protocol((20.0, 20.0))
    if protocol == "auto":
        pytest.skip("this uvicorn's auto transport has no sansio keepalive to extend "
                    "(the pyproject floor); the fallback test below pins that verdict")
    assert AutoWebSocketsProtocol in protocol.__mro__


def test_a_transport_without_sansio_keepalive_keeps_its_own_ping_verdict(caplog):
    """The pyproject floor (``uvicorn>=0.31``) predates the sansio keepalive. On 0.41 ``ws="auto"``
    resolves to the LEGACY ``websockets_impl.WebSocketProtocol`` (whose own ping does the killing)
    and the sansio class carries no keepalive at all, so subclassing the sansio class silently
    turned half-open detection OFF — the reviewer measured a never-ponging peer left alive with 0
    pings sent where ``auto`` killed it. Both shapes of that uvicorn must fall back to ``"auto"``.

    Each arm runs on a cold protocol cache: the class cache is the process's answer to "what does
    this uvicorn give us", so a warm entry would return the verdict of an earlier arm. And the
    fallback must SAY SO — one WARNING naming the uvicorn version, not a silent downgrade.
    """
    import uvicorn.protocols.websockets.auto as auto_mod
    import uvicorn.protocols.websockets.websockets_sansio_impl as sansio_mod
    from hermes_cli import web_server_keepalive as keepalive
    from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol as _Legacy

    class _SansioPreKeepalive:  # what the sansio class looked like before its ping existed
        pass

    # Arm 1: auto resolves to the legacy transport (uvicorn before the sansio switch).
    with _cold_strike_cache(keepalive), _fresh_fallback_warning(keepalive), \
            caplog.at_level(logging.WARNING, logger="hermes_cli.web_server_keepalive"):
        with _swapped(auto_mod, "AutoWebSocketsProtocol", _Legacy):
            assert strike_keepalive_protocol((20.0, 20.0)) == "auto"
        # Arm 2: auto IS sansio, but that sansio has no keepalive hooks to override (the floor).
        with _swapped(sansio_mod, "WebSocketsSansIOProtocol", _SansioPreKeepalive), \
                _swapped(auto_mod, "AutoWebSocketsProtocol", _SansioPreKeepalive):
            assert strike_keepalive_protocol((20.0, 20.0)) == "auto"
        # Warn-once: one static verdict per uvicorn install, not one per listener build.
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "no sansio keepalive" in r.getMessage()]
        assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
        assert uvicorn.__version__ in warnings[0].getMessage()


@contextlib.contextmanager
def _cold_strike_cache(keepalive_module):
    """Empty the per-process strike-class cache for the duration of the block."""
    with _swapped(keepalive_module, "_STRIKE_PROTOCOLS", {}):
        yield


@contextlib.contextmanager
def _fresh_fallback_warning(keepalive_module):
    """Clear the warn-once latch so this arm sees the warning a cold process would emit."""
    with _swapped(keepalive_module, "_no_base_fallback_logged", False):
        yield


@contextlib.contextmanager
def _swapped(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def test_strike_policy_does_not_disturb_a_healthy_client():
    """A live client that always pongs rides the normal chain for many cycles: the policy must
    not kill, not stutter the cadence, and not change anything for everyone who never misses."""
    protocol = strike_keepalive_protocol((PING_INTERVAL_S, PING_TIMEOUT_S))
    with served_gateway(ws=protocol) as (_uvicorn, port, lines):

        class _Always(_ScriptedWsClient):
            def answer(self, ping_index, payload):
                return True

        client = _Always(_upgraded_raw_socket(port))
        try:
            client.pump(until=client.t0 + 5.0)
            assert not lines, f"a healthy pinging client was closed: {lines[0]}"
            # interval - rtt chaining: ~2.5s of life should carry at least 5 pings at 0.4s.
            assert len(client.ping_times) >= 5, client.ping_times
            gaps = [b - a for a, b in zip(client.ping_times, client.ping_times[1:])]
            assert all(0.2 < g < 0.8 for g in gaps), f"cadence stuttered: {gaps}"
        finally:
            client.sock.close()
