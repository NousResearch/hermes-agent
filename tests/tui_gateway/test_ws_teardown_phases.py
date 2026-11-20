"""C1 (t_c1f387b2): the ``ws closed`` teardown line names who ended the connection
and what the socket looked like when detection happened — the two fields that turn
production ``gui.log`` into a reconnect-latency baseline:

* ``life_s``   accept → close detection (the churn denominator).
* ``silent_s`` last inbound frame → detection: how long the socket sat orphaned and
  silent before the server declared it dead (the server-visible half of detection lag).
* ``initiated`` who closed it.

These drive ``handle_ws`` over a fake socket, so they pin the CLASSIFIER: which label each
ASGI disconnect maps to, and that no label claims a cause the evidence does not carry.
Which disconnect the production transport actually hands the app — and therefore which
label C2a reads in production — is pinned against real uvicorn in
``test_ws_close_initiated.py``; each case below names the scenario there that produces it.

Log-only; no timer or teardown semantics change.
"""

import asyncio
import logging

from tui_gateway import server
from tui_gateway import ws as ws_mod


def _close_line(caplog, monkeypatch, exc_factory):
    """Drive handle_ws through accept → read failure → teardown; return the new ``ws closed`` line."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    server._sessions.clear()
    server._live_transports.clear()
    try:
        class FakeWS:
            async def accept(self, subprotocol=None):
                pass

            async def send_text(self, line):
                pass

            async def receive_text(self):
                raise exc_factory()

            async def close(self):
                pass

        before = len(caplog.records)
        with caplog.at_level(logging.INFO, logger="tui_gateway.ws"):
            asyncio.run(ws_mod.handle_ws(FakeWS()))
    finally:
        server._sessions.clear()
        server._live_transports.clear()

    lines = [r.getMessage() for r in caplog.records[before:] if r.getMessage().startswith("ws closed ")]
    assert len(lines) == 1, caplog.text
    return lines[0]


def _field(line, name):
    token = [part for part in line.split() if part.startswith(f"{name}=")]
    assert len(token) == 1, f"{name} missing from: {line}"
    return token[0][len(name) + 1:]


def test_graceful_shutdown_code_is_shutdown_not_keepalive(caplog, monkeypatch):
    # 1012 is the code uvicorn's sansio protocol queues on a graceful shutdown, so every deploy
    # or restart lands here (test_ws_close_initiated.py::test_graceful_shutdown_is_shutdown).
    # It was listed as a keepalive code before, which read every deploy as a keepalive kill.
    line = _close_line(caplog, monkeypatch, lambda: ws_mod._WebSocketDisconnect(1012, ""))
    assert _field(line, "initiated") == "shutdown"
    assert float(_field(line, "life_s")) >= 0.0
    assert float(_field(line, "silent_s")) >= 0.0


def test_a_connection_lost_without_a_close_frame_claims_no_cause(caplog, monkeypatch):
    # 1005 is what the app is handed whenever the TCP connection dies with no Close frame — both a
    # keepalive ping that went unanswered and a peer that vanished
    # (test_ws_close_initiated.py::test_keepalive_kill_is_not_claimed_as_the_peer_leaving and
    # ::test_a_vanished_client_is_not_claimed_as_the_peer_leaving). The line must offer both
    # readings; silent_s against the surface's ping budget is what splits them.
    line = _close_line(caplog, monkeypatch, lambda: ws_mod._WebSocketDisconnect(1005, ""))
    assert _field(line, "initiated") == "peer_or_keepalive"


def test_a_client_goodbye_frame_is_the_peer(caplog, monkeypatch):
    # A real Close frame from the client (test_ws_close_initiated.py::test_a_client_goodbye_is_peer).
    assert _field(_close_line(caplog, monkeypatch, lambda: ws_mod._WebSocketDisconnect(1000, "")), "initiated") == "peer"
    assert _field(_close_line(caplog, monkeypatch, lambda: ws_mod._WebSocketDisconnect(1001, "")), "initiated") == "peer"


def test_a_keepalive_reason_text_is_not_laundered_into_a_cause(caplog, monkeypatch):
    # Legacy websockets implementations do report ``1011 keepalive ping timeout``; the stack
    # production runs never does. Rather than trust a code, the label stays the honest bucket and
    # the verbatim code and reason stay on the line for whoever needs them.
    line = _close_line(caplog, monkeypatch,
                       lambda: ws_mod._WebSocketDisconnect(1011, "keepalive ping timeout"))
    assert _field(line, "initiated") == "unknown"
    assert "keepalive ping timeout" in line  # raw reason retained verbatim


def test_abnormal_close_codes_are_not_claimed(caplog, monkeypatch):
    # 1002 (protocol error) and 1006 (TCP lost before the handshake completed) name no cause the
    # C2a hypothesis predicts; the line must not launder either into a causal bucket.
    assert _field(_close_line(caplog, monkeypatch, lambda: ws_mod._WebSocketDisconnect(1002, "protocol")), "initiated") == "unknown"
    assert _field(_close_line(caplog, monkeypatch, lambda: ws_mod._WebSocketDisconnect(1006, "")), "initiated") == "unknown"


def test_the_servers_own_socket_close_is_attributed_to_the_server():
    # The transport latches "this server asked the socket to close" (send deadline, fanout
    # overflow), so a server-initiated teardown needs no close code at all — on the production
    # stack it arrives as 1011 anyway
    # (test_ws_close_initiated.py::test_the_servers_own_close_is_the_server).
    fields = ws_mod._ws_close_fields(None, None, None, "client_disconnect(code=1011,reason=)", 1011,
                                     shutting_down=False, server_closed=True)
    assert fields[2] == "server"


def test_a_cancelled_handler_is_a_shutdown():
    fields = ws_mod._ws_close_fields(None, None, None, "client_disconnect(code=1005,reason=)", 1005,
                                     shutting_down=True, server_closed=False)
    assert fields[2] == "shutdown"


def test_server_send_failure_is_not_labeled_connected(caplog, monkeypatch):
    """A refused ready-frame write used to tear down with reason/initiated indistinguishable
    from a healthy connection: it must be attributed to the server."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    server._sessions.clear()
    server._live_transports.clear()
    try:
        class DeadSendWS:
            async def accept(self, subprotocol=None):
                pass

            async def send_text(self, line):
                raise RuntimeError("transport is gone")

            async def receive_text(self):  # pragma: no cover - never reached
                raise AssertionError("read loop must not start after a failed ready write")

            async def close(self):
                pass

        with caplog.at_level(logging.INFO, logger="tui_gateway.ws"):
            asyncio.run(ws_mod.handle_ws(DeadSendWS()))
    finally:
        server._sessions.clear()
        server._live_transports.clear()

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("ws closed "))
    assert _field(line, "reason") == "ready_send_failed"
    assert _field(line, "initiated") == "server"
