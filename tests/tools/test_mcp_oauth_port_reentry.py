"""A completed OAuth callback flow on a pinned port must not wedge the next flow (#73997).

`hermes mcp login` against servers with a pinned callback port (CIMD or a configured
``oauth.redirect_port``) failed with ``Errno 98 Address already in use`` on the SECOND
authorization attempt — right after the user had consented and the code WAS delivered.

Mechanism (verified empirically, mcp 2.0.0, Linux): the consent tab often holds its side
of the callback connection open past the waiter's ``server_close()`` (the waiter polls the
result at 0.5s and closes the listener as soon as the code lands, while the browser is
still sending/receiving). The server then performs the ACTIVE close of the accepted
connection, leaving it in TIME_WAIT (or FIN-WAIT while the peer drains). On the adopted-
reservation path (pinned ports) the listener socket came from ``_bind_reserved``, which
set no ``SO_REUSEADDR`` — and Linux only honours a ``SO_REUSEADDR`` rebind against a
TIME_WAIT socket when the socket that created that TIME_WAIT itself had the flag set.
The next flow's bind on the same pinned port therefore fails for the whole ~60s window
and the re-entered authorization is lost.
"""

import asyncio
import socket
import threading
import time

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)

import tools.mcp_oauth as mod
from tools.mcp_oauth import _make_callback_handler, _make_callback_waiter

# Port mechanics use a free port rather than the real CIMD range: test files run as
# concurrent subprocesses and the pinned range is bound for real by _pick_cimd_port.


def _hit_callback_when_ready(url: str, timeout: float = 15.0, hold_open: float = 1.2) -> None:
    """Consent tab: GET the callback once it answers and hold the socket open (late close)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((socket.parse_ns(f"//{url.split('/')[2]}").host
                                          if False else "127.0.0.1",
                                          int(url.split(":")[2].split("/")[0])), timeout=5)
            break
        except OSError:
            time.sleep(0.01)
    else:
        raise AssertionError(f"callback listener never came up: {url}")
    s.sendall(b"GET /callback?code=abc123&state=xyz HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
    time.sleep(hold_open)  # browser tab lingers past the waiter's server_close()
    try:
        s.recv(4096)
    except OSError:
        pass
    s.close()


def _complete_flow_on(port: int) -> None:
    """One full callback flow: reserve -> adopt/listen -> consent round-trip -> waiter's finally."""
    assert mod._bind_reserved(port) == port
    waiter = _make_callback_waiter(port)

    async def _drive():
        threading.Thread(
            target=_hit_callback_when_ready,
            args=(f"http://127.0.0.1:{port}/callback?code=abc123&state=xyz",),
            daemon=True,
        ).start()
        return await asyncio.wait_for(waiter(), timeout=20)

    outcome = asyncio.run(_drive())
    assert outcome.code == "abc123"


@pytest.fixture
def private_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestCompletedFlowPortIsImmediatelyReusable:
    """The port a completed flow listened on is rebindable by the next flow immediately."""

    def test_second_flow_after_completed_flow_binds_the_same_pinned_port(
        self, private_port, monkeypatch
    ):
        """RED on base: the adopted reserved socket carries no SO_REUSEADDR, so the TIME_WAIT
        its completed connection leaves behind makes the next bind fail EADDRINUSE for the
        full kernel window — the 'port already in use' failure right after successful consent.
        """
        port = private_port
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        _complete_flow_on(port)

        # Flow B (re-entry / next login attempt) starts immediately on the SAME port.
        assert mod._bind_reserved(port) == port  # must not raise
        handler_cls, _result = _make_callback_handler()
        server = mod._start_callback_server(port, handler_cls)  # EADDRINUSE on base
        try:
            assert server.server_address[1] == port
        finally:
            server.server_close()

    def test_completed_flow_releases_the_port_for_a_plain_bind(self, private_port, monkeypatch):
        """The waiter's finally must leave the port bindable (SO_REUSEADDR on) immediately,
        even when the consent tab holds its connection open past server_close."""
        port = private_port
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        _complete_flow_on(port)

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))  # EADDRINUSE on base
        finally:
            probe.close()
