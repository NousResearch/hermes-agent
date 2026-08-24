"""Regression tests for WebSocket dead-peer detection via TCP keepalive.

Without SO_KEEPALIVE a silently-dropped client (SSH tunnel reset, laptop
sleep, NAT timeout) leaves the TCP leg half-open forever: ``receive_text()``
blocks indefinitely and the disconnect teardown (detach, orphan reap, resume
replay) never runs.  ``_disable_nagle`` already reaches the raw socket, so
keepalive is enabled there too.
"""

from __future__ import annotations

import socket

from tui_gateway.ws import _disable_nagle


class _FakeSocket:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def setsockopt(self, level: int, optname: int, value: int) -> None:
        self.calls.append((level, optname, value))


class _FakeTransport:
    def __init__(self, sock: _FakeSocket) -> None:
        self._sock = sock

    def get_extra_info(self, name: str):
        return self._sock if name == "socket" else None


class _FakeWS:
    def __init__(self, sock: _FakeSocket) -> None:
        self.scope = {"extensions": {"transport": _FakeTransport(sock)}}


def test_ws_socket_enables_keepalive() -> None:
    sock = _FakeSocket()
    _disable_nagle(_FakeWS(sock))

    opts = {(level, optname): value for level, optname, value in sock.calls}
    # Nagle still disabled (pre-existing behavior preserved).
    assert opts[(socket.IPPROTO_TCP, socket.TCP_NODELAY)] == 1
    # Keepalive always on.
    assert opts[(socket.SOL_SOCKET, socket.SO_KEEPALIVE)] == 1
    # Idle/interval/count tuning is platform-specific: Linux exposes
    # TCP_KEEPIDLE/INTVL/CNT, macOS only TCP_KEEPALIVE (idle seconds).
    if hasattr(socket, "TCP_KEEPIDLE"):
        assert opts[(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)] == 30
        assert opts[(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL)] == 10
        assert opts[(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)] == 3
    elif hasattr(socket, "TCP_KEEPALIVE"):
        assert opts[(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE)] == 30


def test_ws_socket_unreachable_is_silent() -> None:
    """No transport / no socket must not raise (best-effort tuning)."""

    class _BareWS:
        scope: dict = {}

    _disable_nagle(_BareWS())  # no exception
