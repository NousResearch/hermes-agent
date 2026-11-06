"""The selector loop's ``_write_send`` must keep a chunk the socket refused (#140246).

CPython 3.12-3.14 pop the head chunk before ``send()``; a ``BlockingIOError`` then drops it
while the writer callback stays registered, and every later loop iteration trips
``assert self._buffer, 'Data should not be empty'`` — the traceback storm that grew the
Windows dashboard to ~22 GB. The shipped and the backported method are driven through the
same socket states so the replacement is proven to differ only where python/cpython#146518
does. Host-independent: the method exists on every platform, Windows is merely the one whose
selector loop has no sendmsg alternative to it.
"""

from __future__ import annotations

import asyncio
import asyncio.selector_events
import collections
import socket
import time
import types

import pytest

from tui_gateway import selector_write_backport as backport

_Transport = asyncio.selector_events._SelectorSocketTransport
_SHIPPED = _Transport.__dict__.get("_write_send")


def _shipped_or_skip():
    if _SHIPPED is None:
        pytest.skip("3.11 has no _write_send (its bytearray buffer never dropped chunks)")
    if not backport.write_send_drops_blocked_chunk(_SHIPPED):
        pytest.skip("this interpreter already carries python/cpython#146518")
    return _SHIPPED


class _Socket:
    """Scripted ``send`` outcomes: an int is the byte count accepted, an exception is raised."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.shutdowns = []

    def send(self, data):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def shutdown(self, how):
        self.shutdowns.append(how)


class _Waiter:
    def __init__(self):
        self.result = None
        self.exception = None

    def set_result(self, value):
        self.result = ("result", value)

    def set_exception(self, exc):
        self.exception = type(exc)


def _transport(chunks, outcomes=(), *, closing=False, eof=False, waiter=False, conn_lost=0):
    """A stand-in with every attribute ``_write_send`` touches, recording what it did."""
    t = types.SimpleNamespace(
        _buffer=collections.deque(memoryview(c) for c in chunks),
        _conn_lost=conn_lost,
        _sock=_Socket(outcomes),
        _sock_fd=7,
        _loop=types.SimpleNamespace(removed=[]),
        _empty_waiter=_Waiter() if waiter else None,
        _closing=closing,
        _eof=eof,
        resumed=0,
        fatal=[],
        lost=[],
    )
    t._loop._remove_writer = t._loop.removed.append
    t._maybe_resume_protocol = lambda: setattr(t, "resumed", t.resumed + 1)
    t._fatal_error = lambda exc, message: t.fatal.append((type(exc), message))
    t._call_connection_lost = t.lost.append
    return t


def _state(t):
    """Everything observable after one ``_write_send`` call."""
    waiter = t._empty_waiter
    return {
        "buffer": [bytes(c) for c in t._buffer],
        "writer_removed": list(t._loop.removed),
        "resumed": t.resumed,
        "fatal": list(t.fatal),
        "lost": list(t.lost),
        "shutdowns": list(t._sock.shutdowns),
        "waiter": None if waiter is None else (waiter.result, waiter.exception),
    }


@pytest.mark.parametrize("refusal", [BlockingIOError, InterruptedError])
def test_backport_keeps_the_chunk_the_socket_refused(refusal):
    t = _transport([b"payload", b"next"], [refusal()])
    backport._write_send_keeping_blocked_chunk(t)
    assert [bytes(c) for c in t._buffer] == [b"payload", b"next"]
    # The writer stays registered and nothing was reported: the next write-ready retries.
    assert t._loop.removed == [] and t.fatal == [] and t.resumed == 0


def test_shipped_method_drops_the_chunk_then_asserts_forever():
    shipped = _shipped_or_skip()
    t = _transport([b"payload"], [BlockingIOError(), BlockingIOError()])
    shipped(t)
    assert list(t._buffer) == []  # the chunk is gone while the writer is still registered
    with pytest.raises(AssertionError, match="Data should not be empty"):
        shipped(t)  # what every later loop iteration then logs (#140246)


_SCENARIOS = {
    "partial-send": dict(chunks=[b"payload"], outcomes=[3]),
    "complete-send": dict(chunks=[b"payload"], outcomes=[7], waiter=True),
    "complete-send-more-queued": dict(chunks=[b"pay", b"load"], outcomes=[3]),
    "complete-send-closing": dict(chunks=[b"payload"], outcomes=[7], closing=True),
    "complete-send-eof": dict(chunks=[b"payload"], outcomes=[7], eof=True),
    "fatal-error": dict(
        chunks=[b"payload", b"next"], outcomes=[ConnectionResetError()], waiter=True
    ),
    "connection-lost": dict(chunks=[b"payload"], conn_lost=1),
}


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_backport_matches_the_shipped_method_on_every_other_path(scenario):
    if _SHIPPED is None:
        pytest.skip("3.11 has no _write_send to mirror")
    spec = _SCENARIOS[scenario]
    expected = _transport(**spec)
    _SHIPPED(expected)
    actual = _transport(**spec)
    backport._write_send_keeping_blocked_chunk(actual)
    assert _state(actual) == _state(expected)


def _dropping_write_send(self):
    self._buffer.popleft()
    try:
        self._sock.send(b"")
    except BlockingIOError:
        pass


def test_probe_tells_a_dropping_method_from_a_keeping_one():
    def unknown_shape(self):
        raise AttributeError("an attribute some future transport grew")

    assert backport.write_send_drops_blocked_chunk(_dropping_write_send) is True
    keeping = backport._write_send_keeping_blocked_chunk
    assert backport.write_send_drops_blocked_chunk(keeping) is False
    assert backport.write_send_drops_blocked_chunk(unknown_shape) is False


def test_install_patches_only_a_dropping_class_and_only_once():
    class _Buggy:
        _write_send = _dropping_write_send

    class _Fixed:
        _write_send = backport._write_send_keeping_blocked_chunk

    class _Legacy:  # 3.11: no _write_send at all
        pass

    assert backport.backport_selector_write_send(_Buggy) is True
    assert _Buggy._write_send is backport._write_send_keeping_blocked_chunk
    assert backport.backport_selector_write_send(_Buggy) is False
    assert backport.backport_selector_write_send(_Fixed) is False
    assert _Fixed._write_send is backport._write_send_keeping_blocked_chunk
    assert backport.backport_selector_write_send(_Legacy) is False
    assert not hasattr(_Legacy, "_write_send")


def _drain(sock):
    out = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except BlockingIOError:
            return bytes(out)
        if not chunk:
            return bytes(out)
        out += chunk


@pytest.fixture
def stalled_transport():
    """A real ``_SelectorSocketTransport`` whose peer stopped reading: the next ``send`` blocks."""
    loop = asyncio.SelectorEventLoop()
    ours, peer = socket.socketpair()
    ours.setblocking(False)
    peer.setblocking(False)
    transport = _Transport(loop, ours, asyncio.Protocol())
    for _ in range(10_000):  # loopback buffers are a few MB at most
        try:
            ours.send(b"x" * 65536)
        except BlockingIOError:
            break
    else:
        pytest.fail("could not fill the socket send buffer")
    try:
        yield transport, peer
    finally:
        transport._buffer.clear()
        transport.close()
        loop.run_until_complete(asyncio.sleep(0))
        peer.close()
        loop.close()


def test_shipped_method_storms_on_a_stalled_client(stalled_transport):
    shipped = _shipped_or_skip()
    transport, _peer = stalled_transport
    transport._buffer.append(memoryview(b"payload"))
    shipped(transport)
    assert list(transport._buffer) == []
    with pytest.raises(AssertionError, match="Data should not be empty"):
        shipped(transport)


def test_backport_delivers_the_chunk_once_the_client_reads_again(stalled_transport, monkeypatch):
    shipped = _shipped_or_skip()
    monkeypatch.setattr(_Transport, "_write_send", shipped)
    assert backport.backport_selector_write_send() is True
    transport, peer = stalled_transport
    transport._buffer.append(memoryview(b"payload"))

    transport._write_send()
    transport._write_send()  # a second write-ready on the still-stalled client: no assertion
    assert [bytes(c) for c in transport._buffer] == [b"payload"]

    received = bytearray()
    for _ in range(200):
        received += _drain(peer)
        transport._write_send()
        if not transport._buffer:
            break
        time.sleep(0.01)
    assert not transport._buffer, "the retained chunk never went out"
    received += _drain(peer)
    assert received.endswith(b"payload")
