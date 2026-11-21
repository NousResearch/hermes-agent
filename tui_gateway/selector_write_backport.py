"""Backport of the ``_SelectorSocketTransport._write_send`` fix from python/cpython#146518.

CPython 3.12-3.14 pop the head chunk off ``self._buffer`` *before* calling ``sock.send()`` on
the sendmsg-less write path — the only path a Windows selector loop has, since
``socket.sendmsg`` does not exist there. When ``send()`` raises ``BlockingIOError`` (a stalled
Desktop client, kernel send buffer full) the chunk is gone but the writer callback stays
registered, so every later loop iteration trips ``assert self._buffer, 'Data should not be
empty'``: tens of thousands of identical tracebacks per minute and unbounded memory growth
(#140246 saw ~22 GB). CPython fixed it on main (3.15) by putting the chunk back; this module
installs that same method body on interpreters whose shipped method still drops the chunk.

``_run_serve`` calls :func:`backport_selector_write_send` on Windows before uvicorn starts,
because ``_SelectorSocketTransport.__init__`` binds ``self._write_ready`` to ``self._write_send``
at construction — a transport that already exists keeps the method it was born with.
"""

from __future__ import annotations

import asyncio.selector_events
import collections
import logging
import socket
import types
from typing import Callable

_log = logging.getLogger(__name__)


def _write_send_keeping_blocked_chunk(self) -> None:
    """``_write_send`` as fixed by python/cpython#146518.

    Byte-for-byte the 3.12-3.14 method except that ``BlockingIOError``/``InterruptedError``
    puts the popped chunk back instead of dropping it. The ``_buffer_size`` counter that PR
    also introduced does not exist on these interpreters and is deliberately left out.
    """
    assert self._buffer, "Data should not be empty"
    if self._conn_lost:
        return
    try:
        buffer = self._buffer.popleft()
        n = self._sock.send(buffer)
        if n != len(buffer):
            # Not all data was written
            self._buffer.appendleft(buffer[n:])
    except (BlockingIOError, InterruptedError):
        self._buffer.appendleft(buffer)
        return
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # health: allow BLE001 -- mirrors the stdlib handler; _fatal_error forwards exc to the loop's exception handler
        self._loop._remove_writer(self._sock_fd)
        self._buffer.clear()
        self._fatal_error(exc, "Fatal write error on socket transport")
        if self._empty_waiter is not None:
            self._empty_waiter.set_exception(exc)
    else:
        self._maybe_resume_protocol()  # May append to buffer.
        if not self._buffer:
            self._loop._remove_writer(self._sock_fd)
            if self._empty_waiter is not None:
                self._empty_waiter.set_result(None)
            if self._closing:
                self._call_connection_lost(None)
            elif self._eof:
                self._sock.shutdown(socket.SHUT_WR)


def write_send_drops_blocked_chunk(write_send: Callable[[object], None]) -> bool:
    """True when ``write_send`` loses the head chunk on a ``BlockingIOError``.

    Drives the method against a stub transport whose socket refuses the write — the state a
    stalled client leaves a real transport in. A method of an unexpected shape (anything else
    raised) counts as "not the bug we know", so nothing gets patched blind.
    """

    class _RefusingSocket:
        def send(self, data):
            raise BlockingIOError

    probe = types.SimpleNamespace(
        _buffer=collections.deque([memoryview(b"probe")]),
        _conn_lost=0,
        _sock=_RefusingSocket(),
    )
    try:
        write_send(probe)
    except Exception:  # health: allow BLE001 -- an unknown method shape must not be patched; the traceback is the diagnostic
        _log.debug("_write_send probe raised; leaving the shipped method alone", exc_info=True)
        return False
    return not probe._buffer


def backport_selector_write_send(transport_cls=None) -> bool:
    """Install the fixed ``_write_send`` on ``transport_cls`` when its own still drops chunks.

    Returns True when the method body was replaced. False when the class has no
    ``_write_send`` (3.11's bytearray buffer never had the bug) or when the shipped method
    already keeps the chunk — a fixed interpreter, or an earlier call of this function — so
    it is safe to call on every serve entry.
    """
    if transport_cls is None:
        transport_cls = asyncio.selector_events._SelectorSocketTransport
    shipped = getattr(transport_cls, "_write_send", None)
    if shipped is None or not write_send_drops_blocked_chunk(shipped):
        return False
    transport_cls._write_send = _write_send_keeping_blocked_chunk
    _log.debug("installed python/cpython#146518 _write_send backport on %s", transport_cls.__name__)
    return True
