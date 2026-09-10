"""Transport abstraction for the tui_gateway JSON-RPC server.

A :class:`Transport` forwards a JSON-serialisable dict to its peer, so one dispatcher runs over stdio
(``tui_gateway.entry``) or WebSocket (``tui_gateway.ws``). The request's transport lives in a
``ContextVar`` so pool-dispatched handlers write to the right peer; with nothing bound
``server.write_json`` falls back to the module-level :class:`StdioTransport`, which resolves
``_real_stdout`` lazily so tests that monkey-patch it keep working.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import contextlib
import contextvars
import errno
import json
import logging
import os
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Errno values that mean "the peer is gone" rather than "the host has a real I/O problem". Anything
# outside this set re-raises so it surfaces in the crash log instead of looking like a clean disconnect.
_PEER_GONE_ERRNOS = frozenset({
    errno.EPIPE, errno.ECONNRESET, errno.EBADF, errno.ESHUTDOWN,
    getattr(errno, "WSAECONNRESET", -1), getattr(errno, "WSAESHUTDOWN", -1),  # win32 (no-op on POSIX)
} - {-1})

logger = logging.getLogger(__name__)

# When true, StdioTransport skips ``stream.flush`` after writing: on a half-closed pipe (TUI Node parent quit
# while the gateway still emits) flush can block long enough to starve the worker pool. Python text stdout is
# fully buffered on a pipe, so this ONLY makes sense with ``-u``/``PYTHONUNBUFFERED=1``; otherwise the TUI hangs.
_DISABLE_FLUSH = (os.environ.get("HERMES_TUI_GATEWAY_NO_FLUSH", "") or "").strip().lower() in {"1", "true", "yes", "on"}

@runtime_checkable
class Transport(Protocol):
    """Minimal interface every transport implements."""

    def write(self, obj: dict) -> bool:
        """Emit one JSON frame. Return ``False`` when the peer is gone."""

    def close(self) -> None:
        """Release any resources owned by this transport."""


_current_transport: contextvars.ContextVar[Optional[Transport]] = contextvars.ContextVar(
    "hermes_gateway_transport", default=None
)


def current_transport() -> Optional[Transport]:
    return _current_transport.get()


def bind_transport(transport: Optional[Transport]):
    """Bind *transport* for the current context; returns a token for :func:`reset_transport`."""
    return _current_transport.set(transport)


def reset_transport(token) -> None:
    _current_transport.reset(token)


def _raise_unless_peer_gone(exc: Exception, what: str) -> None:
    """Return when *exc* from a stream write/flush means the peer is gone; re-raise anything else.
    ``False`` from :meth:`StdioTransport.write` is the dispatcher's "broken stdout pipe" signal (``entry.py``
    exits cleanly on it), so programming errors and real host I/O bugs (UnicodeEncodeError from a misconfigured
    locale, ENOSPC, EACCES, ...) MUST re-raise so the crash log records them instead of masquerading as a clean
    disconnect. Peer-gone: BrokenPipeError, ValueError("...closed file..."), OSError errno in _PEER_GONE_ERRNOS."""
    if isinstance(exc, BrokenPipeError):
        return
    if isinstance(exc, ValueError):
        if isinstance(exc, UnicodeEncodeError) or "closed file" not in str(exc):
            raise exc
        return
    if not isinstance(exc, OSError) or exc.errno not in _PEER_GONE_ERRNOS:
        raise exc
    logger.debug("StdioTransport %s peer gone: %s", what, exc)


class StdioTransport:
    """Writes JSON frames to a stream (usually ``sys.stdout``) resolved via a callable, so runtime
    monkey-patches of the stream keep working."""

    __slots__ = ("_stream_getter", "_lock")

    def __init__(self, stream_getter: Callable[[], Any], lock: threading.Lock) -> None:
        self._stream_getter = stream_getter
        self._lock = lock

    def write(self, obj: dict) -> bool:
        """Return ``True`` on success, ``False`` ONLY when the peer is gone (see :func:`_raise_unless_peer_gone`)."""
        # Serialization is OUTSIDE the lock so a large payload can't block other threads' frames. A
        # non-JSON-safe payload is a programming error: re-raise.
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            stream = self._stream_getter()
            try:
                stream.write(line)
            except Exception as e:
                _raise_unless_peer_gone(e, "write")
                return False
            # A flush that *raises* peer-gone means the dispatcher should exit cleanly; one that *hangs*
            # on a half-closed pipe holds the lock until it returns — ``_DISABLE_FLUSH`` skips it entirely.
            if not _DISABLE_FLUSH:
                try:
                    stream.flush()
                except Exception as e:
                    _raise_unless_peer_gone(e, "flush")
                    return False
        return True

    def close(self) -> None:
        return None


@dataclass(eq=False)
class _FanoutPeer:
    transport: Transport
    pending: deque = field(default_factory=deque)
    pending_bytes: int = 0
    writing: bool = False
    attached: bool = True
    generation: int = 0


class FanoutTransport:
    """Ordered, bounded session-event mailboxes; RPC replies remain request-local.

    One slow socket must not stop the emitting turn or any healthy subscriber.
    Each peer has at most one daemon writer and a bounded backlog. On overflow
    it loses its subscription (history/replay is the recovery path), not other
    sessions sharing its socket. A write already in the OS cannot be revoked.
    """

    _MAX_PENDING_FRAMES = 256
    _MAX_PENDING_BYTES = 4 * 1024 * 1024

    def __init__(self, *transports: Transport) -> None:
        self._lock = threading.Lock()
        self._peers: list[_FanoutPeer] = []
        for transport in transports:
            self.attach(transport)

    def attach(self, transport: Transport) -> bool:
        if transport is None or transport is self:
            return False
        with self._lock:
            for peer in self._peers:
                if peer.transport is transport:
                    if peer.attached:
                        return False
                    # Reuse the in-flight writer: reconnect cannot spawn more
                    # threads or overtake a write already inside this socket.
                    peer.attached = True
                    peer.generation += 1
                    return True
            self._peers.append(_FanoutPeer(transport))
            return True

    def _remove(self, peer: _FanoutPeer) -> None:
        # Membership lock held; identity fences a stale writer from removing
        # a later attachment of the same transport.
        peer.attached = False
        peer.pending.clear()
        peer.pending_bytes = 0
        if not peer.writing and peer in self._peers:
            self._peers.remove(peer)

    def detach(self, transport: Transport) -> bool:
        with self._lock:
            for peer in self._peers:
                if peer.attached and peer.transport is transport:
                    self._remove(peer)
                    return True
        return False

    def contains(self, transport: Transport) -> bool:
        with self._lock:
            return any(peer.attached and peer.transport is transport for peer in self._peers)

    def transports(self) -> list[Transport]:
        with self._lock:
            return [peer.transport for peer in self._peers if peer.attached]

    def has_transports(self, *, excluding: Transport | None = None) -> bool:
        return any(peer is not excluding for peer in self.transports())

    def _drain(self, peer: _FanoutPeer) -> None:
        while True:
            with self._lock:
                if not peer.attached or not peer.pending:
                    peer.writing = False
                    if not peer.attached:
                        self._remove(peer)
                    return
                generation = peer.generation
                frame, size = peer.pending.popleft()
                peer.pending_bytes -= size
            try:
                from tui_gateway.ws import WSTransport
                if isinstance(peer.transport, WSTransport):
                    # write() acknowledges buffered tokens/timeouts, not socket
                    # progress. Await the real send so WS cannot move an
                    # unbounded backlog underneath this bounded mailbox.
                    from agent.async_utils import safe_schedule_threadsafe
                    future = safe_schedule_threadsafe(
                        peer.transport.write_async(frame), peer.transport._loop)
                    ok = future is not None and future.result()
                else:
                    ok = peer.transport.write(frame)
            except Exception:
                logger.debug("fanout write failed; pruning peer", exc_info=True)
                ok = False
            if not ok:
                with self._lock:
                    if peer.generation != generation:
                        continue
                    peer.writing = False
                    self._remove(peer)
                return

    def write(self, obj: dict) -> bool:
        # Freeze the queued frame so a caller cannot mutate it after admission.
        encoded = json.dumps(obj, ensure_ascii=False)
        size = len(encoded.encode("utf-8", errors="surrogatepass"))
        frame = json.loads(encoded)
        with self._lock:
            for peer in list(self._peers):
                if not peer.attached:
                    continue
                if (len(peer.pending) >= self._MAX_PENDING_FRAMES
                        or peer.pending_bytes + size > self._MAX_PENDING_BYTES):
                    logger.warning("fanout subscriber backlog full; detaching peer")
                    self._remove(peer)
                    continue
                peer.pending.append((frame, size))
                peer.pending_bytes += size
                if not peer.writing:
                    peer.writing = True
                    threading.Thread(target=self._drain, args=(peer,),
                                     name="tui-fanout", daemon=True).start()
            return any(peer.attached for peer in self._peers)

    def close(self) -> None:
        """Detach without closing sockets owned by the connection handlers."""
        with self._lock:
            for peer in list(self._peers):
                self._remove(peer)


class TeeTransport:
    """Mirrors writes to one primary plus N best-effort secondaries. The primary's return value (and
    exceptions) determine the result; secondaries swallow failures so a wedged sidecar never stalls the
    main IO path. Used by the PTY child: every emit lands on stdio (Ink) AND a back-WS for the dashboard."""

    __slots__ = ("_primary", "_secondaries")

    def __init__(self, primary: "Transport", *secondaries: "Transport") -> None:
        self._primary = primary
        self._secondaries = secondaries

    def write(self, obj: dict) -> bool:
        # Primary first so a slow sidecar (WS publisher) never delays Ink/stdio.
        ok = self._primary.write(obj)
        for sec in self._secondaries:
            with contextlib.suppress(Exception):
                sec.write(obj)
        return ok

    def close(self) -> None:
        try:
            self._primary.close()
        finally:
            for sec in self._secondaries:
                with contextlib.suppress(Exception):
                    sec.close()
