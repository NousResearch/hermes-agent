"""Transport abstraction for the tui_gateway JSON-RPC server.

Historically the gateway wrote every JSON frame directly to real stdout.  This
module decouples the I/O sink from the handler logic so the same dispatcher
can be driven over stdio (``tui_gateway.entry``) or WebSocket
(``tui_gateway.ws``) without duplicating code.

A :class:`Transport` is anything that can accept a JSON-serialisable dict and
forward it to its peer.  The active transport for the current request is
tracked in a :class:`contextvars.ContextVar` so handlers — including those
dispatched onto the worker pool — route their writes to the right peer.

Backward compatibility
----------------------
``tui_gateway.server.write_json`` still works without any transport bound.
When nothing is on the contextvar and no session-level transport is found,
it falls back to the module-level :class:`StdioTransport`, which wraps the
original ``_real_stdout`` + ``_stdout_lock`` pair.  Tests that monkey-patch
``server._real_stdout`` continue to work because the stdio transport resolves
the stream lazily through a callback.
"""

from __future__ import annotations

import contextvars
import errno
import json
import logging
import os
import queue
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Errno values that mean "the peer is gone" rather than "the host has a
# real I/O problem".  Anything outside this set re-raises so it surfaces
# in the crash log instead of looking like a clean disconnect.
_PEER_GONE_ERRNOS = frozenset({
    errno.EPIPE,        # write to closed pipe (POSIX)
    errno.ECONNRESET,   # peer reset the connection
    errno.EBADF,        # fd closed under us
    errno.ESHUTDOWN,    # transport endpoint shut down
    getattr(errno, "WSAECONNRESET", -1),  # win32 mapping (no-op on POSIX)
    getattr(errno, "WSAESHUTDOWN", -1),
} - {-1})

logger = logging.getLogger(__name__)


def _is_peer_gone(exc: BaseException) -> bool:
    """True when *exc* means "the peer is gone" rather than a host problem.

    Mirrors StdioTransport.write's classification so the async writer
    thread treats exactly the same errors as a clean disconnect and keeps
    programming/host errors (non-JSON-safe payloads, encoding misconfig,
    ENOSPC, ...) visible as exceptions instead of swallowing them.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, ValueError):
        # ValueError("I/O operation on closed file") is the ONLY ValueError
        # that means "peer gone".  UnicodeEncodeError (a ValueError subclass
        # for misconfigured locales) is a real bug — keep it visible.
        if isinstance(exc, UnicodeEncodeError):
            return False
        return "closed file" in str(exc)
    if isinstance(exc, OSError):
        return exc.errno in _PEER_GONE_ERRNOS
    return False

# Optional knob: when true, StdioTransport does not call ``stream.flush``
# after writing.  Use this on environments where a half-closed pipe (TUI
# Node parent quit while the gateway is still emitting events) makes
# flush block long enough to starve the rest of the worker pool.
#
# IMPORTANT: Python text stdout is fully buffered when attached to a
# pipe (the TUI case), so this knob ONLY makes sense when the gateway
# is launched with ``-u`` or ``PYTHONUNBUFFERED=1``.  Without one of
# those, JSON-RPC frames will accumulate in the buffer and the TUI
# will hang waiting for ``gateway.ready``.  Default stays off so the
# existing flush-after-write behaviour is unchanged.
_DISABLE_FLUSH = (os.environ.get("HERMES_TUI_GATEWAY_NO_FLUSH", "") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@runtime_checkable
class Transport(Protocol):
    """Minimal interface every transport implements."""

    def write(self, obj: dict) -> bool:
        """Emit one JSON frame. Return ``False`` when the peer is gone."""

    def close(self) -> None:
        """Release any resources owned by this transport."""


_current_transport: contextvars.ContextVar[Optional[Transport]] = (
    contextvars.ContextVar(
        "hermes_gateway_transport",
        default=None,
    )
)


def current_transport() -> Optional[Transport]:
    """Return the transport bound for the current request, if any."""
    return _current_transport.get()


def bind_transport(transport: Optional[Transport]):
    """Bind *transport* for the current context. Returns a token for :func:`reset_transport`."""
    return _current_transport.set(transport)


def reset_transport(token) -> None:
    """Restore the transport binding captured by :func:`bind_transport`."""
    _current_transport.reset(token)


class StdioTransport:
    """Writes JSON frames to a stream (usually ``sys.stdout``).

    The stream is resolved via a callable so runtime monkey-patches of the
    underlying stream continue to work — this preserves the behaviour the
    existing test suite relies on (``monkeypatch.setattr(server, "_real_stdout", ...)``).
    """

    __slots__ = ("_stream_getter", "_lock")

    def __init__(self, stream_getter: Callable[[], Any], lock: threading.Lock) -> None:
        self._stream_getter = stream_getter
        self._lock = lock

    def write(self, obj: dict) -> bool:
        """Return ``True`` on success, ``False`` ONLY when the peer is gone.

        Returning ``False`` is the dispatcher's "broken stdout pipe" signal
        — ``entry.py`` calls ``sys.exit(0)`` when ``write_json`` reports
        ``False``.  So programming errors (non-JSON-safe payloads, encoding
        misconfig, unexpected ValueErrors, host I/O bugs like ENOSPC) MUST
        NOT return ``False``, otherwise a real bug looks like a clean
        disconnect and is harder to diagnose.  Those re-raise so the
        existing crash-log infrastructure records the traceback.

        Peer-gone branches:
          * ``BrokenPipeError``
          * ``ValueError("...closed file...")``
          * ``OSError`` whose errno is in :data:`_PEER_GONE_ERRNOS`
            (EPIPE / ECONNRESET / EBADF / ESHUTDOWN; plus WSA mappings
            on Windows).  Other OSError errnos (ENOSPC, EACCES, ...) are
            real host problems and re-raise.
        """
        # Serialization is OUTSIDE the lock so a large payload can't
        # block other threads emitting their own frames.  A non-JSON-safe
        # payload is a programming error: re-raise so the crash log
        # captures it instead of silently exiting via the False path.
        line = json.dumps(obj, ensure_ascii=False) + "\n"

        with self._lock:
            stream = self._stream_getter()
            try:
                stream.write(line)
            except BrokenPipeError:
                return False
            except ValueError as e:
                # ValueError("I/O operation on closed file") is the
                # ONLY ValueError that means "peer gone".  Anything
                # else — including UnicodeEncodeError, which is a
                # ValueError subclass for misconfigured locales —
                # is a real bug; re-raise so it surfaces in the crash log.
                if isinstance(e, UnicodeEncodeError) or "closed file" not in str(e):
                    raise
                return False
            except OSError as e:
                if e.errno not in _PEER_GONE_ERRNOS:
                    raise
                logger.debug("StdioTransport write peer gone: %s", e)
                return False

            # A flush that *raises* with a peer-gone errno means the
            # dispatcher should exit cleanly.  A flush that *hangs* on
            # a half-closed pipe holds the lock until it returns — see
            # ``_DISABLE_FLUSH`` for the "skip flush entirely" escape
            # hatch.
            if not _DISABLE_FLUSH:
                try:
                    stream.flush()
                except BrokenPipeError:
                    return False
                except ValueError as e:
                    if isinstance(e, UnicodeEncodeError) or "closed file" not in str(e):
                        raise
                    return False
                except OSError as e:
                    if e.errno not in _PEER_GONE_ERRNOS:
                        raise
                    logger.debug("StdioTransport flush peer gone: %s", e)
                    return False

        return True

    def close(self) -> None:
        return None


# ── Non-blocking stream writer (TUI backpressure fix) ─────────────────
# The TUI gateway streams JSON-RPC frames to the Ink client over a stdio
# pipe.  When the client stops draining stdout (heavy re-render, React
# reconciliation pause, frozen terminal), the OS pipe buffer fills and a
# synchronous ``stream.write()`` blocks the thread that is consuming the
# provider stream — the interrupt check never runs and the turn appears
# hung with no way to cancel.
#
# BufferedStreamWriter decouples producers from the sink with a bounded
# queue drained by one dedicated writer thread:
#
#   * Streaming frames (``message.delta`` / ``reasoning.delta`` /
#     ``thinking.delta``) coalesce into a single pending batch and are
#     pushed WITHOUT ever blocking the producer.  When the writer can't
#     keep up, accumulated deltas are dropped — cosmetic only, because the
#     canonical response text always arrives in the terminal
#     ``message.complete`` frame.
#   * Control frames (RPC responses, ``message.complete``, tool events,
#     ...) drain the pending deltas ahead of themselves into ONE queue
#     item, so on-the-wire order is preserved.  A control push waits at
#     most ``_STREAM_CONTROL_PUSH_TIMEOUT_S`` for queue space; a sink that
#     is still wedged then degrades to the transport's normal peer-gone
#     signal (``False``) instead of deadlocking the agent thread.
#   * Control ordering fence: a control frame that drains the pending
#     batch claims it (``_control_claimed``) until its queue item has been
#     written by the writer.  The writer's opportunistic delta flush is
#     held while any claim is outstanding, so a delta produced *after* a
#     control push began can never reach the sink before that control —
#     even when the control push is still blocked waiting for queue space.
#   * ``close()`` signals the writer and joins with a bounded timeout, so
#     a wedged write can never leak a thread past process teardown (the
#     thread is daemon and dies with the process if the pipe never drains).
#     If the queue is full at close time (so ``_STOP`` can't be enqueued),
#     the writer still exits on its own once queued + pending work drains.
#
# This mirrors the coalescing semantics the WebSocket transport already
# has (``tui_gateway.ws.WSTransport``) for the same high-frequency frame
# types; here the flush boundary is the writer thread instead of an event
# loop.  Only the TUI entry point installs it (``tui_gateway.entry``), so
# non-TUI transports and the synchronous module default are unchanged.

# High-frequency, display-only frames eligible for coalescing.  Keep in
# sync with ``tui_gateway.ws._STREAMING_EVENT_TYPES``.
_STREAMING_EVENT_TYPES = frozenset({
    "message.delta",
    "reasoning.delta",
    "thinking.delta",
})

# Bounded queue depth, in flush batches (not frames — deltas coalesce into
# one batch).  Control frames are the typical occupants once the writer is
# wedged behind a full pipe.
_STREAM_QUEUE_MAXSIZE = 256
# Max time a control-frame push waits for queue space before treating the
# peer as dead.  Matches the WS transport's slow-write bound
# (``tui_gateway.ws._WS_WRITE_TIMEOUT_S``); a local stdio pipe that stays
# full for this long means the Ink client is effectively gone.
_STREAM_CONTROL_PUSH_TIMEOUT_S = 10.0
# Max time a streamed token waits in the pending batch before flush
# (~30 fps cadence; mirrors ``tui_gateway.ws._TOKEN_COALESCE_S``).
_STREAM_TOKEN_COALESCE_S = 0.033
# Memory bound for the pending delta batch while the writer is wedged:
# once reached, older accumulated deltas are dropped (display-only).
_STREAM_MAX_PENDING_DELTAS = 512
# Bounded join budget for close(); a writer wedged on a full pipe cannot
# strand the caller.
_STREAM_CLOSE_JOIN_TIMEOUT_S = 1.0

_STOP = object()


class BufferedStreamWriter:
    """Transport wrapper that keeps producer threads off a blocking sink.

    Implements the :class:`Transport` protocol (``write``/``close``) so it
    can stand in for any transport the gateway already uses.  See the
    module comment above for the backpressure rationale and ordering
    guarantees.
    """

    __slots__ = (
        "_inner",
        "_queue",
        "_pending",
        "_pending_lock",
        "_thread_lock",
        "_thread",
        "_closed",
        "_control_claimed",
        "_coalesce_s",
        "_control_push_timeout_s",
        "_max_pending_deltas",
        "_close_join_timeout_s",
    )

    def __init__(
        self,
        inner: Transport,
        *,
        queue_maxsize: int = _STREAM_QUEUE_MAXSIZE,
        control_push_timeout_s: float = _STREAM_CONTROL_PUSH_TIMEOUT_S,
        coalesce_s: float = _STREAM_TOKEN_COALESCE_S,
        max_pending_deltas: int = _STREAM_MAX_PENDING_DELTAS,
        close_join_timeout_s: float = _STREAM_CLOSE_JOIN_TIMEOUT_S,
    ) -> None:
        self._inner = inner
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._pending: list[dict] = []
        self._pending_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        # Number of control batches that have drained the pending delta
        # batch but have not yet been written to the sink.  While nonzero,
        # the writer's opportunistic delta flush is held so a later delta
        # can never overtake a control frame (see _drain_loop).
        self._control_claimed = 0
        self._coalesce_s = coalesce_s
        self._control_push_timeout_s = control_push_timeout_s
        self._max_pending_deltas = max_pending_deltas
        self._close_join_timeout_s = close_join_timeout_s

    @staticmethod
    def _is_streaming_frame(obj: dict) -> bool:
        """True for high-frequency per-token frames eligible for coalescing."""
        params = obj.get("params") if isinstance(obj, dict) else None
        if not isinstance(params, dict):
            return False
        return params.get("type") in _STREAMING_EVENT_TYPES

    def _ensure_writer(self) -> None:
        """Lazily start the writer thread on first frame.

        Starting on first write keeps creation side-effect free for the
        module default (which is never wrapped in production) and for tests
        that construct writers but emit nothing.
        """
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._drain_loop,
                    name="tui-stdio-writer",
                    daemon=True,
                )
                self._thread.start()

    def write(self, obj: dict) -> bool:
        """Enqueue one frame.  Streaming frames never block; control frames
        wait at most ``control_push_timeout_s`` for queue space, then return
        ``False`` (peer gone) so the caller's normal dead-transport handling
        applies instead of deadlocking the agent."""
        if self._closed:
            return False
        self._ensure_writer()

        if self._is_streaming_frame(obj):
            # Coalesce into the pending batch; return immediately.  The
            # writer thread flushes the batch on its coalesce cadence or
            # when a control frame drains it.
            with self._pending_lock:
                if self._closed:
                    return False
                if len(self._pending) >= self._max_pending_deltas:
                    # Writer is wedged behind a full pipe — keep only the
                    # newest delta and drop the backlog.  Display-only; the
                    # terminal message.complete frame carries the canonical
                    # text.
                    self._pending = [obj]
                else:
                    self._pending.append(obj)
            return True

        # Control frame: append it behind the pending deltas in ONE queue
        # item so it can never overtake the tokens it follows.  Claim the
        # batch before the (possibly blocking) put: while the claim is
        # outstanding the writer holds its opportunistic delta flush, so a
        # delta produced after this control push began can never reach the
        # sink before the control (see _drain_loop).
        with self._pending_lock:
            batch = self._pending
            self._pending = []
            batch = batch + [obj]
            self._control_claimed += 1
        try:
            self._queue.put(batch, timeout=self._control_push_timeout_s)
        except queue.Full:
            logger.warning(
                "stdio stream writer wedged: no queue space after %.0fs — "
                "treating peer as dead",
                self._control_push_timeout_s,
            )
            # The claim never made it to the queue — release it so a
            # subsequent close/drain does not wait on a phantom control.
            with self._pending_lock:
                self._control_claimed -= 1
            self._closed = True
            return False
        return not self._closed

    def _write_batch(self, batch: list[dict]) -> bool:
        """Write one batch to the inner transport.  Returns False when the
        peer is gone (inner returned False or raised a peer-gone error); the
        writer loop then stops instead of churning against a dead pipe.

        Programming/host errors (non-JSON-safe payloads, encoding misconfig,
        ENOSPC, ...) are NOT treated as a clean disconnect: they are logged
        with their traceback and re-raised so the gateway's thread panic
        hook records them in the crash log — the same visibility contract
        StdioTransport.write documents for its synchronous path.
        """
        inner = self._inner
        for obj in batch:
            try:
                if not inner.write(obj):
                    return False
            except Exception as exc:
                if _is_peer_gone(exc):
                    logger.debug("stdio stream writer peer gone: %s", exc)
                    return False
                self._closed = True
                logger.exception("stdio stream writer inner write failed")
                raise
        return True

    def _drain_loop(self) -> None:
        q = self._queue
        while True:
            try:
                batch = q.get(timeout=self._coalesce_s)
            except queue.Empty:
                batch = None
            if batch is _STOP:
                break
            if batch:
                if not self._write_batch(batch):
                    self._closed = True
                    return
                # Every queued item is a control batch (deltas never enter
                # the queue), so a successful write releases its claim.
                with self._pending_lock:
                    self._control_claimed -= 1
            # Opportunistically flush deltas that coalesced while the writer
            # was busy, so a pure delta stream still reaches the client even
            # with no control frame in sight.  Held while any control batch
            # is claimed but not yet written — a later delta must never
            # reach the sink before the control that precedes it.
            with self._pending_lock:
                if self._control_claimed:
                    continue
                pending = self._pending
                self._pending = []
            if pending and not self._write_batch(pending):
                self._closed = True
                return
            # close() may not have been able to enqueue _STOP (queue was
            # full); once everything queued + pending has drained, a closed
            # writer exits on its own instead of waiting forever.  With
            # _closed set, no new claims can arrive, so reaching this check
            # with an empty queue implies all claims were written above.
            if self._closed and q.empty() and self._control_claimed == 0:
                return
        # Stop requested: drain everything already queued + still-outstanding
        # claims + the pending batch.  A control whose queue.put was blocked
        # when close() enqueued _STOP may land AFTER the sentinel (put order
        # is not claim order); keep draining until every claim is written so
        # no claimed control — or the deltas held behind it — is dropped.
        while self._control_claimed > 0 or not q.empty():
            try:
                batch = q.get(timeout=min(self._coalesce_s, 0.05))
            except queue.Empty:
                batch = None
            if batch is _STOP or batch is None:
                continue
            if not self._write_batch(batch):
                break
            with self._pending_lock:
                self._control_claimed -= 1
        with self._pending_lock:
            if self._control_claimed == 0:
                pending = self._pending
                self._pending = []
            else:
                # A control batch is still claimed but unwritten — hold the
                # deltas rather than let them overtake it (display-only;
                # message.complete carries the canonical text).
                pending = None
        if pending:
            self._write_batch(pending)

    def close(self) -> None:
        """Signal the writer to stop and join it (bounded).

        Remaining queued/pending frames are drained best-effort.  If the
        writer is wedged on a full pipe the join times out and the daemon
        thread is left to die with the process — it can never strand
        shutdown.
        """
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put(_STOP, timeout=self._close_join_timeout_s)
        except queue.Full:
            pass  # Writer is wedged; it will exit when the pipe drains.
        thread.join(timeout=self._close_join_timeout_s)


class TeeTransport:
    """Mirrors writes to one primary plus N best-effort secondaries.

    The primary's return value (and exceptions) determine the result —
    secondaries swallow failures so a wedged sidecar never stalls the
    main IO path.  Used by the PTY child so every dispatcher emit lands
    on stdio (Ink) AND on a back-WS feeding the dashboard sidebar.
    """

    __slots__ = ("_primary", "_secondaries")

    def __init__(self, primary: "Transport", *secondaries: "Transport") -> None:
        self._primary = primary
        self._secondaries = secondaries

    def write(self, obj: dict) -> bool:
        # Primary first so a slow sidecar (WS publisher) never delays Ink/stdio.
        ok = self._primary.write(obj)
        for sec in self._secondaries:
            try:
                sec.write(obj)
            except Exception:
                pass
        return ok

    def close(self) -> None:
        try:
            self._primary.close()
        finally:
            for sec in self._secondaries:
                try:
                    sec.close()
                except Exception:
                    pass
