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
import json
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable


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
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                stream = self._stream_getter()
                stream.write(line)
                stream.flush()
            return True
        except BrokenPipeError:
            return False

    def close(self) -> None:
        return None


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
