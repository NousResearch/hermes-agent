"""Transport abstraction for the tui_gateway JSON-RPC server.

Keeps the existing stdio protocol intact while allowing the same dispatcher
logic to run over alternate transports like WebSocket.
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
        """Emit one JSON frame. Return False when the peer is gone."""

    def close(self) -> None:
        """Release any resources owned by this transport."""


_current_transport: contextvars.ContextVar[Optional[Transport]] = contextvars.ContextVar(
    "hermes_gateway_transport",
    default=None,
)


def current_transport() -> Optional[Transport]:
    return _current_transport.get()


def bind_transport(transport: Optional[Transport]):
    return _current_transport.set(transport)


def reset_transport(token) -> None:
    _current_transport.reset(token)


class StdioTransport:
    """Writes JSON frames to a stream resolved lazily at write time."""

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
