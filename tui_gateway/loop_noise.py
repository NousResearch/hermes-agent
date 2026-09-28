"""Suppress benign event-loop teardown noise on the gateway serving loop.

When the Desktop client forcibly closes its WebSocket while the gateway still has pending socket
operations, asyncio logs a traceback per pending ``_call_connection_lost`` callback —
``ConnectionResetError`` (WinError 10054), ``ConnectionAbortedError`` (10053) or ``BrokenPipeError``
(POSIX); one disconnect can emit 50+. They are the expected side effect of the peer hanging up
before our writes drained, so the handler here collapses exactly that class to one debug line and
forwards everything else to the previous handler unchanged.

On Python 3.14, ``asyncio.shield()`` also reports an exception from a shielded future whose outer
await was already cancelled, even when the owner retrieves it (cpython gh-156321). uvicorn's
websockets protocol shields its close/keepalive futures, so a client that stops answering pings
(laptop asleep behind a tunnel) logs ``ConnectionClosedError exception in shielded future`` at
ERROR. A closed websocket there is the same peer hangup and gets the same debug line.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

_log = logging.getLogger(__name__)

# Connection-teardown errors that mean "the peer hung up mid-write".
_BENIGN_TEARDOWN_ERRORS = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)

_SHIELDED_FUTURE_MESSAGE = "exception in shielded future"


def _is_closed_websocket(exc: object) -> bool:
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:
        return False
    return isinstance(exc, ConnectionClosed)


def _is_benign_teardown(context: dict[str, Any]) -> bool:
    """True when the loop error is a peer-hangup during transport teardown.

    The two cases are gated differently. A ``_call_connection_lost`` error needs BOTH the
    exception type AND the callback marker (matched on repr), so the same error type raised
    elsewhere still reaches the default handler. The shielded-future case is gated on
    asyncio's message and a websockets ``ConnectionClosed`` only, with no location check,
    by design: a closed websocket is a peer hangup whichever ``shield()`` retrieved it.
    """
    exc = context.get("exception")
    if _SHIELDED_FUTURE_MESSAGE in str(context.get("message", "")):
        return _is_closed_websocket(exc)
    if not isinstance(exc, _BENIGN_TEARDOWN_ERRORS):
        return False
    marker = "_call_connection_lost"
    return marker in repr(context.get("callback")) or marker in repr(context.get("handle"))


def install_loop_noise_filter(loop: asyncio.AbstractEventLoop) -> None:
    """Chain a teardown-noise filter ahead of the loop's existing handler.

    Idempotent: a loop already carrying the filter is left alone, so it's safe to call
    on every reconnect/serve entry without stacking handlers.
    """
    if getattr(loop, "_hermes_noise_filter_installed", False):
        return
    previous = loop.get_exception_handler()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        if _is_benign_teardown(context):
            _log.debug("ws peer hangup during teardown (suppressed): %s", context.get("exception"))
            return
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover - exotic loop impls
        loop._hermes_noise_filter_installed = True  # type: ignore[attr-defined]
