"""Suppress benign event-loop teardown noise on the gateway serving loop.

When the Desktop client forcibly closes its WebSocket while the gateway still has pending socket
operations, asyncio logs a traceback per pending ``_call_connection_lost`` callback —
``ConnectionResetError`` (WinError 10054), ``ConnectionAbortedError`` (10053) or ``BrokenPipeError``
(POSIX); one disconnect can emit 50+. They are the expected side effect of the peer hanging up
before our writes drained, so the handler here collapses exactly that class to one debug line and
forwards everything else to the previous handler unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

_log = logging.getLogger(__name__)

# Connection-teardown errors that mean "the peer hung up mid-write".
_BENIGN_TEARDOWN_ERRORS = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


def _is_benign_teardown(context: dict[str, Any]) -> bool:
    """True when the loop error is a peer-hangup during transport teardown.

    Gated on BOTH the exception type AND the ``_call_connection_lost`` callback (matched
    on repr) so the same error type raised elsewhere still reaches the default handler.
    """
    if not isinstance(context.get("exception"), _BENIGN_TEARDOWN_ERRORS):
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
