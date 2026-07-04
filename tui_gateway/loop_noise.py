"""Suppress benign event-loop teardown noise on the gateway serving loop.

When the Desktop client forcibly closes its WebSocket while the gateway still
has pending socket operations, asyncio's transport teardown logs a full
traceback for every pending ``_call_connection_lost`` callback. On Windows this
surfaces as ``ConnectionResetError: [WinError 10054]`` (and the rarer
``ConnectionAbortedError: [WinError 10053]``); on POSIX it is the equivalent
``ConnectionResetError``/``BrokenPipeError``. A single client disconnect can
emit 50+ identical tracebacks into ``errors.log`` (#50005).

These are not actionable — they are the expected side effect of the peer
hanging up before our writes drained. We install a loop exception handler that
collapses exactly this class of teardown error to one debug line and forwards
everything else to asyncio's default handler unchanged, so genuine loop bugs
still surface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_log = logging.getLogger(__name__)

# Connection-teardown errors that mean "the peer hung up mid-write". WinError
# 10054 (connection reset) and 10053 (connection aborted) raise as these.
_BENIGN_TEARDOWN_ERRORS = (
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def _is_benign_teardown(context: dict[str, Any]) -> bool:
    """True when the loop error is a peer-hangup during transport teardown.

    Gated on BOTH the exception type AND the ``_call_connection_lost``
    callback so we only swallow the disconnect flood — any other place these
    errors surface (a real handler, a custom callback) still goes to the
    default handler.
    """
    exc = context.get("exception")
    if not isinstance(exc, _BENIGN_TEARDOWN_ERRORS):
        return False
    # The flood originates from the transport's connection-lost callback. Match
    # on its repr so we don't suppress the same error type raised elsewhere.
    callback = context.get("callback")
    handle = context.get("handle")
    marker = "_call_connection_lost"
    return marker in repr(callback) or marker in repr(handle)


def install_loop_noise_filter(loop: asyncio.AbstractEventLoop) -> None:
    """Chain a teardown-noise filter ahead of the loop's existing handler.

    Idempotent: re-installing on a loop that already has the filter is a no-op,
    so it's safe to call on every reconnect/serve entry.
    """
    if getattr(loop, "_hermes_noise_filter_installed", False):
        return

    previous = loop.get_exception_handler()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        if _is_benign_teardown(context):
            _log.debug(
                "ws peer hangup during teardown (suppressed): %s",
                context.get("exception"),
            )
            return
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    # Mark on the loop instance so a second install (reconnect, re-serve) is a
    # no-op rather than stacking handlers.
    try:
        loop._hermes_noise_filter_installed = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # pragma: no cover - exotic loop impls
        pass
