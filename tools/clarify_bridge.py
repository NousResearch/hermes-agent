"""
Gateway bridge for the ``clarify`` tool.

The interactive CLI has always been able to pause the agent and ask the user a
question via ``tools.clarify_tool`` + ``HermesCLI._clarify_callback``.  The
gateway did not wire that callback — any attempt to use ``clarify`` from a
messaging platform (Telegram, Discord, …) returns
``"Clarify tool is not available in this execution context."``.

This module provides the gateway-side glue, mirroring the proven pattern used
for dangerous-command approvals in ``tools/approval.py``:

1. The gateway registers a per-session *notify* callback via
   :func:`register_gateway_clarify_notify` before starting an agent turn.
2. The gateway injects :func:`gateway_clarify_callback` as the agent's
   ``clarify_callback``.
3. When the agent calls ``clarify``, that callback creates a
   :class:`_ClarifyEntry` (question + choices + ``threading.Event``), invokes
   the notify callback (which the gateway uses to send the question to the
   user), then blocks on the event.
4. When the next user message arrives, the gateway routes it to
   :func:`resolve_gateway_clarify` instead of starting a new turn.  That sets
   the event and the agent thread resumes with the user's answer.

Thread safety: all shared state is guarded by ``_lock``.  Each blocked agent
thread has its own ``threading.Event``; parallel clarifies are supported via a
FIFO queue per session.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default max time an agent thread will wait for the user to answer a clarify
# question before giving up.  30 minutes matches typical messaging UX — users
# may step away from a chat for a while but won't meaningfully answer after
# longer than that.  Configurable via :func:`set_clarify_timeout`.
_DEFAULT_CLARIFY_TIMEOUT_SECS = 30 * 60

_lock = threading.Lock()
_timeout_secs: float = _DEFAULT_CLARIFY_TIMEOUT_SECS


class _ClarifyEntry:
    """One pending clarify question waiting for a user answer."""

    __slots__ = ("event", "question", "choices", "result")

    def __init__(self, question: str, choices: Optional[list]):
        self.event = threading.Event()
        self.question = question
        self.choices = choices
        # ``None`` means "no answer yet".  Empty string is a legitimate
        # open-ended answer, so ``None`` is the sentinel.
        self.result: Optional[str] = None


# session_key -> FIFO queue of pending entries.  A session can have at most
# one live agent thread in practice (the gateway serialises turns), but
# parallel subagents could in theory trigger multiple clarifies; we keep the
# queue to match approval.py's shape and avoid surprises.
_queues: dict[str, list[_ClarifyEntry]] = {}

# session_key -> notify callable.  Signature: ``cb(question, choices) -> None``.
# The gateway uses this to dispatch the question to the user (via Telegram
# send, Discord webhook, etc.).  The callback runs in the agent thread and is
# expected to be non-blocking (bridge to the event loop via
# ``asyncio.run_coroutine_threadsafe`` if needed).
_notify_cbs: dict[str, Callable[[str, Optional[list]], None]] = {}


# =========================================================================
# Gateway-facing API
# =========================================================================


def register_gateway_clarify_notify(
    session_key: str,
    cb: Callable[[str, Optional[list]], None],
) -> None:
    """Register a per-session callback that delivers clarify questions to the user.

    The callback is invoked synchronously from the agent thread each time the
    ``clarify`` tool is called.  It must return quickly (typically by
    scheduling an async send on the event loop); the actual blocking wait
    happens inside :func:`gateway_clarify_callback`.
    """
    if not session_key:
        return
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_gateway_clarify_notify(session_key: str) -> None:
    """Unregister the notify callback and drain any pending waits.

    Call this at the end of every agent turn.  Any still-blocked threads are
    woken with ``result=None`` so they can propagate a graceful error instead
    of hanging until the 30-minute timeout.
    """
    if not session_key:
        return
    with _lock:
        _notify_cbs.pop(session_key, None)
        entries = _queues.pop(session_key, [])
    for entry in entries:
        # Don't set a result — ``gateway_clarify_callback`` treats ``None`` as
        # "unregistered / cancelled" and raises the appropriate error.
        entry.event.set()


def resolve_gateway_clarify(session_key: str, answer: str) -> int:
    """Resolve the oldest pending clarify for a session with ``answer``.

    Called by the gateway's inbound message handler when it detects that a
    user message is meant to answer a live clarify question.

    Returns the number of clarifies resolved (``0`` if nothing was pending,
    ``1`` if one was resolved).  Multiple parallel clarifies are resolved
    FIFO, one user message per clarify.
    """
    if not session_key:
        return 0
    with _lock:
        queue = _queues.get(session_key)
        if not queue:
            return 0
        entry = queue.pop(0)
        if not queue:
            _queues.pop(session_key, None)
    entry.result = answer
    entry.event.set()
    return 1


def has_blocking_clarify(session_key: str) -> bool:
    """Return True if the session has at least one clarify waiting for a user answer."""
    if not session_key:
        return False
    with _lock:
        return bool(_queues.get(session_key))


def clear_session_clarifies(session_key: str) -> None:
    """Cancel any pending clarifies for a session (e.g. on ``/new`` or shutdown).

    Each waiting thread is woken with ``result=None`` so it returns a clean
    error instead of hanging.
    """
    if not session_key:
        return
    with _lock:
        entries = _queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def set_clarify_timeout(seconds: float) -> None:
    """Override the default wait-for-answer timeout (useful for tests)."""
    global _timeout_secs
    if seconds <= 0:
        raise ValueError("clarify timeout must be positive")
    _timeout_secs = float(seconds)


def get_clarify_timeout() -> float:
    """Return the current wait-for-answer timeout in seconds."""
    return _timeout_secs


# =========================================================================
# Agent-facing callback factory
# =========================================================================


class ClarifyUnavailable(RuntimeError):
    """Raised internally when no notify callback is registered for the session."""


class ClarifyTimeout(RuntimeError):
    """Raised internally when the user did not answer within the timeout."""


def gateway_clarify_callback(
    session_key: str,
) -> Callable[[str, Optional[list]], str]:
    """Build an ``AIAgent.clarify_callback``-compatible callable for a gateway session.

    Returned callable signature: ``(question, choices) -> str``.  It blocks the
    caller thread until the user answers (via a message routed through
    :func:`resolve_gateway_clarify`) or the timeout elapses.

    Raises exceptions for abnormal paths.  The caller (``clarify_tool``) turns
    the exception into a JSON error — see ``tests/tools/test_clarify_gateway.py``
    for the expected contract.
    """

    def _callback(question: str, choices: Optional[list]) -> str:
        with _lock:
            notify = _notify_cbs.get(session_key)
            if notify is None:
                raise ClarifyUnavailable(
                    "No gateway clarify notifier registered for this session."
                )
            entry = _ClarifyEntry(question=question, choices=choices)
            _queues.setdefault(session_key, []).append(entry)

        # Notify OUTSIDE the lock to avoid deadlocks if the gateway adapter's
        # send path re-enters this module (it shouldn't, but be safe).
        try:
            notify(question, choices)
        except Exception as exc:
            # Pop the entry we just enqueued so ``has_blocking_clarify`` stays
            # accurate and the inbound router doesn't swallow the user's next
            # message.
            with _lock:
                queue = _queues.get(session_key)
                if queue and entry in queue:
                    queue.remove(entry)
                    if not queue:
                        _queues.pop(session_key, None)
            logger.exception("Clarify notify failed")
            raise RuntimeError(f"Failed to send clarify question to user: {exc}") from exc

        # Block until a user message resolves the entry or the timeout fires.
        got_signal = entry.event.wait(timeout=_timeout_secs)

        if not got_signal:
            # Timeout — remove ourselves from the queue so a late user message
            # doesn't bind to a dead entry.
            with _lock:
                queue = _queues.get(session_key)
                if queue and entry in queue:
                    queue.remove(entry)
                    if not queue:
                        _queues.pop(session_key, None)
            raise ClarifyTimeout(
                f"User did not answer the clarify question within {_timeout_secs:.0f}s."
            )

        if entry.result is None:
            # Woken by unregister/clear with no answer (session ended, /new, etc.).
            raise ClarifyUnavailable(
                "Clarify was cancelled before the user answered "
                "(session ended or was reset)."
            )

        return entry.result

    return _callback
