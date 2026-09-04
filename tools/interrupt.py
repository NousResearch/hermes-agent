"""Per-thread interrupt signaling for all tools: thread-scoped so interrupting one
agent session does not kill tools in other sessions (the gateway runs many agents in one
process). The agent passes its execution thread id to set_interrupt(); tools call
is_interrupted(), which checks the CURRENT thread."""

import logging
import os
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Opt-in debug tracing — pairs with HERMES_DEBUG_INTERRUPT in tools/environments/base.py.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))
if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode forces the `tools` logger to ERROR on CLI startup;
    # force ours back to INFO so the trace is visible in agent.log.
    logger.setLevel(logging.INFO)

# Interrupted thread idents + optional user-safe cause (never the user's message text).
_interrupted_threads: set[int] = set()
_interrupt_reasons: dict[int, str] = {}
_lock = threading.Lock()


def set_interrupt(active: bool, thread_id: int | None = None, *, reason: str | None = None) -> None:
    """Set or clear the interrupt for *thread_id* (default: current thread); ``reason`` is
    an optional user-safe cause."""
    tid = thread_id if thread_id is not None else threading.current_thread().ident
    with _lock:
        (_interrupted_threads.add if active else _interrupted_threads.discard)(tid)
        if active and reason:
            _interrupt_reasons[tid] = reason
        else:
            _interrupt_reasons.pop(tid, None)
        _snapshot = set(_interrupted_threads) if _DEBUG_INTERRUPT else None
    if _DEBUG_INTERRUPT:
        logger.info(
            "[interrupt-debug] set_interrupt(active=%s, target_tid=%s) "
            "called_from_tid=%s current_set=%s",
            active, tid, threading.current_thread().ident, _snapshot)


def is_interrupted() -> bool:
    return is_thread_interrupted(threading.current_thread().ident)


def is_thread_interrupted(thread_id: int | None) -> bool:
    """Whether *thread_id* has an interrupt bit set (``None`` never is). Used when
    a wait moves onto a deadline worker (``run_bounded_sync``) so ``/stop``
    targeting the original tool-worker tid still kills the subprocess.

    See #94285.
    """
    if thread_id is None:
        return False
    with _lock:
        return thread_id in _interrupted_threads


def run_if_not_interrupted(callback: Callable[[], None]) -> bool:
    """Run a state transition atomically with current-thread interruption.

    Returns ``False`` without calling ``callback`` when the current thread is
    already interrupted. The callback runs under the interrupt lock and must
    not block or re-enter any interrupt API.
    """
    tid = threading.current_thread().ident
    with _lock:
        if tid in _interrupted_threads:
            return False
        callback()
        return True


def get_interrupt_reason() -> str | None:
    """User-safe interrupt cause for the current thread, if known."""
    with _lock:
        return _interrupt_reasons.get(threading.current_thread().ident)


def clear_current_thread_interrupt() -> None:
    """Clear any interrupt bit on the CURRENT thread: gives a user-approved command a clean
    slate right before it spawns its child, so a stale bit that landed during the blocking
    approval-wait cannot SIGINT the just-approved run. A *genuine* interrupt arriving after
    this call re-sets the bit and is still observed by the executor's poll loop. Call
    directly on the executing thread."""
    set_interrupt(False)
