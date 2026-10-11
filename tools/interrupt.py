"""Per-thread interrupt signaling for all tools: thread-scoped so interrupting one
agent session does not kill tools in other sessions (the gateway runs many agents in one
process). The agent passes its execution thread id to set_interrupt(); tools call
is_interrupted(), which checks the CURRENT thread."""

import contextlib
import contextvars
import logging
import threading
from collections.abc import Callable

from utils import env_var_enabled

logger = logging.getLogger(__name__)

# Opt-in debug tracing — pairs with HERMES_DEBUG_INTERRUPT in tools/environments/base.py.
_DEBUG_INTERRUPT = env_var_enabled("HERMES_DEBUG_INTERRUPT")
if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode forces the `tools` logger to ERROR on CLI startup;
    # force ours back to INFO so the trace is visible in agent.log.
    logger.setLevel(logging.INFO)

# Interrupted thread idents + optional user-safe cause (never the user's message text).
_interrupted_threads: set[int] = set()
_interrupt_reasons: dict[int, str] = {}
# Threads asked to YIELD: hand a long-running foreground command to the background
# instead of killing it, so a mid-turn user message is not parked behind it.
_yield_threads: set[int] = set()
# Why a yield was requested (e.g. ``"user_detach"`` for the explicit detach key / ``/detach``);
# absent = the historical mid-turn-message redirect. Read by the terminal tool to pick its note.
_yield_reasons: dict[int, str] = {}
_lock = threading.Lock()
# Tool-worker tid a deadline worker acts for. ``run_bounded_sync`` runs its worker under
# ``contextvars.copy_context()``, so a guard chain moved onto that worker still honours
# ``/stop`` aimed at the tool thread that spawned it (``is_interrupted`` checks both).
acting_for_tid: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "hermes_interrupt_acting_for_tid", default=None,
)


def set_interrupt(active: bool, thread_id: int | None = None, *, reason: str | None = None) -> None:
    """Set or clear the interrupt for *thread_id* (default: current thread); ``reason`` is
    an optional user-safe cause. Clearing also drops a pending yield request."""
    tid = thread_id if thread_id is not None else threading.current_thread().ident
    with _lock:
        (_interrupted_threads.add if active else _interrupted_threads.discard)(tid)
        if active and reason:
            _interrupt_reasons[tid] = reason
        else:
            _interrupt_reasons.pop(tid, None)
        if not active:
            _yield_threads.discard(tid)
            if tid is not None:
                _yield_reasons.pop(tid, None)
        _snapshot = set(_interrupted_threads) if _DEBUG_INTERRUPT else None
    if _DEBUG_INTERRUPT:
        logger.info(
            "[interrupt-debug] set_interrupt(active=%s, target_tid=%s) "
            "called_from_tid=%s current_set=%s",
            active, tid, threading.current_thread().ident, _snapshot)


def is_interrupted() -> bool:
    return is_thread_interrupted(threading.current_thread().ident) or is_thread_interrupted(acting_for_tid.get())


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


def request_yield(thread_id: int, reason: str | None = None) -> None:
    """Ask the tool running on *thread_id* to yield: a foreground terminal command hands
    its live process to the background registry and returns at once, so a user's mid-turn
    message (``redirect()`` during tool execution) is delivered instead of parked behind it.
    The command itself is never killed; that is what ``set_interrupt`` is for. ``reason``
    (e.g. ``"user_detach"``) lets the tool tell the model why it was moved."""
    with _lock:
        _yield_threads.add(thread_id)
        if reason:
            _yield_reasons[thread_id] = reason
        else:
            _yield_reasons.pop(thread_id, None)


def pop_yield_reason(thread_id: int | None) -> str | None:
    """Take the reason recorded with the last yield request for *thread_id* (``None`` if none)."""
    if thread_id is None:
        return None
    with _lock:
        return _yield_reasons.pop(thread_id, None)


# Tool-worker tids currently blocked in a wait that CAN yield (a foreground terminal command on
# the local backend, which arms ``yield_handler``). Lets an explicit detach tell "nothing
# detachable is running" apart from "detaching" instead of setting a bit nobody will consume.
_yield_armed: dict[int, int] = {}


@contextlib.contextmanager
def yield_armed(thread_id: int | None = None):
    """Mark *thread_id* (default: current thread) as in a yieldable wait for the block's duration."""
    tid: int = thread_id if thread_id is not None else threading.get_ident()
    with _lock:
        _yield_armed[tid] = _yield_armed.get(tid, 0) + 1
    try:
        yield tid
    finally:
        with _lock:
            left = _yield_armed.get(tid, 0) - 1
            if left > 0:
                _yield_armed[tid] = left
            else:
                _yield_armed.pop(tid, None)


def is_yield_armed(thread_id: int | None) -> bool:
    """Whether *thread_id* is currently in a wait that honours a yield request."""
    if thread_id is None:
        return False
    with _lock:
        return thread_id in _yield_armed


def is_thread_yield_requested(thread_id: int | None) -> bool:
    """Whether a yield is pending for *thread_id* (``None`` never is)."""
    if thread_id is None:
        return False
    with _lock:
        return thread_id in _yield_threads


def consume_yield(thread_id: int | None) -> bool:
    """Atomically take the pending yield for *thread_id*; True if one was pending."""
    if thread_id is None:
        return False
    with _lock:
        if thread_id in _yield_threads:
            _yield_threads.discard(thread_id)
            return True
        return False


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
