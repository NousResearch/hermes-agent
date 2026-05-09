"""Per-thread interrupt signaling for all tools.

In gateway mode multiple agent sessions run concurrently in the same
process. The previous implementation used a single global
``threading.Event``, which meant that interrupting *one* session
interrupted *all* running sessions simultaneously.

This module replaces the global event with a **per-thread** set keyed by
``threading.get_ident()``. Each session runs in its own
``ThreadPoolExecutor`` thread, so interrupt state is naturally isolated.

Backward-compatibility notes:
- ``set_interrupt(active)`` still works; it now targets the calling thread.
- ``set_interrupt(active, thread_id=<int>)`` lets the gateway/agent set the
  flag on a *specific* thread (required because the interrupt arrives on the
  HTTP-handler thread, not the agent thread).
- ``is_interrupted()`` is unchanged (checks the current thread).
- ``_interrupt_event`` shim provides ``.is_set()`` / ``.set()`` /
  ``.clear()`` / ``.wait()`` for legacy callers that imported it directly.

Usage in tools:
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))
if _DEBUG_INTERRUPT:
    logger.setLevel(logging.INFO)

# Set of thread idents that have been interrupted.
_interrupted_threads: set[int] = set()
_lock = threading.Lock()


def set_interrupt(active: bool, thread_id: int | None = None) -> None:
    """Set or clear the interrupt flag for a specific thread.

    Args:
        active: True to signal interrupt, False to clear it.
        thread_id: Target thread ident. When ``None``, targets the
                   calling thread (backward-compatible default for
                   CLI / single-session usage).
    """
    tid = thread_id if thread_id is not None else threading.current_thread().ident
    with _lock:
        if active:
            _interrupted_threads.add(tid)
        else:
            _interrupted_threads.discard(tid)
        _snapshot = set(_interrupted_threads) if _DEBUG_INTERRUPT else None
    if _DEBUG_INTERRUPT:
        logger.info(
            "[interrupt-debug] set_interrupt(active=%s, target_tid=%s) "
            "called_from_tid=%s current_set=%s",
            active, tid, threading.current_thread().ident, _snapshot,
        )


def is_interrupted() -> bool:
    """Check if an interrupt has been requested for the current thread.

    Safe to call from any thread — each thread only sees its own
    interrupt state.
    """
    tid = threading.current_thread().ident
    with _lock:
        return tid in _interrupted_threads


# ---------------------------------------------------------------------------
# Backward-compatible _interrupt_event shim
# ---------------------------------------------------------------------------
# Legacy call sites import ``_interrupt_event`` directly and call
# ``.is_set()`` / ``.set()`` / ``.clear()`` / ``.wait()``.  This proxy
# maps those calls to the per-thread functions above.


class _ThreadAwareEventProxy:
    """Drop-in proxy that maps threading.Event API to per-thread state."""

    def is_set(self) -> bool:  # noqa: D102
        return is_interrupted()

    def set(self) -> None:  # noqa: A003
        set_interrupt(True)

    def clear(self) -> None:  # noqa: D102
        set_interrupt(False)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the current thread is interrupted or *timeout* elapses.

        Returns True if the interrupt was set, False on timeout.
        Note: the legacy threading.Event.wait() semantics are preserved —
        callers that rely on blocking up to *timeout* seconds will work
        correctly instead of spinning.
        """
        if timeout is None:
            # Block indefinitely — poll at 50 ms to stay interruptible.
            while not is_interrupted():
                time.sleep(0.05)
            return True

        deadline = time.monotonic() + timeout
        while not is_interrupted():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return True


_interrupt_event = _ThreadAwareEventProxy()
