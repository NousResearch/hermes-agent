"""Per-thread interrupt signaling for all tools.

Provides thread-scoped interrupt tracking so that interrupting one agent
session does not kill tools running in other sessions.  This is critical
in the gateway where multiple agents run concurrently in the same process.

The agent stores its execution thread ID at the start of run_conversation()
and passes it to set_interrupt()/clear_interrupt().  Tools call
is_interrupted() which checks the CURRENT thread — no argument needed.

Usage in tools:
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Exit code a user-interrupted command returns (128 + SIGINT(2)).  Mirrors the
# value produced in tools/environments/base.py when a command is interrupted.
# Failure classifiers treat this as benign (not a real failure) so a manual
# stop never colours the card red or feeds the guardrail halt counter.
INTERRUPT_EXIT_CODE = 130

# Exit code for a command killed by SIGPIPE (128 + SIGPIPE(13)).  A downstream
# pipe reader closed early — e.g. `producer | head`, where head exits after N
# lines and the producer gets SIGPIPE on its next write.  Common under
# `set -o pipefail`, where that 141 propagates as the pipeline's status.  This
# is a benign, expected outcome, not a command failure.
SIGPIPE_EXIT_CODE = 141

# Signal-death exit codes the failure classifiers treat as benign.  Only these
# two: SIGINT (user stop) and SIGPIPE (downstream closed the pipe).  Other
# 128+N exits stay failures — SIGSEGV (139), SIGABRT (134), SIGKILL/OOM (137)
# are genuine crashes.
BENIGN_SIGNAL_EXIT_CODES = frozenset({INTERRUPT_EXIT_CODE, SIGPIPE_EXIT_CODE})

# Opt-in debug tracing — pairs with HERMES_DEBUG_INTERRUPT in
# tools/environments/base.py.  Enables per-call logging of set/check so the
# caller thread, target thread, and current state are visible when
# diagnosing "interrupt signaled but tool never saw it" reports.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode path forces `tools` logger to ERROR on CLI startup.
    # Force our own logger back to INFO so the trace is visible in agent.log.
    logger.setLevel(logging.INFO)

# Set of thread idents that have been interrupted.
_interrupted_threads: set[int] = set()
_lock = threading.Lock()


def set_interrupt(active: bool, thread_id: int | None = None) -> None:
    """Set or clear interrupt for a specific thread.

    Args:
        active: True to signal interrupt, False to clear it.
        thread_id: Target thread ident.  When None, targets the
                   current thread (backward compat for CLI/tests).
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
# Backward-compatible _interrupt_event proxy
# ---------------------------------------------------------------------------
# Some legacy call sites (code_execution_tool, process_registry, tests)
# import _interrupt_event directly and call .is_set() / .set() / .clear().
# This shim maps those calls to the per-thread functions above so existing
# code keeps working while the underlying mechanism is thread-scoped.

class _ThreadAwareEventProxy:
    """Drop-in proxy that maps threading.Event methods to per-thread state."""

    def is_set(self) -> bool:
        return is_interrupted()

    def set(self) -> None:  # noqa: A003
        set_interrupt(True)

    def clear(self) -> None:
        set_interrupt(False)

    def wait(self, timeout: float | None = None) -> bool:
        """Not truly supported — returns current state immediately."""
        return self.is_set()


_interrupt_event = _ThreadAwareEventProxy()
