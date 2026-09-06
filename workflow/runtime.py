"""The plumbing a run needs wherever it is being driven from.

One run is touched by several threads: the worker walking it, a timer firing
under it, and an RPC handler asking it to pause. All three need the same lock,
the same event counter and the same signal mailbox, so those live here rather
than in the loop — the loop is one caller of this, not the owner of it.

``spawn`` imports the loop late, on purpose. Everything else in this file is
below the loop; spawn is the one thing that reaches back up, and a late import
is what keeps the dependency pointing one way at module level.
"""

from __future__ import annotations

import threading

from workflow.store import append_event, load_run, save_run

_threads: dict[str, threading.Thread] = {}
_timer_threads: dict[str, threading.Thread] = {}
_thread_lock = threading.Lock()

_run_locks: dict[str, threading.Lock] = {}
_run_locks_guard = threading.Lock()

# Pause/cancel from the RPC thread. The runner holds its own state dict and
# save_run's it — writing the flag only to disk gets clobbered, so the live
# loop never sees it.
_signals: dict[str, str] = {}
_signals_lock = threading.Lock()


def lock_for(run_id: str) -> threading.Lock:
    with _run_locks_guard:
        lock = _run_locks.get(run_id)
        if lock is None:
            lock = threading.Lock()
            _run_locks[run_id] = lock
        return lock


def signal(run_id: str, kind: str) -> None:
    with _signals_lock:
        _signals[run_id] = kind


def clear_signal(run_id: str) -> None:
    with _signals_lock:
        _signals.pop(run_id, None)


def absorb_signals(state: dict) -> None:
    """Fold a pending pause/cancel into the live state dict."""
    with _signals_lock:
        kind = _signals.get(state["runId"])
    if kind == "pause":
        state["pauseRequested"] = True
    elif kind == "cancel":
        state["status"] = "cancelled"


def emit(state: dict, event_type: str, payload: dict | None = None) -> dict:
    """Append one event, numbering it from the live counter.

    The counter is the caller's, not the file's: reloading the run JSON to
    number a line races an in-flight save, reuses numbers, and the canvas drops
    every event after the collision.
    """
    seq = int(state.get("seq") or 0)
    event = append_event(state["runId"], event_type, payload, seq=seq)
    state["seq"] = seq + 1
    return event


def thread_alive(run_id: str) -> bool:
    with _thread_lock:
        thread = _threads.get(run_id)
    return thread is not None and thread.is_alive()


def fail_dead_run(state: dict) -> dict:
    """A run whose worker is gone. Nothing will ever move it again, so say so
    rather than leaving it spinning at "running" forever."""
    state["status"] = "failed"
    state["failed"] = True
    state["pauseRequested"] = False
    save_run(state)
    emit(state, "RunFinished", {"state": "failed", "error": "runner process died"})
    return load_run(state["runId"]) or state


def spawn(run_id: str, execute_fn=None) -> None:
    """Drive a run on its own thread."""
    from workflow.runner import advance

    def work() -> None:
        try:
            advance(run_id, execute_fn=execute_fn)
        except Exception as exc:
            state = load_run(run_id)
            if state is None:
                return
            state["status"] = "failed"
            state["failed"] = True
            save_run(state)
            emit(state, "RunFinished", {"state": "failed", "error": str(exc)})

    thread = threading.Thread(target=work, name=f"workflow-{run_id}", daemon=True)
    with _thread_lock:
        _threads[run_id] = thread
    thread.start()


def arm(key: str, name: str, seconds: float, fire) -> None:
    """Run `fire` once, `seconds` from now, on a daemon thread."""

    def wait_then_fire() -> None:
        import time

        time.sleep(max(0.0, seconds))
        fire()

    thread = threading.Thread(target=wait_then_fire, name=name, daemon=True)
    with _thread_lock:
        _timer_threads[key] = thread
    thread.start()
