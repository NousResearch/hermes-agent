"""The plumbing a run needs wherever it is being driven from.

One run is touched by several threads: the loop thread scheduling it, step workers computing under
it, timer and poll threads firing for its waits, and the reactor delivering events to it. The loop
thread is the only writer of a live run's state; everyone else posts to the run's mailbox and the
loop folds the mail in between steps. ``post`` starts a loop when none is alive, and ``retire`` (the
loop's last act) refuses while mail is waiting, both under one lock, so mail can never land in a
mailbox nobody reads.

``ensure_running`` imports the loop late, on purpose: everything else here is below the loop, and
the late import keeps the module-level dependency pointing one way.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from workflow import trace
from workflow.store import load_run, save_run

_lock = threading.Lock()
_threads: dict[str, threading.Thread] = {}
_mail: dict[str, list[dict[str, Any]]] = {}
_wakeups: dict[str, threading.Event] = {}
_run_locks: dict[str, threading.Lock] = {}
# What a step worker may need to know mid-step: "pause" or "cancel". The mail is the loop's; this
# is the read-only echo a long-running step (the scripted fake) polls to stop early.
_stopping: dict[str, str] = {}


def lock_for(run_id: str) -> threading.Lock:
    """Held by the loop for a whole pass, and by anyone mutating a run that has no loop."""
    with _lock:
        return _run_locks.setdefault(run_id, threading.Lock())


def wakeup_for(run_id: str) -> threading.Event:
    with _lock:
        return _wakeups.setdefault(run_id, threading.Event())


def post(run_id: str, mail: dict[str, Any], *, start: bool = True, execute_fn=None) -> None:
    """Hand the run something to fold in: a resolved wait, an answer, a fired trigger, a pause.
    ``start=False`` leaves starting the loop to the caller (a synchronous ``advance``)."""
    kind = mail.get("kind")
    with _lock:
        _mail.setdefault(run_id, []).append(dict(mail))
        if kind in {"pause", "cancel"}:
            _stopping[run_id] = kind
        elif kind == "resume":
            _stopping.pop(run_id, None)
    wakeup_for(run_id).set()
    if start:
        ensure_running(run_id, execute_fn)


def take_mail(run_id: str) -> list[dict[str, Any]]:
    with _lock:
        return _mail.pop(run_id, [])


def stopping(run_id: str) -> str | None:
    with _lock:
        return _stopping.get(run_id)


def clear_stopping(run_id: str) -> None:
    with _lock:
        _stopping.pop(run_id, None)


def retire(run_id: str) -> bool:
    """The loop is about to exit. False (and keep looping) when mail arrived in the meantime."""
    with _lock:
        if _mail.get(run_id):
            return False
        if _threads.get(run_id) is threading.current_thread():
            _threads.pop(run_id, None)
        return True


def thread_alive(run_id: str) -> bool:
    with _lock:
        thread = _threads.get(run_id)
    return thread is not None and thread.is_alive()


def emit(state: dict, event_type: str, payload: dict | None = None) -> None:
    """Record one runner event on the run's Relay trace, numbered from the live counter."""
    seq = int(state.get("seq") or 0)
    state["seq"] = seq + 1
    trace.mark(state["runId"], event_type, payload or {}, seq)


def fail_dead_run(state: dict) -> dict:
    """A run that says "running" with no loop alive. Nothing will move it again, so say so rather
    than leaving it spinning forever."""
    state["status"] = "failed"
    state["failed"] = True
    state["pauseRequested"] = False
    emit(state, "RunFinished", {"state": "failed", "error": "runner process died"})
    save_run(state)
    return state


def ensure_running(run_id: str, execute_fn=None) -> None:
    """Start the run's loop thread unless one is already alive."""
    from agent.memory_provider import spawn_context_thread
    from workflow.runner import advance

    def work() -> None:
        try:
            advance(run_id, execute_fn=execute_fn)
        except Exception as exc:
            state = load_run(run_id)
            if state is not None:
                state["status"] = "failed"
                state["failed"] = True
                emit(state, "RunFinished", {"state": "failed", "error": str(exc)})
                save_run(state)
        finally:
            with _lock:
                if _threads.get(run_id) is threading.current_thread():
                    _threads.pop(run_id, None)
                again = bool(_mail.get(run_id))
            if again:
                ensure_running(run_id, execute_fn)

    with _lock:
        thread = _threads.get(run_id)
        if thread is not None and thread.is_alive():
            return
        thread = spawn_context_thread(work, name=f"workflow-{run_id}")
        _threads[run_id] = thread
    thread.start()


def arm(name: str, seconds: float, fire) -> None:
    """Run ``fire`` once, ``seconds`` from now, on a daemon thread under the caller's profile."""
    from agent.memory_provider import spawn_context_thread

    def wait_then_fire() -> None:
        time.sleep(max(0.0, seconds))
        fire()

    spawn_context_thread(wait_then_fire, name=name).start()
