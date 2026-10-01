"""The one place workflow events turn into runs.

Every host process — the desktop backend and the messaging gateway — starts the reactor; a lock
file per Hermes home lets exactly one of them drain the inbox, own the run threads and re-arm
parked runs after a restart. The others only publish. A CLI never runs a graph itself: it
publishes, and the event waits on disk until a host is up.

``workflow.cmd.*`` events are commands (start, pause, resume, cancel, answer); every other name is
handed to ``runner.deliver_event`` for whatever listens.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from workflow import events, runner
from workflow.store import workflows_dir

logger = logging.getLogger(__name__)

_LOCK_NAME = ".reactor.lock"
_HOLDING_POLL = 0.5
_WAITING_POLL = 5.0

_lock = threading.Lock()
_started: set[str] = set()
_held: dict[str, Any] = {}  # home -> open lock file


def _home_key() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _try_lock() -> bool:
    """Take this home's reactor lock without waiting; keep it for the life of the process."""
    from pm.filesystem import lock_fd

    key = _home_key()
    with _lock:
        if key in _held:
            return True
    handle = open(workflows_dir() / _LOCK_NAME, "a+b")  # noqa: SIM115 - held for the process lifetime
    try:
        if not lock_fd(handle.fileno(), wait=False):
            handle.close()
            return False
    except OSError:
        handle.close()
        return False
    with _lock:
        _held[key] = handle
    return True


def host_running() -> bool:
    """Some process holds this home's reactor lock (for a CLI: will a published event be picked up?)."""
    from pm.filesystem import lock_fd

    if _home_key() in _held:
        return True
    with open(workflows_dir() / _LOCK_NAME, "a+b") as handle:
        try:
            taken = lock_fd(handle.fileno(), wait=False)
        except OSError:
            return True
        if taken:
            _unlock(handle)
        return not taken


def _unlock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _answer(payload: dict, *, background: bool, execute_fn) -> None:
    from workflow.waits import find_approval

    run_id, node_id = str(payload.get("runId") or ""), str(payload.get("nodeId") or "")
    if not (run_id and node_id) and payload.get("code"):
        found = find_approval(str(payload["code"]))
        if found is None:
            logger.info("workflow answer for unknown approval code %s", payload["code"])
            return
        run_id, node_id = found[0]["runId"], found[1]["nodeId"]
    runner.respond(run_id, node_id, str(payload.get("decision") or ""), by=payload.get("by"),
                   background=background, execute_fn=execute_fn)


def _start(payload: dict, *, background: bool, execute_fn) -> None:
    runner.start_run(
        str(payload.get("workflowId") or ""), scenario=payload.get("scenario"), payload=payload.get("payload"),
        source=str(payload.get("source") or "manual"), trigger=payload.get("trigger") or None,
        run_id=payload.get("runId") or None, fake=bool(payload.get("fake")),
        background=background, execute_fn=execute_fn,
    )


def _on_run(fn: Callable[..., Any]) -> Callable[..., None]:
    def handle(payload: dict, *, background: bool, execute_fn) -> None:
        fn(str(payload.get("runId") or ""), background=background, execute_fn=execute_fn)

    return handle


_COMMANDS: dict[str, Callable[..., None]] = {
    events.START: _start,
    events.PAUSE: _on_run(runner.request_pause),
    events.RESUME: _on_run(runner.resume_run),
    events.CANCEL: _on_run(runner.cancel_run),
    events.ANSWER: _answer,
}


def handle(event: dict, *, background: bool = True, execute_fn=None) -> None:
    name = str(event.get("name") or "")
    payload = event.get("payload")
    command = _COMMANDS.get(name)
    try:
        if command is not None:
            command(payload if isinstance(payload, dict) else {}, background=background, execute_fn=execute_fn)
        else:
            runner.deliver_event(name, payload, source=str(event.get("source") or "event"),
                                 background=background, execute_fn=execute_fn)
    except ValueError as exc:
        logger.info("workflow event %s: %s", name, exc)
    except Exception:
        logger.warning("workflow event %s failed", name, exc_info=True)


def drain_once(*, background: bool = True, execute_fn=None) -> int:
    """Handle everything queued. The reactor thread's body; tests call it directly."""
    batch = events.drain()
    for event in batch:
        handle(event, background=background, execute_fn=execute_fn)
    return len(batch)


def _in_use() -> bool:
    """A home that never stored a workflow or published an event has nothing to react to (and gets
    no ``workflows/`` directory from merely running a gateway)."""
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / "workflows").is_dir()


def _loop() -> None:
    holding = False
    while True:
        try:
            if not holding and _in_use() and _try_lock():
                holding = True
                runner.rearm_parked()
            if holding:
                drain_once()
        except Exception:
            logger.warning("workflow reactor pass failed", exc_info=True)
        events.wait_for_wake(_HOLDING_POLL if holding else _WAITING_POLL)


def ensure_started() -> None:
    """Start this home's reactor thread once per process (idempotent; cheap with no workflows)."""
    from agent.memory_provider import spawn_context_thread

    key = _home_key()
    with _lock:
        if key in _started:
            return
        _started.add(key)
    spawn_context_thread(_loop, name="workflow-reactor").start()
