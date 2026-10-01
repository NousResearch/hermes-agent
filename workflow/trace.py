"""A run's history is its Relay trace.

Each run is a Relay session (``wf-<runId>``) the trace recorder writes to
``<home>/traces/wf-<runId>.jsonl``. What the runner decides — a step started, a gate routed, a
person was asked — is a Relay mark named ``hermes.workflow.<Type>`` on that session, carrying the
event the canvas folds. An agent step is a delegated child session (``wf-<runId>-<nodeId>``), so its
model calls and tools are the Relay spans every other Hermes session records. There is no second
event log: the canvas, ``hermes trace show`` and the Agents waterfall all read this one.

Recording follows ``telemetry.traces``: with traces off (or no Relay on this platform) a run still
executes from its state file; it just has no history to replay.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

MARK_PREFIX = "hermes.workflow."
_SURFACE = "workflow"

_lock = threading.Lock()
_open: dict[tuple[str, str], list[Any]] = {}  # (profile key, session id) -> [lease, refcount]


def run_session_id(run_id: str) -> str:
    return f"wf-{run_id}"


def step_session_id(run_id: str, node_id: str) -> str:
    return f"wf-{run_id}-{node_id}"


def _profile_key() -> str:
    from agent import relay_runtime

    return relay_runtime.current_profile_key()


@contextlib.contextmanager
def recording(run_id: str) -> Iterator[Any]:
    """Hold the run's Relay session open for the block; nested holds share one scope."""
    from agent import relay_runtime

    key = (_profile_key(), run_session_id(run_id))
    with _lock:
        held = _open.get(key)
        if held is not None:
            held[1] += 1
    if held is None:
        lease = None
        try:
            # Importing the package registers the recorder's session initializer.
            import hermes_cli.observability  # noqa: F401

            lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=key[0], session_id=key[1], platform=_SURFACE,
            )
        except Exception:
            logger.debug("workflow run %s: Relay session unavailable", run_id, exc_info=True)
        with _lock:
            held = _open.setdefault(key, [lease, 0])
            held[1] += 1
            if held[0] is not lease and lease is not None:
                relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
    try:
        yield held[0]
    finally:
        with _lock:
            held[1] -= 1
            last = held[1] <= 0
            if last:
                _open.pop(key, None)
        if last and held[0] is not None:
            try:
                relay_runtime.SESSION_COORDINATOR.finalize_conversation(profile_key=key[0], session_id=key[1])
                relay_runtime.SESSION_COORDINATOR.release_conversation(held[0])
            except Exception:
                logger.debug("workflow run %s: Relay session close failed", run_id, exc_info=True)


def _session(lease: Any) -> tuple[Any, Any] | None:
    host, session = getattr(lease, "host", None), getattr(lease, "session", None)
    if session is None or getattr(session, "handle", None) is None or not hasattr(host, "run_in_session"):
        return None
    return host, session


def mark(run_id: str, event_type: str, payload: dict[str, Any], seq: int) -> None:
    """Record one runner event as a mark on the run's session."""
    with recording(run_id) as lease:
        opened = _session(lease)
        if opened is None:
            return
        host, session = opened
        try:
            host.run_in_session(
                session, host.relay.scope.event, MARK_PREFIX + event_type, handle=session.handle,
                data={"runId": run_id, "seq": seq, "payload": payload},
            )
        except Exception:
            logger.debug("workflow run %s: mark %s failed", run_id, event_type, exc_info=True)


@contextlib.contextmanager
def step_session(run_id: str, node_id: str) -> Iterator[Any]:
    """A child Relay session for a step that is not an agent turn (the scripted fake), so its
    tool spans sit where a real step's would."""
    with recording(run_id) as lease:
        opened = _session(lease)
        child = None
        if opened is not None:
            host, _ = opened
            try:
                child = host.register_subagent(
                    {"parent_session_id": run_session_id(run_id), "child_session_id": step_session_id(run_id, node_id)},
                    metadata={"hermes.execution_surface": _SURFACE},
                )
            except Exception:
                logger.debug("workflow run %s: step session failed", run_id, exc_info=True)
        try:
            yield (opened[0], child) if opened is not None and child is not None else None
        finally:
            if opened is not None and child is not None:
                opened[0].close_session({"session_id": step_session_id(run_id, node_id)})


def tool_span(opened: Any, name: str, arg: str) -> None:
    """One finished tool call on a step session from ``step_session``."""
    if opened is None:
        return
    host, child = opened
    try:
        span = host.run_in_session(child, host.relay.tools.call, name, {"arg": arg}, handle=child.handle)
        host.run_in_session(child, host.relay.tools.call_end, span, host.relay.ToolExecutionResult({"ok": True}))
    except Exception:
        logger.debug("workflow step tool span failed", exc_info=True)


def flush() -> None:
    """Wait for Relay to hand every emitted event to its subscribers (delivery is asynchronous)."""
    from agent import relay_runtime

    host = relay_runtime.get_runtime(create=False)
    if host is not None:
        host.relay.subscribers.flush()


def recorded_events(run_id: str) -> tuple[list[dict[str, Any]], bool]:
    """The run's recorded Relay events (its steps' sessions included) and whether recording is on."""
    from hermes_cli.observability import relay_traces
    from hermes_constants import get_hermes_home

    try:
        on = relay_traces.policy().get("enabled", True) is not False
    except Exception:
        on = True
    events, _truncated = relay_traces.read_session_events(get_hermes_home(), run_session_id(run_id))
    return events, on


def runner_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The runner's own events out of a run's Relay trace, in the canvas's envelope."""
    out = []
    for event in events:
        name = str(event.get("name") or "")
        data = event.get("data")
        if event.get("kind") != "mark" or not name.startswith(MARK_PREFIX) or not isinstance(data, dict):
            continue
        out.append({
            "runId": data.get("runId"), "seq": data.get("seq"), "type": name[len(MARK_PREFIX):],
            "payload": data.get("payload") or {}, "timestamp": event.get("timestamp"),
        })
    out.sort(key=lambda e: (int(e.get("seq") or 0)))
    return out


def run_reply(state: dict | None, run_id: str) -> dict[str, Any]:
    """A run as the canvas reads it: its state file and its recorded Relay events."""
    events, recording = recorded_events(run_id)
    return {"run": state, "events": events, "runId": run_id, "recording": recording}
