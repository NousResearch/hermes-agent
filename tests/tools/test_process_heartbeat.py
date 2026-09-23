"""Heartbeat notifications for long background processes.

A ``heartbeat`` on a background process emits a periodic "still running + output since the last
heartbeat" event on the completion queue so the agent stays current on a long bounded job (merge
train, full suite, deploy) without polling. Invariants: each heartbeat carries only NEW output,
heartbeats stop at exit, and the normal completion notice still fires.
"""
import json
import queue
import time

import pytest

import tools.process_registry as pr
from tools.process_registry import ProcessRegistry


def _drain(q: "queue.Queue") -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def _wait_until(pred, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


@pytest.mark.linux_only
def test_heartbeat_carries_only_new_output_and_stops_at_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "HEARTBEAT_MIN_SECONDS", 1)
    monkeypatch.setattr(pr, "HEARTBEAT_TICK_SECONDS", 0.1)
    registry = ProcessRegistry()
    session = registry.spawn_local("echo first; sleep 2.5; echo second; sleep 2.5", cwd=str(tmp_path))
    session.notify_on_complete = True
    assert registry.arm_heartbeat(session, 1) == 1

    assert _wait_until(lambda: registry.poll(session.id)["status"] != "running", timeout=20)
    # Give the completion event a moment to be enqueued after the reader observes EOF.
    assert _wait_until(lambda: any(e.get("type") == "completion" for e in list(registry.completion_queue.queue)),
                       timeout=5)
    events = _drain(registry.completion_queue)
    beats = [e for e in events if e["type"] == "heartbeat"]
    completion = [e for e in events if e["type"] == "completion"]

    assert len(beats) >= 2, events
    assert [b["seq"] for b in beats] == list(range(1, len(beats) + 1))
    assert all(b["session_id"] == session.id and b["interval"] == 1 for b in beats)
    # Output is a delta: every produced line appears in exactly one heartbeat, never twice.
    joined = "".join(b["output"] for b in beats)
    assert joined.count("first") == 1 and joined.count("second") == 1, [b["output"] for b in beats]
    assert len(completion) == 1
    # Heartbeats never outlive the process: nothing after the completion notice.
    assert events.index(completion[0]) > events.index(beats[-1])
    assert not _wait_until(lambda: any(e.get("type") == "heartbeat" for e in list(registry.completion_queue.queue)),
                           timeout=2.5)


def test_terminal_dispatch_heartbeat_implies_notify_and_refuses_foreground(monkeypatch):
    from tools import terminal_tool as tt

    captured = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "Background process started", "session_id": "proc_x", "exit_code": 0})

    monkeypatch.setattr(tt, "terminal_tool", fake_terminal_tool)
    dispatch = tt._handle_terminal
    fg = json.loads(dispatch({"command": "sleep 1", "heartbeat": 120}))
    assert fg.get("error") and "background" in fg["error"]

    bg = json.loads(dispatch({"command": "sleep 1", "background": True, "heartbeat": 120}))
    assert "error" not in bg or not bg["error"]
    assert captured["heartbeat"] == 120 and captured["notify_on_complete"] is True


# ── #120334 regression: heartbeat must not be enqueued once the session has exited ──
#
# A ``terminal(background=true, notify=true, heartbeat=…)`` heartbeat is a snapshot
# taken while the process is running. On a busy messaging Gateway session the
# heartbeat can be admitted/queued, the process can exit, and the queued heartbeat
# is later promoted into a fresh agent turn as "still running" — triggering an
# unnecessary model call + false status. The fix re-validates liveness atomically
# at the registry before the put (defensive backstop in ``_emit_heartbeat``) and
# again in the gateway before starting a turn. Valid at emission != valid at
# delivery; both sides must drop the stale event.


def _make_synthetic_session(registry, session_id, started_at):
    """Insert a session that's NOT in the reader thread loop — bypass ``spawn_local``."""
    from tools.process_registry import ProcessSession
    session = ProcessSession(
        id=session_id,
        command="sleep 60",
        task_id="t_120334",
        started_at=started_at,
        notify_on_complete=True,
    )
    session.heartbeat_seconds = 0  # set by arm_heartbeat
    session._heartbeat_last = started_at
    session._heartbeat_total_at_last = 0
    session.output_buffer = ""
    session.total_output_chars = 0
    registry._running[session_id] = session
    return session


def test_emit_heartbeat_skips_session_already_exited_before_put(monkeypatch):
    """If a session is ``exited=True`` at the moment ``_emit_heartbeat`` runs, the
    heartbeat must NOT be put on ``completion_queue`` — the process is done, the
    completion notice will fire, and a stale "still running" heartbeat must not
    start a fresh turn. Pre-fix this fails: the registry enqueued the heartbeat
    even though the session had exited (snapshot-then-emit race)."""
    registry = ProcessRegistry()
    session = _make_synthetic_session(registry, "proc_stale_a", started_at=1.0)
    registry.arm_heartbeat(session, 1)
    # Simulate the reader thread having observed process exit BEFORE the
    # heartbeat thread's snapshot-then-emit completed. The pre-fix code read
    # ``not s.exited`` only at snapshot time and never re-checked at put time.
    session.exited = True

    registry._emit_heartbeat(session, time.time())

    events = _drain(registry.completion_queue)
    heartbeat_events = [e for e in events if e.get("type") == "heartbeat"]
    assert heartbeat_events == [], (
        "Stale heartbeat enqueued after session exit: "
        "valid-at-emission must imply valid-at-delivery (#120334)"
    )


def test_emit_heartbeat_skips_session_dropped_from_running_before_put(monkeypatch):
    """If the session is removed from ``_running`` between snapshot and emit
    (e.g. the reader thread reaped it to ``_finished``), the heartbeat must NOT
    be put on ``completion_queue``."""
    registry = ProcessRegistry()
    session = _make_synthetic_session(registry, "proc_stale_b", started_at=2.0)
    registry.arm_heartbeat(session, 1)
    # The session is no longer "running" — the registry already moved it to
    # ``_finished`` (e.g. via the reader's exit handling).
    finished_copy = registry._running.pop(session.id)
    finished_copy.exited = True
    registry._finished[session.id] = finished_copy

    registry._emit_heartbeat(session, time.time())

    events = _drain(registry.completion_queue)
    heartbeat_events = [e for e in events if e.get("type") == "heartbeat"]
    assert heartbeat_events == [], (
        "Stale heartbeat enqueued after session was reaped from _running (#120334)"
    )


def test_emit_heartbeat_uses_fresh_started_at_match(monkeypatch):
    """Sanity: a live session with no exit DOES emit a heartbeat (regression
    guard against an over-eager guard that would silence valid heartbeats)."""
    registry = ProcessRegistry()
    session = _make_synthetic_session(registry, "proc_live", started_at=3.0)
    registry.arm_heartbeat(session, 1)
    session.output_buffer = "still working\n"
    session.total_output_chars = len(session.output_buffer)

    registry._emit_heartbeat(session, time.time())

    events = _drain(registry.completion_queue)
    heartbeat_events = [e for e in events if e.get("type") == "heartbeat"]
    assert len(heartbeat_events) == 1
    assert heartbeat_events[0]["session_id"] == "proc_live"
    assert heartbeat_events[0]["started_at"] == 3.0
    assert "still working" in heartbeat_events[0]["output"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_heartbeat_loop_skips_exited_session_in_due_loop(monkeypatch):
    """End-to-end: the heartbeat loop's snapshot-then-emit race. Set the session
    to ``exited=True`` AFTER the snapshot collection runs but BEFORE the loop
    reaches ``_emit_heartbeat``. The fix re-checks liveness under the registry
    lock right before emit. Pre-fix this fails because the loop emits without
    re-validating.

    The production ``_heartbeat_loop`` is ``while True: time.sleep; ...``, so
    we run it on a worker thread and stop it after one tick. To make the
    snapshot-then-emit race observable we patch ``_emit_heartbeat`` to flip
    ``exited=True`` just before delegating to the real implementation (the
    real implementation then re-checks liveness under the lock)."""
    import threading
    monkeypatch.setattr(pr, "HEARTBEAT_TICK_SECONDS", 0.005)

    registry = ProcessRegistry()
    # Insert a session directly with the heartbeat fields set — no daemon
    # thread is started by us here; the test owns the worker thread.
    session = _make_synthetic_session(registry, "proc_loop_stale", started_at=4.0)
    session.heartbeat_seconds = 1
    session._heartbeat_last = 0.0  # make it due immediately

    # Hook a marker that flips exited=True AFTER the snapshot collected the
    # session but BEFORE emit — emulates the reader thread reaping the process
    # between snapshot and emit. The fix must re-check liveness under the lock
    # and skip.
    original_emit = registry._emit_heartbeat

    def hook_emit(s, now):
        s.exited = True  # reader observed EOF between snapshot and emit
        return original_emit(s, now)

    monkeypatch.setattr(registry, "_emit_heartbeat", hook_emit)

    # Worker thread runs the production loop. The fix's re-check inside the
    # loop should observe ``s.exited = True`` and skip the emit.
    loop_thread = threading.Thread(target=registry._heartbeat_loop, daemon=True)
    loop_thread.start()

    # Give the loop time to take one snapshot+emit tick.
    assert _wait_until(
        lambda: any(e.get("type") == "heartbeat" for e in list(registry.completion_queue.queue))
        or session.exited,
        timeout=5,
    ), "loop did not observe the session exit in time"

    # Stop the worker by raising a benign exception in the next ``time.sleep``
    # call. We patch sleep so the loop breaks out on the next iteration.
    stop = {"stop": False}

    def stopping_sleep(_seconds):
        if stop["stop"]:
            raise StopIteration
        time.sleep(_seconds)

    monkeypatch.setattr(pr.time, "sleep", stopping_sleep)
    stop["stop"] = True
    loop_thread.join(timeout=5)

    # Drain anything that landed. Pre-fix: a heartbeat is enqueued. Post-fix: none.
    events = _drain(registry.completion_queue)
    heartbeat_events = [e for e in events if e.get("type") == "heartbeat"]
    assert heartbeat_events == [], (
        "Loop emitted heartbeat for a session that exited between snapshot and emit (#120334)"
    )
