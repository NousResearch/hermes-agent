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
from unittest.mock import patch

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


@pytest.mark.platforms("linux")
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


def _live_recoverable_entry(tmp_path, registry, *, heartbeat_seconds=30):
    """Spawn a live host process and write a checkpoint entry naming it, so
    ``recover_from_checkpoint`` adopts it as a detached session."""
    import json as _json
    import subprocess as _sp
    import sys as _sys

    proc = _sp.Popen([_sys.executable, "-c", "import time; time.sleep(60)"])
    entry = {
        "session_id": "proc_recover_hb",
        "command": "sleep 60",
        "pid": proc.pid,
        "pid_scope": "host",
        "host_start_time": registry._safe_host_start_time(proc.pid),
        "task_id": "t1",
        "session_key": "session-hb",
        "heartbeat_seconds": heartbeat_seconds,
    }
    checkpoint = tmp_path / "procs.json"
    checkpoint.write_text(_json.dumps([entry]))
    return proc, checkpoint


def test_recovery_rearms_checkpointed_heartbeat(tmp_path, monkeypatch):
    """A recovered session with a checkpointed heartbeat beats again. The checkpoint
    persists ``heartbeat_seconds`` for exactly this, but recovery never re-armed the
    timer thread, so the process ran silent until its completion notice."""
    import tools.process_registry as _pr

    monkeypatch.setattr(_pr, "HEARTBEAT_MIN_SECONDS", 1)
    monkeypatch.setattr(_pr, "HEARTBEAT_TICK_SECONDS", 0.1)
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda pid, start: True)
    proc, checkpoint = _live_recoverable_entry(tmp_path, registry, heartbeat_seconds=1)
    try:
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 1

        session = registry.get("proc_recover_hb")
        assert session is not None and session.heartbeat_seconds == 1
        assert registry._heartbeat_thread is not None and registry._heartbeat_thread.is_alive()
        assert _wait_until(
            lambda: any(e.get("type") == "heartbeat" and e.get("session_id") == "proc_recover_hb"
                        for e in list(registry.completion_queue.queue)),
            timeout=10), list(registry.completion_queue.queue)
    finally:
        proc.kill()
        proc.wait()


def test_recovery_rearm_uses_the_persisted_interval(tmp_path, monkeypatch):
    """The re-armed cadence is the checkpointed value floored at HEARTBEAT_MIN_SECONDS,
    matching arm_heartbeat's contract; nothing beats before the floor elapses."""
    import tools.process_registry as _pr

    monkeypatch.setattr(_pr, "HEARTBEAT_MIN_SECONDS", 60)
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda pid, start: True)
    # Interval below the floor is clamped up, matching arm_heartbeat's contract.
    proc, checkpoint = _live_recoverable_entry(tmp_path, registry, heartbeat_seconds=5)
    try:
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 1

        session = registry.get("proc_recover_hb")
        assert session.heartbeat_seconds == 60
        # No heartbeat was requested before the floor elapses: nothing queued.
        assert not any(e.get("type") == "heartbeat"
                       for e in list(registry.completion_queue.queue))
    finally:
        proc.kill()
        proc.wait()
