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


def test_a_tick_with_no_new_output_queues_nothing():
    """Every queued heartbeat costs the owning session a full model turn, so a quiet tick is
    skipped rather than delivered as "(no new output)"; the next tick with output still carries
    exactly the delta and the sequence counts delivered beats only."""
    registry = ProcessRegistry()
    session = pr.ProcessSession(id="proc_quiet", command="sleep 600", notify_on_complete=True)
    session._heartbeat_last = 0.0

    registry._emit_heartbeat(session, now=100.0)
    registry._emit_heartbeat(session, now=200.0)
    assert _drain(registry.completion_queue) == []
    assert session._heartbeat_last == 200.0 and session._heartbeat_seq == 0

    session.output_buffer += "first line\n"
    session.total_output_chars += len("first line\n")
    registry._emit_heartbeat(session, now=300.0)
    (beat,) = _drain(registry.completion_queue)
    assert beat["type"] == "heartbeat" and beat["seq"] == 1 and beat["output"] == "first line\n"

    registry._emit_heartbeat(session, now=400.0)
    assert _drain(registry.completion_queue) == []


def test_schema_minimum_heartbeat_is_disabled_for_foreground(monkeypatch):
    from tools import terminal_tool as tt

    captured = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "Background process started", "session_id": "proc_x", "exit_code": 0})

    monkeypatch.setattr(tt, "terminal_tool", fake_terminal_tool)
    heartbeat_schema = tt.TERMINAL_SCHEMA["parameters"]["properties"]["heartbeat"]
    generated = {
        "command": "pwd",
        "background": False,
        "timeout": 20,
        "pty": False,
        "notify": False,
        "heartbeat": heartbeat_schema["minimum"],
    }

    result = json.loads(tt._handle_terminal(generated))

    assert not result.get("error")
    assert captured["background"] is False
    assert captured["heartbeat"] == 0
    assert captured["notify_on_complete"] is False


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
