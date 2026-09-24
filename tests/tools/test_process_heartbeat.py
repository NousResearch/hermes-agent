"""Heartbeat notifications for long background processes.

A ``heartbeat`` on a background process emits a periodic "still running + output since the last
heartbeat" event on the completion queue so the agent stays current on a long bounded job (merge
train, full suite, deploy) without polling. Invariants: each heartbeat carries only NEW output,
heartbeats stop at exit, and the normal completion notice still fires.
"""
import json
import queue
import time
from types import SimpleNamespace

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


def test_cli_rejects_queued_heartbeat_after_process_exit(monkeypatch):
    """A heartbeat queued while live must not be delivered after exit."""
    from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin
    from tools.process_registry_notifications import format_process_notification

    heartbeat = {
        "type": "heartbeat", "session_id": "proc_exited", "session_key": "session-a",
        "started_at": 12.0, "interval": 60, "command": "sleep 60",
    }
    completion = {
        "type": "completion", "session_id": "proc_exited", "session_key": "session-a",
        "command": "sleep 60", "exit_code": 0, "output": "done",
    }
    registry = SimpleNamespace(
        get=lambda session_id: SimpleNamespace(exited=True, started_at=12.0),
        drain_notifications=lambda **kwargs: [
            (heartbeat, format_process_notification(heartbeat)),
            (completion, format_process_notification(completion)),
        ],
    )
    delivered = []
    monkeypatch.setattr(pr, "process_registry", registry)
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda event, consumer: delivered.append(event["type"]) or "claimed",
    )
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *args: None)

    class CLI(CLIProcessNotificationsMixin):
        session_id = "session-a"
        _pending_input = queue.Queue()

    cli = CLI()
    cli._drain_process_notifications("cli-idle")

    assert delivered == ["completion"]
    assert cli._pending_input.qsize() == 1
    live_registry = SimpleNamespace(
        get=lambda session_id: SimpleNamespace(exited=False, started_at=12.0),
    )
    assert not CLIProcessNotificationsMixin._process_heartbeat_is_stale(heartbeat, live_registry)
