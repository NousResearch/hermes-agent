"""Delayed process results defer to current conversation instructions."""

from tools.process_registry_notifications import format_process_notification


def test_completion_notification_carries_supersession_context(monkeypatch):
    """Delayed completions must defer to newer conversation instructions."""
    monkeypatch.setattr("tools.process_registry_notifications.time.time", lambda: 1300.0)

    text = format_process_notification({
        "type": "completion",
        "session_id": "proc_old_completion",
        "command": "run old task",
        "exit_code": 0,
        "output": "done",
        "started_at": 1000.0,
    })

    assert text is not None
    assert "this process started 5m ago" in text
    assert "Reconcile this result with the current conversation before acting" in text
    assert "newer user instructions take precedence" in text


def test_legacy_completion_without_timing_still_carries_supersession_context():
    """Recovered/legacy events cannot bypass the newer-instructions boundary."""
    text = format_process_notification({
        "type": "completion",
        "session_id": "proc_legacy",
        "command": "run old task",
        "exit_code": 0,
        "output": "done",
    })

    assert text is not None
    assert "may relate to an earlier turn" in text
    assert "newer user instructions take precedence" in text


def test_watch_notification_carries_supersession_context(monkeypatch):
    """Delayed watch matches have the same precedence contract as completions."""
    monkeypatch.setattr("tools.process_registry_notifications.time.time", lambda: 1300.0)

    text = format_process_notification({
        "type": "watch_match",
        "session_id": "proc_old_watch",
        "command": "watch old task",
        "pattern": "READY",
        "output": "READY",
        "started_at": 1000.0,
    })

    assert text is not None
    assert "this process started 5m ago" in text
    assert "newer user instructions take precedence" in text
