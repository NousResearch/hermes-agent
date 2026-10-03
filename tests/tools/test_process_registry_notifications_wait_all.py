"""Completion guidance must not contradict a parent's strict wait-all policy."""

import pytest

from tools.process_registry_notifications import format_process_notification


@pytest.mark.parametrize("variant", ["batch", "single", "early_failure"])
def test_async_notice_preserves_wait_all_before_action(variant, monkeypatch):
    # Do not consult the user's live provider configuration for a rendering test.
    monkeypatch.setattr("tools.process_registry_notifications._delegation_config", lambda: {})
    event = {
        "type": "async_delegation", "delegation_id": "deleg_policy_test",
        "goal": "first task", "summary": "first report", "status": "completed",
        "dispatched_at": 1.0, "completed_at": 2.0,
    }
    if variant != "single":
        event.update(is_batch=True, goals=["first task", "second task"], results=[
            {"task_index": 0, "status": "completed", "summary": "first report"},
        ])
    if variant == "early_failure":
        event.update(task_failure_notice=True, n_tasks=2)
        event["results"][0].update(status="error", error="test failure")
    text = format_process_notification(event)
    assert text is not None
    assert "When waiting for all reports" in text
    assert "retain this result and end your turn immediately" in text
    assert "Do not analyze or act on it, re-dispatch work, poll, or otherwise advance the task" in text
    assert "until every expected outcome has returned" in text
    assert "verify and integrate the results together" in text
    assert "Failed, cancelled, or timed-out outcomes count as returned outcomes, not successful results" in text
    assert "after acting on this one" not in text
    assert "investigate now instead of then" not in text
    assert "you can act on the result" not in text
    assert "deleg_policy_test" in text
    if variant == "early_failure":
        assert "test failure" in text and "TASK FAILED" in text
    else:
        assert "first report" in text


def test_ordinary_process_notification_keeps_its_existing_contract():
    text = format_process_notification({
        "type": "completion", "session_id": "proc_policy_test",
        "command": "printf done", "exit_code": 0, "output": "done",
    })
    assert text == (
        "[IMPORTANT: Background process proc_policy_test completed normally (exit code 0).\n"
        "Command: printf done\nOutput:\ndone]"
    )
