"""Regression tests for quiet delivery of wholly failed delegation batches."""

from tools.process_registry import format_process_notification


def test_all_failed_batch_is_not_a_chat_nudge():
    text = format_process_notification({
        "type": "async_delegation",
        "delegation_id": "deleg_failed",
        "is_batch": True,
        "goals": ["look up the answer"],
        "results": [{"task_index": 0, "status": "failed", "error": "provider secret"}],
    })
    assert "produced no usable result" in text
    assert "Do not mention" in text
    assert "Redo the lookup yourself only if the user is still waiting" in text
    assert "act on these" not in text
    assert "provider secret" not in text


def test_partial_batch_keeps_success_and_redacts_failed_entry():
    text = format_process_notification({
        "type": "async_delegation",
        "delegation_id": "deleg_partial",
        "is_batch": True,
        "goals": ["good", "bad"],
        "results": [
            {"task_index": 0, "status": "completed", "summary": "usable fact"},
            {"task_index": 1, "status": "timeout", "error": "raw provider error"},
        ],
    })
    assert "usable fact" in text
    assert "no usable result — status=timeout" in text
    assert "raw provider error" not in text
    assert "act on these" not in text
