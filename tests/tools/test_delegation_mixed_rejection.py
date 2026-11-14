"""A configuration hint must preserve successful siblings and their actual outcomes."""
import pytest

from tools import process_registry_notifications as notices


@pytest.mark.parametrize("success_status", ["completed", "success"])
def test_rejection_notice_counts_only_failed_children(monkeypatch, success_status):
    monkeypatch.setattr(notices, "_delegation_config", lambda: {"model": "test/model"})
    rejected = {"task_index": 0, "status": "error", "error": "test/model model_not_found"}
    success = {"task_index": 1, "status": success_status, "summary": "Useful completed work"}
    event = {"type": "async_delegation", "is_batch": True, "results": [rejected, success], "goals": ["a", "b"]}
    text = notices.format_process_notification(event)
    assert "1 of 2 task(s)" in text
    assert "Every task" not in text
    assert "Useful completed work" in text

    success["summary"] = "Resolved test/model model_not_found and completed the work"
    event["results"] = [success]
    assert "SUBAGENT MODEL REJECTED" not in notices.format_process_notification(event)
