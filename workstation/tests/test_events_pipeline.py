from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from workstation.contracts import ExecutionEventKind, RiskLevel
from workstation.events import (
    SystemEvent,
    SystemEventPipeline,
    SystemEventSeverity,
    SystemEventType,
)
from workstation.journal import ExecutionJournal


def test_system_event_types_and_severities():
    assert SystemEventType.PROCESS_CRASH == "process_crash"
    assert SystemEventType.BUILD_FAILURE == "build_failure"
    assert SystemEventType.BUILD_SUCCESS == "build_success"
    assert SystemEventType.DOWNLOAD_COMPLETED == "download_completed"
    assert SystemEventType.REPO_STATE_CHANGED == "repo_state_changed"
    assert SystemEventType.CONTROLLER_HEALTH_CHANGED == "controller_health_changed"
    assert SystemEventType.USER_ATTENTION_REQUIRED == "user_attention_required"

    assert SystemEventSeverity.INFO == "info"
    assert SystemEventSeverity.CRITICAL == "critical"


def test_ingest_event_enriches_existing_task(tmp_path, monkeypatch):
    journal_path = tmp_path / "task-bound-100.jsonl"
    monkeypatch.setattr(
        "workstation.events.ExecutionJournal",
        lambda task_id, session_id: ExecutionJournal(task_id, session_id, file_path=journal_path),
    )

    pipeline = SystemEventPipeline()
    evt = pipeline.ingest_event(
        event_type=SystemEventType.BUILD_FAILURE,
        severity=SystemEventSeverity.ERROR,
        source="vitest_watcher",
        title="Desktop tests failed",
        message="3 tests failed in workstation-browser-runtime.test.ts",
        target_task_id="task-bound-100",
        target_session_id="session-user-abc",
    )

    assert evt.processed is True
    assert evt.target_task_id == "task-bound-100"

    journal = ExecutionJournal("task-bound-100", "session-user-abc", file_path=journal_path)
    events = journal.read_events()
    assert len(events) == 1
    assert events[0].kind == ExecutionEventKind.ERROR
    assert "Desktop tests failed" in events[0].message
    assert events[0].metadata["source"] == "vitest_watcher"


def test_ingest_critical_untracked_event_has_no_intent_authority():
    mock_bridge = MagicMock()
    mock_bridge.promote_request_if_multistep.return_value = "kanban-task-auto-999"

    pipeline = SystemEventPipeline(kanban_bridge=mock_bridge)
    evt = pipeline.ingest_event(
        event_type=SystemEventType.PROCESS_CRASH,
        severity=SystemEventSeverity.CRITICAL,
        source="electron_host",
        title="Chromium renderer crashed unexpectedly",
        message="Segmentation fault in WebContentsView",
    )

    assert evt.processed is True
    assert evt.resulting_task_id is None
    mock_bridge.promote_request_if_multistep.assert_not_called()



def test_custom_handler_and_history():
    pipeline = SystemEventPipeline()
    received_events = []

    pipeline.register_handler(lambda e: received_events.append(e))

    pipeline.ingest_event(
        event_type=SystemEventType.DOWNLOAD_COMPLETED,
        severity=SystemEventSeverity.INFO,
        source="download_manager",
        title="Report downloaded",
        message="Saved to /downloads/report.pdf",
    )

    assert len(received_events) == 1
    assert received_events[0].title == "Report downloaded"
    assert len(pipeline.get_recent_events()) == 1
