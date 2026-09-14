from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db
from workstation.contracts import (
    BrowserTaskReport,
    DiscoveredTask,
    EvidenceRef,
    ExecutionEventKind,
    RiskLevel,
)
from workstation.journal import ExecutionJournal
from workstation.kanban import WorkstationKanbanBridge, is_multistep_request


def test_is_multistep_request():
    assert is_multistep_request("Can you execute this multistep workflow for me?") is True
    assert is_multistep_request("First sign in, then extract and download the report") is True
    assert is_multistep_request("what is the weather today?") is False


def test_journal_append_and_read(tmp_path):
    journal_file = tmp_path / "test-task.jsonl"
    journal = ExecutionJournal("t_test_1", "s_test_1", file_path=journal_file)

    ev1 = journal.record(
        ExecutionEventKind.TASK_CREATED,
        "Task initiated",
        metadata={"step": 1},
    )
    ev2 = journal.record(
        ExecutionEventKind.NAVIGATION,
        "Navigated to dashboard",
        url="https://example.com/dashboard",
        risk=RiskLevel.LOW,
    )
    ev3 = journal.record(
        ExecutionEventKind.SCREENSHOT,
        "Captured state",
        evidence=[EvidenceRef(kind="screenshot", uri="file:///tmp/snap.png", summary="dashboard state")],
    )

    events = journal.read_events()
    assert len(events) == 3
    assert events[0].kind == ExecutionEventKind.TASK_CREATED
    assert events[0].task_id == "t_test_1"
    assert events[0].session_id == "s_test_1"
    assert events[1].kind == ExecutionEventKind.NAVIGATION
    assert events[1].url == "https://example.com/dashboard"
    assert events[2].kind == ExecutionEventKind.SCREENSHOT
    assert len(events[2].evidence) == 1
    assert events[2].evidence[0].uri == "file:///tmp/snap.png"


def test_kanban_bridge_multistep_promotion(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    bridge = WorkstationKanbanBridge()
    task_id = bridge.promote_request_if_multistep(
        "First extract data from the page, then fill out the form and submit it",
        session_id="session-dogfood-1",
    )
    assert task_id is not None
    assert task_id.startswith("t_")

    with bridge.get_connection() as conn:
        task = kanban_db.get_task(conn, task_id)
        assert task is not None
        assert task.session_id == "session-dogfood-1"
        assert task.created_by == "workstation"


def test_kanban_bridge_followup_and_completion(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    bridge = WorkstationKanbanBridge()
    parent_id = bridge.promote_request_if_multistep(
        "Multistep workflow to book travel",
        session_id="session-dogfood-2",
    )
    assert parent_id is not None

    followup = DiscoveredTask(
        title="Approve extra luggage fee",
        parent_task_id=parent_id,
        discovered_by="travel-agent",
        reason="Airline charges $50 for extra luggage",
        origin_session_id="session-dogfood-2",
        required_for_parent=True,
    )

    child_id = bridge.record_discovered_followup(parent_id, followup)
    assert child_id.startswith("t_")
    assert followup.task_id == child_id

    with bridge.get_connection() as conn:
        child = kanban_db.get_task(conn, child_id)
        assert child is not None
        assert child.title == "Approve extra luggage fee"
        assert "Origin session: session-dogfood-2" in child.body
        assert "Discovery evidence:" in child.body
        parent = kanban_db.get_task(conn, parent_id)
        assert parent.status in ("blocked", "todo")

    # Complete the child task
    with bridge.get_connection() as conn:
        kanban_db.complete_task(conn, child_id, result="Approved luggage fee")
        kanban_db.recompute_ready(conn)
        parent_ready = kanban_db.get_task(conn, parent_id)
        assert parent_ready is not None
        assert parent_ready.status == "ready"

    # Now complete the parent task with a Workstation report
    report = BrowserTaskReport(
        task_id=parent_id,
        session_id="session-dogfood-2",
        objective="Book travel",
        result="Travel booked successfully",
        completed=True,
        sites=["https://airline.example.com"],
        discovered_tasks=[followup],
    )
    success = bridge.complete_task_with_report(parent_id, report)
    assert success is True

    with bridge.get_connection() as conn:
        parent_done = kanban_db.get_task(conn, parent_id)
        assert parent_done.status == "done"
        assert parent_done.result == "Travel booked successfully"
        events = kanban_db.list_events(conn, parent_id)
        completed_events = [e for e in events if getattr(e, "kind", None) == "completed" or getattr(e, "event", None) == "completed"]
        assert len(completed_events) > 0


def test_journal_timeline_elapsed_seconds(tmp_path):
    journal_file = tmp_path / "timeline-task.jsonl"
    journal = ExecutionJournal("task-timeline-1", "session-1", file_path=journal_file)

    journal.record(ExecutionEventKind.TASK_CREATED, "Task started")
    import time
    time.sleep(0.01)
    journal.record(ExecutionEventKind.NAVIGATION, "Navigated to home", url="https://example.com")
    time.sleep(0.01)
    journal.record(ExecutionEventKind.TASK_COMPLETED, "Task completed successfully")

    timeline = journal.read_timeline()
    assert len(timeline) == 3
    assert timeline[0]["kind"] == ExecutionEventKind.TASK_CREATED.value
    assert timeline[0]["elapsed_seconds"] == 0.0
    assert timeline[1]["kind"] == ExecutionEventKind.NAVIGATION.value
    assert "elapsed_seconds" in timeline[1]
    assert timeline[2]["kind"] == ExecutionEventKind.TASK_COMPLETED.value
    assert timeline[2]["elapsed_seconds"] >= 0.0
