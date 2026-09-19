"""Regression coverage for rejecting an unclaimed Kanban review (#113010)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def unclaimed_review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Create a review card with no active reviewer run."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Review me", assignee="builder")
        claimed = kb.claim_task(conn, task_id, claimer="builder:1")
        assert claimed is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="Ready for review",
            reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
    return task_id


def test_reject_review_closes_an_unclaimed_review_with_a_terminal_audit_event(
    unclaimed_review: str,
) -> None:
    from tools import kanban_tools as tools

    response = json.loads(tools._handle_reject_review({
        "task_id": unclaimed_review,
        "reason": "The acceptance criteria no longer apply.",
    }))

    assert response["ok"] is True
    assert response["status"] == "done"
    with kbc.connect() as conn:
        task = kb.get_task(conn, unclaimed_review)
        assert task is not None
        assert (task.status, task.current_run_id) == ("done", None)
        event = [
            item for item in kb.list_events(conn, unclaimed_review)
            if item.kind == "review_rejected"
        ][-1]
        assert event.payload == {
            "reason": "The acceptance criteria no longer apply.",
            "reviewer": "reviewer",
            "status": "done",
        }
        assert kb.latest_run(conn, unclaimed_review).outcome == "review_rejected"


def test_reject_review_refuses_a_profile_other_than_the_assigned_reviewer(
    unclaimed_review: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as tools

    monkeypatch.setenv("HERMES_PROFILE", "other-reviewer")
    response = json.loads(tools._handle_reject_review({
        "task_id": unclaimed_review,
        "reason": "Not authorized.",
    }))

    assert "assigned reviewer" in response["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, unclaimed_review).status == "review"


def test_reject_review_refuses_a_review_claimed_after_inspection(
    unclaimed_review: str,
) -> None:
    from tools import kanban_tools as tools

    with kbc.connect() as conn:
        assert kb.claim_review_task(conn, unclaimed_review, claimer="reviewer:1") is not None
    response = json.loads(tools._handle_reject_review({
        "task_id": unclaimed_review,
        "reason": "Too late.",
    }))

    assert "unclaimed review" in response["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, unclaimed_review).status == "running"
