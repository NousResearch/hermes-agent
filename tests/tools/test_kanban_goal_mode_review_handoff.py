"""Regression coverage for goal-mode review handoffs.

A review request is an implementation handoff, not terminal completion. The
worker must be able to enter the native review lane so an independent reviewer
can evaluate the goal and decide whether to complete it.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from hermes_cli import kanban_db as kb


def test_goal_mode_request_review_reaches_review_lane(monkeypatch, tmp_path: Path) -> None:
    """A goal-mode worker can request review even when the completion judge
    would (correctly) say that the handoff summary is not itself completion.

    This is a disposable board reproduction of the deadlock: before the fix,
    ``_handle_request_review`` ran the completion judge first and returned an
    error while the task remained ``running``.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "pending")

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="goal-mode review handoff",
            assignee="test-worker",
            body="Implementation must be independently reviewed.",
            goal_mode=True,
        )
        kb.claim_task(conn, task_id, claimer="test-worker")
        run_id = kb.get_task(conn, task_id).current_run_id
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    # Simulate a reachable judge. Review handoffs still use the judge, but
    # with a phase-scoped note so missing future reviewer evidence does not
    # prevent entering the review lane.
    monkeypatch.setattr("tools.kanban_tools._goal_judge_available", lambda: True)
    monkeypatch.setattr(
        "tools.kanban_tools.judge_goal",
        lambda **kwargs: (
            "done",
            "implementation is ready for review"
            if "do not withhold DONE merely because" in kwargs["goal"]
            else "review evidence is missing",
            False,
            None,
            False,
        ),
    )

    from tools import kanban_tools as kt

    result = json.loads(
        kt._handle_request_review(
            {
                "summary": "Implementation is staged and ready for independent review.",
                "reviewer": "os-reviewer",
            }
        )
    )
    assert result["ok"] is True
    assert result["status"] == "review"

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert kb.latest_run(conn, task_id).outcome == "review_requested"
    finally:
        conn.close()


def test_review_scoping_note_survives_long_goal(monkeypatch) -> None:
    """The phase note must survive judge_goal's 2000-character truncation."""
    from types import SimpleNamespace

    from tools import kanban_tools as kt

    captured = {}
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)

    def capture_judge(*, goal, last_response, **_kwargs):
        captured["goal"] = goal
        return "done", "ready", False, None, False

    monkeypatch.setattr(kt, "judge_goal", capture_judge)
    task = SimpleNamespace(
        title="long goal",
        body="Acceptance criterion. " * 200,
        goal_mode=True,
    )
    verdict, rejection = kt._goal_mode_handoff_rejection(
        task,
        "implementation complete",
        handoff="review",
    )

    assert verdict == "done"
    assert rejection is None
    assert len(captured["goal"]) <= 2000
    assert "do not withhold DONE merely because" in captured["goal"]


def test_cli_goal_mode_request_review_uses_scoped_goal_judge(
    monkeypatch, tmp_path: Path
) -> None:
    """The human CLI request-review surface must share the same handoff rule."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="goal-mode CLI review handoff",
            assignee="test-worker",
            body="Implementation must be independently reviewed.",
            goal_mode=True,
        )
        kb.claim_task(conn, task_id, claimer="test-worker")
        run_id = kb.get_task(conn, task_id).current_run_id
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    import agent.auxiliary_client as auxiliary_client
    from hermes_cli import goals

    monkeypatch.setattr(
        auxiliary_client,
        "get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )

    def scoped_judge(*, goal, last_response, **_kwargs):
        assert "do not withhold DONE merely because" in goal
        return "done", "implementation is ready for review", False, None, False

    monkeypatch.setattr(goals, "judge_goal", scoped_judge)
    from hermes_cli import kanban as kanban_cli

    result = kanban_cli._cmd_request_review(
        Namespace(
            task_id=task_id,
            summary="Implementation is staged and ready for independent review.",
            metadata=None,
            reviewer="os-reviewer",
            force=False,
        )
    )
    assert result == 0

    conn = kb.connect()
    try:
        assert kb.get_task(conn, task_id).status == "review"
    finally:
        conn.close()
