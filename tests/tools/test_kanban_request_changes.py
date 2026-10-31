"""Orchestrator review verdicts use the existing manual reopen path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools  # registers the real handlers
from tools.registry import registry


@pytest.fixture
def review_card(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "lead")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB",
                "HERMES_KANBAN_BOARD", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    from hermes_cli.config import save_config

    save_config({"toolsets": ["kanban"]})
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Review the implementation", assignee="builder")
        run = kb.claim_task(conn, tid)
        assert run is not None
        assert kb.request_review(conn, tid, summary="Ready", reviewer="reviewer",
                                 expected_run_id=run.current_run_id)
    return tid


@pytest.mark.parametrize("parent_status, landing", [(None, "ready"), ("ready", "todo"), ("done", "ready")])
def test_orchestrator_comments_then_restores_implementer(review_card, parent_status, landing):
    from toolsets import resolve_toolset

    definitions = registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)
    assert "kanban_request_changes" in {row["function"]["name"] for row in definitions}
    with kbc.connect() as conn:
        if parent_status:
            parent = kb.create_task(conn, title="Dependency", assignee="builder")
            if parent_status == "done":
                assert kb.complete_task(conn, parent, summary="Done")
            kb.link_tasks(conn, parent_id=parent, child_id=review_card)
        before = kb.get_task(conn, review_card)

    secret = "ghp_" + "A" * 40
    out = json.loads(registry.dispatch("kanban_request_changes", {
        "task_id": review_card, "reason": f"Add a boundary assertion; token={secret}",
    }))

    assert out.get("ok") is True, out
    assert (out["status"], out["implementer"]) == (landing, "builder")
    with kbc.connect() as conn:
        task = kb.get_task(conn, review_card)
        assert (task.status, task.assignee, task.current_run_id) == (landing, "builder", None)
        assert task.consecutive_failures == before.consecutive_failures
        comments = kb.list_comments(conn, review_card)
        assert len(comments) == 1
        assert comments[0].author == "lead"
        assert "Add a boundary assertion" in comments[0].body
        assert secret not in comments[0].body
        kinds = [event.kind for event in kb.list_events(conn, review_card)]
        assert kinds[-2:] == ["commented", "review_reopened"]


@pytest.mark.parametrize("state", ["ready", "running_review", "missing"])
def test_orchestrator_refuses_cards_outside_review(review_card, state):
    with kbc.connect() as conn:
        if state == "ready":
            assert kb.reopen_review_task(conn, review_card)
        elif state == "running_review":
            assert kb.claim_review_task(conn, review_card) is not None
        tid = "missing-task" if state == "missing" else review_card
        before = kb.get_task(conn, review_card)
        events = kb.list_events(conn, review_card)

    out = json.loads(registry.dispatch("kanban_request_changes", {
        "task_id": tid, "reason": "Needs another assertion",
    }))

    assert ("not found" if state == "missing" else "not in review") in out.get("error", ""), out
    with kbc.connect() as conn:
        assert kb.get_task(conn, review_card) == before
        assert kb.list_comments(conn, review_card) == []
        assert kb.list_events(conn, review_card) == events


def test_delegated_child_cannot_request_orchestrator_changes(review_card):
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        out = json.loads(registry.dispatch("kanban_request_changes", {
            "task_id": review_card, "reason": "Child verdict",
        }))
    assert "delegate_task child" in out.get("error", ""), out
    with kbc.connect() as conn:
        assert kb.get_task(conn, review_card).status == "review"
        assert kb.list_comments(conn, review_card) == []
