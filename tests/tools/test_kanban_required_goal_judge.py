from __future__ import annotations

import json

from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt


def _required_worker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="prove it", assignee="worker", goal_mode=True,
            goal_judge_policy="required",
        )
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def test_required_unavailable_judge_blocks_needs_input(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: False)
    out = json.loads(kt._handle_complete({"summary": "done"}))
    assert out.get("ok") is not True
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_required_unavailable_judge_reports_actual_triage_state(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    with kb.connect() as conn:
        conn.execute(
            "UPDATE tasks SET block_kind = 'needs_input', block_recurrences = ? "
            "WHERE id = ?",
            (kb.BLOCK_RECURRENCE_LIMIT, tid),
        )
        conn.commit()
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: False)

    out = json.loads(kt._handle_complete({"summary": "done"}))

    assert out.get("ok") is not True
    assert "triage" in out["error"].lower()
    assert "blocked/needs_input" not in out["error"].lower()
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_required_done_supplies_gate_proof(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.goals.judge_goal",
        lambda **_: ("done", "accepted", False, None, False),
    )
    out = json.loads(kt._handle_complete({"summary": "evidence"}))
    assert out["ok"] is True
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"


def test_required_missing_run_id_does_not_mutate(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: False)
    out = json.loads(kt._handle_complete({"summary": "done"}))
    assert "run" in out["error"].lower()
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_required_stale_run_id_cannot_block_newer_run(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: False)
    out = json.loads(kt._handle_complete({"summary": "done"}))
    assert "refused" in out["error"].lower()
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_kanban_create_persists_required_policy(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    out = json.loads(kt._handle_create({
        "title": "child", "assignee": "worker", "goal_mode": True,
        "goal_judge_policy": "required",
    }))
    with kb.connect() as conn:
        assert kb.get_task(conn, out["task_id"]).goal_judge_policy == "required"


def test_kanban_create_null_policy_uses_best_effort_default(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    out = json.loads(kt._handle_create({
        "title": "child", "assignee": "worker", "goal_mode": True,
        "goal_judge_policy": None,
    }))
    assert out["ok"] is True
    with kb.connect() as conn:
        assert kb.get_task(conn, out["task_id"]).goal_judge_policy == "best_effort"


def test_required_dependency_block_is_rejected_but_needs_input_allowed(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    dependency = json.loads(kt._handle_block({
        "reason": "waiting for child", "kind": "dependency",
    }))
    assert dependency.get("ok") is not True
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"

    needs_input = json.loads(kt._handle_block({
        "reason": "need operator answer", "kind": "needs_input",
    }))
    assert needs_input["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


def test_required_fail_closed_block_stays_sticky_and_unclaimable(tmp_path, monkeypatch):
    tid = _required_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: False)
    json.loads(kt._handle_complete({"summary": "done"}))
    with kb.connect() as conn:
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.claim_task(conn, tid) is None
