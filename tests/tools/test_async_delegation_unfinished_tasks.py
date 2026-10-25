"""A recovered fan-out names the tasks a re-dispatch must re-run."""

import json

import tools.async_delegation as ad
from tools.process_registry_notifications import _recovery_lines


def _recover(monkeypatch, tmp_path, task, result=None):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda pid, started: False)
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at, "
            "delivery_state, delivery_attempts, owner_pid, task_json, result_json) "
            "VALUES ('d1', 's', 'running', 1, 1, 'pending', 0, 4242, ?, ?)",
            (json.dumps(task), json.dumps(result) if result else None))
    assert ad.recover_abandoned_delegations() == 1
    with ad._DB_LOCK, ad._transaction() as conn:
        return json.loads(conn.execute("SELECT event_json FROM async_delegations").fetchone()[0])


def test_partial_batch_lists_only_unknown_tasks(monkeypatch, tmp_path):
    task = {"is_batch": True, "goals": ["a", "b", "c"]}
    done = {"partial": True, "results": [{"task_index": 1, "status": "completed", "summary": "ok"}]}
    evt = _recover(monkeypatch, tmp_path, task, done)
    assert evt["unfinished_tasks"] == [{"task_index": 0, "goal": "a"}, {"task_index": 2, "goal": "c"}]
    text = "\n".join(_recovery_lines(evt))
    assert "RE-DISPATCH: 2 task(s)" in text and "task 1: a" in text and "task 3: c" in text and "task 2: b" not in text


def test_unit_with_no_recorded_results_lists_its_own_tasks(monkeypatch, tmp_path):
    evt = _recover(monkeypatch, tmp_path, {"is_batch": True, "goals": ["a", "b", "c"], "task_indexes": [2]})
    assert evt["unfinished_tasks"] == [{"task_index": 2, "goal": "c"}]


def test_single_delegation_lists_its_goal(monkeypatch, tmp_path):
    assert _recover(monkeypatch, tmp_path, {"goal": "solo"})["unfinished_tasks"] == [{"task_index": 0, "goal": "solo"}]
