"""The failure limit counts failed tasks, not more recent successful delegations."""
import time

import pytest

from tools import async_delegation as ad


def _store(ident, now, status, result, *, ui="history-ui", **task):
    ad._persist_dispatch({
        "delegation_id": ident, "session_key": "", "origin_ui_session_id": ui,
        "parent_session_id": "history-parent", "dispatched_at": now - 60, "goal": "review", **task,
    })
    ad._persist_completion({"delegation_id": ident, "status": status, "completed_at": now}, result)


@pytest.mark.parametrize("batch", [False, True])
def test_recent_successes_do_not_hide_an_older_failure(tmp_path, monkeypatch, batch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    failure = {"status": "error", "error": "review failed", "task_index": 3}
    _store("history-0", now - 3, "completed" if batch else "error",
           {"results": [failure]} if batch else failure,
           is_batch=batch, task_indexes=[3], goals=["review"])
    for index in [1, 2]:
        _store(f"history-{index}", now - 3 + index, "completed", {"status": "completed"})

    failures = ad.failed_delegations_for_session("history-ui", limit=1, now=now)

    assert [(row["delegation_id"], row["error"]) for row in failures] == [("history-0", "review failed")]


def test_limit_counts_failed_tasks_in_newest_first_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    _store("old", now - 3, "error", {"status": "error", "error": "old failure"})
    _store("batch", now - 2, "completed", {"results": [
        {"task_index": 3, "status": "completed"},
        {"task_index": 4, "status": "timeout", "error": "timed out"},
        {"task_index": 5, "status": "error", "error": "tool failed"},
    ]}, is_batch=True, task_indexes=[3, 4, 5], goals=["api", "web", "storage"])
    _store("success", now - 1, "completed", {"status": "completed"})
    _store("foreign", now, "error", {"status": "error"}, ui="another-ui")

    failures = ad.failed_delegations_for_session("history-ui", limit=2, now=now)
    assert [(row["delegation_id"], row["task_index"], row["goal"]) for row in failures] == [
        ("batch", 4, "web"), ("batch", 5, "storage")]
    assert [row["delegation_id"] for row in ad.failed_delegations_for_session("history-ui", limit=3, now=now)] == [
        "batch", "batch", "old"]
    assert ad.failed_delegations_for_session("history-ui", limit=0, now=now) == []
