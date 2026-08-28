"""Focused dispatcher/producer integration tests for task-scoped closeout."""

import json
import queue
import sqlite3
import threading
import time

import pytest

from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    isolated = queue.Queue()
    from tools.process_registry import process_registry

    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    yield isolated
    ad._reset_for_tests()


class _Executor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn):
        self.submitted.append(fn)


def _dispatch(monkeypatch, *, work="work-1", generation=0, turn="turn-1", **kwargs):
    executor = _Executor()
    monkeypatch.setattr(ad, "_get_executor", lambda _limit: executor)
    result = ad.dispatch_async_delegation_batch(
        goals=["inspect the implementation carefully"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="route",
        parent_session_id="parent",
        runner=lambda: {"results": [{"status": "completed", "summary": "ok"}]},
        origin_work_id=work,
        work_generation=generation,
        owner_turn_id=turn,
        **kwargs,
    )
    return result, executor


def _row(sql, values=()):
    with ad._transaction() as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, values).fetchone()


def test_first_and_same_turn_members_register_before_submit(monkeypatch):
    first, first_executor = _dispatch(monkeypatch)
    assert first["status"] == "dispatched"
    assert len(first_executor.submitted) == 1
    member = _row(
        "SELECT origin_work_id, work_generation FROM async_delegations WHERE delegation_id=?",
        (first["delegation_id"],),
    )
    assert dict(member) == {"origin_work_id": "work-1", "work_generation": 0}

    second, second_executor = _dispatch(monkeypatch)
    assert second["status"] == "dispatched"
    assert len(second_executor.submitted) == 1
    assert _row(
        "SELECT COUNT(*) AS n FROM async_delegations WHERE origin_work_id='work-1'"
    )["n"] == 2


def test_registration_failure_fails_closed_without_submit(monkeypatch):
    monkeypatch.setattr(ad, "register_work_group_member", lambda **_kw: False)
    result, executor = _dispatch(monkeypatch)
    assert result["status"] == "rejected"
    assert "no background child was submitted" in result["error"]
    assert executor.submitted == []


def test_submit_failure_removes_phantom_member_and_empty_group(monkeypatch):
    class BrokenExecutor:
        def submit(self, _fn):
            raise RuntimeError("executor unavailable")

    monkeypatch.setattr(ad, "_get_executor", lambda _limit: BrokenExecutor())
    result = ad.dispatch_async_delegation_batch(
        goals=["inspect the implementation carefully"], context=None,
        toolsets=None, role="leaf", model="test", session_key="route",
        runner=lambda: {}, origin_work_id="work-submit-fail",
        owner_turn_id="turn-submit-fail",
    )
    assert result["status"] == "rejected"
    assert _row(
        "SELECT COUNT(*) AS n FROM async_delegations WHERE origin_work_id='work-submit-fail'"
    )["n"] == 0
    assert _row(
        "SELECT COUNT(*) AS n FROM async_delegation_work_groups "
        "WHERE work_id='work-submit-fail'"
    )["n"] == 0


def test_open_completion_persists_without_any_event_then_seal_enqueues_one(
    monkeypatch, _isolated_runtime
):
    result, _executor = _dispatch(monkeypatch)
    delegation_id = result["delegation_id"]
    record = dict(ad._records[delegation_id])
    record["completed_at"] = time.time()
    ad._push_batch_completion_event(
        record, {"results": [{"status": "completed", "summary": "ok"}]}, "completed"
    )
    assert _isolated_runtime.empty()
    assert _row(
        "SELECT state FROM async_delegations WHERE delegation_id=?", (delegation_id,)
    )["state"] == "completed"

    event = ad.seal_and_enqueue_work_group("work-1", "turn-1")
    assert event is not None
    queued = _isolated_runtime.get_nowait()
    assert queued == event
    assert queued["type"] == "async_delegation_work_closeout"
    assert queued["delivery_id"] == queued["envelope"]["delivery_id"]
    assert _isolated_runtime.empty()
    assert ad.seal_and_enqueue_work_group("work-1", "turn-1") is None


def test_completion_after_seal_enqueues_aggregate_not_individual(
    monkeypatch, _isolated_runtime
):
    result, _executor = _dispatch(monkeypatch)
    assert ad.seal_work_group("work-1", "turn-1")
    delegation_id = result["delegation_id"]
    record = dict(ad._records[delegation_id])
    record["completed_at"] = time.time()
    ad._push_batch_completion_event(
        record, {"results": [{"status": "completed", "summary": "ok"}]}, "completed"
    )
    queued = _isolated_runtime.get_nowait()
    assert queued["type"] == "async_delegation_work_closeout"
    assert queued["origin_work_id"] == "work-1"
    assert _isolated_runtime.empty()


def test_seal_completion_race_publishes_exactly_one_aggregate(
    monkeypatch, _isolated_runtime
):
    result, _executor = _dispatch(monkeypatch)
    record = dict(ad._records[result["delegation_id"]])
    record["completed_at"] = time.time()
    barrier = threading.Barrier(2)

    def finish():
        barrier.wait()
        ad._push_batch_completion_event(record, {"results": []}, "completed")

    def seal():
        barrier.wait()
        ad.seal_and_enqueue_work_group("work-1", "turn-1")

    threads = [threading.Thread(target=finish), threading.Thread(target=seal)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert _isolated_runtime.qsize() == 1
    assert _isolated_runtime.get_nowait()["type"] == "async_delegation_work_closeout"


def test_legacy_completion_still_uses_individual_rail(monkeypatch, _isolated_runtime):
    result, _executor = _dispatch(monkeypatch, work="", turn="")
    record = dict(ad._records[result["delegation_id"]])
    record["completed_at"] = time.time()
    ad._push_batch_completion_event(record, {"results": []}, "completed")
    assert _isolated_runtime.get_nowait()["type"] == "async_delegation"


def test_replacement_reopens_next_generation_before_submit(monkeypatch):
    first, _ = _dispatch(monkeypatch)
    delegation_id = first["delegation_id"]
    assert ad.persist_group_member_completion(
        delegation_id,
        {"delegation_id": delegation_id, "status": "completed"},
        {"status": "completed", "summary": "review"},
    ) is False
    assert ad.seal_work_group("work-1", "turn-1")
    claimed = ad.claim_ready_work_group("work-1", "test")
    assert claimed is not None
    delivery = claimed["envelope"]["delivery_id"]
    assert ad.bind_work_group_closeout_turn(
        "work-1", delivery, claimed["claim_id"], "closeout-turn"
    )

    replacement, executor = _dispatch(
        monkeypatch,
        generation=1,
        turn="closeout-turn",
        closeout_delivery_id=delivery,
        closeout_claim_id=claimed["claim_id"],
    )
    assert replacement["status"] == "dispatched"
    assert len(executor.submitted) == 1
    group = _row(
        "SELECT state, generation, owner_turn_id FROM async_delegation_work_groups "
        "WHERE work_id='work-1'"
    )
    assert dict(group) == {
        "state": "open",
        "generation": 1,
        "owner_turn_id": "closeout-turn",
    }


def test_recovery_enqueues_same_delivery_identity(monkeypatch, _isolated_runtime):
    result, _ = _dispatch(monkeypatch)
    delegation_id = result["delegation_id"]
    ad.persist_group_member_completion(
        delegation_id,
        {"delegation_id": delegation_id, "status": "completed"},
        {"status": "completed", "summary": "done"},
    )
    ad.seal_work_group("work-1", "turn-1")
    events = ad.recover_and_enqueue_work_groups()
    assert len(events) == 1
    assert events[0]["delivery_id"] == events[0]["envelope"]["delivery_id"]
    assert _isolated_runtime.get_nowait()["delivery_id"] == events[0]["delivery_id"]
    assert ad.recover_and_enqueue_work_groups() == []
