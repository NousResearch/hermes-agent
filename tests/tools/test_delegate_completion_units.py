"""Phase 1 RED contract for upstream completion-unit delegation semantics."""
from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import Any, cast
import json
import sqlite3

from tools import async_delegation as ad
from tools import delegate_tool_dispatch as dispatch


def _child(task_index):
    return SimpleNamespace(_subagent_id=f"sa-{task_index}")


def _batch(tasks):
    return dispatch._Batch(
        task_list=tasks,
        children=[(i, task, _child(i)) for i, task in enumerate(tasks)],
        parent_agent=MagicMock(),
        creds={"model": "model-under-test"},
        context=None,
        top_role="leaf",
        max_children=2,
        live_deleg_id="deleg_phase1",
        live_writers=[],
        live_paths=[],
        origin_wake_sid="origin",
        origin_ui_session_id="ui",
        origin_owner_transport=None,
        origin_owner_session_record=None,
        overall_start=0.0,
    )


def test_units_partition_ungrouped_tasks_independently():
    batch = _batch(
        [
            {"goal": "fast one"},
            {"goal": "slow two"},
        ]
    )
    partitioner = cast(Any, getattr(dispatch, "_units_of", None))
    assert callable(partitioner)
    units = partitioner(batch)
    assert [[i for i, _, _ in unit.children] for unit in units] == [[0], [1]]


def test_units_keep_same_group_together_but_not_unrelated_tasks():
    batch = _batch(
        [
            {"goal": "group one", "group": "research"},
            {"goal": "independent"},
            {"goal": "group two", "group": "research"},
        ]
    )
    partitioner = cast(Any, getattr(dispatch, "_units_of", None))
    assert callable(partitioner)
    units = partitioner(batch)
    assert [[i for i, _, _ in unit.children] for unit in units] == [[0, 2], [1]]
    assert units[0].group == "research"
    assert units[1].group is None


def test_grouping_controls_delivery_not_execution_order():
    batch = _batch(
        [
            {"goal": "slow grouped", "group": "research"},
            {"goal": "fast unrelated"},
            {"goal": "fast grouped", "group": "research"},
        ]
    )
    partitioner = cast(Any, getattr(dispatch, "_units_of", None))
    assert callable(partitioner)
    units = partitioner(batch)
    assert [i for i, _, _ in units[0].children] == [0, 2]
    assert [i for i, _, _ in units[1].children] == [1]


def test_each_unit_uses_one_shared_slot_key_and_preserves_task_indexes():
    dispatch_unit = cast(Any, getattr(dispatch, "_dispatch_unit", None))
    assert callable(dispatch_unit)
    assert "slot_key" in dispatch_unit.__code__.co_varnames
    assert "task_indexes" in ad.dispatch_async_delegation_batch.__code__.co_varnames


def test_whole_batch_occupies_one_async_slot_but_reports_child_count():
    with ad._records_lock:
        ad._records.clear()
        ad._records.update(
            {
                "one-batch": {
                    "status": "running",
                    "is_batch": True,
                    "goals": ["a", "b", "c"],
                }
            }
        )
    try:
        assert ad.active_count() == 1
        assert ad.active_task_count() == 3
    finally:
        with ad._records_lock:
            ad._records.clear()


def test_child_completion_preserves_task_index_and_group_in_published_event(monkeypatch):
    monkeypatch.setattr(ad, "_persist_completion", lambda event, result: None)
    registry = SimpleNamespace(completion_queue=MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "tools.process_registry", SimpleNamespace(process_registry=registry))
    record = {
        "delegation_id": "deleg_phase1",
        "is_batch": True,
        "goal": "batch",
        "goals": ["slow grouped", "fast independent"],
        "session_key": "session",
        "origin_ui_session_id": "ui",
        "origin_session_id": "origin",
        "parent_session_id": "parent",
        "role": "leaf",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
        "status": "finalizing",
    }
    result = {
        "results": [
            {"task_index": 1, "group": None, "status": "completed", "summary": "fast"},
            {"task_index": 0, "group": "research", "status": "completed", "summary": "slow"},
        ],
        "total_duration_seconds": 1.0,
    }
    ad._push_completion_event(record, result, "completed")
    event = registry.completion_queue.put.call_args.args[0]
    assert [entry["task_index"] for entry in event["results"]] == [1, 0]
    assert [entry["group"] for entry in event["results"]] == [None, "research"]


def test_partial_child_persistence_and_failed_child_notice_are_first_class():
    assert callable(getattr(ad, "record_unit_child", None))
    assert callable(getattr(ad, "publish_child_failure_notice", None))


def test_child_result_is_redacted_and_idempotent_in_durable_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    record = {
        "delegation_id": "durable-unit",
        "session_key": "session",
        "origin_ui_session_id": "ui",
        "parent_session_id": "parent",
        "origin_session_id": "origin",
        "dispatched_at": 1.0,
        "goal": "child",
        "status": "running",
        "goals": ["child"],
        "is_batch": True,
    }
    with ad._records_lock:
        ad._records[record["delegation_id"]] = dict(record)
    try:
        ad._persist_dispatch(record)
        assert ad.record_unit_child(
            "durable-unit", 0, {"status": "completed", "summary": "sk-test-secret"}
        )
        assert ad.record_unit_child(
            "durable-unit", 0, {"status": "completed", "summary": "replacement"}
        )
        with sqlite3.connect(tmp_path / "state.db") as conn:
            raw = conn.execute(
                "SELECT child_results_json FROM async_delegations WHERE delegation_id=?",
                ("durable-unit",),
            ).fetchone()[0]
        stored = json.loads(raw)
        assert list(stored) == ["0"]
        assert stored["0"]["summary"] != "sk-test-secret"
        assert stored["0"]["summary"] != "replacement"
    finally:
        with ad._records_lock:
            ad._records.pop("durable-unit", None)


def test_failed_child_notice_keeps_task_index_group_and_error():
    notice = ad.publish_child_failure_notice("unit-1", 3, "tool failed", "research")
    assert notice["task_index"] == 3
    assert notice["group"] == "research"
    assert notice["error"] == "tool failed"
