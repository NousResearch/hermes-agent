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


def _batch(tasks, monkeypatch=None):
    # Split semantics under test: upstream default (c89f3b8800) is one
    # completion per call; these tests pin independent_completions on.
    if monkeypatch is not None:
        import tools.delegate_tool as dt
        monkeypatch.setattr(dt, "_load_config", lambda: {"independent_completions": True})
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
        origin_session_history_delivery=False,
        overall_start=0.0,
    )


def test_units_partition_ungrouped_tasks_independently(monkeypatch):
    batch = _batch(
        [
            {"goal": "fast one"},
            {"goal": "slow two"},
        ],
        monkeypatch,
    )
    partitioner = cast(Any, getattr(dispatch, "_units_of", None))
    assert callable(partitioner)
    units = partitioner(batch)
    assert [[i for i, _, _ in unit.children] for unit in units] == [[0], [1]]


def test_units_keep_same_group_together_but_not_unrelated_tasks(monkeypatch):
    batch = _batch(
        [
            {"goal": "group one", "group": "research"},
            {"goal": "independent"},
            {"goal": "group two", "group": "research"},
        ],
        monkeypatch,
    )
    partitioner = cast(Any, getattr(dispatch, "_units_of", None))
    assert callable(partitioner)
    units = partitioner(batch)
    assert [[i for i, _, _ in unit.children] for unit in units] == [[0, 2], [1]]
    assert units[0].group == "research"
    assert units[1].group is None


def test_grouping_controls_delivery_not_execution_order(monkeypatch):
    batch = _batch(
        [
            {"goal": "slow grouped", "group": "research"},
            {"goal": "fast unrelated"},
            {"goal": "fast grouped", "group": "research"},
        ],
        monkeypatch,
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


def test_crash_before_unit_join_keeps_recorded_children_and_marks_rest_unknown(tmp_path, monkeypatch):
    """Integrated split-unit seam: a finished child recorded on the unit's own
    row survives an owner crash; the unfinished sibling replays as unknown."""
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    record = {
        "delegation_id": "durable-unit",
        "session_key": "session",
        "origin_ui_session_id": "ui",
        "parent_session_id": "parent",
        "origin_session_id": "origin",
        "dispatched_at": 1.0,
        "goal": "unit",
        "goals": ["child-a", "child-b"],
        "status": "running",
        "is_batch": True,
        "task_indexes": [0, 1],
    }
    with ad._records_lock:
        ad._records[record["delegation_id"]] = dict(record)
    try:
        ad._persist_dispatch(record)
        # one child finished before the crash; the other never ran
        ad.record_unit_child(
            "durable-unit", {"task_index": 0, "status": "completed", "summary": "child-a done"}
        )
        import gateway.status as _gs
        monkeypatch.setattr(_gs, "_pid_exists", lambda pid: False)
        assert ad.recover_abandoned_delegations() == 1
        with sqlite3.connect(tmp_path / "state.db") as conn:
            event_json, result_json, state = conn.execute(
                "SELECT event_json, result_json, state FROM async_delegations WHERE delegation_id=?",
                ("durable-unit",),
            ).fetchone()
        assert state == "unknown"
        event = json.loads(event_json)
        by_index = {r["task_index"]: r for r in event["results"]}
        assert by_index[0]["status"] == "completed"
        assert by_index[0]["summary"] == "child-a done"
        assert by_index[1]["status"] == "unknown"
    finally:
        with ad._records_lock:
            ad._records.pop("durable-unit", None)


def test_failed_child_notice_rides_completion_shape_without_claiming_final(monkeypatch):
    """Interim failure notice: same async_delegation shape as the batch result
    (every drain/route/format path treats it identically), marked interim, and
    never claims or dedups against the final result's durable row."""
    from tools.process_registry import process_registry
    record = {
        "delegation_id": "unit-1",
        "session_key": "session",
        "origin_ui_session_id": "ui",
        "origin_session_id": "origin",
        "parent_session_id": "parent",
        "goal": "0",
        "goals": ["a", "b"],
        "is_batch": True,
        "status": "running",
        "dispatched_at": 1.0,
    }
    with ad._records_lock:
        ad._records["unit-1"] = dict(record)
    try:
        seen = []
        monkeypatch.setattr(
            process_registry.completion_queue, "put", lambda evt: seen.append(evt)
        )
        entry = {"task_index": 0, "status": "failed", "error": "tool failed", "summary": None}
        ad.push_task_failure_notice("unit-1", dict(entry), n_tasks=2)
        assert len(seen) == 1
        notice = seen[0]
        assert notice["type"] == "async_delegation"
        assert notice["task_failure_notice"] is True
        assert notice["results"] == [entry]
        assert notice["status"] == "running"  # batch NOT finalized
        # interim: no durable claim, final row untouched
        assert ad.is_interim_delegation_event(notice) is True
        assert ad.claim_event_delivery(notice, "test") == ""
    finally:
        with ad._records_lock:
            ad._records.pop("unit-1", None)


def test_detached_unit_records_finished_child_and_surfaces_failure_now(tmp_path, monkeypatch):
    """_Batch seam: a detached unit (unit_id set) persists each finished child
    on the unit row and emits an interim notice for a failed non-final child;
    sync batches (unit_id None) persist nothing and emit nothing."""
    from dataclasses import replace as _replace
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    record = {
        "delegation_id": "unit-9",
        "session_key": "session",
        "origin_ui_session_id": "ui",
        "parent_session_id": "parent",
        "origin_session_id": "origin",
        "dispatched_at": 1.0,
        "goal": "unit",
        "goals": ["a", "b"],
        "status": "running",
        "is_batch": True,
        "task_indexes": [0, 1],
    }
    with ad._records_lock:
        ad._records["unit-9"] = dict(record)
    try:
        ad._persist_dispatch(record)
        parent = MagicMock()
        unit = dispatch._Batch(
            task_list=[{"goal": "a"}, {"goal": "b"}],
            children=[(0, {"goal": "a"}, MagicMock()), (1, {"goal": "b"}, MagicMock())],
            parent_agent=parent,
            creds={"model": "m"},
            context=None,
            top_role="leaf",
            max_children=2,
            live_deleg_id="call-9",
            live_writers=[],
            live_paths=[],
            origin_wake_sid="",
            origin_ui_session_id="",
            origin_owner_transport=None,
            origin_owner_session_record=None,
            origin_session_history_delivery=False,
            overall_start=0.0,
            group=None,
            unit_id="unit-9",
        )
        seen = []
        monkeypatch.setattr(
            "tools.process_registry.process_registry.completion_queue.put",
            lambda evt: seen.append(evt),
        )
        entry = {"task_index": 0, "status": "failed", "error": "boom", "summary": None}
        dispatch._record_child_completion(unit, dict(entry))
        dispatch._maybe_emit_failure_notice(unit, dict(entry), remaining=1)
        with sqlite3.connect(tmp_path / "state.db") as conn:
            raw = conn.execute(
                "SELECT result_json FROM async_delegations WHERE delegation_id=?",
                ("unit-9",),
            ).fetchone()[0]
        partial = json.loads(raw)
        assert partial["partial"] is True
        assert [r["task_index"] for r in partial["results"]] == [0]
        assert len(seen) == 1 and seen[0]["task_failure_notice"] is True
        # sync batch: no unit row, no notice
        sync = _replace(unit, unit_id=None)
        dispatch._record_child_completion(sync, dict(entry))
        dispatch._maybe_emit_failure_notice(sync, dict(entry), remaining=1)
        assert len(seen) == 1
    finally:
        with ad._records_lock:
            ad._records.pop("unit-9", None)
