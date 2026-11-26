"""Failed async delegations stay visible after the live roster forgets them (#97202).

A renderer reload drops the in-memory subagent roster, and an ended child leaves it anyway, so a
failed delegation had nowhere to show up. ``failed_delegations_for_session`` reads the durable row.
These tests write real rows into a temp ``state.db`` through the module's own persistence calls.
"""

import threading
import time

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.waiting import wait_until
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def home(tmp_path):
    ad._reset_for_tests()
    token = set_hermes_home_override(str(tmp_path))
    yield tmp_path
    reset_hermes_home_override(token)
    ad._reset_for_tests()


def _delegation(delegation_id, *, status, result, ui="ui-1", parent="agent-1", completed_at=None, **task):
    ad._persist_dispatch({"delegation_id": delegation_id, "session_key": "", "origin_ui_session_id": ui,
                          "parent_session_id": parent, "dispatched_at": time.time() - 60, **task})
    ad._persist_completion({"delegation_id": delegation_id, "status": status,
                            "completed_at": completed_at or time.time()}, result)


def test_a_failed_single_delegation_is_listed_for_its_session():
    _delegation("d-fail", status="error", goal="audit billing",
                result={"status": "error", "error": "interrupted: waiting for model response"})
    _delegation("d-ok", status="completed", goal="write docs", result={"status": "completed", "summary": "done"})

    assert ad.failed_delegations_for_session("ui-1") == [{
        "delegation_id": "d-fail", "task_index": 0, "status": "error", "goal": "audit billing",
        "error": "interrupted: waiting for model response", "dispatched_at": pytest.approx(time.time() - 60, abs=30),
        "completed_at": pytest.approx(time.time(), abs=30)}]


def test_a_batch_that_completed_still_surfaces_its_failed_task():
    # A split unit stores the whole call's goals (dispatch_async_delegation_batch) and its own task indexes.
    _delegation("d-batch", status="completed", is_batch=True, task_indexes=[3, 4],
                goals=["scan auth", "scan db", "scan cache", "scan api", "scan web"],
                result={"results": [{"task_index": 3, "status": "completed", "summary": "ok"},
                                    {"task_index": 4, "status": "timeout", "error": "no progress"}]})

    [row] = ad.failed_delegations_for_session("ui-1")
    assert (row["task_index"], row["goal"], row["status"], row["error"]) == (4, "scan web", "timeout", "no progress")


def test_a_split_unit_labels_its_failed_task_with_that_tasks_goal():
    """Through the real producer: a split unit stores the call's goals, so task 1 is 'B goal', not 'A goal'."""
    from tools.process_registry import process_registry
    handle = ad.dispatch_async_delegation_batch(
        delegation_id="deleg_call-2", task_indexes=[1], goals=["A goal", "B goal"], context=None, toolsets=None,
        role="leaf", model="m", session_key="", origin_ui_session_id="ui-1", max_async_children=2,
        runner=lambda: {"results": [{"task_index": 1, "status": "error", "error": "boom"}],
                        "total_duration_seconds": 0})
    assert handle["status"] == "dispatched"
    assert process_registry.completion_queue.get(timeout=10)["delegation_id"] == "deleg_call-2"

    [row] = ad.failed_delegations_for_session("ui-1")
    assert (row["task_index"], row["goal"], row["error"]) == (1, "B goal", "boom")


@pytest.mark.parametrize("state, result", [
    ("stalled", {"results": [], "error": "stopped responding"}),  # stall monitor; a runner crash writes the same shape
    ("unknown", {"status": "unknown", "error": "owner exited"}),  # owner gone before any child recorded
])
def test_a_split_unit_that_failed_before_any_task_reported_lists_its_own_tasks(state, result):
    _delegation("d-unit", status=state, is_batch=True, task_indexes=[1, 2],
                goals=["A goal", "B goal", "C goal"], result=result)
    _delegation("d-whole", status=state, is_batch=True, goals=["A goal", "B goal"], result=result, ui="ui-2")

    assert [(r["task_index"], r["goal"]) for r in ad.failed_delegations_for_session("ui-1")] == [
        (1, "B goal"), (2, "C goal")]
    assert [r["delegation_id"] for r in ad.failed_delegations_for_session("ui-2")] == ["d-whole"]


def test_a_split_unit_that_stalls_after_one_task_finished_lists_only_the_stalled_task(monkeypatch):
    """Task 1 finishes (recorded on the unit's row), then task 2 stalls: the forced finalization keeps task 1's
    result, so only task 2 surfaces as failed."""
    from tools.process_registry import process_registry
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.3)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", 0.3)
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.1)
    gate = threading.Event()

    def runner():
        ad.record_unit_child("deleg_call-2", {"task_index": 1, "status": "completed", "summary": "done"})
        gate.wait(timeout=10)
        return {"results": []}

    try:
        res = ad.dispatch_async_delegation_batch(
            delegation_id="deleg_call-2", task_indexes=[1, 2], goals=["A goal", "B goal", "C goal"], context=None,
            toolsets=None, role="leaf", model="m", session_key="", origin_ui_session_id="ui-1",
            max_async_children=2, runner=runner, progress_fn=lambda: ((0, None), False))
        assert res["status"] == "dispatched"
        failed = wait_until(lambda: ad.failed_delegations_for_session("ui-1"), timeout=5)
        assert [(r["task_index"], r["goal"], r["status"]) for r in failed] == [(2, "C goal", "stalled")]
    finally:
        gate.set()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


@pytest.mark.parametrize("task_indexes, unfinished", [([1, 2], [2]), (None, [0, 2])], ids=["split-unit", "whole-batch"])
def test_a_unit_whose_runner_raises_after_one_task_finished_lists_only_the_unfinished_tasks(task_indexes, unfinished):
    """The runner-crash twin of the stall test: task 1's recorded result survives, only the rest surface."""
    from tools.process_registry import process_registry
    goals = ["A goal", "B goal", "C goal"]

    def runner():
        ad.record_unit_child("deleg_call-2", {"task_index": 1, "status": "completed", "summary": "done"})
        raise RuntimeError("kaboom")

    ad.dispatch_async_delegation_batch(
        delegation_id="deleg_call-2", task_indexes=task_indexes, goals=goals, context=None, toolsets=None,
        role="leaf", model="m", session_key="", origin_ui_session_id="ui-1", max_async_children=2, runner=runner)
    evt = process_registry.completion_queue.get(timeout=10)

    assert [(r["task_index"], r["status"]) for r in evt["results"]] == sorted(
        [(1, "completed")] + [(i, "error") for i in unfinished])
    assert [(r["task_index"], r["goal"]) for r in ad.failed_delegations_for_session("ui-1")] == [
        (i, goals[i]) for i in unfinished]


def test_a_unit_that_finished_reports_its_runner_results_and_drops_the_recorded_copies():
    from tools.process_registry import process_registry

    def runner():
        ad.record_unit_child("deleg_call-2", {"task_index": 1, "status": "completed", "summary": "recorded"})
        return {"results": [{"task_index": i, "status": "completed", "summary": "final"} for i in (1, 2)]}

    ad.dispatch_async_delegation_batch(
        delegation_id="deleg_call-2", task_indexes=[1, 2], goals=["A goal", "B goal", "C goal"], context=None,
        toolsets=None, role="leaf", model="m", session_key="", origin_ui_session_id="ui-1", max_async_children=2,
        runner=runner)
    evt = process_registry.completion_queue.get(timeout=10)

    assert [(r["task_index"], r["summary"]) for r in evt["results"]] == [(1, "final"), (2, "final")]
    record = wait_until(lambda: (r := ad._records.get("deleg_call-2")) and r.get("status") == "completed" and r)
    assert "_finished_children" not in record  # a retained completed record doesn't keep every child


def test_the_durable_parent_id_claims_rows_after_a_reload_remints_the_ui_session():
    _delegation("d-fail", status="stalled", goal="refactor", result={"status": "error", "error": "stalled"})

    assert [r["delegation_id"] for r in ad.failed_delegations_for_session("ui-reminted", "agent-1")] == ["d-fail"]


def test_other_sessions_and_old_failures_stay_out():
    _delegation("d-other", status="error", goal="x", ui="ui-2", parent="agent-2", result={"status": "error"})
    _delegation("d-old", status="error", goal="y", completed_at=time.time() - 3 * 86400, result={"status": "error"})

    assert ad.failed_delegations_for_session("ui-1", "agent-1") == []
    assert ad.failed_delegations_for_session() == []
