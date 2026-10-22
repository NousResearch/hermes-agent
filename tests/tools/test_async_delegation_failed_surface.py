"""Failed async delegations stay visible after the live roster forgets them (#97202).

A renderer reload drops the in-memory subagent roster, and an ended child leaves it anyway, so a
failed delegation had nowhere to show up. ``failed_delegations_for_session`` reads the durable row.
These tests write real rows into a temp ``state.db`` through the module's own persistence calls.
"""

import time

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
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
    _delegation("d-batch", status="completed", is_batch=True, goals=["scan api", "scan web"], task_indexes=[3, 4],
                result={"results": [{"task_index": 3, "status": "completed", "summary": "ok"},
                                    {"task_index": 4, "status": "timeout", "error": "no progress"}]})

    [row] = ad.failed_delegations_for_session("ui-1")
    assert (row["task_index"], row["goal"], row["status"], row["error"]) == (4, "scan web", "timeout", "no progress")


def test_the_durable_parent_id_claims_rows_after_a_reload_remints_the_ui_session():
    _delegation("d-fail", status="stalled", goal="refactor", result={"status": "error", "error": "stalled"})

    assert [r["delegation_id"] for r in ad.failed_delegations_for_session("ui-reminted", "agent-1")] == ["d-fail"]


def test_other_sessions_and_old_failures_stay_out():
    _delegation("d-other", status="error", goal="x", ui="ui-2", parent="agent-2", result={"status": "error"})
    _delegation("d-old", status="error", goal="y", completed_at=time.time() - 3 * 86400, result={"status": "error"})

    assert ad.failed_delegations_for_session("ui-1", "agent-1") == []
    assert ad.failed_delegations_for_session() == []
