"""``delegate_task(action='result')`` reads a delegation's durable record, scoped to the calling session.

Rows are written into a temp ``state.db`` through the ledger's own persistence calls, so the read goes through the
same path a restarted gateway would use (no in-memory registry involved).
"""

import json
import time
from types import SimpleNamespace

import pytest

from agent.tool_guardrails import _subagent_spawn_count
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad
from tools import delegate_tool
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


@pytest.fixture(autouse=True)
def home(tmp_path):
    ad._reset_for_tests()
    token = set_hermes_home_override(str(tmp_path))
    yield tmp_path
    reset_hermes_home_override(token)
    ad._reset_for_tests()


def _agent(session_id="sess-1", lineage=None):
    db = SimpleNamespace(resolve_resume_session_id=lambda sid: (lineage or {}).get(sid, sid))
    return SimpleNamespace(session_id=session_id, _session_db=db, _delegate_depth=0)


def _store(delegation_id, *, parent="sess-1", result=None, model="m-1", **extra):
    ad._persist_dispatch({"delegation_id": delegation_id, "session_key": "key-1", "origin_ui_session_id": "ui-1",
                          "origin_session_id": "origin-1", "parent_session_id": parent, "model": model,
                          "dispatched_at": time.time() - 5, "goal": "g", **extra})
    if result is not None:
        ad._persist_completion({"delegation_id": delegation_id, "status": "completed", "completed_at": time.time()},
                               result)


def _result(parent, delegation_id):
    return json.loads(delegate_task(action="result", delegation_id=delegation_id, parent_agent=parent))


def test_own_delegation_returns_the_stored_record():
    _store("d-own", result={"status": "completed", "summary": "all done"})

    out = _result(_agent(), "d-own")

    assert out["action"] == "result" and out["delegation_id"] == "d-own"
    assert out["state"] == "completed" and out["delivery_state"] == "pending" and out["delivery_attempts"] == 0
    assert out["parent_session_id"] == "sess-1" and out["origin_session_id"] == "origin-1" and out["model"] == "m-1"
    assert out["dispatched_at"] and out["completed_at"] and out["delivered_at"] is None
    assert out["result"] == {"status": "completed", "summary": "all done"}


def test_partial_per_child_entries_are_returned_while_the_unit_runs():
    _store("d-run", is_batch=True, goals=["a", "b"], task_indexes=[0, 1])
    ad.record_unit_child("d-run", {"task_index": 0, "status": "completed", "summary": "first"})

    out = _result(_agent(), "d-run")

    assert out["state"] == "running" and out["completed_at"] is None
    assert out["result"]["partial"] is True
    assert [r["summary"] for r in out["result"]["results"]] == ["first"]


def test_delivery_fields_follow_the_ledger():
    _store("d-del", result={"status": "completed", "summary": "x"})
    ad.mark_completion_delivered("d-del")

    out = _result(_agent(), "d-del")

    assert out["delivery_state"] == "delivered" and out["delivered_at"]


def test_lineage_tip_of_a_rotated_session_still_owns_the_row():
    _store("d-rot", parent="sess-old", result={"status": "completed", "summary": "x"})

    assert _result(_agent("sess-new", {"sess-old": "sess-new"}), "d-rot")["result"]["summary"] == "x"


def test_foreign_and_unknown_ids_fail_closed_with_the_same_error():
    _store("d-foreign", parent="sess-2", result={"status": "completed", "summary": "secret"})
    _store("d-ownerless", parent=None, result={"status": "completed", "summary": "secret"})
    caller = _agent()

    foreign = delegate_task(action="result", delegation_id="d-foreign", parent_agent=caller)
    unknown = delegate_task(action="result", delegation_id="d-missing", parent_agent=caller)
    ownerless = delegate_task(action="result", delegation_id="d-ownerless", parent_agent=caller)

    assert "secret" not in foreign + ownerless
    assert "error" in json.loads(foreign)
    assert json.loads(foreign)["error"] == json.loads(unknown)["error"].replace("d-missing", "d-foreign")
    assert json.loads(ownerless)["error"] == json.loads(unknown)["error"].replace("d-missing", "d-ownerless")
    assert "result" not in json.loads(foreign) and "sess-2" not in foreign


def test_caller_without_a_session_id_owns_nothing():
    _store("d-own", result={"status": "completed", "summary": "x"})

    assert "error" in _result(_agent(""), "d-own")


def test_ledger_read_failure_is_the_same_not_found_error(monkeypatch):
    def boom(_id):
        raise RuntimeError("db locked")
    monkeypatch.setattr(ad, "get_durable_delegation", boom)

    assert "No delegation 'd-x' found" in _result(_agent(), "d-x")["error"]


def test_worker_ledger_row_without_owner_columns_relies_on_the_store_fence(monkeypatch):
    # The session-worker ledger refuses foreign rows itself (raises) and omits parent_session_id from what it returns.
    row = {"delegation_id": "d-w", "state": "completed", "delivery_state": "pending", "delivery_attempts": 0,
           "result": {"summary": "from worker"}}
    monkeypatch.setattr(ad, "get_durable_delegation", lambda _id: row)

    out = _result(_agent(), "d-w")

    assert out["result"] == {"summary": "from worker"} and out["parent_session_id"] is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_delegation_id_is_rejected_clearly(value):
    out = json.loads(delegate_task(action="result", delegation_id=value, parent_agent=_agent()))

    assert "requires delegation_id" in out["error"]


def test_result_never_consumes_a_spawn_slot():
    assert _subagent_spawn_count({"action": "result", "delegation_id": "d-1"}) == 0
    assert _subagent_spawn_count({"action": " Result ", "delegation_id": "d-1"}) == 0
    assert _subagent_spawn_count({"goal": "x"}) == 1


def test_result_bypasses_the_spawn_gates(monkeypatch):
    _store("d-own", result={"status": "completed", "summary": "x"})
    monkeypatch.setattr(delegate_tool, "is_spawn_paused", lambda: True)
    agent = _agent()
    agent._delegate_depth = 99

    assert _result(agent, "d-own")["state"] == "completed"


def test_schema_advertises_result_and_keeps_spawn_args_optional():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "result" in props["action"]["enum"] and "delegation_id" in props
    assert DELEGATE_TASK_SCHEMA["parameters"]["required"] == []
