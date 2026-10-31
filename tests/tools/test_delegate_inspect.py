"""Durable result recovery is owner-scoped, bounded, and never consumes delivery."""

import json
import os
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ledger
from tools import delegate_tool


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    from gateway import session_context
    token = set_hermes_home_override(str(tmp_path))
    monkeypatch.setattr(session_context, "get_session_env", lambda name, default="": default)
    yield tmp_path
    reset_hermes_home_override(token)


def _saved(*, running=False, batch=False):
    record = {"delegation_id": "deleg_recover", "parent_session_id": "parent-before", "origin_ui_session_id": "ui-own",
              "session_key": "", "dispatched_at": 100.0, "goal": "recover this goal"}
    if batch:
        record.pop("goal")
        record.update(goals=["first", "second"], is_batch=True)
    ledger._persist_dispatch(record)
    if not running:
        result = {"status": "completed", "summary": "recovered answer", "tool_trace": "PRIVATE TRACE"}
        if batch:
            result = {"results": [{"task_index": 3, "status": "completed", "summary": "first answer"},
                                  {"task_index": 4, "status": "error", "error": "second failed"}]}
        ledger._persist_completion({"delegation_id": record["delegation_id"], "status": "completed",
                                    "completed_at": 200.0}, result)
    return record["delegation_id"]


def _inspect(parent, delegation_id="deleg_recover", *, native=False):
    args = {"action": "inspect", "delegation_id": delegation_id}
    if native:
        from run_agent import AIAgent
        return json.loads(AIAgent._dispatch_delegate_task(parent, args))
    return json.loads(delegate_tool.registry.dispatch("delegate_task", args, parent_agent=parent))


def _row():
    with ledger._transaction() as conn:
        return conn.execute("SELECT * FROM async_delegations WHERE delegation_id='deleg_recover'").fetchone()


@pytest.mark.parametrize("owner", ["parent", "lineage", "ui"])
@pytest.mark.parametrize("native", [False, True])
def test_owner_recovers_finished_result_without_changing_row(monkeypatch, owner, native):
    _saved(batch=True)
    parent = SimpleNamespace(session_id="parent-before" if owner == "parent" else "parent-after")
    if owner == "lineage":
        parent._session_db = SimpleNamespace(resolve_resume_session_id=lambda sid: "parent-after")
    if owner == "ui":
        from gateway import session_context
        monkeypatch.setattr(session_context, "get_session_env", lambda name, default="": "ui-own")
    before = _row()
    out = _inspect(parent, native=native)
    assert out == {"delegation_id": "deleg_recover", "state": "completed", "goal": "first; second",
                   "dispatched_at": 100.0, "completed_at": 200.0, "delivery_state": "pending",
                   "results": [{"task_index": 3, "status": "completed", "summary": "first answer"},
                               {"task_index": 4, "status": "error", "summary": "second failed"}]}
    assert _row() == before


def test_foreign_and_missing_ids_are_indistinguishable():
    _saved()
    foreign = SimpleNamespace(session_id="foreign")
    out = _inspect(foreign)
    assert out == _inspect(foreign, "absent")
    assert "not found" in out["error"].lower()
    assert "error" in _inspect(SimpleNamespace(session_id=""))


def test_summary_redacted_before_bounding_and_private_fields_omitted():
    _saved()
    # A synthetic PEM straddles the cap; redaction must see the entire value before slicing.
    summary = "prefix " + "-----BEGIN PRIVATE KEY-----\n" + "synthetic-body\n" * 600 + "-----END PRIVATE KEY-----"
    with ledger._transaction() as conn:
        conn.execute("UPDATE async_delegations SET task_json=?,result_json=? WHERE delegation_id='deleg_recover'",
                     (json.dumps({"goal": "g" * 5000}), json.dumps({"summary": summary + "\n" + "tail " * 2000,
                       "transcript": "PRIVATE TRACE", "tool_trace": "PRIVATE TRACE"})))
    before = _row()
    out = _inspect(SimpleNamespace(session_id="parent-before"))
    assert len(out["goal"]) <= 1000
    assert len(out["results"][0]["summary"]) <= 4000
    assert "prefix" in out["results"][0]["summary"]
    assert "synthetic-body" not in json.dumps(out)
    assert "PRIVATE TRACE" not in json.dumps(out)
    assert _row() == before


def test_running_inspect_returns_only_running_state():
    _saved(running=True)
    before = _row()
    assert _inspect(SimpleNamespace(session_id="parent-before")) == {"state": "running"}
    assert _row() == before


@pytest.mark.parametrize("target", ["deleg_recover", "foreign", "absent"])
def test_worker_inspect_uses_remote_ledger_without_local_connection(monkeypatch, tmp_path, target):
    from agent import runtime_session_store
    from gateway.session_worker_delegation import _read
    from hermes_state import SessionDB
    from hermes_state_runtime import RuntimeStoreError
    from tools import async_delegation_worker as worker

    _saved()
    db = SessionDB(tmp_path / "state.db")
    with ledger._transaction() as conn:
        conn.execute("UPDATE async_delegations SET owner_execution_id='worker-own'")
    before = _row()
    calls = []

    def apply(operation, payload):
        calls.append((operation, payload))
        assert operation == "delegation.read"
        try:
            with db._read_ctx() as conn:
                return _read(db, conn, "parent-before", {"delegation_id": "deleg_recover" if target == "foreign" else target},
                             execution_id="other-worker" if target == "foreign" else "worker-own")
        except RuntimeStoreError as exc:
            raise runtime_session_store.WorkerPersistenceError(str(exc)) from exc

    def forbidden():
        pytest.fail("worker inspect opened local ledger")

    try:
        monkeypatch.setattr(runtime_session_store, "_worker_process", True)
        monkeypatch.setattr(worker, "_store", SimpleNamespace(scope={"pid": os.getpid()}, _apply=apply))
        monkeypatch.setattr(ledger, "_connect", forbidden)
        out = _inspect(SimpleNamespace(session_id="parent-before"), target)
        if target == "deleg_recover":
            assert out["goal"] == "recover this goal"
            assert out["results"][0]["summary"] == "recovered answer"
            assert out["delivery_state"] == "pending"
        else:
            assert "not found" in out["error"].lower()
        assert calls == [("delegation.read", {"delegation_id": target})]
        with db._read_ctx() as conn:
            assert tuple(conn.execute("SELECT * FROM async_delegations WHERE delegation_id='deleg_recover'").fetchone()) == before
    finally:
        db.close()
