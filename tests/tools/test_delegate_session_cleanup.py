"""Admission and cancellation must release only children that never started.

Real durable reservations expose orphaned or prematurely released conversations;
event-controlled workers exercise the scheduling boundary without provider calls.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tools.delegate_session_store import DelegateSessionStore
from tools.delegate_tool_child_run import _close_child
from tools.delegate_tool_dispatch import _Batch, _run_batch, _run_children_parallel


@pytest.fixture
def batch_state(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", "cli")
    store = DelegateSessionStore(db)
    parent = SimpleNamespace(session_id="parent", _active_children=[], _interrupt_requested=False)
    tasks, children = [], []
    for i in range(3):
        sid = f"child-{i}"
        db.create_session(sid, "subagent", parent_session_id="parent", model_config={"_delegate_from": "parent"})
        store.register("parent", sid, model="test-model", provider="test-provider", assignment=f"Task {i}")
        _, token = store.acquire("parent", sid, f"Task {i}")
        child = SimpleNamespace(session_id=sid, model="test-model", _delegate_resumable=True,
                                _delegate_session_id=sid, _delegate_session_token=token,
                                _delegate_store_path=str(db.db_path), closes=0)
        def close(child=child):
            child.closes += 1
        child.close = close
        task = {"goal": f"Task {i}"}
        tasks.append(task)
        children.append((i, task, child))
        parent._active_children.append(child)
    batch = _Batch(tasks, children, parent, {}, None, "leaf", 1, "test-batch", [], [], "parent", "",
                   None, None, True, time.monotonic())
    monkeypatch.setattr("agent.delegated_review.finish_delegated_review", lambda child: None)
    monkeypatch.setattr("tools.delegate_tool_dispatch.begin_delegation_run", lambda *a, **kw: None)
    yield batch, store
    for _, _, child in children:
        if vars(child).get("_delegate_session_token"):
            _close_child(child, "fixture cleanup")
    db.close()


def test_cancelled_queued_children_release_but_active_worker_keeps_reservation(batch_state, monkeypatch):
    batch, store = batch_state
    started, finish, ended = threading.Event(), threading.Event(), threading.Event()
    runs = []

    def run_child(*, child, task_index, **kwargs):
        runs.append(task_index)
        started.set()
        batch.parent_agent._interrupt_requested = True
        try:
            assert finish.wait(20)
            return {"task_index": task_index, "status": "completed"}
        finally:
            _close_child(child, "worker cleanup")
            ended.set()

    monkeypatch.setattr("tools.delegate_tool._run_single_child", run_child)
    try:
        results = []
        _run_children_parallel(batch, results, honor_parent_interrupt=True)
        assert started.is_set()
        assert runs == [0]
        assert store.get("parent", "child-0")["status"] == "running"
        assert store.get("parent", "child-1")["status"] == "idle"
        assert store.get("parent", "child-2")["status"] == "idle"
        assert batch.children[0][2].closes == 0
    finally:
        finish.set()
        assert ended.wait(20)
    assert store.get("parent", "child-0")["status"] == "idle"
    assert all(child.closes == 1 for _, _, child in batch.children)


@pytest.mark.parametrize("failure", ["later_unit", "after_submit", "queued"])
def test_admission_failure_never_frees_an_active_child_or_runs_abandoned_queue(batch_state, monkeypatch, failure):
    batch, store = batch_state
    started, finish = threading.Event(), threading.Event()
    workers, queued, runs = [], [], []
    monkeypatch.setattr("tools.delegate_tool_config._get_independent_completions", lambda: True)
    monkeypatch.setattr("tools.delegate_tool._get_max_async_children", lambda: 3)
    monkeypatch.setattr("tools.delegate_tool_dispatch._resolve_async_wake_sid", lambda *a: "parent")
    monkeypatch.setattr("tools.delegate_tool_dispatch._resolve_async_session_key", lambda *a: ("parent", ""))

    def execute(unit, **kwargs):
        child = unit.children[0][2]
        runs.append(child.session_id)
        started.set()
        try:
            assert finish.wait(20)
            return {"results": [{"task_index": unit.children[0][0], "status": "completed"}]}
        finally:
            _close_child(child, "worker cleanup")

    def dispatch(*, runner, delegation_id, **kwargs):
        if workers:
            raise RuntimeError("Later admission failed")
        if failure == "queued":
            queued.append(runner)
            raise RuntimeError("Post-submit bookkeeping failed before queued worker started")
        worker = threading.Thread(target=runner)
        workers.append(worker)
        worker.start()
        assert started.wait(20)
        if failure == "after_submit":
            raise RuntimeError("Post-submit bookkeeping failed")
        return {"status": "dispatched", "delegation_id": delegation_id}

    monkeypatch.setattr("tools.delegate_tool_dispatch._execute_and_aggregate", execute)
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", dispatch)
    try:
        with pytest.raises(RuntimeError, match="failed"):
            _run_batch(batch, background=True)
        assert store.get("parent", "child-1")["status"] == "idle"
        assert store.get("parent", "child-2")["status"] == "idle"
        if failure == "queued":
            assert store.get("parent", "child-0")["status"] == "idle"
            queued[0]()  # a submitted runner starting late must not touch its closed child
            assert runs == []
        else:
            assert store.get("parent", "child-0")["status"] == "running"
            assert batch.children[0][2].closes == 0
    finally:
        finish.set()
        for worker in workers:
            worker.join(timeout=20)
            assert not worker.is_alive()
    assert all(store.get("parent", f"child-{i}")["status"] == "idle" for i in range(3))
    assert all(child.closes == 1 for _, _, child in batch.children)
