"""Admission, nested ownership and real synchronous wait producer regressions."""
from types import SimpleNamespace

import pytest

from tools import worker_roster as roster
from tools.delegate_tool_child_run import _register_child
from tools.delegate_tool_registry import _unregister_subagent
from tui_gateway import server
from tests.tui_gateway.test_worker_recovery import make_session, call


@pytest.fixture
def scope(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    home = tmp_path / "profile"
    transport = SimpleNamespace(write=lambda _: True)
    owner = make_session(home, transport)
    monkeypatch.setattr(server, "_sessions", {"ui": owner})
    monkeypatch.setattr(server, "_compute_host_supervisor", None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _: False)
    token = set_hermes_home_override(home)
    yield home, transport
    reset_hermes_home_override(token)


def test_admission_nested_join_reopen_and_terminal(scope, monkeypatch):
    from tools import delegate_tool_dispatch as dispatch
    home, transport = scope
    root = SimpleNamespace(session_id="parent")
    first, queued = roster.admit(root, [{"goal": "a" * 1000}, {"goal": "queued"}], "batch")
    rows = call(transport)["workers"]
    assert {r["status"] for r in rows} == {"queued"}
    assert max(len(r["goal"]) for r in rows) == 160
    child = SimpleNamespace(session_id="child-session", _subagent_id="outer", _delegate_depth=1)
    roster.bind(first, child)
    assert roster.claim(child)
    _register_child(child, root, "full goal", owner_session_id=None, owner_transport=None, owner_session_record=None)
    grandchild = SimpleNamespace(_subagent_id="inner", _delegate_depth=2)
    nested = roster.admit(child, [{"goal": "nested"}], "nested-batch")[0]
    roster.bind(nested, grandchild)
    _register_child(grandchild, child, "nested", owner_session_id=None, owner_transport=None, owner_session_record=None)
    try:
        seen = []
        def aggregate(batch):
            rows = call(transport)["workers"]
            seen.extend(rows)
            assert next(r for r in rows if r["subagent_id"] == "outer")["status"] == "waiting"
            assert next(r for r in rows if r["subagent_id"] == "inner")["parent_run_id"] == first["_roster"][1]
            return {"results": []}
        monkeypatch.setattr(dispatch, "_execute_and_aggregate", aggregate)
        dispatch._run_batch(SimpleNamespace(parent_agent=child, task_list=[], children=[]), False)
        assert seen
        assert next(r for r in call(transport)["workers"] if r["subagent_id"] == "outer")["status"] == "running"
        replacement = SimpleNamespace(write=lambda _: True)
        server._sessions["ui"]["transport"] = replacement
        roster.finish(nested, "failed")
        roster.finish(first, "completed")
        roster.finish(queued, "cancelled")
        assert {r["status"] for r in call(replacement)["workers"]} == {"failed", "completed", "cancelled"}
    finally:
        _unregister_subagent("outer", agent=child)
        _unregister_subagent("inner", agent=grandchild)


def test_queued_cancel_prevents_execution_and_terminal_latches(scope):
    _, transport = scope
    child = SimpleNamespace(_subagent_id="cancelled")
    record = roster.admit(SimpleNamespace(session_id="parent"), [{"goal": "cancel"}], None)[0]
    roster.bind(record, child)
    roster.cancel_queued(child)
    assert not roster.claim(child)
    roster.transition(record, "running")
    roster.finish(record, "completed")
    assert call(transport)["workers"][0]["status"] == "cancelled"


def test_terminal_only_retention_preserves_active_unknown_legacy(scope, monkeypatch):
    home, _ = scope
    parent = SimpleNamespace(session_id="parent")
    records = roster.admit(parent, [{"goal": str(i)} for i in range(5)], None)
    roster.finish(records[0], "completed")
    roster.finish(records[1], "failed")
    # Simulate unknown owner by losing its attestation, not marking it dead.
    roster._live.pop(tuple(records[2]["_roster"]))
    with roster._connect(home) as db:
        db.execute("INSERT INTO workers VALUES ('legacy','old','parent','old','completed','{}',1)")
    monkeypatch.setattr(roster, "TERMINAL_RETAIN_COUNT", 1)
    roster.admit(parent, [{"goal": "trigger"}], None)
    rows = roster.observe(home, ["parent"], roster.local_observations(home, ["parent"]))
    assert any(r["run_id"] == "legacy" for r in rows)
    assert len([r for r in rows if r["status"] in roster.TERMINAL]) == 2
    assert len([r for r in rows if r["status"] == "unknown"]) == 1
    assert len([r for r in rows if r["status"] == "queued"]) == 3


def _stub_delegate(monkeypatch):
    from tools import delegate_tool as delegate
    monkeypatch.setattr(delegate, "_resolve_delegation_credentials", lambda *a: {"model": "test", "provider": "test"})
    monkeypatch.setattr(delegate, "_load_config", lambda: {})
    monkeypatch.setattr(delegate, "_get_max_spawn_depth", lambda: 3)
    monkeypatch.setattr(delegate, "_oneshot_spawn_budget", lambda *a: None)
    monkeypatch.setattr(delegate, "_announce_batch", lambda *a: None)
    monkeypatch.setattr(delegate, "_capture_origin", lambda: ("", "", None, None, False))
    return delegate


def _assert_admitted_rows_settled(home, transport, expected):
    """Every admitted row is terminal, timestamped, prunable and no longer attested live."""
    rows = call(transport)["workers"]
    assert len(rows) == expected
    assert {r["status"] for r in rows} == {"failed"}
    assert not [key for key in roster._live if key[0] == str(home.resolve())]
    with roster._connect(home) as db:
        stored = db.execute("SELECT status, json_extract(metadata, '$.finished_at') FROM workers").fetchall()
    assert len(stored) == expected
    assert all(status == "failed" and finished for status, finished in stored)


def test_real_delegate_admission_before_build_and_build_failure(scope, monkeypatch):
    home, transport = scope
    delegate = _stub_delegate(monkeypatch)
    parent = SimpleNamespace(session_id="parent", _delegate_depth=0)
    observed = []
    def build(*args, **kwargs):
        observed.extend(call(transport)["workers"])
        assert len(observed) == 2
        assert {r["status"] for r in observed} == {"queued"}
        assert len(kwargs["worker_records"]) == 2
        return [], "construction refused"
    monkeypatch.setattr(delegate, "_build_children", build)
    result = delegate.delegate_task(tasks=[{"goal": "first specific task"}, {"goal": "second specific task"}], parent_agent=parent)
    assert "construction refused" in result
    _assert_admitted_rows_settled(home, transport, 2)


@pytest.mark.parametrize("exc", [ValueError("preflight refused"), KeyboardInterrupt()])
def test_real_delegate_build_raise_settles_admitted_rows(scope, monkeypatch, exc):
    """A raising constructor must not leave admitted rows queued (never pruned) or live."""
    home, transport = scope
    delegate = _stub_delegate(monkeypatch)
    parent = SimpleNamespace(session_id="parent", _delegate_depth=0)
    def build(*args, **kwargs):
        assert {r["status"] for r in call(transport)["workers"]} == {"queued"}
        raise exc
    monkeypatch.setattr(delegate, "_build_children", build)
    with pytest.raises(type(exc)):
        delegate.delegate_task(tasks=[{"goal": "first specific task"}, {"goal": "second specific task"}], parent_agent=parent)
    _assert_admitted_rows_settled(home, transport, 2)


def test_real_delegate_dispatch_raise_settles_only_queued_rows(scope, monkeypatch):
    """A dispatch failure settles still-queued rows without overwriting a reported verdict."""
    home, transport = scope
    delegate = _stub_delegate(monkeypatch)
    parent = SimpleNamespace(session_id="parent", _delegate_depth=0)
    admitted = []
    def build(*args, **kwargs):
        admitted.extend(kwargs["worker_records"])
        return [object(), object()], None
    def run_batch(batch, background):
        roster.finish(admitted[0], "completed")
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(delegate, "_build_children", build)
    monkeypatch.setattr(delegate, "_run_batch", run_batch)
    with pytest.raises(RuntimeError):
        delegate.delegate_task(tasks=[{"goal": "first specific task"}, {"goal": "second specific task"}], parent_agent=parent)
    statuses = sorted(r["status"] for r in call(transport)["workers"])
    assert statuses == ["completed", "failed"]
    assert not [key for key in roster._live if key[0] == str(home.resolve())]


@pytest.mark.parametrize("child_fails", [False, True])
def test_later_async_unit_inline_join_waits_and_restores(scope, monkeypatch, child_fails):
    """Observe the real aggregate's blocked child, not a mocked wait producer."""
    import json
    import threading
    import time
    from tools import delegate_tool_dispatch as dispatch
    from tools import delegation_live_log

    _, transport = scope
    parent = SimpleNamespace(session_id="nested-parent", _subagent_id="outer")
    record = roster.admit(SimpleNamespace(session_id="parent"), [{"goal": "outer"}], "outer")[0]
    roster.bind(record, parent)
    assert roster.claim(parent)
    entered, release = threading.Event(), threading.Event()
    tasks = [{"goal": "detached"}, {"goal": "inline"}]
    children = [SimpleNamespace(_subagent_id="detached"), SimpleNamespace(_subagent_id="inline")]
    batch = dispatch._Batch(tasks, [(i, t, c) for i, (t, c) in enumerate(zip(tasks, children))],
                            parent, {"model": "test"}, None, "leaf", 2, "batch", [], [],
                            "wake", "ui", None, None, True, time.monotonic())
    monkeypatch.setattr(dispatch, "_units_of", lambda b: [
        dispatch.replace(b, children=[entry]) for entry in b.children])
    monkeypatch.setattr(dispatch, "_resolve_async_wake_sid", lambda *a: "wake")
    monkeypatch.setattr(dispatch, "_detach_child", lambda *a: None)
    monkeypatch.setattr(dispatch, "_attach_child", lambda *a: None)
    attempts = []
    def dispatch_unit(unit, unit_id, *args):
        attempts.append(unit_id)
        return {"status": "dispatched", "delegation_id": unit_id} if len(attempts) == 1 else {
            "status": "rejected", "error": "schedule failure"}
    monkeypatch.setattr(dispatch, "_dispatch_unit", dispatch_unit)
    def run_child(self, i, task, child):
        assert i == 1
        entered.set()
        assert release.wait(5), "test did not release inline child"
        if child_fails:
            raise RuntimeError("inline child failed")
        return {"task_index": i, "status": "completed", "summary": "inline done"}
    monkeypatch.setattr(dispatch._Batch, "run_child", run_child)
    monkeypatch.setattr(dispatch, "_record_finished_child", lambda *a: None)
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *a: None)
    monkeypatch.setattr(delegation_live_log, "update_manifest_statuses", lambda *a: None)
    monkeypatch.setattr(dispatch, "_dispatched_payload", lambda *a: {"status": "dispatched"})
    results, errors = [], []
    def run():
        try:
            results.append(json.loads(dispatch._dispatch_background(batch)))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5), "inline child was not started"
        assert len(attempts) == 2
        assert call(transport)["workers"][0]["status"] == "waiting"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert call(transport)["workers"][0]["status"] == "running"
    assert record["_waiters"] == 0
    if child_fails:
        assert len(errors) == 1 and str(errors[0]) == "inline child failed"
    else:
        assert not errors
        assert results[0]["inline_results"][0]["summary"] == "inline done"


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_run_batch_failed_wait_entry_then_storage_restore_and_join(scope, monkeypatch, failure):
    """Production wait producer and aggregate must recover without phantom joins."""
    import json
    import time
    from tools import delegate_tool_dispatch as dispatch
    from tools import delegation_live_log

    _, transport = scope
    parent = SimpleNamespace(session_id="nested-parent", _subagent_id="outer")
    record = roster.admit(SimpleNamespace(session_id="parent"), [{"goal": "outer"}], None)[0]
    roster.bind(record, parent)
    assert roster.claim(parent)
    task = {"goal": "joined child"}
    batch = dispatch._Batch([task], [(0, task, SimpleNamespace())], parent,
                            {"model": "test"}, None, "leaf", 2, "batch", [], [],
                            "", "", None, None, False, time.monotonic())
    joins = []
    def run_child(self, *args):
        joins.append(call(transport)["workers"][0]["status"])
        assert record["_waiters"] == 1
        return {"task_index": 0, "status": "completed", "summary": "joined"}
    monkeypatch.setattr(dispatch._Batch, "run_child", run_child)
    monkeypatch.setattr(dispatch, "_record_finished_child", lambda *a: None)
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *a: None)
    monkeypatch.setattr(delegation_live_log, "update_manifest_statuses", lambda *a: None)
    with monkeypatch.context() as outage:
        outage.setattr(roster, "_connect", lambda _: (_ for _ in ()).throw(failure("entry unavailable")))
        with pytest.raises(failure, match="entry unavailable"):
            dispatch._run_batch(batch, False)
    assert not joins
    assert record.get("_waiters", 0) == 0
    assert call(transport)["workers"][0]["status"] == "running"
    result = json.loads(dispatch._run_batch(batch, False))
    assert result["results"][0]["summary"] == "joined"
    assert joins == ["waiting"]
    assert record["_waiters"] == 0
    assert call(transport)["workers"][0]["status"] == "running"


@pytest.mark.parametrize("scenario", ["nested", "entry_failure", "body_failure", "exit_failure", "exit_interrupt", "terminal"])
def test_wait_scope_nested_and_exception_accounting(scope, monkeypatch, scenario):
    _, transport = scope
    parent = SimpleNamespace(_subagent_id="outer")
    record = roster.admit(SimpleNamespace(session_id="parent"), [{"goal": "outer"}], None)[0]
    roster.bind(record, parent)
    assert roster.claim(parent)
    def status():
        return call(transport)["workers"][0]["status"]

    if scenario in {"exit_failure", "exit_interrupt"}:
        failure = OSError if scenario == "exit_failure" else KeyboardInterrupt
        with monkeypatch.context() as outage:
            with pytest.raises(failure, match="exit unavailable"):
                with roster.waiting_for_children(parent):
                    assert status() == "waiting"
                    outage.setattr(roster, "_connect", lambda _: (_ for _ in ()).throw(failure("exit unavailable")))
        assert record["_waiters"] == 0
        assert status() == "unknown"  # durable waiting is no longer an active join
        with roster.waiting_for_children(parent):
            assert status() == "waiting"
        assert status() == "running"
        return

    with roster.waiting_for_children(parent):
        assert status() == "waiting"
        if scenario == "entry_failure":
            with monkeypatch.context() as outage:
                outage.setattr(roster, "_connect", lambda _: (_ for _ in ()).throw(OSError("nested entry unavailable")))
                with pytest.raises(OSError):
                    with roster.waiting_for_children(parent):
                        pytest.fail("failed entry must not execute its body")
        elif scenario == "body_failure":
            with pytest.raises(KeyboardInterrupt, match="body failed"):
                with roster.waiting_for_children(parent):
                    assert record["_waiters"] == 2
                    raise KeyboardInterrupt("body failed")
        else:
            with roster.waiting_for_children(parent):
                assert record["_waiters"] == 2
                if scenario == "terminal":
                    roster.finish(record, "completed")
        assert record["_waiters"] == 1
        assert status() == ("completed" if scenario == "terminal" else "waiting")
    assert record["_waiters"] == 0
    assert status() == ("completed" if scenario == "terminal" else "running")


def test_storage_failure_rejects_admission(scope, monkeypatch):
    monkeypatch.setattr(roster, "_connect", lambda _: (_ for _ in ()).throw(OSError("unavailable")))
    with pytest.raises(OSError):
        roster.admit(SimpleNamespace(session_id="parent"), [{"goal": "no dispatch"}], None)
