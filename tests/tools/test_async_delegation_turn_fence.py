"""The delegation ledger's completion write is fenced by the turn that dispatched the unit.

The durable ledger shares ``state.db`` with the session store, and a delegation reports back in
the context the dispatch copied from its turn. That completion write checks the turn's fence
inside its own transaction, like any session write (``hermes_state_fence``): a takeover (another
owner epoch claimed the profile) refuses a superseded owner's result, and the session's next claim
by the same owner does not, because a delegation unit is work designed to outlive its turn
(``agent.runtime_scope.outlive_turn``). With no turn bound (a local CLI dispatch, the owner's own
work), or a fence issued by another store, the ledger behaves exactly as before.
"""
import contextvars
import threading
import time
from contextlib import closing

import pytest

import hermes_state_runtime as rt
from agent.runtime_scope import bind_turn, claim_fence, outlive_turn
from hermes_state import SessionDB
from hermes_state_errors import TurnFenceSupersededError, classify_persistence_error
from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    # Give just-released workers a beat to finalize BEFORE draining, so their completion events
    # land now instead of leaking into the next test's queue.
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


@pytest.fixture
def claimed(tmp_path, monkeypatch):
    """A session store with a claimed turn, and the delegation ledger on the same ``state.db``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("s", "cli")
        epoch = rt.begin_runtime_epoch(db, instance_id="one")
        rt.admit_session_input(db, epoch=epoch, principal_id="u", session_id="s",
                               request_id="r1", payload={})
        row = rt.claim_session_input(db, epoch=epoch, session_id="s")
        yield db, row


def _elsewhere(fn, *args):
    """Run *fn* as another writer would: in a fresh context, outside any turn."""
    return contextvars.Context().run(fn, *args)


def _take_over(db):
    """Another owner takes the profile over: a new owner epoch."""
    _elsewhere(lambda: rt.begin_runtime_epoch(db, instance_id="two"))


def _next_claim(db, row):
    """The same owner settles *row*'s claim and claims the session's next input: a new generation."""
    def run():
        epoch, sid = row["owner_epoch"], row["target_session_id"]
        rt.settle_session_input(db, epoch=epoch, admission_id=row["admission_id"],
                                generation=row["generation"], outcome="completed", result={})
        rt.admit_session_input(db, epoch=epoch, principal_id="u", session_id=sid,
                               request_id="r2", payload={})
        claimed = rt.claim_session_input(db, epoch=epoch, session_id=sid)
        assert claimed["generation"] > row["generation"]
    _elsewhere(run)


def _dispatch(db, row, runner, **kwargs):
    """Dispatch one unit from inside the claimed turn, as the gateway does."""
    with bind_turn("s", row["admission_id"], claim_fence(row), store=db.db_path):
        return ad.dispatch_async_delegation(goal="g", context=None, toolsets=None, role="leaf",
                                            model=None, session_key="", parent_session_id="s",
                                            runner=runner, **kwargs)


def _drain_for(delegation_id, timeout=10.0):
    """Drain until the completion event for *delegation_id* appears (discarding others)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def _durable(delegation_id):
    record = ad.get_durable_delegation(delegation_id)
    assert record is not None
    return record


def test_a_completion_while_the_turn_still_holds_its_lease_lands(claimed):
    db, row = claimed
    handle = _dispatch(db, row, lambda: {"status": "completed", "summary": "on time"})
    evt = _drain_for(handle["delegation_id"])
    assert evt is not None and evt["status"] == "completed"
    record = _durable(handle["delegation_id"])
    assert record["state"] == "completed"
    assert record["result"]["summary"] == "on time"


def test_a_takeover_refuses_a_superseded_owners_result(claimed):
    db, row = claimed
    gate = threading.Event()

    def runner():
        gate.wait(10)
        return {"status": "completed", "summary": "late result"}

    handle = _dispatch(db, row, runner)
    _take_over(db)  # another owner claims the profile while the unit still runs
    gate.set()
    evt = _drain_for(handle["delegation_id"])
    assert evt is not None  # delivered in memory, as on any failed durable write
    record = _durable(handle["delegation_id"])
    assert record["state"] == "running"  # but the superseded owner saved nothing
    assert record["result"] is None


def test_the_sessions_next_claim_still_lands_the_delegations_result(claimed):
    db, row = claimed
    gate = threading.Event()

    def runner():
        gate.wait(10)
        return {"status": "completed", "summary": "late but current"}

    handle = _dispatch(db, row, runner)
    _next_claim(db, row)  # the same owner settles this turn and claims the next input
    gate.set()
    evt = _drain_for(handle["delegation_id"])
    assert evt is not None and evt["status"] == "completed"
    record = _durable(handle["delegation_id"])
    assert record["state"] == "completed"
    assert record["result"]["summary"] == "late but current"


def test_a_delegation_dispatched_outside_any_turn_writes_as_before(claimed):
    db, _row = claimed
    _take_over(db)
    handle = ad.dispatch_async_delegation(goal="g", context=None, toolsets=None, role="leaf",
                                          model=None, session_key="", parent_session_id="s",
                                          runner=lambda: {"status": "completed", "summary": "cli result"})
    evt = _drain_for(handle["delegation_id"])
    assert evt is not None and evt["status"] == "completed"
    record = _durable(handle["delegation_id"])
    assert record["state"] == "completed"
    assert record["result"]["summary"] == "cli result"


def test_a_fence_issued_by_another_store_does_not_reach_this_ledger(claimed, tmp_path):
    db, row = claimed
    other = tmp_path / "other"
    other.mkdir()
    gate = threading.Event()

    def runner():
        gate.wait(10)
        return {"status": "completed", "summary": "another store's turn"}

    with bind_turn("s", row["admission_id"], claim_fence(row), store=other / "state.db"):
        handle = ad.dispatch_async_delegation(goal="g", context=None, toolsets=None, role="leaf",
                                              model=None, session_key="", parent_session_id="s",
                                              runner=runner)
    _take_over(db)
    gate.set()
    evt = _drain_for(handle["delegation_id"])
    assert evt is not None and evt["status"] == "completed"
    record = _durable(handle["delegation_id"])
    assert record["state"] == "completed"
    assert record["result"]["summary"] == "another store's turn"


def test_a_forced_stall_outcome_survives_the_sessions_next_claim(claimed, monkeypatch):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.15)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", 0.3)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.15)
    db, row = claimed
    gate = threading.Event()
    frozen = object()

    def runner():
        gate.wait(10)
        return {"status": "completed", "summary": "never reached"}

    def progress_fn():
        return frozen, False  # a token that never changes: the unit stalls

    handle = _dispatch(db, row, runner, progress_fn=progress_fn)
    _next_claim(db, row)
    evt = _drain_for(handle["delegation_id"], timeout=15)
    gate.set()
    assert evt is not None and evt["status"] == "stalled"
    record = _durable(handle["delegation_id"])
    assert record["state"] == "stalled"


def test_the_refused_completion_write_lands_nothing_and_names_the_lease(claimed):
    db, row = claimed
    dispatch = {"delegation_id": "fenced-completion", "session_key": "", "origin_ui_session_id": "",
                "parent_session_id": "s", "dispatched_at": time.time(), "origin_session_id": ""}
    with bind_turn("s", row["admission_id"], claim_fence(row), store=db.db_path):
        ad._persist_dispatch(dispatch)  # the dispatching turn's own running row
        copied = contextvars.copy_context()
    _take_over(db)
    event = {"type": "async_delegation", "delegation_id": "fenced-completion", "status": "completed",
             "completed_at": time.time(), "session_key": "", "origin_ui_session_id": "",
             "origin_session_id": "", "goal": "g"}

    def report():
        with outlive_turn():
            ad._persist_completion(event, {"summary": "superseded owner"})

    with pytest.raises(TurnFenceSupersededError) as refused:
        copied.run(report)
    assert classify_persistence_error(refused.value) == "turn_lease"
    saved = _durable("fenced-completion")
    assert saved["state"] == "running"
    assert saved["result"] is None
