"""Invariant tests for the durable-convergence patch in ``tools/async_delegation.py``.

Two defects, both unaddressed on upstream v2026.9.11, are pinned behaviourally
here: no source-text/regex assertions, no snapshot or enumeration-count
assertions. Each test asserts a relationship between two pieces of runtime data.

M1 - the bare ``_persist_completion(...)`` call site
---------------------------------------------------
The terminal durable write (``UPDATE async_delegations ...``) ran unguarded and
its success was ignored. When that UPDATE raised - locked db, I/O error, disk
full - the exception propagated out of ``_push_completion_event`` BEFORE
``process_registry.completion_queue.put(evt)``, and out of ``_finalize`` BEFORE
the in-memory record was moved off ``finalizing``. Consequences, in order:

  * the parent session never received the child's result (that queue is the only
    delivery path);
  * the record stayed ``finalizing``, which ``active_count()`` counts as a live
    unit, so one concurrency slot leaked for the life of the process and
    permanently ate into the effective ``delegation.max_concurrent_children``;
  * the durable row stayed ``running``, so the next process start's
    ``recover_abandoned_delegations()`` rewrote it to ``unknown`` even though
    the child had actually finished successfully.

M2 - ``_prune_completed_locked``'s "is this record finished?" predicate
----------------------------------------------------------------------
The pruner selected candidates with ``status != "running"`` while the module's
own ``_LIVE_STATES`` is ``{"running", "stalling", "finalizing"}``: the two
contradicted each other. ``stalling``/``finalizing`` are live work, yet they
were prunable - and the sort key falls back to ``dispatched_at`` when a record
has no ``completed_at`` yet, which is exactly the shape of a record that is
still stalling. Such a record therefore sorted FIRST and was popped as soon as
the retained cap overflowed. Its runner then reached ``_finalize``'s
missing-record path: a real (non-synthetic) result was silently dropped, and
the durable row was left on ``running`` for the restart path to misreport.

Base vs patch
-------------
On the pre-patch base BOTH tests fail, one assertion each, on the contracts
named in their docstrings. With the durable-convergence patch applied BOTH
pass. Raw output for both runs: ``upstream-pr/RED-GREEN.md``.

Run with the canonical runner (not bare pytest - CI uses the per-file
subprocess runner)::

    scripts/run_tests.sh tests/tools/test_async_delegation_durable_converge.py

Isolation: the autouse ``_hermetic_environment`` / ``_isolate_hermes_home``
fixtures in ``tests/conftest.py`` point ``HERMES_HOME`` at a per-test tmpdir, so
every durable row these tests write lands in that tmpdir's ``state.db`` and
nothing can touch ``~/.hermes``. No child agent, provider call, or network I/O
is involved: ``runner`` is the module's own documented injection seam, and only
the in-process state machine plus the ledger are exercised.
"""

import sqlite3
import threading
import time

import pytest

from gateway import status as gateway_status
from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_state():
    """Per-test isolation: empty registry, empty shared completion queue.

    Same shape as ``tests/tools/test_async_delegation.py``: the drain after the
    yield waits for in-flight finalizers first, so a just-released worker lands
    its event now instead of leaking it into the next test.
    """
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 3.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _records_by_id():
    """Registry contents through the public listing path, keyed by delegation id."""
    return {r["delegation_id"]: r for r in ad.list_async_delegations()}


def _await_status(delegation_id, wanted, timeout=15.0):
    """Block until the record's status is one of *wanted*; return it, else None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = _records_by_id().get(delegation_id)
        if record is not None and record.get("status") in wanted:
            return record.get("status")
        time.sleep(0.01)
    return None


def _drain_for(delegation_id, timeout=5.0):
    """Drain the shared completion queue until the event for *delegation_id* appears."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def _dispatch(goal, runner, capacity, **kwargs):
    """Dispatch one background unit through the public dispatch entry point."""
    return ad.dispatch_async_delegation(
        goal=goal, context=None, toolsets=None, role="leaf", model="test-model",
        session_key="", runner=runner, max_async_children=capacity, **kwargs)


def test_prune_never_evicts_a_live_record(monkeypatch, tmp_path):
    """M2: while the retained cap overflows, a still-live record is not a candidate.

    Contract: the set of records the pruner evicts is a subset of the records
    that were already terminal - checked continuously, because the finalizer
    prunes after every terminal unit, not only on the explicit call below. A
    ``stalling`` record with no ``completed_at`` and the OLDEST ``dispatched_at``
    (the exact shape the pre-patch predicate reached first) is still registered,
    still stalling, and the ``running`` unit beside it is still registered too.
    """
    # Ledger isolation, the tests/conftest.py way: state files resolve through
    # HERMES_HOME (here this test's tmpdir), never the developer's ~/.hermes.
    assert str(tmp_path) in str(ad._db_path())

    # Trip the stall clock quickly, but park the grace window far out of reach:
    # the record must STAY "stalling" (live, no completed_at) for the whole test.
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.15)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 3600.0)

    release = threading.Event()
    capacity = ad._MAX_RETAINED_COMPLETED + 10

    def wedged():
        # Never returns on its own; only this test's `release` lets it unwind.
        release.wait(timeout=60)
        return {"status": "completed", "summary": "returned late"}

    # Both live units are dispatched FIRST, so they are older than every terminal
    # record that follows - and the stalling one has no completed_at at all.
    running = _dispatch("live unit, still running", wedged, capacity)
    stalling = _dispatch(
        "live unit, stalling", wedged, capacity,
        interrupt_fn=lambda: None,                    # ignores the stall interrupt
        progress_fn=lambda: ((7, None), False))       # frozen progress token => wedges
    assert running["status"] == "dispatched" and stalling["status"] == "dispatched"
    running_id = running["delegation_id"]
    stalling_id = stalling["delegation_id"]
    assert _await_status(stalling_id, {"stalling"}) == "stalling", (
        "the stale monitor never marked the wedged unit as stalling")

    terminal_ids = {}
    for i in range(ad._MAX_RETAINED_COMPLETED + 5):
        handle = _dispatch(
            f"terminal unit {i}",
            lambda i=i: {"status": "completed", "summary": f"result {i}", "api_calls": 1},
            capacity)
        assert handle["status"] == "dispatched"
        status = _await_status(
            handle["delegation_id"], {"completed", "error", "stalled", "interrupted"})
        assert status is not None, "a dispatched unit never reached a terminal status"
        terminal_ids[handle["delegation_id"]] = status

        # The cap overflows inside this loop; at no point may a live record fall
        # into the eviction prefix.
        live_now = _records_by_id()
        assert stalling_id in live_now, (
            f"the stalling unit was evicted after {i + 1} terminal units: its runner's real "
            "result is now dropped and the durable row stays on 'running'")
        assert running_id in live_now, (
            f"the running unit was evicted after {i + 1} terminal units")

    # ... and the explicit call, under the registry lock exactly as the finalizer
    # makes it, must not change that verdict.
    with ad._records_lock:
        ad._prune_completed_locked()

    after = _records_by_id()
    assert stalling_id in after and after[stalling_id]["status"] == "stalling"
    assert running_id in after and after[running_id]["status"] == "running"

    evicted = ({running_id, stalling_id} | set(terminal_ids)) - set(after)
    assert evicted, "nothing was evicted: the retained cap never engaged, so this proves nothing"
    assert not (evicted & {running_id, stalling_id}), (
        "the pruner evicted a live record: " + repr(sorted(evicted & {running_id, stalling_id})))
    retained_terminal = [r for r in after.values() if r["status"] not in ad._LIVE_STATES]
    assert len(retained_terminal) <= ad._MAX_RETAINED_COMPLETED

    release.set()  # let both live units unwind so teardown is not left waiting


def test_persist_failure_neither_drops_the_result_nor_invents_unknown(monkeypatch, tmp_path):
    """M1: a raising durable completion write must not lose the outcome.

    Contracts asserted, in two parts:

    (a) the completion event still reached the shared queue, OR the record is no
        longer counted live - never "event dropped AND slot leaked";
    (b) the durable row is not rewritten to ``unknown`` by the restart recovery
        path, and - when the parent was told an outcome - the row and the
        delivered event agree on it.
    """
    assert str(tmp_path) in str(ad._db_path())

    def _raising_persist(event, result):
        # The failure M1 is about: the UPDATE itself raises.
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(ad, "_persist_completion", _raising_persist)

    handle = _dispatch(
        "child that succeeds",
        lambda: {"status": "completed", "summary": "the child really finished",
                 "api_calls": 2, "duration_seconds": 1.5},
        capacity=1)
    assert handle["status"] == "dispatched"
    delegation_id = handle["delegation_id"]

    evt = _drain_for(delegation_id, timeout=5.0)
    record = _records_by_id().get(delegation_id)
    live_status = record.get("status") if record else None
    left_live = live_status in ad._LIVE_STATES

    # (a)
    assert evt is not None or not left_live, (
        "the durable completion write failed and the unit was left on %r: the parent never "
        "received the result (queue empty) and active_count() still counts the record as a "
        "live unit, so one concurrency slot is leaked" % (live_status,))
    if evt is not None:
        assert evt["summary"] == "the child really finished"

    # (b) restart recovery must not narrate this successful child as 'unknown'.
    # Pretend the process that owned the delegation is gone: that is the only
    # condition under which recovery selects 'running'/'finalizing' rows.
    monkeypatch.setattr(gateway_status, "_pid_exists", lambda pid: False)
    ad.recover_abandoned_delegations()

    row = ad.get_durable_delegation(delegation_id)
    assert row is None or row["state"] != "unknown", (
        "restart recovery rewrote a delegation whose child had already succeeded as 'unknown'")
    if evt is not None and row is not None:
        assert row["state"] == evt["status"], (
            "the ledger and the delivered outcome disagree: row says %r, the parent was told %r"
            % (row["state"], evt["status"]))