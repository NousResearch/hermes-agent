"""Invariant tests for the durable-convergence patch in ``tools/async_delegation.py``.

Three defects, all unaddressed on upstream ``origin/main``, are pinned
behaviourally here: no source-text/regex assertions on the module under test, no
snapshot or enumeration-count assertions. Each test asserts a relationship
between two pieces of runtime data.

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

P1#1 - "degraded" is not the same as "settled", and the caller must be told
--------------------------------------------------------------------------
The converge retry covers the *transient* write failure (M1's shape). It does
NOT cover a failure that takes every bottom-level ledger write with it -
SQLITE_FULL, a read-only ``state.db``, a lock that outlives the retry. In that
case ``_shield_unconverged_durable_row()``'s tombstone ``UPDATE`` raises, its
fallback ``DELETE`` raises, and nothing in the ledger changes: the row is still
``running``. The v2 patch discarded that verdict (``_converge_durable_completion``'s
return value was logged and dropped) and published the result as an unqualified
terminal success anyway, so the next process start's
``recover_abandoned_delegations()`` still rewrote a *delivered* success to
``unknown`` - the exact self-contradiction the PR exists to remove.

Settlement authority is now explicit. ``_converge_durable_completion`` returns
True only when the ledger can no longer contradict the in-process outcome: the
terminal write landed, or the converge retry landed, or the shield landed its
minimal terminal mark / removed the row. When it returns False - every
bottom-level ledger write raised - the result is STILL delivered (dropping a
finished child's result is the one thing that must never happen), but:

  * the event carries the additive marker ``durable_settlement == "unconfirmed"``
    ("delivered, durable settlement not confirmed - reconcile");
  * a WARNING names the delegation and the failed durable settlement;
  * ``recover_abandoned_delegations()`` reports ``indeterminate`` for that id
    instead of a bare ``unknown``.

The test for it below fails the BOTTOM-LEVEL writes - ``sqlite3`` ``UPDATE`` and
``DELETE`` statements against ``async_delegations``, reached through the
module's own ``_connect`` - so the shield UPDATE, the tombstone DELETE and the
converge retry all raise, instead of only the first monkeypatch wrapper.

Base vs patch
-------------
On the unpatched base all three tests fail, on the contracts named in their
docstrings. With the durable-convergence patch applied all three pass. Raw
output for both runs is pasted in ``REVISION-NOTES-v3.md``.

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

import logging
import re
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


class _NoLedgerWrites:
    """A ``sqlite3`` connection wrapper whose UPDATE/DELETE on the ledger raises.

    Deliberately BELOW every wrapper in ``tools/async_delegation.py``: the
    failure is injected at ``execute`` on the connection handed out by the
    module's own ``_connect``, so ``_persist_completion``, the converge retry
    ``_write_durable_terminal``, ``_shield_unconverged_durable_row``'s tombstone
    UPDATE *and* its fallback DELETE all hit it. Monkeypatching the first
    wrapper (``_persist_completion``) proves nothing about this class of
    failure - the retry lands on a healthy tmp db and converges.
    """

    _WRITE_ON_LEDGER = re.compile(r"\b(UPDATE|DELETE)\b", re.IGNORECASE)

    def __init__(self, conn, armed):
        self._conn = conn
        self._armed = armed

    def _maybe_fail(self, sql):
        if self._armed[0] and "async_delegations" in sql and self._WRITE_ON_LEDGER.search(sql):
            raise sqlite3.OperationalError("database or disk is full")

    def execute(self, sql, *args, **kwargs):
        self._maybe_fail(sql)
        return self._conn.execute(sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        self._maybe_fail(sql)
        return self._conn.executemany(sql, *args, **kwargs)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_total_write_failure_delivers_but_flags_unsettled_and_reconciles(
        monkeypatch, tmp_path, caplog):
    """P1#1: when NO bottom-level ledger write lands, be honest about it.

    The failure is injected under every wrapper (see ``_NoLedgerWrites``): the
    terminal UPDATE, the converge retry, the shield's tombstone UPDATE and its
    fallback DELETE all raise. Contracts asserted:

    (a) the parent session still receives the result - a finished child's result
        is never dropped because the ledger was unwritable;
    (b) that result is NOT presented as an unqualified terminal success: the
        event carries ``durable_settlement == "unconfirmed"`` and a warning names
        the delegation and the failed durable settlement;
    (c) the follow-up ``recover_abandoned_delegations()`` does not narrate the
        already-delivered success as a bare ``unknown``: it reports
        ``indeterminate`` and says to reconcile against the delivered event.
    """
    assert str(tmp_path) in str(ad._db_path())

    real_connect = ad._connect
    armed = [False]  # the dispatch INSERT and its prune DELETEs must still land
    monkeypatch.setattr(ad, "_connect", lambda: _NoLedgerWrites(real_connect(), armed))

    release = threading.Event()

    def gated_runner():
        # Hold the worker until the fault is armed, then succeed - the failure
        # class under test is the TERMINAL write, not the run itself.
        assert release.wait(timeout=30), "the runner was never released"
        return {"status": "completed", "summary": "the child really finished",
                "api_calls": 2, "duration_seconds": 1.5}

    handle = _dispatch("child that succeeds", gated_runner, capacity=1)
    assert handle["status"] == "dispatched"
    delegation_id = handle["delegation_id"]

    armed[0] = True
    try:
        with caplog.at_level(logging.WARNING, logger="tools.async_delegation"):
            release.set()
            evt = _drain_for(delegation_id, timeout=10.0)
    finally:
        armed[0] = False

    # (a) delivery is unconditional.
    assert evt is not None, (
        "every bottom-level ledger write failed and the result was never delivered: a finished "
        "child's result must reach the parent even when the ledger is unwritable")
    assert evt["summary"] == "the child really finished"
    assert evt["status"] == "completed"

    # (b) ... but not as an unqualified terminal success.
    assert evt.get("durable_settlement") == ad._DURABLE_SETTLEMENT_UNCONFIRMED, (
        "the result was published as a clean terminal success although nothing was settled durably")
    assert ad.is_unsettled_delivery(delegation_id), (
        "the id was not registered, so restart recovery can still invent an 'unknown' for it")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        delegation_id in message and "durable settlement failed" in message for message in warnings), (
        "no WARNING named the delegation and its failed durable settlement; got: " + repr(warnings))

    # The row is exactly what the defect is about: still selectable on restart.
    row = ad.get_durable_delegation(delegation_id)
    assert row is not None and row["state"] in ad._LIVE_STATES, (
        "the fixture did not reproduce the failure class: the ledger row is %r"
        % (row and row["state"],))

    # (c) restart recovery must not turn a delivered success into a bare 'unknown'.
    monkeypatch.setattr(gateway_status, "_pid_exists", lambda pid: False)
    recovered = ad.recover_abandoned_delegations()
    assert recovered >= 1, "restart recovery did not classify the abandoned row at all"

    row = ad.get_durable_delegation(delegation_id)
    assert row is not None
    assert row["state"] != "unknown", (
        "restart recovery narrated a delegation whose result the parent already had as 'unknown'")
    assert row["state"] == "indeterminate", (
        "expected an explicit needs-reconciliation verdict, got %r" % (row["state"],))
    assert "reconcile" in (row["result"] or {}).get("error", "").lower(), (
        "the indeterminate verdict does not say what to do about it")


def test_healthy_completion_is_not_marked_unsettled(tmp_path):
    """The marker is additive: a normal completion stays byte-for-byte the old contract."""
    assert str(tmp_path) in str(ad._db_path())

    handle = _dispatch(
        "child that succeeds",
        lambda: {"status": "completed", "summary": "clean", "api_calls": 1},
        capacity=1)
    assert handle["status"] == "dispatched"
    delegation_id = handle["delegation_id"]

    evt = _drain_for(delegation_id, timeout=5.0)
    assert evt is not None and evt["summary"] == "clean"
    assert "durable_settlement" not in evt, (
        "a healthy completion carried the unsettled marker")
    assert not ad.is_unsettled_delivery(delegation_id)
    row = ad.get_durable_delegation(delegation_id)
    assert row is not None and row["state"] == "completed"
