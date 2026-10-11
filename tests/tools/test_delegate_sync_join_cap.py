"""Sync delegation joins must be bounded by a wall clock, and the bound must NOT re-hang on the join.

Two distinct failure shapes are covered, both from #109749:

1. ``_run_children_parallel`` / the single-child path: a child that never resolves. The cap must
   fabricate a ``timeout`` entry so the parent turn continues.
2. The re-hang trap: once we abandon a wedged worker we must not then ``shutdown(wait=True)`` on it.
   A ``with DaemonThreadPoolExecutor(...)`` block joins every worker on exit, so the abandon is only
   real if the shutdown passes ``wait=False``. This is asserted directly - a test that only checked the
   returned result would pass against the broken ``wait=True`` version.
"""

import concurrent.futures
import threading
import time

import pytest

from tools import delegate_tool_dispatch as dispatch


@pytest.fixture
def no_cap(monkeypatch):
    """Default: no wall cap, i.e. today's shipped behaviour."""
    monkeypatch.setattr(dispatch, "_join_wall_cap", lambda: None)


@pytest.fixture
def tiny_cap(monkeypatch):
    """A cap small enough to trip instantly, so tests need no real sleeping."""
    monkeypatch.setattr(dispatch, "_join_wall_cap", lambda: 0.05)


def _never_resolving_child(gate: threading.Event):
    """A child that blocks on a gate nobody opens - the wedged-future shape."""

    def run_child(i, task, child):
        gate.wait(timeout=30)
        return {"task_index": i, "status": "completed", "summary": "unreachable"}

    return run_child


def _batch(children, run_child, honor_parent_interrupt=True):
    batch = dispatch._Batch(
        task_list=[{"goal": f"task {i}"} for i, _, _ in children],
        children=children,
        parent_agent=type("A", (), {"_interrupt_requested": False, "_delegate_spinner": None})(),
        creds={},
        context=None,
        top_role="worker",
        max_children=len(children),
        live_deleg_id=None,
        live_writers=[],
        live_paths=[],
        origin_wake_sid="",
        origin_ui_session_id="",
        origin_owner_transport=None,
        origin_owner_session_record=None,
        origin_session_history_delivery=False,
        overall_start=time.monotonic(),
        unit_id="unit-test",
    )
    # ``run_child`` is a real method on _Batch; swap it for the stub under test.
    batch.run_child = run_child
    return batch


def test_parallel_join_fabricates_timeout_entries_when_capped(tiny_cap):
    gate = threading.Event()
    children = [(0, {"goal": "a"}, object()), (1, {"goal": "b"}, object())]
    batch = _batch(children, _never_resolving_child(gate))

    results: list = []
    started = time.monotonic()
    dispatch._run_children_parallel(batch, results, honor_parent_interrupt=True)
    elapsed = time.monotonic() - started

    assert elapsed < 10, "the capped join returned far too slowly to be bounded"
    assert len(results) == 2
    assert all(r["status"] == "timeout" for r in results), results
    assert {r["task_index"] for r in results} == {0, 1}


def test_capped_parallel_join_does_not_join_the_wedged_worker(tiny_cap):
    """The abandon must be real: shutdown must not wait on workers that never finish.

    ``_run_children_parallel`` returns while the worker is still parked on the gate. If the shutdown
    passed ``wait=True`` this call itself would block for the full 30s gate timeout - so a return
    well inside that window is the assertion.
    """
    gate = threading.Event()
    children = [(0, {"goal": "a"}, object()), (1, {"goal": "b"}, object())]
    batch = _batch(children, _never_resolving_child(gate))

    results: list = []
    started = time.monotonic()
    dispatch._run_children_parallel(batch, results, honor_parent_interrupt=True)
    elapsed = time.monotonic() - started

    assert elapsed < 10, (
        "capped join blocked on shutdown(wait=True) - the wedged worker was joined instead of abandoned"
    )
    try:
        assert gate.is_set() is False
    finally:
        gate.set()  # let the daemon workers unwind so the test process can exit


def test_single_child_call_is_capped_and_fabricates_a_timeout(tiny_cap):
    """The one-child path has no join loop, so the cap must be applied around the call itself."""
    gate = threading.Event()
    child = object()
    batch = _batch([(0, {"goal": "solo"}, child)], _never_resolving_child(gate))

    started = time.monotonic()
    payload = dispatch._execute_and_aggregate(batch)
    elapsed = time.monotonic() - started

    assert elapsed < 10, "the capped single-child call returned far too slowly to be bounded"
    entries = payload.get("results") or []
    assert len(entries) == 1, payload
    assert entries[0]["status"] == "timeout", entries[0]
    gate.set()  # let the abandoned daemon worker unwind so the test process can exit


def test_no_cap_preserves_the_shipped_join_path(no_cap):
    """With the setting off, behaviour is unchanged: a resolvable child still returns its own result."""
    children = [(0, {"goal": "a"}, object())]

    def run_child(i, task, child):
        return {"task_index": i, "status": "completed", "summary": "done"}

    batch = _batch(children, run_child)
    results: list = []
    dispatch._run_children_parallel(batch, results, honor_parent_interrupt=True)

    assert [r["status"] for r in results] == ["completed"]


def test_join_wall_cap_is_opt_in_and_floored(monkeypatch):
    """Off by default; a set value is floored so a silly-small config cannot kill healthy subagents."""
    monkeypatch.setattr(dispatch, "_JOIN_CAP_FLOOR", 60.0)

    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: None, raising=False)
    assert dispatch._join_wall_cap() is None

    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: 0, raising=False)
    assert dispatch._join_wall_cap() is None, "0 means disabled, not 'cap at 0s'"

    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: 5, raising=False)
    assert dispatch._join_wall_cap() == 60.0, "a tiny configured value must be raised to the floor"

    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: 7200, raising=False)
    assert dispatch._join_wall_cap() == 7200.0
