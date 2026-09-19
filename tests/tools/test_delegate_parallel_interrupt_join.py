"""A wedged child must not hold the parent past an interrupt.

``_run_children_parallel`` polled futures with a timeout and fabricated an
'interrupted' entry for still-pending children, but the executor's ``with``
exit then joined every worker — a child stuck in an uninterruptible call
(hung socket read) hung the parent thread forever, defeating the fast path.
The interrupt path now shuts the pool down without waiting.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from tools.delegate_tool_dispatch import _Batch, _run_children_parallel


def _batch(children, parent):
    tasks = [{"goal": f"task {i}"} for i in range(len(children))]
    return _Batch(
        task_list=tasks, children=[(i, tasks[i], c) for i, c in enumerate(children)],
        parent_agent=parent, creds={}, context=None, top_role="leaf",
        max_children=len(children), live_deleg_id=None, live_writers=[], live_paths=[],
        origin_wake_sid="", origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False,
        overall_start=time.monotonic(),
    )


def test_interrupt_does_not_join_wedged_child():
    wedged_release = threading.Event()
    wedged_started = threading.Event()
    parent = SimpleNamespace(
        _interrupt_requested=False, _delegate_spinner=None, quiet_mode=True,
    )

    def run_child(i, task, child):
        if i == 0:
            wedged_started.set()
            wedged_release.wait()  # parked until test teardown
            return {"task_index": i, "status": "completed"}
        return {"task_index": i, "status": "completed", "summary": "done",
                "error": None, "api_calls": 1, "duration_seconds": 0}

    children = [SimpleNamespace(_delegate_role="leaf") for _ in range(2)]
    batch = _batch(children, parent)
    batch.run_child = run_child
    results = []
    returned = threading.Event()

    def drive():
        _run_children_parallel(batch, results, honor_parent_interrupt=True)
        returned.set()

    worker = threading.Thread(target=drive, daemon=True)
    worker.start()
    try:
        assert wedged_started.wait(5), "wedged child never started"
        assert parent._interrupt_requested is False
        parent._interrupt_requested = True
        # The poll loop ticks every 0.5s; 3s is generous and still catches a join.
        assert returned.wait(3), "interrupt path joined the wedged worker"
        assert len(results) == 2
        by_index = {e["task_index"]: e for e in results}
        assert by_index[0]["status"] == "interrupted"
        assert by_index[1]["status"] in ("completed", "interrupted")
    finally:
        wedged_release.set()
        worker.join(timeout=5)


def test_normal_completion_still_joins_cleanly():
    """No interrupt: all children finish and results sort by task_index."""
    parent = SimpleNamespace(
        _interrupt_requested=False, _delegate_spinner=None, quiet_mode=True,
    )

    def run_child(i, task, child):
        return {"task_index": i, "status": "completed", "summary": f"s{i}",
                "error": None, "api_calls": 1, "duration_seconds": 0}

    children = [SimpleNamespace(_delegate_role="leaf") for _ in range(3)]
    batch = _batch(children, parent)
    batch.run_child = run_child
    results = []
    _run_children_parallel(batch, results, honor_parent_interrupt=True)
    assert [e["task_index"] for e in results] == [0, 1, 2]
    assert all(e["status"] == "completed" for e in results)


def test_interrupt_cancels_queued_children_before_they_start():
    """More children than workers: on interrupt, cancel_futures drops the
    queued child instead of letting it start once a worker frees up."""
    wedged_release = threading.Event()
    wedged_started = threading.Event()
    parent = SimpleNamespace(
        _interrupt_requested=False, _delegate_spinner=None, quiet_mode=True,
    )
    ran = []

    def run_child(i, task, child):
        ran.append(i)
        if i == 0:
            wedged_started.set()
            wedged_release.wait()  # parked until test teardown
        return {"task_index": i, "status": "completed"}

    children = [SimpleNamespace(_delegate_role="leaf") for _ in range(2)]
    batch = _batch(children, parent)
    batch.max_children = 1  # child 1 stays queued behind the wedged child 0
    batch.run_child = run_child
    results = []
    returned = threading.Event()

    def drive():
        _run_children_parallel(batch, results, honor_parent_interrupt=True)
        returned.set()

    worker = threading.Thread(target=drive, daemon=True)
    worker.start()
    try:
        assert wedged_started.wait(5), "wedged child never started"
        parent._interrupt_requested = True
        assert returned.wait(3), "interrupt path joined the wedged worker"
        assert {e["task_index"] for e in results} == {0, 1}
        assert all(e["status"] == "interrupted" for e in results)
        assert ran == [0], "cancelled queued child must never run"
    finally:
        wedged_release.set()
        worker.join(timeout=5)
