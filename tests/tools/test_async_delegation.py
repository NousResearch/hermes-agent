"""Tests for async (background) delegation — tools/async_delegation.py.

Covers the dispatch handle, non-blocking behavior, completion-event delivery
onto the shared process_registry.completion_queue, the rich re-injection block
formatting, capacity rejection, and crash handling.
"""

import queue
import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry, format_process_notification


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_one(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def test_dispatch_returns_immediately_without_blocking():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "done", "api_calls": 1,
                "duration_seconds": 0.1, "model": "m"}

    t0 = time.monotonic()
    res = ad.dispatch_async_delegation(
        goal="g", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=3,
    )
    elapsed = time.monotonic() - t0

    assert res["status"] == "dispatched"
    assert res["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: dispatch returned while the runner is still
    # gated (active), so it cannot have waited on the gate. The active_count
    # check is the environment-independent proof; the generous wall-clock
    # bound is a loose sanity backstop, not the primary assertion (a loaded
    # CI runner can be slow but never anywhere near the runner's 5s gate).
    assert ad.active_count() == 1
    assert elapsed < 4.0, f"dispatch blocked {elapsed:.2f}s (gate is 5s)"
    gate.set()


def test_async_executor_workers_are_daemon_threads():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "done"}

    res = ad.dispatch_async_delegation(
        goal="daemon check", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1,
    )
    assert res["status"] == "dispatched"

    deadline = time.monotonic() + 2
    worker = None
    while time.monotonic() < deadline:
        worker = next(
            (t for t in threading.enumerate() if t.name.startswith("async-delegate")),
            None,
        )
        if worker is not None:
            break
        time.sleep(0.02)
    assert worker is not None
    assert worker.daemon is True
    gate.set()
    assert _drain_one() is not None


def test_completion_event_lands_on_shared_queue_with_session_key():
    def runner():
        return {"status": "completed", "summary": "the result",
                "api_calls": 3, "duration_seconds": 2.0, "model": "test-model"}

    res = ad.dispatch_async_delegation(
        goal="compute X", context="some context", toolsets=["web", "file"],
        role="leaf", model="test-model", session_key="agent:main:cli:dm:local",
        runner=runner, max_async_children=3,
    )
    assert res["status"] == "dispatched"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["summary"] == "the result"
    assert evt["session_key"] == "agent:main:cli:dm:local"
    assert evt["delegation_id"] == res["delegation_id"]


def test_rich_reinjection_block_is_self_contained():
    def runner():
        return {"status": "completed", "summary": "The answer is 42.",
                "api_calls": 7, "duration_seconds": 3.5, "model": "test-model"}

    ad.dispatch_async_delegation(
        goal="Compute the meaning of life",
        context="User is a philosopher. Respond tersely.",
        toolsets=["web"], role="leaf", model="test-model",
        session_key="", runner=runner, max_async_children=3,
    )
    evt = _drain_one()
    assert evt is not None
    text = format_process_notification(evt)
    assert text is not None
    for needle in [
        "ASYNC DELEGATION COMPLETE",
        "Compute the meaning of life",
        "User is a philosopher",
        "Toolsets: web",
        "The answer is 42.",
        "Status: completed",
        "API calls: 7",
    ]:
        assert needle in text, f"missing {needle!r}"


def test_dispatch_rejected_at_capacity():
    ev = threading.Event()

    def blocker():
        ev.wait(timeout=5)
        return {"status": "completed", "summary": "x"}

    for i in range(2):
        r = ad.dispatch_async_delegation(
            goal=f"task{i}", context=None, toolsets=None, role="leaf",
            model="m", session_key="", runner=blocker, max_async_children=2,
        )
        assert r["status"] == "dispatched"

    r3 = ad.dispatch_async_delegation(
        goal="task3", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=blocker, max_async_children=2,
    )
    assert r3["status"] == "rejected"
    assert "capacity reached" in r3["error"]
    ev.set()


def test_crashed_runner_produces_error_completion():
    def boom():
        raise RuntimeError("subagent exploded")

    r = ad.dispatch_async_delegation(
        goal="risky", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=boom, max_async_children=3,
    )
    assert r["status"] == "dispatched"
    evt = _drain_one()
    assert evt is not None
    assert evt["status"] == "error"
    text = format_process_notification(evt)
    assert text is not None
    assert "did not complete successfully" in text
    assert "subagent exploded" in text


def test_interrupt_all_signals_running_children():
    ev = threading.Event()
    interrupted = {"count": 0}

    def blocker():
        ev.wait(timeout=5)
        return {"status": "interrupted", "summary": None,
                "error": "cancelled"}

    def interrupt_fn():
        interrupted["count"] += 1
        ev.set()

    ad.dispatch_async_delegation(
        goal="long task", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=blocker,
        interrupt_fn=interrupt_fn, max_async_children=3,
    )
    n = ad.interrupt_all(reason="test")
    assert n == 1
    assert interrupted["count"] == 1
    # child still emits a completion event after interrupt
    evt = _drain_one()
    assert evt is not None
    assert evt["status"] == "interrupted"


def test_interrupt_by_id_only_signals_target_and_is_idempotent():
    gates = [threading.Event(), threading.Event()]
    interrupts = [0, 0]
    handles = []

    for index in range(2):
        def runner(i=index):
            gates[i].wait(timeout=5)
            return {"status": "interrupted" if interrupts[i] else "completed"}

        def interrupt_fn(i=index):
            interrupts[i] += 1
            gates[i].set()

        handles.append(ad.dispatch_async_delegation(
            goal=f"task-{index}", context=None, toolsets=None, role="leaf",
            model="m", session_key="", runner=runner,
            interrupt_fn=interrupt_fn, max_async_children=3,
        )["delegation_id"])

    result = ad.interrupt_async_delegation(handles[0], reason="test")
    assert result["status"] == "cancelling"
    assert result["delegation_id"] == handles[0]
    assert interrupts == [1, 0]

    repeated = ad.interrupt_async_delegation(handles[0], reason="test again")
    assert repeated["status"] == "cancelling"
    assert repeated["already_requested"] is True
    assert interrupts == [1, 0]

    evt = _drain_one()
    assert evt is not None
    assert evt["delegation_id"] == handles[0]
    assert evt["status"] == "interrupted"
    gates[1].set()


def test_cancel_callback_failure_rolls_back_request_and_preserves_truth():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed"}

    def interrupt_fn():
        raise RuntimeError("cannot stop")

    handle = ad.dispatch_async_delegation(
        goal="uncancellable", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, interrupt_fn=interrupt_fn,
    )["delegation_id"]

    result = ad.interrupt_async_delegation(handle)
    assert result["status"] == "error"
    snapshot = ad.get_async_delegation(handle)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["cancel_requested"] is False

    gate.set()
    event = _drain_one()
    assert event is not None
    assert event["status"] == "completed"


def test_cancel_request_does_not_force_false_interrupted_status():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed"}

    handle = ad.dispatch_async_delegation(
        goal="ignores cancel", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner,
        interrupt_fn=gate.set,
    )["delegation_id"]

    assert ad.interrupt_async_delegation(handle)["status"] == "cancelling"
    event = _drain_one()
    assert event is not None
    assert event["status"] == "completed"
    snapshot = ad.get_async_delegation(handle)
    assert snapshot is not None
    assert snapshot["status"] == "completed"


def test_snapshot_nested_values_cannot_mutate_registry():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed"}

    handle = ad.dispatch_async_delegation(
        goal="immutable", context=None, toolsets=["web"], role="leaf",
        model="m", session_key="s", runner=runner, max_async_children=1,
    )
    snap = ad.get_async_delegation(handle["delegation_id"])
    assert snap is not None
    snap["toolsets"].append("MUTATED")
    fresh = ad.get_async_delegation(handle["delegation_id"])
    assert fresh is not None
    assert fresh["toolsets"] == ["web"]
    gate.set()
    assert _drain_one() is not None


def test_finalize_is_compare_and_set_and_emits_once():
    handle = ad.dispatch_async_delegation(
        goal="once", context=None, toolsets=None, role="leaf", model="m",
        session_key="s", runner=lambda: {"status": "completed"},
        max_async_children=1,
    )
    first = _drain_one()
    assert first is not None and first["status"] == "completed"
    original = ad.get_async_delegation(handle["delegation_id"])
    assert original is not None

    ad._finalize(handle["delegation_id"], {"status": "error"}, "error")

    assert _drain_one(timeout=0.05) is None
    fresh = ad.get_async_delegation(handle["delegation_id"])
    assert fresh is not None
    assert fresh["status"] == "completed"
    assert fresh["completed_at"] == original["completed_at"]


def test_single_empty_or_unknown_result_fails_closed():
    for result in ({}, {"status": "surprise"}):
        handle = ad.dispatch_async_delegation(
            goal="invalid", context=None, toolsets=None, role="leaf", model="m",
            session_key="s", runner=lambda value=result: value,
            max_async_children=1,
        )
        event = _drain_one()
        assert event is not None
        assert event["delegation_id"] == handle["delegation_id"]
        assert event["status"] == "error"


@pytest.mark.parametrize(
    "results",
    [
        [],
        [None, {"task_index": 1, "status": "completed"}],
        [{"task_index": 0, "status": "completed"}],
        [
            {"task_index": 0, "status": "completed"},
            {"task_index": 0, "status": "completed"},
        ],
        [
            {"task_index": 0, "status": "completed"},
            {"task_index": 1, "status": "surprise"},
        ],
    ],
)
def test_batch_malformed_or_incomplete_results_fail_closed(results):
    handle = ad.dispatch_async_delegation_batch(
        goals=["a", "b"], context=None, toolsets=None, role="leaf", model="m",
        session_key="s", runner=lambda: {"results": results},
        max_async_children=1,
    )
    event = _drain_one()
    assert event is not None
    assert event["delegation_id"] == handle["delegation_id"]
    assert event["status"] == "error"
    assert all(isinstance(item, dict) for item in event["results"])


def test_formatter_tolerates_malformed_batch_items():
    from tools.process_registry import _format_async_delegation

    text = _format_async_delegation({
        "type": "async_delegation",
        "delegation_id": "deleg_bad",
        "is_batch": True,
        "goals": ["a", "b"],
        "results": [None, {"task_index": 1, "status": "completed", "summary": "ok"}],
        "status": "error",
    })
    assert "deleg_bad" in text
    assert "TASK 2/2" in text
    assert "ok" in text


def test_interrupt_by_id_reports_unknown_and_completed():
    assert ad.interrupt_async_delegation("deleg_missing")["status"] == "not_found"

    handle = ad.dispatch_async_delegation(
        goal="quick", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: {"status": "completed"},
    )["delegation_id"]
    assert _drain_one() is not None

    result = ad.interrupt_async_delegation(handle)
    assert result["status"] == "completed"
    assert result["active"] is False


def test_stalled_status_is_derived_from_child_activity_and_can_recover():
    gate = threading.Event()
    activity = {"last_activity_ts": 100.0, "last_activity_desc": "waiting", "current_tool": None}
    handle = ad.dispatch_async_delegation(
        goal="slow", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: (gate.wait(timeout=5) or {"status": "completed"}),
        activity_fn=lambda: dict(activity),
    )["delegation_id"]

    stalled = ad.get_async_delegation(
        handle, now=400.0, stalled_after_seconds=180.0
    )
    assert stalled["status"] == "stalled"
    assert stalled["last_activity_at"] == 100.0
    assert stalled["seconds_since_activity"] == 300.0

    activity["last_activity_ts"] = 390.0
    running = ad.get_async_delegation(
        handle, now=400.0, stalled_after_seconds=180.0
    )
    assert running["status"] == "running"
    assert running["seconds_since_activity"] == 10.0
    gate.set()


@pytest.mark.parametrize("workspace_path", [None, "/definitely/missing/hermes-workspace"])
def test_write_dispatch_rejects_unavailable_workspace(workspace_path):
    result = ad.dispatch_async_delegation(
        goal="unsafe write", context=None, toolsets=["file"], role="leaf",
        model="m", session_key="", runner=lambda: {"status": "completed"},
        workspace_path=workspace_path, workspace_mode="write",
    )
    assert result["status"] == "rejected"
    assert result["reason_code"] == "workspace_unavailable"
    assert ad.active_count() == 0


def test_workspace_lock_allows_read_read_and_different_workspaces(tmp_path):
    gate = threading.Event()
    (tmp_path / "repo").mkdir()
    (tmp_path / "other").mkdir()
    kwargs = dict(
        context=None, toolsets=["web"], role="leaf", model="m", session_key="",
        runner=lambda: (gate.wait(timeout=5) or {"status": "completed"}),
        max_async_children=4,
    )
    first = ad.dispatch_async_delegation(
        goal="read-a", workspace_path=str(tmp_path / "repo"), workspace_mode="read", **kwargs
    )
    second = ad.dispatch_async_delegation(
        goal="read-b", workspace_path=str(tmp_path / "repo"), workspace_mode="read", **kwargs
    )
    third = ad.dispatch_async_delegation(
        goal="write-other", workspace_path=str(tmp_path / "other"), workspace_mode="write", **kwargs
    )
    assert [first["status"], second["status"], third["status"]] == [
        "dispatched", "dispatched", "dispatched"
    ]
    gate.set()


@pytest.mark.parametrize(
    ("first_mode", "second_mode"),
    [("write", "read"), ("read", "write"), ("write", "write")],
)
def test_workspace_lock_rejects_conflicting_modes(tmp_path, first_mode, second_mode):
    gate = threading.Event()
    (tmp_path / "repo").mkdir()
    kwargs = dict(
        context=None, toolsets=None, role="leaf", model="m", session_key="",
        runner=lambda: (gate.wait(timeout=5) or {"status": "completed"}),
        max_async_children=3,
    )
    first = ad.dispatch_async_delegation(
        goal="first", workspace_path=str(tmp_path / "repo"), workspace_mode=first_mode, **kwargs
    )
    second = ad.dispatch_async_delegation(
        goal="second", workspace_path=str(tmp_path / "repo"), workspace_mode=second_mode, **kwargs
    )
    assert first["status"] == "dispatched"
    assert second["status"] == "rejected"
    assert second["reason_code"] == "workspace_locked"
    assert second["holder_delegation_id"] == first["delegation_id"]
    gate.set()


def test_workspace_lock_is_released_after_completion(tmp_path):
    (tmp_path / "repo").mkdir()
    repo = str(tmp_path / "repo")
    first = ad.dispatch_async_delegation(
        goal="first", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: {"status": "completed"},
        workspace_path=repo, workspace_mode="write",
    )
    assert first["status"] == "dispatched"
    assert _drain_one() is not None

    gate = threading.Event()
    second = ad.dispatch_async_delegation(
        goal="second", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: (gate.wait(timeout=5) or {"status": "completed"}),
        workspace_path=repo, workspace_mode="write",
    )
    assert second["status"] == "dispatched"
    gate.set()


def test_session_owner_filters_listing_status_and_cancel(tmp_path):
    gates = [threading.Event(), threading.Event()]
    interrupted = [0, 0]
    handles = []

    for index, owner in enumerate(("session:a", "session:b")):
        (tmp_path / owner).mkdir()

        def runner(i=index):
            gates[i].wait(timeout=5)
            return {"status": "interrupted" if interrupted[i] else "completed"}

        def interrupt_fn(i=index):
            interrupted[i] += 1
            gates[i].set()

        handles.append(ad.dispatch_async_delegation(
            goal=f"owner-{owner}", context="private", toolsets=["file"],
            role="leaf", model="m", session_key=owner, runner=runner,
            interrupt_fn=interrupt_fn, workspace_path=str(tmp_path / owner),
            workspace_mode="write", max_async_children=3,
        )["delegation_id"])

    owned = ad.list_async_delegations(owner_session_key="session:a")
    assert [record["delegation_id"] for record in owned] == [handles[0]]
    assert "session_key" not in owned[0]
    assert ad.get_async_delegation(
        handles[1], owner_session_key="session:a"
    ) is None

    denied = ad.interrupt_async_delegation(
        handles[1], owner_session_key="session:a"
    )
    assert denied["status"] == "not_found"
    assert interrupted == [0, 0]

    allowed = ad.interrupt_async_delegation(
        handles[0], owner_session_key="session:a"
    )
    assert allowed["status"] == "cancelling"
    assert interrupted == [1, 0]
    assert _drain_one() is not None
    gates[1].set()


def test_snapshot_is_internally_consistent_during_finalize():
    runner_gate = threading.Event()
    activity_started = threading.Event()
    activity_continue = threading.Event()

    def runner():
        runner_gate.wait(timeout=5)
        return {"status": "completed"}

    def activity_fn():
        activity_started.set()
        activity_continue.wait(timeout=5)
        return {"last_activity_ts": time.time(), "api_call_count": 1}

    handle = ad.dispatch_async_delegation(
        goal="race", context=None, toolsets=None, role="leaf", model="m",
        session_key="session:a", runner=runner, activity_fn=activity_fn,
    )["delegation_id"]
    result = {}

    thread = threading.Thread(
        target=lambda: result.update(ad.get_async_delegation(handle) or {})
    )
    thread.start()
    assert activity_started.wait(timeout=2)
    runner_gate.set()
    assert _drain_one() is not None
    activity_continue.set()
    thread.join(timeout=2)

    if result["status"] == "running":
        assert result["active"] is True
        assert result["completed_at"] is None
    else:
        assert result["status"] == "completed"
        assert result["active"] is False
        assert result["completed_at"] is not None


@pytest.mark.parametrize(("first_suffix", "second_suffix"), [("", "/sub"), ("/sub", "")])
def test_workspace_lock_rejects_ancestor_descendant_overlap(
    tmp_path, first_suffix, second_suffix
):
    gate = threading.Event()
    (tmp_path / "repo" / "sub").mkdir(parents=True)
    root = str(tmp_path / "repo")
    kwargs = dict(
        context=None, toolsets=["file"], role="leaf", model="m", session_key="",
        runner=lambda: (gate.wait(timeout=5) or {"status": "completed"}),
        workspace_mode="write", max_async_children=3,
    )
    first = ad.dispatch_async_delegation(
        goal="first", workspace_path=root + first_suffix, **kwargs
    )
    second = ad.dispatch_async_delegation(
        goal="second", workspace_path=root + second_suffix, **kwargs
    )
    assert first["status"] == "dispatched"
    assert second["status"] == "rejected"
    assert second["reason_code"] == "workspace_locked"
    gate.set()


@pytest.mark.parametrize(
    ("child_statuses", "expected"),
    [
        (["completed", "completed"], "completed"),
        (["interrupted", "interrupted"], "interrupted"),
        (["completed", "interrupted"], "interrupted"),
        (["completed", "error"], "error"),
    ],
)
def test_batch_terminal_status_preserves_child_outcomes(child_statuses, expected):
    result = ad.dispatch_async_delegation_batch(
        goals=[f"task-{i}" for i in range(len(child_statuses))],
        context=None, toolsets=None, role="leaf", model="m", session_key="",
        runner=lambda: {
            "results": [
                {"task_index": i, "status": status}
                for i, status in enumerate(child_statuses)
            ]
        },
    )
    event = _drain_one()
    assert event is not None
    assert event["delegation_id"] == result["delegation_id"]
    assert event["status"] == expected
    snapshot = ad.get_async_delegation(result["delegation_id"])
    assert snapshot is not None
    assert snapshot["status"] == expected


def test_cancel_completion_race_returns_completed_truth():
    runner_gate = threading.Event()
    callback_started = threading.Event()
    callback_release = threading.Event()
    cancel_result = {}

    def runner():
        runner_gate.wait(timeout=5)
        return {"status": "completed"}

    def interrupt_fn():
        callback_started.set()
        callback_release.wait(timeout=5)

    handle = ad.dispatch_async_delegation(
        goal="race", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, interrupt_fn=interrupt_fn,
    )["delegation_id"]

    thread = threading.Thread(
        target=lambda: cancel_result.update(ad.interrupt_async_delegation(handle))
    )
    thread.start()
    assert callback_started.wait(timeout=2)
    runner_gate.set()
    event = _drain_one()
    assert event is not None and event["status"] == "completed"
    callback_release.set()
    thread.join(timeout=2)

    assert cancel_result["status"] == "completed"
    assert cancel_result["active"] is False


def test_cancel_callback_error_after_completion_returns_completed_truth():
    runner_gate = threading.Event()
    callback_started = threading.Event()
    callback_release = threading.Event()
    cancel_result = {}

    def runner():
        runner_gate.wait(timeout=5)
        return {"status": "completed"}

    def interrupt_fn():
        callback_started.set()
        callback_release.wait(timeout=5)
        raise RuntimeError("late failure")

    handle = ad.dispatch_async_delegation(
        goal="race error", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, interrupt_fn=interrupt_fn,
    )["delegation_id"]
    thread = threading.Thread(
        target=lambda: cancel_result.update(ad.interrupt_async_delegation(handle))
    )
    thread.start()
    assert callback_started.wait(timeout=2)
    runner_gate.set()
    event = _drain_one()
    assert event is not None and event["status"] == "completed"
    callback_release.set()
    thread.join(timeout=2)

    assert cancel_result["status"] == "completed"
    assert cancel_result["active"] is False


def test_completed_records_pruned_to_cap():
    # Run more than the retention cap quickly; ensure list doesn't grow forever.
    for i in range(ad._MAX_RETAINED_COMPLETED + 10):
        ad.dispatch_async_delegation(
            goal=f"t{i}", context=None, toolsets=None, role="leaf", model="m",
            session_key="", runner=lambda: {"status": "completed", "summary": "ok"},
            max_async_children=ad._MAX_RETAINED_COMPLETED + 20,
        )
    # let workers finish
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and ad.active_count() > 0:
        time.sleep(0.05)
    assert len(ad.list_async_delegations()) <= ad._MAX_RETAINED_COMPLETED


# ---------------------------------------------------------------------------
# Integration: delegate_task(background=True) routing
# ---------------------------------------------------------------------------

def test_delegate_task_background_routes_async_and_does_not_block(monkeypatch):
    """delegate_task(background=True) returns a handle without running the
    child synchronously, and the child completes on the background thread.
    A single task is dispatched as a one-item background batch unit."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=5)  # a sync impl would hang delegate_task here
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    # monkeypatch (not `with`) so patches outlive delegate_task's return and
    # remain active while the background worker runs.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", slow_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    out = dt.delegate_task(
        goal="the real task", context="ctx", toolsets=["web"],
        background=True, parent_agent=parent,
    )

    import json
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: delegate_task returned while the child is STILL
    # blocked on the closed gate, so no completion event exists yet.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1  # one background batch unit, not finished

    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # Single task rides the batch path → carries a 1-item results list.
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 1
    assert evt["results"][0]["summary"] == "done: the real task"
    text = format_process_notification(evt)
    assert text is not None
    assert "the real task" in text


def test_delegate_task_background_batch_runs_as_one_unit(monkeypatch, tmp_path):
    """A multi-item batch with background=True dispatches the WHOLE fan-out as
    ONE background unit (one handle, one async slot). The children run in
    parallel and join; the consolidated results come back as a single
    completion event when ALL of them finish."""
    import json
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    gate = threading.Event()

    def _blocking_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=5)
        return {
            "task_index": task_index, "status": "completed",
            "summary": f"done: {goal}", "api_calls": 1,
            "duration_seconds": 0.1, "model": "m", "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }

    # Use monkeypatch (not a `with` block) so the patches stay active while the
    # background worker thread runs _execute_and_aggregate AFTER delegate_task
    # has already returned.
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", _blocking_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    out = dt.delegate_task(
        tasks=[{"goal": "a"}, {"goal": "b"}, {"goal": "c"}],
        background=True,
        parent_agent=parent,
    )

    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["count"] == 3
    assert parsed["delegation_id"].startswith("deleg_")
    assert parsed["goals"] == ["a", "b", "c"]
    # ONE background unit for the whole fan-out (not three), and the call
    # returned while all children are still blocked → chat not blocked.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1

    # Release the children; the whole batch joins and emits ONE event.
    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 3
    summaries = sorted(r["summary"] for r in evt["results"])
    assert summaries == ["done: a", "done: b", "done: c"]
    # The consolidated notification names all three tasks in one block.
    text = format_process_notification(evt)
    assert text is not None
    assert "TASK 1/3" in text and "TASK 2/3" in text and "TASK 3/3" in text
    assert "done: a" in text and "done: b" in text and "done: c" in text
    # No more events — it's a single combined completion, not N of them.
    assert _drain_one() is None


def test_model_dispatch_forces_background():
    """The MODEL-facing dispatch path forces background=True for any top-level
    delegation (single task OR batch), and keeps it off for an orchestrator
    subagent (depth > 0). Direct delegate_task() callers are unaffected (they
    keep the synchronous default)."""
    import tools.delegate_tool as dt
    from unittest.mock import MagicMock

    top = MagicMock()
    top._delegate_depth = 0
    sub = MagicMock()
    sub._delegate_depth = 1

    # Registry-fallback helper: top-level always background, regardless of
    # single vs batch; subagent never.
    assert dt._model_background_value({"goal": "x"}, top) is True
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, top
    ) is True
    assert dt._model_background_value({"tasks": [{"goal": "a"}]}, top) is True
    assert dt._model_background_value({"goal": "x"}, sub) is False
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, sub
    ) is False


def test_run_agent_dispatch_forces_background():
    """run_agent._dispatch_delegate_task — the live model path — forces
    background on for any top-level delegation (single OR batch) and off for a
    subagent."""
    from unittest.mock import patch
    import run_agent

    class _FakeAgent:
        _delegate_depth = 0

    captured = {}

    def _fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", _fake_delegate):
        agent = _FakeAgent()
        run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "x"})
        assert captured["background"] is True

        run_agent.AIAgent._dispatch_delegate_task(
            agent, {"tasks": [{"goal": "a"}, {"goal": "b"}]}
        )
        assert captured["background"] is True

        sub = _FakeAgent()
        sub._delegate_depth = 1
        run_agent.AIAgent._dispatch_delegate_task(sub, {"goal": "x"})
        assert captured["background"] is False


def test_delegate_task_background_detaches_child_from_parent(monkeypatch, tmp_path):
    """A background child must NOT remain in parent._active_children —
    otherwise parent-turn interrupts / cache evicts / session close would
    kill the detached subagent mid-run."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=5)
        return {"task_index": 0, "status": "completed", "summary": "ok"}

    def build_and_register(**kw):
        # Mirror what the real _build_child_agent does: register the child
        # for interrupt propagation.
        parent._active_children.append(fake_child)
        return fake_child

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    with patch.object(dt, "_build_child_agent", side_effect=build_and_register), \
         patch.object(dt, "_run_single_child", side_effect=slow_child), \
         patch.object(dt, "_resolve_delegation_credentials", return_value=creds):
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)

    import json
    assert json.loads(out)["status"] == "dispatched"
    # Child detached immediately at dispatch, while it is still running.
    assert fake_child not in parent._active_children
    gate.set()
    assert _drain_one() is not None


def test_concurrent_dispatch_respects_capacity():
    """Two threads racing dispatch with cap=1 must yield exactly one accept
    (capacity check and record insert are atomic under the records lock)."""
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "x"}

    results = []
    barrier = threading.Barrier(2)

    def racer():
        barrier.wait(timeout=5)
        results.append(
            ad.dispatch_async_delegation(
                goal="race", context=None, toolsets=None, role="leaf",
                model="m", session_key="", runner=blocker,
                max_async_children=1,
            )
        )

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["dispatched", "rejected"]
    gate.set()


# ---------------------------------------------------------------------------
# Gateway routing: session_key -> platform/chat_id, rich formatting, injection
# ---------------------------------------------------------------------------

def _make_async_evt(**over):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x1",
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "context": "repo /tmp/p",
        "toolsets": ["terminal"],
        "role": "leaf",
        "model": "m",
        "status": "completed",
        "summary": "Found the bug in test_foo",
        "api_calls": 4,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_gateway_enriches_routing_from_session_key():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt()
    runner._enrich_async_delegation_routing(evt)
    assert evt["platform"] == "telegram"
    assert evt["chat_id"] == "12345"
    assert evt["thread_id"] == "678"


def test_gateway_formatter_renders_async_block():
    from gateway.run import _format_gateway_process_notification

    txt = _format_gateway_process_notification(_make_async_evt())
    assert txt is not None
    assert "ASYNC DELEGATION COMPLETE" in txt
    assert "Found the bug in test_foo" in txt
    assert "Investigate flaky test" in txt


def test_gateway_watch_drain_requeues_async_without_looping():
    from gateway.run import _drain_gateway_watch_events

    q = queue.Queue()
    async_evt = _make_async_evt()
    watch_evt = {
        "type": "watch_match",
        "session_id": "proc_1",
        "command": "pytest",
        "pattern": "READY",
        "output": "READY",
    }
    q.put(async_evt)
    q.put(watch_evt)

    watch_events = _drain_gateway_watch_events(q)

    assert watch_events == [watch_evt]
    assert q.qsize() == 1
    assert q.get_nowait() == async_evt


def test_gateway_builds_routable_source_from_enriched_event():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt()
    runner._enrich_async_delegation_routing(evt)
    src = runner._build_process_event_source(evt)
    assert src is not None
    assert src.platform.value == "telegram"
    assert src.chat_id == "12345"


def test_gateway_cli_origin_event_left_unrouted():
    """An empty session_key (CLI origin) is left without routing fields."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt(session_key="")
    runner._enrich_async_delegation_routing(evt)
    assert "platform" not in evt


