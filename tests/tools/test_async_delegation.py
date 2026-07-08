"""Tests for async (background) delegation — tools/async_delegation.py.

Covers the dispatch handle, non-blocking behavior, completion-event delivery
onto the shared process_registry.completion_queue, the rich re-injection block
formatting, capacity rejection, and crash handling.
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# scripts/run_tests.sh intentionally scrubs HOME for hermetic subprocesses.
# process_registry resolves Hermes home at import time, so provide a module
# collection fallback before importing Hermes modules; per-test state is still
# isolated by the autouse fixture below.
_TEST_HOME_FALLBACK = Path(tempfile.gettempdir()) / f"hermes-test-home-{os.getpid()}"
os.environ.setdefault("HERMES_HOME", str(_TEST_HOME_FALLBACK))
os.environ.setdefault("MESSAGING_CWD", str(_TEST_HOME_FALLBACK))

from tools import async_delegation as ad
from tools.process_registry import process_registry, format_process_notification


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    # Some tests release daemon workers right before teardown. Give any worker
    # that already captured its record a moment to enqueue, then drain so the
    # next test cannot see a late completion from a prior dispatch.
    time.sleep(0.05)
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


def test_completed_batch_event_recovers_from_durable_checkpoint():
    def runner():
        return {
            "results": [
                {
                    "task_index": 0,
                    "status": "completed",
                    "summary": "durable result",
                    "api_calls": 1,
                    "duration_seconds": 0.1,
                    "child_session_id": "child-session-1",
                }
            ],
            "total_duration_seconds": 0.1,
        }

    res = ad.dispatch_async_delegation_batch(
        goals=["recover me"],
        context="ctx",
        toolsets=None,
        role="leaf",
        model="m",
        session_key="agent:main:telegram:dm:12345",
        runner=runner,
        max_async_children=3,
        parent_session_id="parent-session",
        parent_turn_id="turn-1",
        child_session_ids=["child-session-1"],
    )
    assert res["status"] == "dispatched"
    evt = _drain_one()
    assert evt is not None
    assert evt["results"][0]["summary"] == "durable result"

    # Simulate process/queue loss after the child finished but before the
    # parent turn consumed the completion. The durable record must requeue it.
    ad._reset_for_tests(clear_persistent=False)
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    assert ad.recover_pending_delegations(origin="test restart") == 1
    recovered = _drain_one()
    assert recovered is not None
    assert recovered["delegation_id"] == res["delegation_id"]
    assert recovered["results"][0]["summary"] == "durable result"
    assert recovered["child_session_ids"] == ["child-session-1"]

    assert ad.mark_delivered(res["delegation_id"]) is True
    ad._reset_for_tests(clear_persistent=False)
    assert ad.recover_pending_delegations(origin="after delivered") == 0


def test_recovered_event_retries_until_mark_delivered_same_process():
    def runner():
        return {
            "results": [
                {
                    "task_index": 0,
                    "status": "completed",
                    "summary": "retryable durable result",
                }
            ],
            "total_duration_seconds": 0.1,
        }

    res = ad.dispatch_async_delegation_batch(
        goals=["retry me"],
        context="ctx",
        toolsets=None,
        role="leaf",
        model="m",
        session_key="agent:main:telegram:dm:12345",
        runner=runner,
        max_async_children=3,
        parent_session_id="parent-session",
        parent_turn_id="turn-retry",
        child_session_ids=["child-retry"],
    )
    assert res["status"] == "dispatched"
    assert _drain_one() is not None

    # Simulate a restart, recover the durable result, then simulate a delivery
    # path that consumes/drops the queue item before mark_delivered(). A second
    # same-process recovery must requeue it; the old _recovered_event_ids guard
    # returned 0 here and lost the result until process restart.
    ad._reset_for_tests(clear_persistent=False)
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    assert ad.recover_pending_delegations(origin="first recover") == 1
    assert ad.recover_pending_delegations(origin="already queued") == 0
    assert process_registry.completion_queue.qsize() == 1

    dropped = process_registry.completion_queue.get_nowait()
    assert dropped["delegation_id"] == res["delegation_id"]

    assert ad.recover_pending_delegations(origin="retry after dropped delivery") == 1
    retried = _drain_one()
    assert retried is not None
    assert retried["delegation_id"] == res["delegation_id"]
    assert retried["results"][0]["summary"] == "retryable durable result"

    assert ad.mark_delivered(res["delegation_id"]) is True
    assert ad.recover_pending_delegations(origin="after delivered") == 0


def test_process_registry_drain_skips_already_delivered_async_event():
    def runner():
        return {
            "results": [
                {"task_index": 0, "status": "completed", "summary": "delivered"}
            ],
            "total_duration_seconds": 0.1,
        }

    res = ad.dispatch_async_delegation_batch(
        goals=["deliver once"],
        context="ctx",
        toolsets=None,
        role="leaf",
        model="m",
        session_key="",
        runner=runner,
        max_async_children=3,
    )
    assert res["status"] == "dispatched"
    evt = _drain_one()
    assert evt is not None
    assert ad.mark_delivered(res["delegation_id"]) is True

    # A stale duplicate can remain queued if recovery raced with the delivery
    # turn. CLI/TUI post-turn drains must skip it after the durable delivered
    # bit is set, rather than injecting the same subagent result twice.
    process_registry.completion_queue.put(evt)
    assert process_registry.drain_notifications() == []


def test_running_checkpoint_recovers_as_lost_with_packet(tmp_path):
    gate = threading.Event()

    def runner():
        gate.wait(timeout=10)
        return {
            "results": [
                {"task_index": 0, "status": "completed", "summary": "late"}
            ],
            "total_duration_seconds": 0.1,
        }

    try:
        res = ad.dispatch_async_delegation_batch(
            goals=["slow a", "slow b"],
            context="ctx",
            toolsets=["terminal"],
            role="leaf",
            model="m",
            session_key="agent:main:telegram:dm:12345",
            runner=runner,
            max_async_children=3,
            parent_session_id="parent-session",
            parent_turn_id="turn-2",
            child_session_ids=["child-a", "child-b"],
        )
        assert res["status"] == "dispatched"
        assert ad.active_count() == 1

        # Simulate a hard parent process restart: the daemon worker is no longer
        # adoptable, but its checkpoint survives.
        ad._reset_for_tests(clear_persistent=False)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()

        assert ad.recover_pending_delegations(origin="test restart") == 1
        evt = _drain_one()
        assert evt is not None
        assert evt["delegation_id"] == res["delegation_id"]
        assert evt["status"] == "lost"
        assert evt["child_session_ids"] == ["child-a", "child-b"]
        assert "silently dropping" in evt["error"]
        packet_path = Path(evt["recovery_packet_path"])
        assert packet_path.exists()
        text = format_process_notification(evt)
        assert "Recovery packet" in text
        assert "child-a" in text
    finally:
        gate.set()


def test_running_checkpoint_owned_by_live_sibling_is_not_marked_lost():
    """A sibling Hermes process sharing HERMES_HOME may own a running delegation."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    delegation_id = "deleg_livebeef"
    try:
        ad._persist_record({
            "delegation_id": delegation_id,
            "goal": "still running elsewhere",
            "goals": ["still running elsewhere"],
            "context": "ctx",
            "toolsets": None,
            "role": "leaf",
            "model": "m",
            "session_key": "agent:main:telegram:dm:12345",
            "status": "running",
            "dispatched_at": time.time(),
            "completed_at": None,
            "delivered": False,
            "owner_pid": proc.pid,
            "owner_start_time": ad._get_process_start_time(proc.pid),
            "owner_hermes_home": os.environ["HERMES_HOME"],
            "owner_session_key": "agent:main:telegram:dm:12345",
        })

        assert ad.recover_pending_delegations(origin="sibling poller") == 0
        assert process_registry.completion_queue.empty()
        state = ad._read_persistent_state_unlocked()
        assert state["records"][delegation_id]["status"] == "running"
        assert "result_event" not in state["records"][delegation_id]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_drain_notifications_checks_delivered_state_in_explicit_home(tmp_path):
    """TUI/Desktop callers can dedupe against a session's profile_home."""
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    delegation_id = "deleg_profile1"
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "session-key",
        "goal": "done in profile",
        "status": "completed",
        "summary": "already delivered",
        "dispatched_at": 1000.0,
        "completed_at": 1001.0,
    }
    ad._persist_record(
        {
            "delegation_id": delegation_id,
            "goal": "done in profile",
            "status": "completed",
            "dispatched_at": 1000.0,
            "completed_at": 1001.0,
            "delivered": True,
            "delivered_at": 1002.0,
            "result_event": evt,
        },
        hermes_home=profile_home,
    )

    process_registry.completion_queue.put(evt)
    assert process_registry.drain_notifications(hermes_home=profile_home) == []


def test_persistence_failure_rejects_background_handle(monkeypatch):
    def boom(_record):
        raise OSError("disk full")

    monkeypatch.setattr(ad, "_persist_record", boom)
    res = ad.dispatch_async_delegation_batch(
        goals=["a"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="",
        runner=lambda: {"results": []},
        max_async_children=3,
    )
    assert res["status"] == "rejected"
    assert "Failed to persist" in res["error"]
    assert ad.active_count() == 0
    assert process_registry.completion_queue.empty()


def test_async_formatter_bounds_huge_summaries(tmp_path):
    full_path = tmp_path / "full-summary.txt"
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_abcdef12",
        "session_key": "",
        "goal": "huge",
        "goals": ["huge"],
        "status": "completed",
        "is_batch": True,
        "results": [
            {
                "task_index": 0,
                "status": "completed",
                "summary": "x" * 80_000,
                "summary_full_path": str(full_path),
                "child_session_id": "child-huge",
            }
        ],
        "dispatched_at": 100.0,
        "completed_at": 101.0,
    }
    text = format_process_notification(evt)
    assert len(text) < 62_000
    assert "ASYNC SUMMARY TRUNCATED" in text
    assert str(full_path) in text
    assert "child-huge" in text
    assert ad.extract_delegation_ids_from_text(text) == ["deleg_abcdef12"]


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
        goal="the real task", context="ctx",
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


def test_delegate_task_background_batch_runs_as_one_unit(monkeypatch):
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


def test_dispatch_never_forwards_model_toolsets():
    """The model has no toolsets argument — subagents always inherit the
    parent's toolsets. Even if a model smuggles a `toolsets` key into the
    tool-call args, the live dispatch path must NOT forward it to
    delegate_task (which no longer accepts it) and must not crash."""
    from unittest.mock import patch
    import run_agent

    class _FakeAgent:
        _delegate_depth = 0

    captured = {}

    def _fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", _fake_delegate):
        run_agent.AIAgent._dispatch_delegate_task(
            _FakeAgent(), {"goal": "x", "toolsets": ["web", "terminal"]}
        )
    assert "toolsets" not in captured


def test_delegate_task_background_detaches_child_from_parent(monkeypatch):
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


