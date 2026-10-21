"""Regression coverage for timed-out delegation teardown."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import tools.terminal_tool as tt
from tools import delegate_tool
from tools import delegate_tool_child_run as dcr


class _SlowUnwindingChild:
    def __init__(self) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.model = "test-model"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.started = threading.Event()
        self.interrupted = threading.Event()
        self.unwinding = threading.Event()
        self.allow_finish = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.close_while_running = False

    def run_conversation(self, **_kwargs):
        self.started.set()
        # Generous bounds: these gate on events the test sets promptly; a tight bound only
        # turns a load-starved test thread into a spurious early exit that fires close().
        assert self.interrupted.wait(timeout=30)
        # Model the real child turn's finally path: it still performs session
        # activity/SQLite cleanup after the parent requests interruption.
        self.unwinding.set()
        assert self.allow_finish.wait(timeout=30)
        self.finished.set()
        return {
            "final_response": "",
            "completed": False,
            "interrupted": True,
            "api_calls": 1,
            "messages": [],
        }

    def hard_interrupt(self, _reason=None):
        self.interrupted.set()

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def close(self):
        if not self.finished.is_set():
            self.close_while_running = True
        self.closed.set()


def test_timeout_does_not_close_child_while_worker_is_unwinding(monkeypatch):
    child = _SlowUnwindingChild()
    parent = SimpleNamespace(
        session_id="parent-timeout-test",
        _current_task_id=None,
        _active_children=[child],
        _active_children_lock=threading.Lock(),
    )
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.5)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)

    from tools.daemon_pool import DaemonThreadPoolExecutor

    submit = DaemonThreadPoolExecutor.submit

    def submit_started(executor, *args, **kwargs):
        future = submit(executor, *args, **kwargs)
        assert child.started.wait(timeout=10)
        return future

    # Timeout accounting starts only after this test's child is running.
    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit_started)
    result = delegate_tool._run_single_child(
        task_index=0,
        goal="exercise timeout teardown",
        child=child,
        parent_agent=parent,
    )

    assert result["status"] == "timeout"
    assert child.unwinding.wait(timeout=10)
    try:
        assert not child.closed.is_set(), (
            "timed-out child.close() ran before its conversation thread unwound"
        )
    finally:
        child.allow_finish.set()
    assert child.finished.wait(timeout=10)
    assert child.closed.wait(timeout=10)
    assert not child.close_while_running, (
        "timed-out child.close() raced its still-running conversation thread"
    )


# ── Child terminal-state lifecycle (clear_task_env_overrides on cleanup) ─────


class _NoopHeartbeat:
    """Stand-in for _Heartbeat in _ChildRun.cleanup (stop() is the only touch)."""

    def stop(self):
        pass


def _seed_child_run(task_index: int, subagent_id=None, parent_task_id="parent-task-lifecycle"):
    """Drive the real seed_workspace() so the child's cwd record + container alias exist."""
    parent = SimpleNamespace(_current_task_id=parent_task_id, session_id="parent-session-lifecycle")
    child = SimpleNamespace(session_id=f"child-session-lifecycle-{task_index}", close=lambda: None)
    run = dcr._ChildRun(child, parent, task_index, "goal", subagent_id, None)
    run.seed_workspace()
    assert run.child_task_id
    return run, child


def _child_has_terminal_state(child_task_id: str) -> bool:
    return bool(tt._container_aliases.get(child_task_id)) or tt.get_session_cwd(child_task_id) is not None


@pytest.fixture(autouse=True)
def _lifecycle_terminal_state():
    """Snapshot and restore the process-global terminal state the lifecycle touches.

    seed_workspace()/cleanup() mutate module-level dicts in tools.terminal_tool; restoring
    the exact snapshots keeps fixture leakage out of neighboring tests in this file.
    """
    cwd_before = dict(tt._session_cwd)
    aliases_before = dict(tt._container_aliases)
    overrides_before = dict(tt._task_env_overrides)
    yield
    tt._session_cwd.clear()
    tt._session_cwd.update(cwd_before)
    with tt._container_alias_lock:
        tt._container_aliases.clear()
        tt._container_aliases.update(aliases_before)
    tt._task_env_overrides.clear()
    tt._task_env_overrides.update(overrides_before)


def test_normal_cleanup_clears_child_terminal_state_and_preserves_parent():
    tt.record_session_cwd("parent-task-lifecycle", "/tmp/parent-lifecycle")
    tt.register_container_alias("parent-task-lifecycle", None)
    run, _child = _seed_child_run(0)

    assert _child_has_terminal_state(run.child_task_id), "seed_workspace must seed the child record"
    run.cleanup(heartbeat=_NoopHeartbeat(), child_pool=None, leased_cred_id=None, close_deferred=False)

    assert not _child_has_terminal_state(run.child_task_id), (
        "normal cleanup must clear the child's cwd record and container alias"
    )
    assert tt.get_session_cwd("parent-task-lifecycle") == "/tmp/parent-lifecycle"
    assert tt._container_aliases.get("parent-task-lifecycle") == "default"


def test_timed_out_child_keeps_terminal_state_until_worker_future_completes():
    tt.record_session_cwd("parent-task-lifecycle", "/tmp/parent-lifecycle")
    tt.register_container_alias("parent-task-lifecycle", None)
    run, _child = _seed_child_run(1)
    child_task_id = run.child_task_id

    worker_future = Future()
    worker_future.set_running_or_notify_cancel()
    dcr._defer_close_after_timeout(_child, worker_future, child_task_id)

    run.cleanup(heartbeat=_NoopHeartbeat(), child_pool=None, leased_cred_id=None, close_deferred=True)

    assert _child_has_terminal_state(child_task_id), (
        "a timed-out child keeps its terminal state while its worker future is still running"
    )

    worker_future.set_result(None)
    deadline = threading.Event()
    # The done-callback runs on the future's completing thread; poll briefly for it.
    for _ in range(100):
        if not _child_has_terminal_state(child_task_id):
            deadline.set()
            break
        deadline.wait(0.05)
    assert deadline.is_set(), (
        "the deferred done-callback must clear the child's terminal state once the worker future completes"
    )
