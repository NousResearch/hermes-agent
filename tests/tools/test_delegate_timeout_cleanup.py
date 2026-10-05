"""Regression coverage for timed-out delegation teardown."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from agent.credential_pool import CredentialPool, PooledCredential
from tools import delegate_tool


class _SlowUnwindingChild:
    def __init__(self) -> None:
        self.tool_progress_callback = None
        self._credential_pool: object | None = None
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


class _RecordingCredentialPool(CredentialPool):
    def __init__(self) -> None:
        super().__init__(
            "test",
            [
                PooledCredential("test", "primary", "primary", "api_key", 0, "manual", "key-1"),
                PooledCredential("test", "backup", "backup", "api_key", 1, "manual", "key-2"),
            ],
        )
        self.release_calls: list[str] = []
        self.primary_released = threading.Event()

    def release_lease(self, credential_id: str) -> None:
        self.release_calls.append(credential_id)
        super().release_lease(credential_id)
        if credential_id == "primary":
            self.primary_released.set()


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


def test_timeout_keeps_credential_lease_until_worker_finishes(monkeypatch):
    child = _SlowUnwindingChild()
    pool = _RecordingCredentialPool()
    child._credential_pool = pool
    parent = SimpleNamespace(
        session_id="parent-timeout-lease-test",
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

    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit_started)
    result = delegate_tool._run_single_child(
        task_index=0,
        goal="exercise timeout lease ownership",
        child=child,
        parent_agent=parent,
    )

    assert result["status"] == "timeout"
    assert child.unwinding.wait(timeout=10)
    successor_lease = None
    try:
        assert not pool.primary_released.is_set(), (
            "the timed-out child released its credential while its worker still owned the run"
        )
        successor_lease = pool.acquire_lease()
        assert successor_lease == "backup", (
            "a successor reused the timed-out child's credential while another credential was idle"
        )
    finally:
        if successor_lease is not None:
            pool.release_lease(successor_lease)
        child.allow_finish.set()

    assert child.finished.wait(timeout=10)
    assert pool.primary_released.wait(timeout=10)
    assert pool.release_calls.count("primary") == 1


def test_timeout_worker_settling_before_cleanup_closes_before_releasing(monkeypatch):
    child = _SlowUnwindingChild()
    pool = _RecordingCredentialPool()
    child._credential_pool = pool
    parent = SimpleNamespace(
        session_id="parent-timeout-completion-race-test",
        _current_task_id=None,
        _active_children=[child],
        _active_children_lock=threading.Lock(),
    )
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.5)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)

    order: list[str] = []
    close = child.close
    release_lease = pool.release_lease

    def record_close():
        close()
        order.append("close")

    def record_release(credential_id: str):
        release_lease(credential_id)
        if credential_id == "primary":
            order.append("release")

    monkeypatch.setattr(child, "close", record_close)
    monkeypatch.setattr(pool, "release_lease", record_release)

    from tools.daemon_pool import DaemonThreadPoolExecutor

    submit = DaemonThreadPoolExecutor.submit
    future_ref = {}

    def submit_started(executor, *args, **kwargs):
        future = submit(executor, *args, **kwargs)
        future_ref["future"] = future
        assert child.started.wait(timeout=10)
        return future

    owner_thread = threading.current_thread()

    def settle_after_timeout_classification():
        if child.interrupted.is_set() and threading.current_thread() is owner_thread:
            child.allow_finish.set()
            future_ref["future"].result(timeout=10)
        return {"api_call_count": 1}

    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit_started)
    monkeypatch.setattr(child, "get_activity_summary", settle_after_timeout_classification)

    result = delegate_tool._run_single_child(
        task_index=0,
        goal="exercise timeout completion race",
        child=child,
        parent_agent=parent,
    )

    assert result["status"] == "timeout"
    assert order == ["close", "release"]
    assert pool.release_calls.count("primary") == 1
