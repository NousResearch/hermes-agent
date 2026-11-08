"""Regression coverage for timed-out delegation teardown."""

from __future__ import annotations

import os
import subprocess
import threading
from types import SimpleNamespace

from tools import delegate_tool


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


def test_timeout_keeps_the_worktree_its_running_worker_still_uses(monkeypatch, tmp_path):
    """A timed-out worker that has not unwound still runs in its isolated worktree. Inspecting it then measures a
    tree that is still being written, and the prune (or archive-then-remove) deletes it under the worker, so the
    parent reads "0 commits, clean, pruned" for work that may still land. The tree must be kept and reported as
    not inspected until the worker is gone."""
    from tools.daemon_pool import DaemonThreadPoolExecutor
    from tools.terminal_tool import clear_session_cwd, record_session_cwd

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "test@test"], ["config", "user.name", "Test"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True, capture_output=True)

    child = _SlowUnwindingChild()
    parent = SimpleNamespace(
        session_id="parent-worktree-timeout",
        _current_task_id="parent-worktree-timeout",
        _active_children=[child],
        _active_children_lock=threading.Lock(),
    )
    record_session_cwd(parent._current_task_id, str(repo))
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.5)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: True)
    submit = DaemonThreadPoolExecutor.submit

    def submit_started(executor, *args, **kwargs):
        future = submit(executor, *args, **kwargs)
        assert child.started.wait(timeout=10)
        return future

    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit_started)
    try:
        result = delegate_tool._run_single_child(
            task_index=0, goal="exercise worktree teardown on timeout", child=child, parent_agent=parent,
        )
        assert result["status"] == "timeout"
        worktree = result["worktree"]
        assert os.path.isdir(worktree["path"]), "the worktree was removed while its timed-out worker still ran"
        assert worktree["pruned"] is False
        assert worktree["inspection_failed"] is True
    finally:
        child.allow_finish.set()
        clear_session_cwd(parent._current_task_id)
    assert child.finished.wait(timeout=10)
    assert child.closed.wait(timeout=10)
