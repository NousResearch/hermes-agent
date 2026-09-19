"""Regression coverage for timed-out delegation teardown."""

from __future__ import annotations

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
        assert self.interrupted.wait(timeout=10)
        # Model the real child turn's finally path: it still performs session
        # activity/SQLite cleanup after the parent requests interruption.
        self.unwinding.set()
        # The parent-side assertion controls when the worker may complete.
        # Keep a generous timeout only as a safeguard against a broken test
        # leaving a non-daemon worker blocked forever.
        assert self.allow_finish.wait(timeout=10)
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

    parent_returned = threading.Event()
    result_holder = {}

    def run_parent():
        try:
            result_holder["result"] = delegate_tool._run_single_child(
                task_index=0,
                goal="exercise timeout teardown",
                child=child,
                parent_agent=parent,
            )
        except BaseException as exc:
            result_holder["error"] = exc
        finally:
            parent_returned.set()

    parent_thread = threading.Thread(target=run_parent)
    parent_thread.start()
    try:
        assert child.unwinding.wait(timeout=1)
        assert parent_returned.wait(timeout=1), (
            "timed-out parent did not return while its worker was unwinding"
        )
        if "error" in result_holder:
            raise result_holder["error"]
        assert result_holder["result"]["status"] == "timeout"
        assert not child.closed.is_set(), (
            "timed-out child.close() ran before its conversation thread unwound"
        )
    finally:
        child.allow_finish.set()
    parent_thread.join(timeout=1)
    assert not parent_thread.is_alive(), "timed-out parent did not finish after worker unwound"
    assert child.finished.wait(timeout=1)
    assert child.closed.wait(timeout=1)
    assert not child.close_while_running, (
        "timed-out child.close() raced its still-running conversation thread"
    )
