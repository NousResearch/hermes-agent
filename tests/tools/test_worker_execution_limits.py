"""Runtime checks for profile execution ceilings beyond route parsing."""

import time
from types import SimpleNamespace

from agent.subagent_lifecycle import _iteration_limit
from tools.delegate_tool_child_run import _ChildRun


def test_profile_iteration_limit_narrows_global_budget():
    assert _iteration_limit(
        {"max_iterations": 20}, {"max_iterations": 4}, default=250,
    ) == 4
    assert _iteration_limit(
        {"max_iterations": 3}, {"max_iterations": 10}, default=250,
    ) == 3


def test_profile_timeout_stops_the_real_child_await_boundary(monkeypatch):
    class BlockingChild:
        _worker_timeout_seconds = 0.02
        _delegate_role = "leaf"
        session_id = "timeout-fixture"

        def run_conversation(self, **_kwargs):
            time.sleep(0.15)
            return {"completed": True, "final_response": "too late"}

        def hard_interrupt(self, _reason=None, **_kwargs):
            self.interrupted = True
            return True

        def interrupt(self, _reason=None):
            self.interrupted = True
            return True

        def get_activity_summary(self):
            return {"api_call_count": 1}

        def close(self):
            return None

    child = BlockingChild()
    parent = SimpleNamespace(_interrupt_requested=False)
    run = _ChildRun(
        child=child, parent_agent=parent, task_index=0, goal="bounded",
        subagent_id=None, child_progress_cb=None,
    )
    _result, error, close_deferred = run.await_child()
    assert error["status"] == "timeout"
    assert error["timeout_seconds"] == 0.02
    assert error["timeout_phase"] == "after_llm_calls"
    assert close_deferred is True
    assert child.interrupted is True
