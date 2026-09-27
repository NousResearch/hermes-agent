"""The opt-in lease must not replace normal inactivity or stale-worker behavior."""
import threading
import time
from types import SimpleNamespace

import pytest

from tools import delegate_tool
from tools.delegate_tool_child_run import _ChildRun


class ActiveChild:
    def __init__(self):
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.ticks = 0

    def run_conversation(self, **kwargs):
        try:
            end = time.monotonic() + 0.4
            while time.monotonic() < end and not self.stopped.wait(0.01):
                self.ticks += 1
            return {"final_response": "done", "completed": True}
        finally:
            self.finished.set()

    def get_activity_summary(self):
        return {"api_call_count": 1, "last_activity_ts": self.ticks}

    def hard_interrupt(self, *args, **kwargs):
        self.stopped.set()

    def close(self):
        self.stopped.set()


@pytest.mark.parametrize("reviewed", [False, True])
def test_activity_renews_only_the_ordinary_inactivity_budget(monkeypatch, reviewed):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"reviewed_timeout": reviewed})
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.15)
    monkeypatch.setattr("tools.delegate_tool_child_run._LIVENESS_POLL_SECONDS", 0.02)
    child = ActiveChild()
    result, error, _ = _ChildRun(child, None, 0, "test", None, None).await_child()
    assert child.finished.wait(2)
    assert child.ticks > 1
    if reviewed:
        assert result is None and error["status"] == "timeout"
        assert "Parent-reviewed deadline expired" in error["error"]
        assert child._delegate_reviewed_deadline.snapshot()["renewals"] == 0
    else:
        assert error is None and result["final_response"] == "done"
        assert not hasattr(child, "_delegate_reviewed_deadline")


def test_heartbeat_stale_verdict_preempts_a_reviewed_window(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"reviewed_timeout": True})
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 60)
    child = ActiveChild()
    heartbeat = SimpleNamespace(settled=threading.Event(), stale_threshold_seconds=0.05)
    heartbeat.settled.set()
    result, error, _ = _ChildRun(child, None, 0, "test", None, None, heartbeat=heartbeat).await_child()
    assert child.finished.wait(2)
    assert result is None and error["status"] == "timeout"
    assert error["timeout_seconds"] == heartbeat.stale_threshold_seconds
    assert "heartbeat stale threshold" in error["error"]
    assert child._delegate_reviewed_deadline.snapshot()["closed"]
