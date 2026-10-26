"""A child whose worker never returns after the heartbeat declares it stale must not hold
the parent forever (#109749: sync delegation in a -Q one-shot kept the Bot Chat lease)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from tools import delegate_tool


class _WedgedAfterFinalAnswer:
    """Frozen activity (final answer already written); the worker only unwinds when released."""

    def __init__(self) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.session_id = "wedged-child"
        self.release = threading.Event()
        self.interrupted = threading.Event()

    def run_conversation(self, **_kwargs):
        self.release.wait()
        return {"final_response": "VERDICT", "completed": True, "api_calls": 28, "messages": []}

    def get_activity_summary(self):
        return {"api_call_count": 28, "current_tool": None, "last_activity_ts": 1000.0, "max_iterations": 50}

    def hard_interrupt(self, *_args, **_kwargs):
        self.interrupted.set()

    def close(self):
        self.release.set()


def test_stale_heartbeat_ends_the_wait_without_a_configured_timeout(monkeypatch):
    child = _WedgedAfterFinalAnswer()
    parent = SimpleNamespace(
        session_id="parent", _current_task_id=None, _active_children=[child],
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
    )
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 2)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: None)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    # Safety valve so an unfixed tree fails (status "completed") instead of hanging the suite.
    valve = threading.Timer(30.0, child.release.set)
    valve.daemon = True
    valve.start()
    try:
        entry = delegate_tool._run_single_child(0, "review", child=child, parent_agent=parent)
    finally:
        valve.cancel()
        child.release.set()

    assert entry["status"] == "timeout", entry
    assert "stopped making progress" in entry["error"]
    assert child.interrupted.is_set()


def test_stale_verdict_under_a_configured_cap_reports_the_stale_threshold_not_the_cap(monkeypatch):
    """The stale verdict pre-empts the cap, so the entry must name the threshold that actually ended
    the wait — not a 3600s cap the child never reached."""
    child = _WedgedAfterFinalAnswer()
    parent = SimpleNamespace(
        session_id="parent", _current_task_id=None, _active_children=[child],
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
    )
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 2)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 3600)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    valve = threading.Timer(30.0, child.release.set)
    valve.daemon = True
    valve.start()
    try:
        entry = delegate_tool._run_single_child(0, "review", child=child, parent_agent=parent)
    finally:
        valve.cancel()
        child.release.set()

    assert entry["status"] == "timeout", entry
    assert "stopped making progress" in entry["error"] and "3600" not in entry["error"], entry["error"]
    assert entry["timeout_seconds"] == 2 * 0.01
