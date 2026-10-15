"""A child whose worker settles shortly AFTER the heartbeat's stale verdict must be
COLLECTED — its recorded result wins over a synthesized timeout (#113222).

The stale verdict ends the waiter's liveness wait. Once it has, there is a short window
where the worker (which already wrote its final answer) finishes unwinding: the wait must
poll for that real result during a grace period instead of raising immediately, so an
async batch never strands a finished child as `running`.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from tools import delegate_tool
from tools.delegate_tool_dispatch import _Batch, _execute_and_aggregate


class _Child:
    """Test double: worker wedges (frozen activity) until released, then returns a result."""

    def __init__(self, session_id: str) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.session_id = session_id
        self.release = threading.Event()
        self.interrupted = threading.Event()

    def run_conversation(self, **_kwargs):
        self.release.wait()
        return {"final_response": f"DONE-{self.session_id}", "completed": True, "api_calls": 3, "messages": []}

    def get_activity_summary(self):
        # Frozen fingerprint: the stale verdict trips at the idle threshold.
        return {"api_call_count": 3, "current_tool": None, "last_activity_ts": 1000.0, "max_iterations": 50}

    def hard_interrupt(self, *_args, **_kwargs):
        self.interrupted.set()

    def close(self):
        self.release.set()


class _CompletingChild(_Child):
    """Worker that returns immediately (a healthy sibling)."""

    def __init__(self) -> None:
        super().__init__("sibling")
        self.release.set()

    def run_conversation(self, **_kwargs):
        self.release.set()  # idempotent safety
        return super().run_conversation(**_kwargs)


def _parent():
    return SimpleNamespace(
        session_id="parent",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _touch_activity=lambda _desc: None,
        _interrupt_requested=False,
    )


def _fast_watchdog(monkeypatch, *, grace: float):
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 2)
    # raising=False: on the unfixed base the constant does not exist yet, so the
    # assertions fail on BEHAVIOR (synthesized timeout vs. collected result).
    monkeypatch.setattr(delegate_tool, "_STALE_RESULT_GRACE_SECONDS", grace, raising=False)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: None)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)


def _finish_soon(child: _Child, delay: float) -> threading.Timer:
    """Timer that releases the wedged worker shortly after the stale verdict fires."""
    timer = threading.Timer(delay, child.release.set)
    timer.daemon = True
    timer.start()
    return timer


def test_stale_verdict_collects_a_result_that_lands_within_the_grace_window(monkeypatch):
    """The wedge: verdict fires while the worker is mid-unwind; the real result must win."""
    child = _Child("grace")
    _fast_watchdog(monkeypatch, grace=1.0)
    timer = _finish_soon(child, 0.15)  # after the verdict (~0.02s), well inside the grace
    try:
        entry = delegate_tool._run_single_child(0, "review", child=child, parent_agent=_parent())
    finally:
        timer.cancel()
        child.release.set()
    assert entry["status"] == "completed", entry
    assert entry["summary"] == "DONE-grace", entry
    # The stop signal still fired: an abandoned worker must be told to unwind.
    assert child.interrupted.is_set()


def test_stale_verdict_without_a_result_in_the_grace_window_is_a_timeout(monkeypatch):
    """A worker that never lands inside the grace window gets the existing timeout entry."""
    child = _Child("wedged")
    _fast_watchdog(monkeypatch, grace=0.05)
    # Safety valve: an unfixed-forever wait must not hang the suite; a fix that never
    # times out fails on the elapsed-time assertion instead.
    valve = _finish_soon(child, 5.0)
    started = time.monotonic()
    try:
        entry = delegate_tool._run_single_child(0, "review", child=child, parent_agent=_parent())
    finally:
        valve.cancel()
        child.release.set()
    assert time.monotonic() - started < 4
    assert entry["status"] == "timeout", entry
    assert "stopped making progress" in (entry.get("error") or ""), entry


def test_async_batch_reports_when_the_last_child_settles_after_its_verdict(monkeypatch):
    """The #113222 batch shape: the finished-late child's result lands in the combined
    payload; the batch does not strand it as `running`."""
    _fast_watchdog(monkeypatch, grace=1.0)
    late = _Child("late")
    done = _CompletingChild()
    timer = _finish_soon(late, 0.15)
    try:
        batch = _Batch(
            task_list=[{"goal": "late"}, {"goal": "done"}],
            children=[(0, {"goal": "late"}, late), (1, {"goal": "done"}, done)],
            parent_agent=_parent(),
            creds={"model": "test"},
            context=None,
            top_role="leaf",
            max_children=2,
            live_deleg_id=None,
            live_writers=[None, None],
            live_paths=[],
            origin_wake_sid="",
            origin_ui_session_id="",
            origin_owner_transport=None,
            origin_owner_session_record=None,
            origin_session_history_delivery=False,
            overall_start=time.monotonic(),
        )
        combined = _execute_and_aggregate(batch, honor_parent_interrupt=False)
    finally:
        timer.cancel()
        late.release.set()
    by_index = {entry["task_index"]: entry for entry in combined["results"]}
    assert by_index[0]["status"] == "completed", by_index[0]
    assert by_index[0]["summary"] == "DONE-late", by_index[0]
    assert by_index[1]["status"] == "completed", by_index[1]
    assert by_index[1]["summary"] == "DONE-sibling", by_index[1]
