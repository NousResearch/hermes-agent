"""A child that never returns a result (timed out, or abandoned when the parent is interrupted mid-batch) has
still been billed for the calls it made: its spend reaches the parent's session cost and its result entry, like
a completed or failed child's does — including the call still in flight when it was abandoned, whose usage is
recorded only when it returns."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tools import delegate_tool
from tools.delegate_tool_child_run import _defer_child_cost_rollup
from tools.delegate_tool_dispatch import _Batch, _execute_and_aggregate
from tools.delegate_tool_results import _run_child_lifecycle


class _BilledChild:
    """Made ``calls`` paid API calls, then stopped moving (no call, tool change or activity tick); the call in
    flight costs ``late`` more when it returns after ``release``."""

    def __init__(self, calls: int, cost: float, release: threading.Event, late: float = 0.0):
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.session_id = f"child-{calls}"
        self.model = "test/model"
        self.session_estimated_cost_usd = cost
        self.session_cost_status = "estimated"
        self._calls, self._ts, self._release, self._late = calls, time.time(), release, late
        self.started = threading.Event()

    def run_conversation(self, **_kwargs):
        self.started.set()
        self._release.wait(10)
        self.session_estimated_cost_usd += self._late
        return {"final_response": "late", "completed": True, "api_calls": self._calls, "messages": []}

    def get_activity_summary(self):
        return {"api_call_count": self._calls, "current_tool": None, "last_activity_ts": self._ts, "max_iterations": 50}

    def hard_interrupt(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


def _parent(children):
    return SimpleNamespace(
        session_id="parent", _current_task_id=None, _active_children=list(children),
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
        _delegate_spinner=None, quiet_mode=True, session_estimated_cost_usd=0.0, session_cost_source="none",
        session_cost_status="unknown",
    )


def _wait_for_cost(parent, expected: float) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and parent.session_estimated_cost_usd != pytest.approx(expected):
        time.sleep(0.01)
    assert parent.session_estimated_cost_usd == pytest.approx(expected)


@pytest.fixture
def release():
    event = threading.Event()
    yield event
    event.set()


def test_timed_out_child_spend_reaches_the_parent(monkeypatch, release):
    child = _BilledChild(calls=12, cost=0.37, release=release, late=0.09)
    parent = _parent([child])
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.4)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)

    entry = _run_child_lifecycle(0, "audit", child, parent)

    assert entry["status"] == "timeout", entry
    assert entry["cost_usd"] == pytest.approx(0.37)
    assert parent.session_estimated_cost_usd == pytest.approx(0.37)
    release.set()
    _wait_for_cost(parent, 0.46)


def test_children_abandoned_on_parent_interrupt_are_billed(monkeypatch, release):
    children = [_BilledChild(calls=3, cost=0.05, release=release, late=0.01),
                _BilledChild(calls=4, cost=0.07, release=release, late=0.02)]
    parent = _parent(children)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    tasks = [{"goal": f"task {i}"} for i in range(2)]
    batch = _Batch(
        task_list=tasks, children=[(i, tasks[i], c) for i, c in enumerate(children)], parent_agent=parent,
        creds={}, context=None, top_role="leaf", max_children=2, live_deleg_id=None, live_writers=[],
        live_paths=[], origin_wake_sid="", origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False, overall_start=time.monotonic(),
    )
    threading.Timer(0.3, lambda: setattr(parent, "_interrupt_requested", True)).start()

    combined = _execute_and_aggregate(batch)

    assert [e["status"] for e in combined["results"]] == ["interrupted", "interrupted"]
    assert parent.session_estimated_cost_usd == pytest.approx(0.12)
    assert [e["api_calls"] for e in combined["results"]] == [3, 4]
    release.set()
    _wait_for_cost(parent, 0.15)


def test_overlapping_deferred_watchers_settle_each_late_dollar_once():
    child = SimpleNamespace(session_estimated_cost_usd=0.03, session_cost_status="estimated")
    parent = _parent([child])
    inner = Future()
    outer = Future()

    _defer_child_cost_rollup(child, inner, parent, 0.03)
    child.session_estimated_cost_usd = 0.05
    inner.set_result(None)

    # A second abandonment layer can register later with an earlier snapshot.
    _defer_child_cost_rollup(child, outer, parent, 0.02)
    outer.set_result(None)

    # This helper owns only the post-snapshot delta; the original 0.02 is
    # rolled by _finalize_child_results from the result entry itself.
    assert parent.session_estimated_cost_usd == pytest.approx(0.03)


def test_deferred_cost_does_not_cross_parent_session_boundary():
    child = SimpleNamespace(session_estimated_cost_usd=0.0, session_cost_status="estimated")
    parent = _parent([child])
    future = Future()

    _defer_child_cost_rollup(child, future, parent, 0.0)
    parent.session_id = "new-parent-session"
    child.session_estimated_cost_usd = 0.11
    future.set_result(None)

    assert parent.session_estimated_cost_usd == 0.0
