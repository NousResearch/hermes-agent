"""Regression tests for the streaming-aware workflow-leaf timeout.

Workflow leaves run on a *streaming-aware* timeout (see
``tools.delegate_tool._run_single_child`` + ``_get_workflow_leaf_timeouts``):
instead of one fixed wall-clock cap, a leaf stays alive while it makes progress
— streamed reasoning/text deltas (which stamp a shared ``activity_ts``) OR an
advancing/active tool/iteration — and is only failed after ``idle_timeout`` of
true silence or once the absolute ``hard_timeout`` ceiling is reached.

This pins the three behaviors that matter:

* a leaf that keeps streaming past the idle window is NOT killed (the real bug:
  a big single-turn generation was being axed at the old fixed cap and retried);
* a genuinely silent/wedged leaf IS killed promptly at the idle cap, well before
  the hard ceiling;
* a leaf sitting inside a long tool call is NOT killed by the idle cap (in-tool
  leniency), the hard ceiling backstops it.

``delegate_task`` (no ``idle_timeout``) keeps its plain fixed-cap behavior — that
path is covered by ``test_delegate_subagent_timeout_diagnostic.py``.
"""
from __future__ import annotations

import threading
import time

from unittest.mock import MagicMock


class _StreamingStubChild:
    """Minimal AIAgent stand-in with controllable progress signals.

    The worker thread runs ``run_conversation`` for ``run_seconds`` while
    optionally stamping the shared ``activity_ts`` dict every ``stream_every``
    seconds (simulating streamed deltas) and/or reporting a ``current_tool``.
    """

    def __init__(
        self,
        *,
        run_seconds: float,
        activity_ts: dict | None = None,
        stream_every: float | None = None,
        current_tool: str | None = None,
        api_call_count: int = 1,
        subagent_id: str = "sa-0-streamstub",
    ):
        self._subagent_id = subagent_id
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self.model = "test/model"
        self.max_iterations = 30
        self.quiet_mode = True
        self._api_call_count = api_call_count
        self._current_tool = current_tool
        self._run_seconds = run_seconds
        self._activity_ts = activity_ts
        self._stream_every = stream_every
        self._stop = threading.Event()
        self.completed = False

    def get_activity_summary(self):
        return {
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "current_tool": self._current_tool,
            "seconds_since_activity": 0,
        }

    def run_conversation(self, user_message, task_id=None):
        deadline = time.monotonic() + self._run_seconds
        while time.monotonic() < deadline:
            if self._stop.wait(self._stream_every or 0.05):
                break
            if self._stream_every and self._activity_ts is not None:
                # Simulate a streamed delta landing.
                self._activity_ts["t"] = time.monotonic()
        self.completed = True
        return {"final_response": "done", "completed": True,
                "api_calls": self._api_call_count}

    def interrupt(self):
        self._stop.set()


def _run_leaf(child, *, idle_timeout, hard_timeout, activity_ts):
    from tools import delegate_tool

    parent = MagicMock()
    parent._touch_activity = MagicMock()
    parent._current_task_id = None
    return delegate_tool._run_single_child(
        task_index=0,
        goal="test goal",
        child=child,
        parent_agent=parent,
        idle_timeout=idle_timeout,
        hard_timeout=hard_timeout,
        activity_ts=activity_ts,
    )


def test_active_streaming_is_not_timed_out():
    """A leaf that keeps streaming past the idle window runs to completion."""
    ts = {"t": time.monotonic()}
    # Runs 2.4s, streaming every 0.15s; idle cap is 1.0s. Each silent gap
    # (0.15s) is far below the idle cap, so it must NOT time out.
    child = _StreamingStubChild(run_seconds=2.4, activity_ts=ts, stream_every=0.15)
    res = _run_leaf(child, idle_timeout=1.0, hard_timeout=30.0, activity_ts=ts)
    assert res["status"] != "timeout", res
    assert child.completed is True


def test_silent_leaf_hits_idle_timeout_before_hard_cap():
    """A wedged leaf (no streaming, no tool/iteration progress) is killed at idle."""
    ts = {"t": time.monotonic()}
    # Hangs 30s with no activity; idle cap 1.0s, hard cap 30s → idle fires first.
    child = _StreamingStubChild(run_seconds=30.0, activity_ts=ts, stream_every=None,
                                current_tool=None)
    start = time.monotonic()
    res = _run_leaf(child, idle_timeout=1.0, hard_timeout=30.0, activity_ts=ts)
    elapsed = time.monotonic() - start
    assert res["status"] == "timeout", res
    # Fired on the idle cap (~1-1.5s with a 0.5s poll floor), not the 30s ceiling.
    assert elapsed < 6.0, f"idle timeout took too long: {elapsed:.1f}s"


def test_in_tool_leaf_is_not_killed_by_idle_cap():
    """A leaf sitting inside a long tool call (current_tool set) survives idle."""
    ts = {"t": time.monotonic()}
    # No streamed deltas at all, but current_tool stays set → in-tool leniency
    # keeps it alive. Runs 2.4s with a 1.0s idle cap; must complete.
    child = _StreamingStubChild(run_seconds=2.4, activity_ts=ts, stream_every=None,
                                current_tool="webfetch")
    res = _run_leaf(child, idle_timeout=1.0, hard_timeout=30.0, activity_ts=ts)
    assert res["status"] != "timeout", res
    assert child.completed is True


def test_hard_ceiling_backstops_a_busy_tool():
    """Even an always-'busy' tool is killed once the absolute hard cap is hit."""
    ts = {"t": time.monotonic()}
    # current_tool always set (would survive idle forever), runs 30s, but the
    # hard ceiling is 1.0s → must time out near the ceiling.
    child = _StreamingStubChild(run_seconds=30.0, activity_ts=ts, stream_every=None,
                                current_tool="webfetch")
    start = time.monotonic()
    res = _run_leaf(child, idle_timeout=10.0, hard_timeout=1.0, activity_ts=ts)
    elapsed = time.monotonic() - start
    assert res["status"] == "timeout", res
    assert elapsed < 6.0, f"hard ceiling took too long: {elapsed:.1f}s"
