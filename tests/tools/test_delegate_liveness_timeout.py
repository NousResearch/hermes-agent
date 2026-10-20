"""``delegation.child_timeout_seconds`` bounds INACTIVITY, not total runtime (#116001).

A configured cap used to be a dispatch-to-death stopwatch: every child that ran longer than the budget was
abandoned even while the provider was actively serving it. Measured over 219 tasks / 75 deaths, 0 of 75 deaths
happened mid-tool — all of them were waiting on an in-flight LLM completion, so the cap converted transient
provider slowness into lost work (a nearly-finished context discarded every time).

These tests pin both directions of the fix:
- a child whose activity signals keep advancing outlives a cap far shorter than its runtime;
- a child with no progress at all is still abandoned when the cap elapses (the knob keeps its teeth).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from tools import delegate_tool

_CAP_SECONDS = 0.4

class _SlowButLiveChild:
    """A provider serving multi-minute completions: the runtime outlasts the cap, but progress never stops.

    ``advance=False`` models the opposite case — a genuinely frozen child (no completed call, no tool change,
    no activity-clock tick).
    """

    def __init__(self, *, total_seconds: float = 1.2, advance: bool = True, initial_calls: int = 0) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.session_id = "live-child"
        self.model = "test/model"
        self._total_seconds = total_seconds
        self._advance = advance
        self._calls = initial_calls
        self._activity_ts = time.time()
        self.interrupted = threading.Event()
        self.steers: list[tuple[float, str]] = []  # (monotonic time, text) of every budget warning received

    def steer(self, text: str) -> bool:
        self.steers.append((time.monotonic(), text))
        return True

    def run_conversation(self, **_kwargs):
        deadline = time.monotonic() + self._total_seconds
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if self._advance:
                self._calls += 1
                self._activity_ts = time.time()
        return {
            "final_response": "FINISHED AFTER OUTLIVING THE CAP",
            "completed": True,
            "api_calls": self._calls,
            "messages": [],
        }

    def get_activity_summary(self):
        return {
            "api_call_count": self._calls,
            "current_tool": None,
            "last_activity_ts": self._activity_ts,
            "max_iterations": 50,
        }

    def hard_interrupt(self, *_args, **_kwargs):
        self.interrupted.set()

    def close(self):
        pass

def _run(child, monkeypatch, cap=_CAP_SECONDS):
    parent = SimpleNamespace(
        session_id="parent", _current_task_id=None, _active_children=[child],
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
    )
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: cap)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    return delegate_tool._run_single_child(0, "watch the slow provider", child=child, parent_agent=parent)

def test_progressing_child_outlives_a_cap_shorter_than_its_runtime(monkeypatch):
    child = _SlowButLiveChild(total_seconds=1.2, advance=True)

    entry = _run(child, monkeypatch)

    assert entry["status"] == "completed", entry
    assert entry["summary"] == "FINISHED AFTER OUTLIVING THE CAP", entry
    # The cap really was armed and the child really did run past it: the budget reset on progress.
    assert entry["duration_seconds"] > _CAP_SECONDS, entry
    assert not child.interrupted.is_set()
    # Progress kept resetting the window, so the 80% budget warning never had cause to fire.
    assert child.steers == [], child.steers

def test_frozen_child_is_still_abandoned_when_the_cap_elapses(monkeypatch):
    """The reported death shape: calls completed earlier, then the child stops moving entirely."""
    child = _SlowButLiveChild(total_seconds=1.2, advance=False, initial_calls=49)

    entry = _run(child, monkeypatch)

    assert entry["status"] == "timeout", entry
    assert "no progress in that window" in entry["error"], entry["error"]
    assert entry["timeout_seconds"] == _CAP_SECONDS
    assert entry["last_event_age"] is not None and entry["last_event_age"] > 0.3, entry
    assert child.interrupted.is_set()


def test_default_child_timeout_is_a_bounded_inactivity_cap(monkeypatch):
    """W1: the shipped default must bound the wait even when no heartbeat tick can fire.

    ``wait_liveness_aware`` with ``child_timeout=None`` is an unbounded ``settled.wait()``;
    its only exits are the worker's done-callback and the heartbeat's stale verdict. A dead
    scheduler or a tick that never counts staleness therefore wedged the run (and its
    registry entry) for process life. The default must be a finite inactivity cap.
    """
    from tools import delegate_tool_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert delegate_tool_config.DEFAULT_CHILD_TIMEOUT is not None, (
        "an unbounded default re-opens W1: settled.wait() has no independent fallback"
    )
    assert delegate_tool_config.DEFAULT_CHILD_TIMEOUT >= 1200, delegate_tool_config.DEFAULT_CHILD_TIMEOUT
    # The production read path deep-merges DEFAULT_CONFIG into every config load, so a
    # disabling 0/None shipped default would mask the constant above entirely.
    shipped = DEFAULT_CONFIG["delegation"].get("child_timeout_seconds")
    assert shipped is None or (isinstance(shipped, (int, float)) and shipped > 0), (
        f"DEFAULT_CONFIG ships a disabling child_timeout_seconds={shipped!r}"
    )
    # ...and the fallback wiring resolves to the bounded constant when nothing overrides it.
    monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: {})
    monkeypatch.delenv("DELEGATION_CHILD_TIMEOUT_SECONDS", raising=False)
    assert delegate_tool_config._get_child_timeout() == delegate_tool_config.DEFAULT_CHILD_TIMEOUT


def test_default_cap_bounds_a_frozen_child_when_the_heartbeat_cannot_fire(monkeypatch):
    """W1 behavior: the default cap is an independent second authority — with the stale verdict
    unreachable (huge stale cycles), a genuinely frozen child still ends in timeout naming the
    cap that ended it."""
    from tools import delegate_tool_config

    child = _SlowButLiveChild(total_seconds=1.2, advance=False, initial_calls=49)
    parent = SimpleNamespace(
        session_id="parent", _current_task_id=None, _active_children=[child],
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
    )
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 10**9)  # heartbeat cannot fire
    monkeypatch.setattr(delegate_tool_config, "DEFAULT_CHILD_TIMEOUT", 0.5)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    monkeypatch.delenv("DELEGATION_CHILD_TIMEOUT_SECONDS", raising=False)

    entry = delegate_tool._run_single_child(0, "frozen without a heartbeat", child=child, parent_agent=parent)

    assert entry["status"] == "timeout", entry
    assert "no progress in that window" in entry["error"], entry["error"]
    assert entry["timeout_seconds"] == 0.5, entry
    assert child.interrupted.is_set()
