"""Inactivity-window behavior of ``delegation.child_timeout_seconds`` — direct coverage for the design spec's
acceptance criteria (issue #116001).

``test_delegate_liveness_timeout.py`` pins the end-to-end shape (healthy child outlives the cap, frozen child
dies, 80% warning fires). These tests go one level down: they exercise the real symbols directly
(``_ChildRun.wait_liveness_aware``, ``_BUDGET_WARNING_FRACTION``, ``_child_activity_fingerprint``), pin the
warning's re-arm semantics (once per window, re-armed by progress, never double-firing inside one window),
the mid-tool window suspension (spec §5 option b: a window that expires while a tool runs defers to the
heartbeat's in-tool stale authority instead of killing the child), and the config parse/backward-compatibility
contract (unset → no cap, <= 0 → no cap, < 30 → floored, env fallback, garbage → default + warning).
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from tools import delegate_tool
from tools import delegate_tool_child_run as dcr

_POLL = 0.05  # tests shrink the production 5s liveness poll so windows elapse in fractions of a second


# ── fake children ─────────────────────────────────────────────────────────────

class _FakeChild:
    """Activity-summary stub: frozen or advancing, optionally mid-tool. Records budget warnings via steer()."""

    def __init__(self, *, calls: int = 3, tool: str | None = None, advance: bool = False) -> None:
        self._calls = calls
        self._tool = tool
        self._ts = time.time()
        self._advance = advance
        self.steers: list[tuple[float, str]] = []  # (monotonic, text)
        self.advanced_at: float | None = None  # monotonic time of the deliberate progress injection

    def get_activity_summary(self):
        return {
            "api_call_count": self._calls, "current_tool": self._tool,
            "last_activity_ts": self._ts, "max_iterations": 50,
        }

    def bump(self) -> None:
        """Inject one unit of progress (a completed call + activity-clock tick)."""
        self._calls += 1
        self._ts = time.time()
        self.advanced_at = time.monotonic()

    def steer(self, text: str) -> bool:
        self.steers.append((time.monotonic(), text))
        return True


class _WaitHarness:
    """Real ``_ChildRun.wait_liveness_aware`` unbound to a minimal ``self``: the method reads only ``self.child``."""

    wait_liveness_aware = dcr._ChildRun.wait_liveness_aware

    def __init__(self, child) -> None:
        self.child = child


def _run_wait_in_thread(child, timeout: float) -> tuple[threading.Thread, threading.Event]:
    settled = threading.Event()
    harness = _WaitHarness(child)
    t = threading.Thread(
        target=lambda: harness.wait_liveness_aware(settled, None, timeout), daemon=True,
    )
    t.start()
    return t, settled


# ── _child_activity_fingerprint (spec criterion 6: real symbols) ─────────────

def test_fingerprint_is_call_tool_clock_tuple():
    child = _FakeChild(calls=7, tool="terminal")
    assert dcr._child_activity_fingerprint(child) == (
        7, "terminal", child.get_activity_summary()["last_activity_ts"],
    )


def test_fingerprint_of_child_without_a_clock_is_the_none_tuple():
    """No summary / failing summary degrades to (None, None, None): the cap falls back to wall-clock behavior."""
    assert dcr._child_activity_fingerprint(SimpleNamespace(get_activity_summary=lambda: {})) == (None, None, None)

    def _boom():
        raise RuntimeError("no activity summary")

    assert dcr._child_activity_fingerprint(SimpleNamespace(get_activity_summary=_boom)) == (None, None, None)


# ── wait_liveness_aware: both directions of the window (spec criterion 1) ─────

def test_idle_child_expires_the_window_without_settled(monkeypatch):
    monkeypatch.setattr(dcr, "_LIVENESS_POLL_SECONDS", _POLL)
    child = _FakeChild()
    t, settled = _run_wait_in_thread(child, timeout=0.3)
    t.join(timeout=2.0)
    assert not t.is_alive(), "idle child should expire the window"
    assert not settled.is_set(), "window expiry must not depend on the settled event"


def test_advancing_fingerprint_never_expires_the_window(monkeypatch):
    """The #116001 shape: progress inside the window keeps resetting it, so the wait outlives many caps."""
    monkeypatch.setattr(dcr, "_LIVENESS_POLL_SECONDS", _POLL)
    child = _FakeChild()
    t, settled = _run_wait_in_thread(child, timeout=0.3)
    deadline = time.monotonic() + 1.0  # > 3x the cap
    while time.monotonic() < deadline:
        time.sleep(0.1)
        child.bump()
    assert t.is_alive(), "a child whose fingerprint keeps moving must never expire the window"
    assert not child.steers, "a progressing child must never see the budget warning"
    settled.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


# ── escalation: warn once per window, re-arm on progress (spec criterion 3) ──

def test_warning_fires_once_per_window_and_rearms_after_progress(monkeypatch):
    monkeypatch.setattr(dcr, "_LIVENESS_POLL_SECONDS", _POLL)
    assert dcr._BUDGET_WARNING_FRACTION == 0.8  # knob deliberately rejected; fraction is a constant
    cap = 0.6
    child = _FakeChild()
    t, settled = _run_wait_in_thread(child, timeout=cap)
    started = time.monotonic()

    def _progress_then_freeze():
        time.sleep(0.55)  # first warning (~0.48) already fired; reset the window just before expiry
        child.bump()

    threading.Thread(target=_progress_then_freeze, daemon=True).start()
    t.join(timeout=3.0)
    settled.set()

    assert not t.is_alive()
    assert len(child.steers) == 2, f"expected exactly one warning per window, got {child.steers}"
    warn1_at, warn2_at = child.steers[0][0] - started, child.steers[1][0]
    # First warning inside the first window, at ~80% of it, not at expiry.
    assert 0.8 * cap - 0.05 <= warn1_at < cap + 0.05, warn1_at
    # Re-armed: the second warning belongs to the SECOND window (measured from the progress injection).
    assert child.advanced_at is not None
    assert warn2_at >= child.advanced_at + 0.8 * cap - 0.05, (warn2_at, child.advanced_at)
    # Never double-fires inside one window: the two warnings are a full window apart.
    assert warn2_at - warn1_at >= 0.8 * cap - 0.05, (warn1_at, warn2_at)
    assert "[delegation budget warning]" in child.steers[0][1]


# ── spec §5 option b: window expiry suspended while a tool runs ──────────────

def test_window_expiry_is_suspended_while_a_tool_is_running(monkeypatch):
    """A cap below a silent tool's duration must NOT kill the mid-tool child: the window suspends and the
    heartbeat's in-tool stale threshold becomes the sole authority (the two must not overlap)."""
    monkeypatch.setattr(dcr, "_LIVENESS_POLL_SECONDS", _POLL)
    child = _FakeChild(tool="terminal_run")
    t, settled = _run_wait_in_thread(child, timeout=0.2)
    time.sleep(1.0)  # 5x the cap, fingerprint fully frozen
    assert t.is_alive(), "window expiry must be suspended while current_tool is set"
    settled.set()  # what the heartbeat's in-tool stale verdict does in production
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_in_tool_frozen_child_is_killed_by_the_stale_threshold_not_the_cap(monkeypatch):
    """End-to-end: cap (0.1s) expires mid-tool and suspends; the heartbeat's in-tool stale verdict (0.25s)
    is what actually ends the wait, and the timeout entry must name THAT cause."""
    child = SimpleNamespace(
        tool_progress_callback=None, _credential_pool=None, _delegate_saved_tool_names=[],
        _delegate_role="leaf", _delegate_depth=1, _subagent_id=None, session_id="mid-tool-child",
        model="test/model", interrupted=threading.Event(), _released=threading.Event(),
        _summary={"api_call_count": 7, "current_tool": "terminal_run",
                  "last_activity_ts": time.time(), "max_iterations": 50},
    )

    def run_conversation(**_kwargs):
        child._released.wait(30)
        return {"final_response": "", "completed": False, "api_calls": 7, "messages": []}

    child.run_conversation = run_conversation
    child.get_activity_summary = lambda: dict(child._summary)
    child.hard_interrupt = lambda *a, **k: child.interrupted.set()
    child.close = lambda: child._released.set()

    parent = SimpleNamespace(
        session_id="parent", _current_task_id=None, _active_children=[child],
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
    )
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.1)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IN_TOOL", 25)  # stale verdict value: 0.25s
    valve = threading.Timer(30.0, child.close)
    valve.daemon = True
    valve.start()
    try:
        entry = delegate_tool._run_single_child(0, "run the long tool", child=child, parent_agent=parent)
    finally:
        valve.cancel()
        child.close()

    assert entry["status"] == "timeout", entry
    # The stale threshold fired, not the 0.1s cap — without suspension the cap would have killed at 0.1s.
    assert entry["timeout_seconds"] == pytest.approx(0.25), entry
    assert "heartbeat stale threshold" in entry["error"], entry["error"]
    assert "no progress in that window" not in entry["error"], entry["error"]  # cap-expiry phrasing must NOT appear
    assert entry["duration_seconds"] > 0.1, entry
    # Spec criterion 5: the entry carries api_calls and last_event_age.
    assert entry["api_calls"] == 7, entry
    assert entry["last_event_age"] is not None, entry
    assert child.interrupted.is_set()


# ── config backward compatibility (spec criterion 2) ─────────────────────────

class TestChildTimeoutConfigBackwardCompatibility:
    @pytest.fixture(autouse=True)
    def _isolated_config(self, monkeypatch):
        self.cfg: dict = {}
        monkeypatch.setattr(delegate_tool, "_load_config", lambda: dict(self.cfg))
        monkeypatch.delenv("DELEGATION_CHILD_TIMEOUT_SECONDS", raising=False)
        self.warnings: list[str] = []
        from tools import delegate_tool_config as dtc
        monkeypatch.setattr(
            dtc.logger, "warning",
            lambda msg, *args: self.warnings.append(msg % args if args else msg),
        )
        from tools.delegate_tool_config import _get_child_timeout
        self.get = _get_child_timeout

    def test_unset_means_no_cap(self):
        assert self.get() is None

    def test_zero_and_negative_disable_the_cap(self):
        self.cfg["child_timeout_seconds"] = 0
        assert self.get() is None
        self.cfg["child_timeout_seconds"] = -5
        assert self.get() is None

    def test_sub_30_value_is_floored_at_30(self):
        self.cfg["child_timeout_seconds"] = 10
        assert self.get() == 30.0

    def test_positive_value_passes_through(self):
        self.cfg["child_timeout_seconds"] = 4800
        assert self.get() == 4800.0

    def test_env_fallback_used_when_key_absent(self):
        import os
        os.environ["DELEGATION_CHILD_TIMEOUT_SECONDS"] = "600"
        assert self.get() == 600.0

    def test_config_value_wins_over_env(self):
        import os
        os.environ["DELEGATION_CHILD_TIMEOUT_SECONDS"] = "600"
        self.cfg["child_timeout_seconds"] = 120
        assert self.get() == 120.0

    def test_garbage_value_falls_back_to_default_with_a_warning(self):
        self.cfg["child_timeout_seconds"] = "fast"
        assert self.get() is None
        assert any("is not a valid number" in w for w in self.warnings), self.warnings
