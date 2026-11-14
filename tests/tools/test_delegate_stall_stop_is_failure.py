"""A stuck child that honours the watchdog's stop is still a stuck child.

Both watchdogs end a stuck child with the cooperative stop a user's ``/stop`` uses. A child that unwinds
inside the grace window used to report ``interrupted`` exactly like a user stop, so its parent could not
tell a stall (a failure to recover from) from a stop someone asked for. The heartbeat's stale verdict now
reports the child as the timeout it is, and the async stall monitor stamps its stall context on the unit.
"""
import threading

from tests.tools.test_async_delegation import _drain_for, _fast_stale_monitor
from tests.tools.test_delegate_stale_heartbeat_collect import _Child, _fast_watchdog, _parent
from tools import async_delegation as ad
from tools import delegate_tool
from tools.process_registry_notifications import delegation_wakes_parent

_INTERRUPTED = {"final_response": "Operation interrupted.", "interrupted": True, "api_calls": 3,
                "messages": [{"role": "assistant", "content": "PARTIAL-WORK"}]}


class _UnwindingChild(_Child):
    """Wedged (frozen activity) until the stop arrives, then unwinds the way an interrupted loop does."""

    def run_conversation(self, **_kwargs):
        self.interrupted.wait(timeout=10)
        return dict(_INTERRUPTED)


class _StoppedChild(_Child):
    """Stopped by its parent or the user: interrupted with no stale verdict behind it."""

    def run_conversation(self, **_kwargs):
        return dict(_INTERRUPTED)


def test_child_that_unwinds_on_the_stale_verdict_is_a_timeout(monkeypatch):
    _fast_watchdog(monkeypatch, grace=5.0)
    entry = delegate_tool._run_single_child(0, "review", child=_UnwindingChild("stuck"), parent_agent=_parent())
    assert (entry["status"], entry["exit_reason"]) == ("timeout", "timeout"), entry
    assert "stopped making progress" in entry["error"], entry
    assert entry["summary"] == "PARTIAL-WORK", "what it had so far is kept"
    assert delegation_wakes_parent({"type": "async_delegation", "is_batch": True, "results": [entry]})


def test_child_stopped_without_a_stale_verdict_stays_interrupted(monkeypatch):
    _fast_watchdog(monkeypatch, grace=5.0)
    entry = delegate_tool._run_single_child(0, "review", child=_StoppedChild("stopped"), parent_agent=_parent())
    assert (entry["status"], entry["exit_reason"]) == ("interrupted", "interrupted"), entry
    assert not delegation_wakes_parent({"type": "async_delegation", "is_batch": True, "results": [entry]})


def _dispatch(interrupted: threading.Event, **kwargs):
    """A unit whose runner is wedged until stopped, then returns its children as interrupted."""
    def runner():
        interrupted.wait(timeout=10)
        return {"results": [{"task_index": 0, "status": "interrupted", "summary": "partial"}]}
    res = ad.dispatch_async_delegation_batch(
        goals=["stuck"], context=None, toolsets=None, role="leaf", model="m", session_key="", runner=runner,
        interrupt_fn=interrupted.set, max_async_children=1, **kwargs)
    assert res["status"] == "dispatched", res
    return res["delegation_id"]


def test_unit_that_unwinds_on_the_stall_monitors_stop_carries_its_stall_context(monkeypatch):
    _fast_stale_monitor(monkeypatch, grace=5.0)
    evt = _drain_for(_dispatch(threading.Event(), progress_fn=lambda: ((3, None), False)), timeout=5.0)
    assert evt is not None and [r["status"] for r in evt["results"]] == ["interrupted"]
    assert evt["stalled_after_quiet_seconds"] >= 0.15 and evt["stall_phase"] == "idle", evt
    assert delegation_wakes_parent(evt)


def test_unit_that_finished_as_the_stall_monitor_stopped_it_is_not_called_stuck():
    """A real result that lands in the grace window stays a success: no wake, no stuck-task line."""
    from tools.process_registry_notifications import format_process_notification
    evt = {"type": "async_delegation", "is_batch": True, "delegation_id": "deleg_x", "goals": ["triage"],
           "stalled_after_quiet_seconds": 600.0, "dispatched_at": 1700000000.0, "completed_at": 1700000060.0,
           "results": [{"task_index": 0, "status": "completed", "summary": "REAL-RESULT"}]}
    assert not delegation_wakes_parent(evt)
    assert "stall monitor" not in (format_process_notification(evt) or "")
    evt["results"][0].update(status="interrupted", summary="partial")
    assert "stall monitor stopped this unit after 600s" in (format_process_notification(evt) or "")


def test_unit_a_user_stopped_carries_no_stall_context(monkeypatch):
    _fast_stale_monitor(monkeypatch, idle=60, in_tool=60)
    delegation_id = _dispatch(threading.Event(), progress_fn=lambda: ((3, None), False))
    assert ad.interrupt_all(reason="test") == 1
    evt = _drain_for(delegation_id, timeout=5.0)
    assert evt is not None and [r["status"] for r in evt["results"]] == ["interrupted"]
    assert "stalled_after_quiet_seconds" not in evt
    assert not delegation_wakes_parent(evt)
