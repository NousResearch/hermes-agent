"""The TUI must not re-announce a completion the agent already holds.

The TUI drains with ``skip_poll_observed=False`` — it owns the decision — and then routes
each completion through ``_notif_handle_event`` / ``_notif_dispatch_completions``. A poll()
taken after the exit already handed the agent the exit code and the output tail, so turning
that completion into another user-visible notice is the stale replay the user reported: the
notice landed right after the report the agent had written about the same process.
"""

import threading
import time

import pytest

from tui_gateway import server
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification

SESSION_KEY = "agent:main:telegram:dm:123:"


def _registry(monkeypatch) -> ProcessRegistry:
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    return registry


def _drive_one_completion(monkeypatch, registry, *, poll: bool) -> list:
    """Run one completion through the real TUI router; return the dispatched texts."""
    sid = f"proc_{'polled' if poll else 'fresh'}"
    session = ProcessSession(
        id=sid, command=f"echo {sid}", task_id="t1", started_at=time.time(),
        exited=False, output_buffer="", notify_on_complete=True, session_key=SESSION_KEY,
    )
    registry._running[sid] = session
    session.mark_exited(0)
    session.output_buffer = "build ok\n"
    registry._move_to_finished(session)
    if poll:
        registry.poll(sid)

    drained = registry.drain_notifications(
        session_key=SESSION_KEY, owns_event=lambda e: True, skip_poll_observed=False)
    assert drained, "the completion reaches the TUI; the surface, not the drain, decides"

    started: list = []
    monkeypatch.setattr(
        server, "_notif_submit",
        lambda rid, sid_, session_, text, what, **kw: started.append(text))
    ui_session = {"history_lock": threading.RLock(), "running": False, "history": [],
                  "_notification_emitted": set()}
    server._notif_handle_ready(
        "sid", ui_session, [event for event, _text in drained],
        ui_session["_notification_emitted"], registry, format_process_notification, None,
        owned=True)
    return started


def test_a_polled_completion_is_not_reannounced_in_the_tui(monkeypatch):
    registry = _registry(monkeypatch)
    assert _drive_one_completion(monkeypatch, registry, poll=True) == []


def test_a_fresh_completion_is_still_announced_in_the_tui(monkeypatch):
    registry = _registry(monkeypatch)
    started = _drive_one_completion(monkeypatch, registry, poll=False)
    assert len(started) == 1
    assert "proc_fresh" in started[0]
