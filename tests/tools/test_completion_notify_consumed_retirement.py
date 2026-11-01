"""sys-9cloc7: the pending background-process completion notice must be RETIRED once
the agent holds the receipt (wait/log consumed, or process_manage poll() observed the
exit inline), and an unread completion must still deliver exactly once.

Before this, gateway/tui watchers checked only ``is_completion_consumed``, so a receipt
reconciled inline via ``process_manage`` poll/status did not retire the notice and the
still-injected ``[IMPORTANT: ... background processes completed ...]`` forced a no-op
acknowledgement turn after the work was already reconciled (recurs since 2026-10-03,
sys-3xmzmk). Retirement is a dedicated registry state (``_notice_retired``), not
drain-time marker clearing; delivery stays at-most-once.
"""

import time
from unittest.mock import patch

from tools.process_registry import ProcessRegistry, ProcessSession


def _make_session(sid, *, exited=False, exit_code=None, output=""):
    return ProcessSession(
        id=sid, command="pytest -q", task_id="t1", started_at=time.time(),
        exited=exited, exit_code=exit_code, output_buffer=output,
        notify_on_complete=True,
    )


def _park_exited_completion(registry, session):
    """Register a running session, flip it exited and move it to _finished so the
    completion event lands in the queue (the watcher's starting state)."""
    registry._running[session.id] = session
    with patch.object(registry, "_write_checkpoint"):
        registry._move_to_finished(session)
    assert registry.completion_queue.qsize() == 1


def test_exited_poll_retires_the_notice_and_suppresses_watcher_delivery():
    """(a) consumed-suppresses: poll() returning the exit fields retires the notice on
    the watcher path — the watcher-style drain (skip_poll_observed=False, as gateway
    run_notifications / tui prompt_turn pass) must drop the stale event."""
    registry = ProcessRegistry()
    session = _make_session("proc_poll", exited=True, exit_code=0, output="all passed")
    _park_exited_completion(registry, session)

    result = registry.poll(session.id)  # the agent reconciles the receipt inline
    assert result["status"] == "exited"
    assert registry.is_completion_notice_retired(session.id)
    # Narrow consumed semantics stay untouched (wait/log contract).
    assert not registry.is_completion_consumed(session.id)
    # Watcher-style drain: retired completions are never re-injected.
    assert registry.drain_notifications(skip_poll_observed=False) == []


def test_wait_consumption_retires_on_the_watcher_path_too():
    registry = ProcessRegistry()
    session = _make_session("proc_wait", exited=True, exit_code=0, output="all passed")
    _park_exited_completion(registry, session)

    assert registry.wait(session.id, timeout=1)["status"] == "exited"
    assert registry.is_completion_notice_retired(session.id)
    assert registry.drain_notifications(skip_poll_observed=False) == []


def test_unread_completion_delivers_exactly_once():
    """(b) unread-delivered-once: nobody polled, so the watcher drain gets the event —
    once; a second drain must not re-offer it."""
    registry = ProcessRegistry()
    session = _make_session("proc_unread", exited=True, exit_code=0, output="all passed")
    _park_exited_completion(registry, session)

    assert not registry.is_completion_notice_retired(session.id)
    drained = registry.drain_notifications(skip_poll_observed=False)
    assert len(drained) == 1
    assert session.id in drained[0][1]
    assert registry.drain_notifications(skip_poll_observed=False) == []


def test_running_poll_stays_read_only_and_later_exit_still_delivers():
    """A poll of a RUNNING session is a status check: it must not retire the
    completion-to-come (#10156's original concern). The exit lands after, unread, and
    is delivered once."""
    registry = ProcessRegistry()
    session = _make_session("proc_running")
    registry._running[session.id] = session

    assert registry.poll(session.id)["status"] == "running"
    assert not registry.is_completion_notice_retired(session.id)

    session.exited = True
    session.exit_code = 0
    with patch.object(registry, "_write_checkpoint"):
        registry._move_to_finished(session)
    assert len(registry.drain_notifications(skip_poll_observed=False)) == 1
