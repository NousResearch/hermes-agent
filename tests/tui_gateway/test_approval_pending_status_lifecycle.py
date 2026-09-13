"""Regression for #86565: an approval-parked session must stop reading as pending.

``_session_pending_kind`` is one decision point with three consumers: the live status
("waiting" is what the Desktop sidebar paints as the amber "needs input" dot) and both
reapers in ``tui_gateway/session_reaper.py``, which treat any pending kind as
eviction-exempt. Making the approval queue authoritative there is what fixes the
status dot — and it is also what would let a *stale* queue entry outlive the approval it
describes: the session would keep reporting "waiting" and keep its eviction exemption
after the prompt can no longer be answered.

Invariant: an approval counts as pending only while a live waiter can still consume a
response. Every terminal exit of the blocked wait — resolved by /approve or /deny, timed
out, interrupted by /stop — and session teardown must drop the queue entry.
"""

import threading

import pytest

from tools import approval as approval_mod
from tools import approval_context
from tools import interrupt as interrupt_mod
from tools.interrupt import set_interrupt
from tui_gateway import server

SESSION_KEY = "approval-status-lifecycle-key"
SID = "approval-status-lifecycle-sid"

APPROVAL_DATA = {
    "command": "rm -rf .git",
    "description": "dangerous command",
    "pattern_key": "dangerous",
    "pattern_keys": ["dangerous"],
}


def _reset() -> None:
    """Approval queues and per-thread interrupt bits are process-global. A bit set on a
    now-dead thread can leak onto a fresh thread that reuses its ident."""
    approval_mod.clear_session(SESSION_KEY)
    with interrupt_mod._lock:
        interrupt_mod._interrupted_threads.clear()
    set_interrupt(False)
    server._sessions.pop(SID, None)


@pytest.fixture(autouse=True)
def _isolate_approval_and_interrupt_state():
    _reset()
    yield
    _reset()


def _park_approval(result: dict) -> threading.Thread:
    """Block a real approval waiter and return once it is enqueued and notified."""
    notified = threading.Event()

    def _worker():
        result["decision"] = approval_mod._await_gateway_decision(
            SESSION_KEY, lambda _data: notified.set(), dict(APPROVAL_DATA)
        )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert notified.wait(timeout=5), "approval was never enqueued"
    return thread


@pytest.mark.parametrize("exit_path", ["approved", "denied", "timed out", "interrupted", "torn down"])
def test_pending_only_while_a_waiter_can_answer(monkeypatch, exit_path):
    monkeypatch.setattr(
        approval_context,
        "_get_approval_config",
        lambda: {"timeout": 2 if exit_path == "timed out" else 300},
    )
    session = {
        "session_key": SESSION_KEY,
        "running": False,
        "transport": server._detached_ws_transport,
    }
    server._sessions[SID] = session
    result: dict = {}
    thread = _park_approval(result)

    try:
        # Parked: the queue entry describes a waiter that can still answer.
        assert server._session_pending_kind(SID) == "approval"
        assert server._session_live_status(SID, session) == "waiting"
        assert server._session_is_lru_evictable(SID, session) is False

        if exit_path == "approved":
            assert approval_mod.resolve_gateway_approval(SESSION_KEY, "session") == 1
        elif exit_path == "denied":
            assert approval_mod.resolve_gateway_approval(SESSION_KEY, "deny") == 1
        elif exit_path == "interrupted":
            set_interrupt(True, thread.ident)
        elif exit_path == "torn down":
            approval_mod.clear_session(SESSION_KEY)
        # "timed out" releases itself through the configured deadline.

        thread.join(timeout=10)
        assert not thread.is_alive(), f"the waiter never returned ({exit_path})"

        # Terminal: no waiter, so no pending kind — and no eviction exemption
        # held open by an approval nobody can answer.
        assert approval_mod.has_blocking_approval(SESSION_KEY) is False
        assert server._session_pending_kind(SID) == ""
        assert server._session_live_status(SID, session) == "idle"
        assert server._session_is_lru_evictable(SID, session) is True
    finally:
        # A failure before the exit was triggered must not leave the thread parked.
        if thread.is_alive():
            approval_mod.resolve_gateway_approval(SESSION_KEY, "deny")
            thread.join(timeout=2)
