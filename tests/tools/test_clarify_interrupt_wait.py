"""wait_for_response must observe the thread-scoped interrupt (#83889 RC1).

Salvage of #25506 (@liuhao1024): the wait loop checks
``tools.interrupt.is_interrupted()`` once per slice. Without it, an
interrupted agent thread holds until the full clarify timeout (600s default),
or forever in unlimited mode — the session-boundary ``clear_session`` cleanup
only runs after the turn ends, which cannot happen while the thread is
blocked here. Family E anchor in the stall triage (#84047).
"""

from __future__ import annotations

import threading
import time

import tools.clarify_gateway as cg
from tools.interrupt import set_interrupt


_ID_COUNTER = iter(range(10_000))


def _register(session_key: str = "agent:main:test:dm:1"):
    clarify_id = f"clarify-test-{next(_ID_COUNTER)}"
    return cg.register(
        clarify_id=clarify_id,
        session_key=session_key,
        question="q",
        choices=None,
    )


def _wait_in_thread(clarify_id: str, timeout: float):
    """Run the wait in a worker thread; expose its ident for set_interrupt."""
    result = {}
    ready = threading.Event()

    def run():
        result["tid"] = threading.get_ident()
        ready.set()
        result["response"] = cg.wait_for_response(clarify_id, timeout)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert ready.wait(timeout=5.0)
    return t, result


def test_interrupt_unblocks_bounded_wait():
    entry = _register()
    t, result = _wait_in_thread(entry.clarify_id, timeout=600.0)
    try:
        time.sleep(0.1)
        assert t.is_alive(), "wait returned before any interrupt/resolve"
        set_interrupt(True, thread_id=result["tid"])
        t.join(timeout=5.0)
        assert not t.is_alive(), "interrupt did not unblock the bounded wait"
        assert result["response"] is None
    finally:
        set_interrupt(False, thread_id=result["tid"])


def test_interrupt_unblocks_unlimited_wait():
    """timeout<=0 is the forever-wedge: no deadline ever fires."""
    entry = _register()
    t, result = _wait_in_thread(entry.clarify_id, timeout=0.0)
    try:
        time.sleep(0.1)
        assert t.is_alive()
        set_interrupt(True, thread_id=result["tid"])
        t.join(timeout=5.0)
        assert not t.is_alive(), "interrupt did not unblock the unlimited wait"
        assert result["response"] is None
    finally:
        set_interrupt(False, thread_id=result["tid"])


def test_resolve_still_wins_and_returns_response():
    entry = _register()
    t, result = _wait_in_thread(entry.clarify_id, timeout=600.0)
    time.sleep(0.05)
    assert cg.resolve_gateway_clarify(entry.clarify_id, "picked A") is True
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert result["response"] == "picked A"


def test_response_resolved_before_interrupt_wins_deterministically():
    entry = _register()
    assert cg.resolve_gateway_clarify(entry.clarify_id, "picked A") is True
    set_interrupt(True)
    try:
        assert cg.wait_for_response(entry.clarify_id, timeout=600.0) == "picked A"
    finally:
        set_interrupt(False)


def test_timeout_still_works_without_interrupt():
    entry = _register()
    t, result = _wait_in_thread(entry.clarify_id, timeout=0.2)
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert result["response"] is None


def test_entry_cleaned_up_after_interrupt():
    """An interrupted wait must not leak its entry/session-index rows."""
    key = "agent:main:test:dm:cleanup"
    entry = _register(session_key=key)
    set_interrupt(True)  # current thread; wait aborts on first slice check
    try:
        response = cg.wait_for_response(entry.clarify_id, 600.0)
    finally:
        set_interrupt(False)
    assert response is None
    # A second resolve should find nothing — the entry is gone.
    assert cg.resolve_gateway_clarify(entry.clarify_id, "late") is False
