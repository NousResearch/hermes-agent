"""Unit tests for the split-runtime client-tool relay primitive.

Exercises the suspend/resume/timeout/cancel semantics of
``tools.client_tool_gateway`` without a live model — the risky new code path
for split-runtime.  Mirrors the shape of the clarify_gateway it is cloned from.
"""

import threading
import time

import pytest

from tools import client_tool_gateway as ctg


@pytest.fixture(autouse=True)
def _clean_state():
    # Isolate module-level state between tests.
    ctg._entries.clear()
    ctg._session_index.clear()
    ctg._notify_cbs.clear()
    yield
    ctg._entries.clear()
    ctg._session_index.clear()
    ctg._notify_cbs.clear()


class _EnteredEvent(threading.Event):
    """The entry's Event, also reporting when a waiter has started blocking on it."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()

    def wait(self, timeout=None):
        self.entered.set()
        return super().wait(timeout)


def test_register_then_resolve_roundtrip():
    run_id = "run_test1"
    ctg.register("call_1", run_id, "set_timer", {"minutes": 5})

    out = {}
    started = threading.Event()

    def _agent_thread():
        started.set()
        out["result"] = ctg.wait_for_result(run_id, "call_1", timeout=5.0)

    t = threading.Thread(target=_agent_thread)
    t.start()
    # The waiter is running; resolve from the "HTTP" side.
    assert started.wait(timeout=5.0)
    assert ctg.has_pending(run_id) is True
    assert ctg.resolve_client_tool(run_id, "call_1", '{"ok": true}') is True
    t.join(timeout=3.0)

    assert out["result"] == '{"ok": true}'
    # Entry is popped after wait_for_result returns.
    assert ctg.has_pending(run_id) is False


def test_timeout_returns_none():
    run_id = "run_test2"
    ctg.register("call_2", run_id, "set_timer", {"minutes": 1})
    start = time.monotonic()
    result = ctg.wait_for_result(run_id, "call_2", timeout=0.5)
    elapsed = time.monotonic() - start
    assert result is None
    assert elapsed >= 0.5
    assert ctg.has_pending(run_id) is False


def test_clear_session_cancels_with_error_result():
    run_id = "run_test3"
    entries = [
        ctg.register("call_3a", run_id, "set_timer", {"minutes": 1}),
        ctg.register("call_3b", run_id, "set_alarm", {"hour": 7}),
    ]
    for entry in entries:
        entry.event = _EnteredEvent()

    out = {}

    def _agent_thread(cid):
        out[cid] = ctg.wait_for_result(run_id, cid, timeout=5.0)

    threads = [threading.Thread(target=_agent_thread, args=(c,))
               for c in ("call_3a", "call_3b")]
    for t in threads:
        t.start()
    # Cancel only once both waiters are blocked on their entry.
    for entry in entries:
        assert entry.event.entered.wait(timeout=5.0)

    cancelled = ctg.clear_session(run_id)
    assert cancelled == 2
    for t in threads:
        t.join(timeout=3.0)

    # Both waiters unblocked with a JSON error sentinel, not None.
    for cid in ("call_3a", "call_3b"):
        assert out[cid] is not None
        assert "cancelled" in out[cid]
    assert ctg.has_pending(run_id) is False


def test_resolve_unknown_call_is_false():
    assert ctg.resolve_client_tool("run_missing", "nope", '{"x":1}') is False


# =========================================================================
# Cross-run correlation — entries are scoped to (run_id, call_id)
# =========================================================================

def test_resolve_from_wrong_run_does_not_unblock():
    """A result POSTed to another live run must not resolve this run's call.

    Regression: the registry was keyed by call_id alone, so a caller who knew
    a pending call_id could unblock a *different* run by posting to its
    /tool_result endpoint.
    """
    run_a, run_b = "run_a", "run_b"
    ctg.register("call_shared", run_a, "set_timer", {"minutes": 5})
    # run_b is live but has no pending call with that id.
    ctg.register("call_b_only", run_b, "get_location", {})

    out = {}
    started = threading.Event()

    def _agent_thread():
        started.set()
        out["result"] = ctg.wait_for_result(run_a, "call_shared", timeout=2.0)

    t = threading.Thread(target=_agent_thread)
    t.start()
    assert started.wait(timeout=5.0)

    # The wrong-run POST is rejected and leaves run_a still waiting.
    assert ctg.resolve_client_tool(run_b, "call_shared", '{"hijacked": true}') is False
    assert ctg.has_pending(run_a) is True

    # The correctly-scoped POST resolves it.
    assert ctg.resolve_client_tool(run_a, "call_shared", '{"ok": true}') is True
    t.join(timeout=3.0)
    assert out["result"] == '{"ok": true}'


def test_duplicate_call_id_across_runs_are_independent():
    """The same call_id in two runs must not collide.

    tool_call_id is only unique within a run. Keyed by call_id alone the
    second register() clobbered the first entry, stranding that agent thread
    until its timeout.
    """
    run_a, run_b = "run_dup_a", "run_dup_b"
    entry_a = ctg.register("call_same", run_a, "set_timer", {"minutes": 5})
    entry_b = ctg.register("call_same", run_b, "set_alarm", {"hour": 7})
    assert entry_a is not entry_b

    out = {}
    started = threading.Barrier(3)

    def _agent_thread(run_id):
        started.wait(timeout=5.0)
        out[run_id] = ctg.wait_for_result(run_id, "call_same", timeout=3.0)

    threads = [threading.Thread(target=_agent_thread, args=(r,))
               for r in (run_a, run_b)]
    for t in threads:
        t.start()
    started.wait(timeout=5.0)  # both waiters are running

    assert ctg.has_pending(run_a) is True
    assert ctg.has_pending(run_b) is True

    # Resolving one leaves the other pending with its own payload.
    assert ctg.resolve_client_tool(run_a, "call_same", '{"which": "a"}') is True
    threads[0].join(timeout=3.0)  # run_a's waiter has unwound
    assert out[run_a] == '{"which": "a"}'
    assert ctg.has_pending(run_b) is True
    assert ctg.resolve_client_tool(run_b, "call_same", '{"which": "b"}') is True

    for t in threads:
        t.join(timeout=3.0)
    assert out[run_a] == '{"which": "a"}'
    assert out[run_b] == '{"which": "b"}'


def test_clear_session_only_affects_its_own_run():
    """Run-boundary cleanup must not cancel another run's pending calls."""
    run_a, run_b = "run_clr_a", "run_clr_b"
    ctg.register("call_same", run_a, "set_timer", {"minutes": 5})
    ctg.register("call_same", run_b, "set_timer", {"minutes": 9})

    assert ctg.clear_session(run_a) == 1
    assert ctg.has_pending(run_a) is False
    assert ctg.has_pending(run_b) is True

    pending_b = ctg.get_pending_for_session(run_b)
    assert pending_b is not None
    assert pending_b.arguments == {"minutes": 9}


def test_double_resolve_is_rejected():
    """A duplicate POST for an already-resolved call returns False (→ 409)."""
    run_id = "run_double"
    ctg.register("call_d", run_id, "set_timer", {"minutes": 1})

    out = {}
    started = threading.Event()

    def _agent_thread():
        started.set()
        out["result"] = ctg.wait_for_result(run_id, "call_d", timeout=3.0)

    t = threading.Thread(target=_agent_thread)
    t.start()
    assert started.wait(timeout=5.0)

    assert ctg.resolve_client_tool(run_id, "call_d", '{"first": true}') is True
    t.join(timeout=3.0)
    # Entry is gone once the waiter unwound, so a replay cannot resolve again.
    assert ctg.resolve_client_tool(run_id, "call_d", '{"second": true}') is False
    assert out["result"] == '{"first": true}'


def test_notify_callback_receives_entry():
    run_id = "run_test4"
    seen = {}

    def _cb(entry):
        seen["sig"] = entry.signature()

    ctg.register_notify(run_id, _cb)
    entry = ctg.register("call_4", run_id, "start_navigation", {"destination": "home"})
    cb = ctg.get_notify(run_id)
    assert cb is not None
    cb(entry)

    assert seen["sig"]["call_id"] == "call_4"
    assert seen["sig"]["name"] == "start_navigation"
    assert seen["sig"]["arguments"] == {"destination": "home"}


def test_unregister_notify_clears_pending():
    run_id = "run_test5"
    ctg.register_notify(run_id, lambda e: None)
    ctg.register("call_5", run_id, "get_location", {})
    assert ctg.has_pending(run_id) is True
    ctg.unregister_notify(run_id)
    # unregister_notify → clear_session drops the pending entry.
    assert ctg.has_pending(run_id) is False
    assert ctg.get_notify(run_id) is None
