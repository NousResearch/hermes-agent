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


def test_register_then_resolve_roundtrip():
    run_id = "run_test1"
    ctg.register("call_1", run_id, "set_timer", {"minutes": 5})

    out = {}

    def _agent_thread():
        out["result"] = ctg.wait_for_result("call_1", timeout=5.0)

    t = threading.Thread(target=_agent_thread)
    t.start()
    # Let the waiter enter its poll loop, then resolve from the "HTTP" side.
    time.sleep(0.2)
    assert ctg.has_pending(run_id) is True
    assert ctg.resolve_client_tool("call_1", '{"ok": true}') is True
    t.join(timeout=3.0)

    assert out["result"] == '{"ok": true}'
    # Entry is popped after wait_for_result returns.
    assert ctg.has_pending(run_id) is False


def test_timeout_returns_none():
    run_id = "run_test2"
    ctg.register("call_2", run_id, "set_timer", {"minutes": 1})
    start = time.monotonic()
    result = ctg.wait_for_result("call_2", timeout=0.5)
    elapsed = time.monotonic() - start
    assert result is None
    assert elapsed >= 0.5
    assert ctg.has_pending(run_id) is False


def test_clear_session_cancels_with_error_result():
    run_id = "run_test3"
    ctg.register("call_3a", run_id, "set_timer", {"minutes": 1})
    ctg.register("call_3b", run_id, "set_alarm", {"hour": 7})

    out = {}

    def _agent_thread(cid):
        out[cid] = ctg.wait_for_result(cid, timeout=5.0)

    threads = [threading.Thread(target=_agent_thread, args=(c,))
               for c in ("call_3a", "call_3b")]
    for t in threads:
        t.start()
    time.sleep(0.2)

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
    assert ctg.resolve_client_tool("nope", '{"x":1}') is False


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
