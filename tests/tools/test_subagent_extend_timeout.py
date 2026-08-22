"""extend_subagent_timeout / _wait_for_child — sparing one running subagent
from delegation.child_timeout_seconds without touching the global config.

Registry-level coverage for extend_subagent_timeout() (mirrors the shape of
the interrupt_subagent/steer_subagent tests) plus the _wait_for_child polling
loop that lets a live extend land while the wait is already in flight.
"""

import threading
import time

import tools.delegate_tool as delegate_tool
from tools.delegate_tool import (
    FuturesTimeoutError,
    _register_subagent,
    _unregister_subagent,
    _wait_for_child,
    extend_subagent_timeout,
    subagent_timeout_active,
)

_UNSET = object()


def _with_registered(sid: str, *, timeout_at, child_timeout_seconds=_UNSET) -> None:
    # A configured cap defaults to matching timeout_at's presence — pass
    # child_timeout_seconds explicitly to register a case where the two
    # diverge (e.g. a restore test's re-arm duration).
    if child_timeout_seconds is _UNSET:
        child_timeout_seconds = None if timeout_at is None else 10.0
    _register_subagent(
        {
            "subagent_id": sid,
            "parent_id": "root",
            "depth": 1,
            "goal": "test goal",
            "status": "running",
            "agent": None,
            "owner_session_id": None,
            "owner_transport": None,
            "owner_session_record": None,
            "timeout_at": timeout_at,
            "child_timeout_seconds": child_timeout_seconds,
        }
    )


class _FakeFuture:
    """Stands in for the child's concurrent.futures.Future in these tests."""

    def __init__(self):
        self._done = threading.Event()
        self._value = "child-result"

    def finish(self):
        self._done.set()

    def result(self, timeout=None):
        if self._done.wait(timeout=timeout):
            return self._value
        raise FuturesTimeoutError()


def test_extend_unknown_subagent_is_false_not_an_error():
    assert extend_subagent_timeout("sid-not-registered", disable=True) is False


def test_extend_with_no_active_deadline_is_a_noop():
    # child_timeout_seconds is off for this child: nothing to extend.
    _with_registered("sid-extend-1", timeout_at=None)
    try:
        assert extend_subagent_timeout("sid-extend-1", seconds=60) is False
    finally:
        _unregister_subagent("sid-extend-1")


def test_disable_clears_the_deadline():
    _with_registered("sid-extend-2", timeout_at=time.time() + 10)
    try:
        assert extend_subagent_timeout("sid-extend-2", disable=True) is True
        with delegate_tool._active_subagents_lock:
            assert delegate_tool._active_subagents["sid-extend-2"]["timeout_at"] is None
    finally:
        _unregister_subagent("sid-extend-2")


def test_omitting_seconds_defaults_to_disable():
    _with_registered("sid-extend-3", timeout_at=time.time() + 10)
    try:
        assert extend_subagent_timeout("sid-extend-3") is True
        with delegate_tool._active_subagents_lock:
            assert delegate_tool._active_subagents["sid-extend-3"]["timeout_at"] is None
    finally:
        _unregister_subagent("sid-extend-3")


def test_seconds_pushes_the_deadline_out():
    original = time.time() + 5
    _with_registered("sid-extend-4", timeout_at=original)
    try:
        assert extend_subagent_timeout("sid-extend-4", seconds=100) is True
        with delegate_tool._active_subagents_lock:
            new_deadline = delegate_tool._active_subagents["sid-extend-4"]["timeout_at"]
        assert new_deadline > original
    finally:
        _unregister_subagent("sid-extend-4")


def test_restore_re_arms_a_fresh_window_not_the_stale_deadline():
    # The original deadline is already in the past — restoring must count a
    # fresh child_timeout_seconds from now, not resurrect a dead timestamp
    # that would fire immediately.
    _with_registered("sid-extend-5", timeout_at=time.time() - 5, child_timeout_seconds=30.0)
    try:
        assert extend_subagent_timeout("sid-extend-5", restore=True) is True
        with delegate_tool._active_subagents_lock:
            new_deadline = delegate_tool._active_subagents["sid-extend-5"]["timeout_at"]
        assert new_deadline > time.time() + 25
    finally:
        _unregister_subagent("sid-extend-5")


def test_restore_on_a_child_with_no_configured_cap_is_a_noop():
    _with_registered("sid-extend-6", timeout_at=None)
    try:
        assert extend_subagent_timeout("sid-extend-6", restore=True) is False
    finally:
        _unregister_subagent("sid-extend-6")


def test_disable_then_restore_round_trip():
    _with_registered("sid-extend-7", timeout_at=time.time() + 10, child_timeout_seconds=10.0)
    try:
        assert extend_subagent_timeout("sid-extend-7", disable=True) is True
        assert subagent_timeout_active("sid-extend-7") is False
        assert extend_subagent_timeout("sid-extend-7", restore=True) is True
        assert subagent_timeout_active("sid-extend-7") is True
    finally:
        _unregister_subagent("sid-extend-7")


def test_subagent_timeout_active_is_none_when_never_capped():
    _with_registered("sid-extend-8", timeout_at=None)
    try:
        assert subagent_timeout_active("sid-extend-8") is None
    finally:
        _unregister_subagent("sid-extend-8")


def test_subagent_timeout_active_is_none_for_unknown_subagent():
    assert subagent_timeout_active("sid-not-registered") is None


def test_wait_for_child_without_subagent_id_is_a_plain_passthrough():
    future = _FakeFuture()
    future.finish()
    assert _wait_for_child(future, None, 5) == "child-result"


def test_wait_for_child_without_a_cap_is_a_plain_passthrough():
    future = _FakeFuture()
    future.finish()
    assert _wait_for_child(future, "sid-not-registered", None) == "child-result"


def test_wait_for_child_raises_when_the_deadline_passes_unextended(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_TIMEOUT_POLL_SECONDS", 0.02)
    sid = "sid-wait-1"
    _with_registered(sid, timeout_at=time.time() + 0.05)
    try:
        future = _FakeFuture()  # never finishes
        start = time.monotonic()
        try:
            _wait_for_child(future, sid, 0.05)
            raise AssertionError("expected FuturesTimeoutError")
        except FuturesTimeoutError:
            pass
        assert time.monotonic() - start < 1.0
    finally:
        _unregister_subagent(sid)


def test_wait_for_child_honors_a_live_extension():
    sid = "sid-wait-2"
    original_poll = delegate_tool._TIMEOUT_POLL_SECONDS
    delegate_tool._TIMEOUT_POLL_SECONDS = 0.02
    _with_registered(sid, timeout_at=time.time() + 0.05)
    try:
        future = _FakeFuture()

        def _bump_then_finish():
            time.sleep(0.03)
            assert extend_subagent_timeout(sid, seconds=1) is True
            time.sleep(0.05)
            future.finish()

        threading.Thread(target=_bump_then_finish).start()

        # The original 0.05s deadline alone would have raised by now; the
        # extension applied at ~0.03s must be what lets this reach the result.
        assert _wait_for_child(future, sid, 0.05) == "child-result"
    finally:
        delegate_tool._TIMEOUT_POLL_SECONDS = original_poll
        _unregister_subagent(sid)


def test_wait_for_child_honors_a_live_disable():
    sid = "sid-wait-3"
    original_poll = delegate_tool._TIMEOUT_POLL_SECONDS
    delegate_tool._TIMEOUT_POLL_SECONDS = 0.02
    _with_registered(sid, timeout_at=time.time() + 0.05)
    try:
        future = _FakeFuture()

        def _disable_then_finish():
            time.sleep(0.03)
            assert extend_subagent_timeout(sid, disable=True) is True
            time.sleep(0.05)
            future.finish()

        threading.Thread(target=_disable_then_finish).start()

        assert _wait_for_child(future, sid, 0.05) == "child-result"
    finally:
        delegate_tool._TIMEOUT_POLL_SECONDS = original_poll
        _unregister_subagent(sid)


def test_wait_for_child_still_enforces_a_deadline_restored_after_a_disable():
    # A disable that later gets restored must go back to actually killing a
    # wedged child — not settle into the unbounded wait a bare disable alone
    # would have been fine to fall into.
    sid = "sid-wait-4"
    original_poll = delegate_tool._TIMEOUT_POLL_SECONDS
    delegate_tool._TIMEOUT_POLL_SECONDS = 0.02
    _with_registered(sid, timeout_at=time.time() + 0.05, child_timeout_seconds=0.05)
    try:
        future = _FakeFuture()  # never finishes

        def _disable_then_restore():
            time.sleep(0.02)
            assert extend_subagent_timeout(sid, disable=True) is True
            time.sleep(0.1)  # comfortably past the original 0.05s deadline
            assert extend_subagent_timeout(sid, restore=True) is True

        threading.Thread(target=_disable_then_restore).start()

        start = time.monotonic()
        try:
            _wait_for_child(future, sid, 0.05)
            raise AssertionError("expected FuturesTimeoutError once restored")
        except FuturesTimeoutError:
            pass
        elapsed = time.monotonic() - start
        # Must have outlived the original deadline (proving the disable held)
        # and still have actually fired once restored (proving the restore
        # re-armed real enforcement, not a no-op).
        assert 0.1 < elapsed < 1.0
    finally:
        delegate_tool._TIMEOUT_POLL_SECONDS = original_poll
        _unregister_subagent(sid)
