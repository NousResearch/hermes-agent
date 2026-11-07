"""Verification must use capture ownership and nonempty predicates."""
import json

import pytest

from tests.tools.test_computer_use_delivery_ladder import _FakeSession, _make_backend
from tools.computer_use.tool import _dispatch

EXPECT = [{"element": {"selector": {"role": "CheckBox", "label_contains": "Advanced"}, "exists": True}}]


@pytest.mark.parametrize("selector", [{}, {"pid": 4242}, {"window_id": 7}, {"pid": 4242, "window_id": 7}])
def test_capture_then_verify_uses_same_window(selector):
    session = _FakeSession({"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}})
    backend = _make_backend(session)
    # Keep real capture selection and dispatch; replace only the driver transport.
    session._out = {"isError": False, "data": 'AXWindow "target"', "structuredContent": {"windows": [
        {"pid": 4242, "window_id": 7, "app_name": "Safari", "title": "target"},
    ]}}
    backend._clear_active_target()
    _dispatch(backend, "capture", {"mode": "ax", "app": "Safari"})
    session._out = {"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}}
    result = json.loads(_dispatch(backend, "verify_state", {"expect": EXPECT, **selector}))
    assert result["verdict"]["decision"] == "done"
    assert session.calls[-1] == ("verify_state", {"pid": 4242, "window_id": 7, "expect": EXPECT, "session": "test-run"})


@pytest.mark.parametrize("expect", [None, [], "predicate", {}, [None], [{}]])
def test_invalid_expect_never_reaches_driver(expect):
    session = _FakeSession({"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}})
    backend = _make_backend(session)
    args = {} if expect is None else {"expect": expect}
    result = json.loads(_dispatch(backend, "verify_state", args))
    assert "error" in result
    assert result.get("verdict", {}).get("decision") != "done"
    assert session.calls == []
    # The backend's public entry point also rejects invalid predicates.
    assert backend.verify_state(expect).ok is False
    assert session.calls == []


def test_no_target_refuses_without_driver_call():
    session = _FakeSession({"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}})
    backend = _make_backend(session)
    backend._clear_active_target()
    result = json.loads(_dispatch(backend, "verify_state", {"expect": EXPECT}))
    assert result["ok"] is False
    assert session.calls == []


def test_explicit_pair_works_without_capture_and_does_not_retarget_inputs():
    session = _FakeSession({"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}})
    backend = _make_backend(session)
    result = backend.verify_state(EXPECT, pid=12, window_id=13)
    assert result.ok is True
    assert session.calls[-1][1]["pid"] == 12
    assert session.calls[-1][1]["window_id"] == 13
    assert backend._active_pid == 4242
    assert backend._active_window_id == 7


def test_changed_pid_does_not_inherit_previous_window():
    session = _FakeSession({"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}})
    backend = _make_backend(session)
    assert backend.verify_state(EXPECT, pid=12).ok is False
    assert session.calls == []
