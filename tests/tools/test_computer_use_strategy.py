"""Unit tests for Computer Use targeting strategies and native HWND targeting."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use.backend import ActionResult, CaptureResult
from tools.computer_use.cua_backend_capture import _CaptureMixin
from tools.computer_use.tool import _dispatch


class FakeBackend:
    def __init__(self):
        self.clicked_args = []
        self.dragged_args = []
        self.scrolled_args = []
        self._last_app = None

    def click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="click", message="clicked")

    def double_click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="double_click", message="double_clicked")

    def right_click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="right_click", message="right_clicked")

    def middle_click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="middle_click", message="middle_clicked")

    def drag(self, **kwargs):
        self.dragged_args.append(kwargs)
        return ActionResult(ok=True, action="drag", message="dragged")

    def scroll(self, **kwargs):
        self.scrolled_args.append(kwargs)
        return ActionResult(ok=True, action="scroll", message="scrolled")

    def set_value(self, **kwargs):
        return ActionResult(ok=True, action="set_value", message="value_set")


def test_invalid_strategy_rejected():
    backend = FakeBackend()
    res = _dispatch(backend, "click", {"element": 1, "strategy": "invalid_strat"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "invalid_strategy"
    assert "invalid strategy" in parsed["error"]


def test_strategy_a11y_requires_element():
    backend = FakeBackend()

    # Reject coordinate-only click under strategy='a11y'
    res = _dispatch(backend, "click", {"coordinate": [100, 200], "strategy": "a11y"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_a11y_required"
    assert "strategy='a11y' requires targeting by `element`" in parsed["error"]

    # Allow element click under strategy='a11y' and strip coordinates
    res2 = _dispatch(backend, "click", {"element": 5, "coordinate": [100, 200], "strategy": "a11y"})
    assert json.loads(res2)["ok"] is True
    assert backend.clicked_args[-1]["element"] == 5
    assert backend.clicked_args[-1]["x"] is None
    assert backend.clicked_args[-1]["y"] is None


def test_strategy_a11y_drag_and_scroll():
    backend = FakeBackend()

    # Drag without element indices fails
    res = _dispatch(backend, "drag", {"from_coordinate": [0, 0], "to_coordinate": [10, 10], "strategy": "a11y"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_a11y_required"

    # Drag with element indices succeeds
    res2 = _dispatch(backend, "drag", {"from_element": 1, "to_element": 2, "strategy": "a11y"})
    assert json.loads(res2)["ok"] is True

    # Scroll with coordinate-only fails
    res3 = _dispatch(backend, "scroll", {"coordinate": [50, 50], "direction": "down", "strategy": "a11y"})
    parsed3 = json.loads(res3)
    assert parsed3["ok"] is False
    assert parsed3["code"] == "strategy_a11y_required"

    # Scroll with element succeeds
    res4 = _dispatch(backend, "scroll", {"element": 3, "direction": "down", "strategy": "a11y"})
    assert json.loads(res4)["ok"] is True


def test_strategy_event_requires_coordinates_and_strips_element():
    backend = FakeBackend()

    # Reject element-only click under strategy='event'
    res = _dispatch(backend, "click", {"element": 4, "strategy": "event"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_event_required"
    assert "strategy='event' requires `coordinate=[x, y]`" in parsed["error"]

    # When both are given, strip element and use coordinates
    res2 = _dispatch(backend, "click", {"element": 4, "coordinate": [150, 250], "strategy": "event"})
    assert json.loads(res2)["ok"] is True
    assert backend.clicked_args[-1]["element"] is None
    assert backend.clicked_args[-1]["x"] == 150
    assert backend.clicked_args[-1]["y"] == 250


def test_strategy_auto_permits_both():
    backend = FakeBackend()

    # Element targeting
    res1 = _dispatch(backend, "click", {"element": 2, "strategy": "auto"})
    assert json.loads(res1)["ok"] is True
    assert backend.clicked_args[-1]["element"] == 2

    # Coordinate targeting
    res2 = _dispatch(backend, "click", {"coordinate": [10, 20], "strategy": "auto"})
    assert json.loads(res2)["ok"] is True
    assert backend.clicked_args[-1]["x"] == 10
    assert backend.clicked_args[-1]["y"] == 20


class FakeCaptureBackend(_CaptureMixin):
    def __init__(self, sample_windows=None):
        self._sample_windows = sample_windows or []
        self._active_target = None
        self._active_pid = None
        self._active_window_id = None
        self._last_app = None

    def _clear_active_target(self):
        self._active_target = None
        self._active_pid = None
        self._active_window_id = None

    def list_windows(self):
        return list(self._sample_windows)


def test_window_id_alone_lookup_matching_window():
    sample_windows = [
        {"app_name": "Notepad", "pid": 1234, "window_id": 4321, "off_screen": False, "title": "Untitled", "z_index": 1},
        {"app_name": "Calculator", "pid": 5678, "window_id": 8765, "off_screen": False, "title": "Calc", "z_index": 2},
    ]
    backend = FakeCaptureBackend(sample_windows)

    # Resolve by window_id alone matching Calculator
    candidates = backend._resolve_capture_windows("som", None, pid=None, window_id=8765)
    assert isinstance(candidates, list)
    assert len(candidates) == 1
    assert candidates[0]["app_name"] == "Calculator"
    assert candidates[0]["pid"] == 5678
    assert candidates[0]["window_id"] == 8765


def test_window_id_alone_fails_closed_when_not_in_list():
    sample_windows = [
        {"app_name": "Notepad", "pid": 1234, "window_id": 4321, "off_screen": False, "title": "Untitled", "z_index": 1},
    ]
    backend = FakeCaptureBackend(sample_windows)

    # Window ID not enumerated in list_windows must fail closed
    res = backend._resolve_capture_windows("som", "GhostApp", pid=None, window_id=99999)
    assert isinstance(res, CaptureResult)
    assert "<window_id 99999 not found in active windows>" in res.window_title


def test_pid_alone_still_fails_closed():
    backend = FakeCaptureBackend([])
    res = backend._resolve_capture_windows("som", "App", pid=1234, window_id=None)
    assert isinstance(res, CaptureResult)
    assert "requires both pid and window_id" in res.window_title


def test_strategy_event_rejects_set_value():
    backend = FakeBackend()
    res = _dispatch(backend, "set_value", {"value": "text", "strategy": "event"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_event_unsupported"
    assert "strategy='event' is not supported for set_value" in parsed["error"]


def test_session_declared_id_syncs_to_pipe_transport():
    from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession
    session = _CuaDriverSession(_AsyncBridge())
    session._started = True
    mock_pipe = MagicMock()
    mock_pipe.session_id = None
    session._pipe_transport = mock_pipe

    # Call start_session
    mock_resp = {"isError": False}
    with patch.object(session._bridge, "run", return_value=mock_resp):
        session.call_tool("start_session", {"session": "test-session-xyz"})
    assert session._declared_session_id == "test-session-xyz"
    assert mock_pipe.session_id == "test-session-xyz"

    # Call end_session
    with patch.object(session._bridge, "run", return_value=mock_resp):
        session.call_tool("end_session", {"session": "test-session-xyz"})
    assert session._declared_session_id is None
    assert mock_pipe.session_id is None

