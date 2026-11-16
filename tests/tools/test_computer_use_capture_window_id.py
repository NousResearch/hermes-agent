"""capture(window_id=...) without pid, and list_apps errors: capture discovery that must fail closed."""

from unittest.mock import MagicMock

import pytest

from tools.computer_use.backend import CaptureResult
from tools.computer_use.cua_backend_capture import _CaptureMixin


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


def test_list_apps_raises_on_a_driver_error_instead_of_returning_nothing():
    backend = FakeCaptureBackend()
    backend._session, backend._session_id = MagicMock(), "sess-test"
    backend._session.call_tool.return_value = {"isError": True, "data": "daemon proxy error", "images": [],
                                               "structuredContent": None}
    with pytest.raises(RuntimeError, match="cua-driver list_apps failed: daemon proxy error"):
        backend.list_apps()

    backend._session.call_tool.return_value = {"isError": False, "data": None, "images": [],
                                               "structuredContent": {"apps": [{"name": "calculator.exe", "pid": 1234}]}}
    assert backend.list_apps() == [{"name": "calculator.exe", "pid": 1234}]
