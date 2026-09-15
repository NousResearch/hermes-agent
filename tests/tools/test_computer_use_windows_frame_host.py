"""UWP frame-host windows must resolve without broadening title matching to other apps."""

import json
from unittest.mock import MagicMock

import pytest

from tools import computer_use_tool  # noqa: F401 -- register the real handler
from tools.computer_use import tool
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.registry import registry

pytestmark = pytest.mark.windows_only


def _capture(monkeypatch, windows, requested="Calculator", **extra_args):
    backend = CuaDriverBackend()
    backend._session = MagicMock()

    def call_driver(name, args):
        if name == "list_windows":
            payload = {"windows": windows}
        elif name == "list_apps":
            payload = {"apps": [{"name": "Windows Calculator", "pid": 0, "running": False,
                                  "bundle_id": "Microsoft.WindowsCalculator_8wekyb3d8bbwe",
                                  "kind": "uwp", "windows": []}]}
        elif name == "get_window_state":
            payload = {"elements": [{"element_index": 8, "role": "Text", "label": "Display is 0",
                                     "frame": {"x": 9, "y": 419, "w": 400, "h": 92}}]}
        else:
            raise AssertionError(f"Unexpected native operation: {name}")
        return {"data": "", "images": [], "isError": False, "structuredContent": payload}

    backend._session.call_tool.side_effect = call_driver
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": backend)
    args = {"action": "capture", "mode": "ax"}
    if requested is not None:
        args["app"] = requested
    args.update(extra_args)
    result = registry.dispatch("computer_use", args)
    return json.loads(result), backend


def _window(app="ApplicationFrameHost.exe", title="Calculator", window_id=20, pid=10):
    return {"app_name": app, "pid": pid, "window_id": window_id,
            "is_on_screen": True, "title": title, "z_index": 1}


def test_frame_host_capture_preserves_the_requested_app_identity(monkeypatch):
    result, backend = _capture(monkeypatch, [_window()], "cAlCuLaToR")
    assert result["total_elements"] == 1
    assert result["elements"][0]["label"] == "Display is 0"
    assert result["app"] == "Calculator"
    assert backend._active_window_id == 20
    assert tool._input_target_mismatch(backend, "Calculator") is None


def test_frame_host_tolerates_name_without_exe(monkeypatch):
    result, backend = _capture(monkeypatch, [_window(app="ApplicationFrameHost")], "Calculator")
    assert result["total_elements"] == 1
    assert result["app"] == "Calculator"
    assert backend._active_window_id == 20
    assert tool._input_target_mismatch(backend, "Calculator") is None


def test_default_capture_on_frame_host_normalizes_app_name_to_title(monkeypatch):
    # When capture is invoked without app parameter (default frontmost window capture),
    # ApplicationFrameHost with title 'Calculator' must normalize app_name to 'Calculator',
    # so subsequent actions with app='Calculator' pass the input target guard.
    result, backend = _capture(monkeypatch, [_window()], requested=None)
    assert result["total_elements"] == 1
    assert result["app"] == "Calculator"
    assert backend._last_app == "Calculator"
    assert tool._input_target_mismatch(backend, "Calculator") is None


@pytest.mark.parametrize("windows", [
    [_window(app="notepad.exe")],
    [_window(app="計算機")],
    [_window(title="Calculator history")],
    [_window(window_id=20), _window(window_id=21)],
])
def test_unrelated_or_ambiguous_titles_never_arm_a_capture_target(monkeypatch, windows):
    result, backend = _capture(monkeypatch, windows)
    assert result["total_elements"] == 0
    assert backend._active_pid is None
    assert backend._active_window_id is None


def test_unlisted_stale_hwnd_capture_fails_closed(monkeypatch):
    """(a) An unlisted/stale HWND is refused (fails closed) and never sets _active_window_id."""
    windows = [_window(app="notepad.exe", title="Notes", window_id=20, pid=100)]
    result, backend = _capture(monkeypatch, windows, requested=None, window_id=999)

    assert result["total_elements"] == 0
    assert "<window_id 999 not found in active windows>" in (result.get("window_title") or "")
    assert backend._active_window_id is None
    assert backend._active_pid is None


def test_observed_hwnd_capture_binds_discovered_pid_and_hwnd(monkeypatch):
    """(b) An observed HWND binds the discovered live PID and HWND."""
    windows = [
        _window(app="notepad.exe", title="Notes", window_id=20, pid=100),
        _window(app="code.exe", title="Editor", window_id=42, pid=555),
    ]
    result, backend = _capture(monkeypatch, windows, requested=None, window_id=42)

    assert result["total_elements"] == 1
    assert backend._active_window_id == 42
    assert backend._active_pid == 555


