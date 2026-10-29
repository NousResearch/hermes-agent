"""app= matching against Windows app names, which cua-driver reports as executable names (``Notepad.exe``).

``app="Notepad"`` must resolve as an exact match of ``Notepad.exe``: falling through to the substring tier picks
whichever ``*notepad*`` window is frontmost (``notepad++.exe``) and spends an extra ``list_apps`` round trip.
The same suffix hid the driver's own ``cua-driver.exe`` cursor overlay from the #94527 self-window filter.
"""

from typing import Any
from unittest.mock import MagicMock

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_capture import _select_capture_target


def _window(app_name: str, pid: int, window_id: int, z_index: int) -> dict[str, Any]:
    return {"app_name": app_name, "pid": pid, "window_id": window_id, "title": "", "is_on_screen": True,
            "z_index": z_index}


def _backend(windows: list[dict[str, Any]]) -> tuple[CuaDriverBackend, list[str]]:
    """A backend whose driver serves *windows*, with list_apps reporting the same executable names."""
    calls: list[str] = []
    apps = [{"name": w["app_name"], "pid": w["pid"]} for w in windows]

    def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append(name)
        content = {"list_windows": {"windows": windows}, "list_apps": {"apps": apps}}[name]
        return {"data": "", "images": [], "structuredContent": content, "isError": False}

    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.side_effect = call_tool
    return backend, calls


def test_app_name_without_exe_targets_that_app_not_a_frontmost_lookalike():
    backend, _ = _backend([_window("notepad++.exe", 2, 20, 9), _window("Notepad.exe", 1, 10, 5)])

    result = backend.focus_app("Notepad")

    assert result.ok
    assert (backend._active_pid, backend._active_window_id) == (1, 10)


def test_app_name_without_exe_resolves_without_listing_apps():
    backend, calls = _backend([_window("Notepad.exe", 1, 10, 5)])

    assert backend.focus_app("notepad").ok
    assert calls == ["list_windows"]


def test_implicit_capture_skips_the_frontmost_cua_driver_exe_overlay():
    overlay = {**_window("cua-driver.exe", 3, 30, 5), "off_screen": False}
    app = {**_window("claude.exe", 1, 10, 4), "off_screen": False}

    assert _select_capture_target([overlay, app], app_requested=False) == app
