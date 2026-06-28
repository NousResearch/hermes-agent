"""Regression tests for Linux cua-driver window metadata quirks."""

from __future__ import annotations


LINUX_LIST_WINDOWS = [
    {
        "app_name": "",
        "pid": 2951331,
        "window_id": 98566147,
        "title": "@!1921,0;BDHF",
        "is_on_screen": None,
        "z_index": 0,
    },
    {
        "app_name": "",
        "pid": 11715,
        "window_id": 81790890,
        "title": "Guides — OMC Docs - Google Chrome",
        "is_on_screen": None,
        "z_index": 0,
    },
    {
        "app_name": "",
        "pid": 11433,
        "window_id": 41943052,
        "title": "README.md - hermes-agent - Visual Studio Code",
        "is_on_screen": False,
        "z_index": 0,
    },
]


def _normalized_windows():
    from tools.computer_use.cua_backend import _window_from_list_windows_entry

    return [_window_from_list_windows_entry(w) for w in LINUX_LIST_WINDOWS]


def test_linux_null_is_on_screen_is_treated_as_unknown_not_offscreen():
    """cua-driver 0.6.x may return JSON null for Linux is_on_screen."""
    windows = _normalized_windows()

    assert windows[0]["off_screen"] is False
    assert windows[1]["off_screen"] is False
    assert windows[2]["off_screen"] is True


def test_linux_empty_app_name_falls_back_to_window_title_for_app_filter():
    """Linux list_windows can omit app_name, so capture(app=...) must use title."""
    from tools.computer_use.cua_backend import _window_matches_app_filter

    chrome = _normalized_windows()[1]

    assert chrome["app_name"] == ""
    assert _window_matches_app_filter(chrome, "chrome") is True
    assert _window_matches_app_filter(chrome, "firefox") is False


def test_default_capture_skips_gnome_shell_background_window():
    """GNOME Shell @!x,y;BDHF windows appear before app windows but screenshot empty."""
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()

    target = _select_capture_target(windows, app_requested=False)

    assert target["pid"] == 11715
    assert target["window_id"] == 81790890
    assert "Google Chrome" in target["title"]


def test_explicit_app_capture_preserves_filtered_target_order():
    """When the caller filters first, target selection should not skip the match."""
    from tools.computer_use.cua_backend import _select_capture_target

    chrome = _normalized_windows()[1]

    assert _select_capture_target([chrome], app_requested=True) == chrome
