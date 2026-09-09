"""Regression tests for Linux/X11 capture target selection (#58026, #54173)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Tied z_index=0 fixture from #58026 (ding ahead of real terminals).
ISSUE_58026_WINDOWS = [
    {
        "app_name": "ding",
        "pid": 4294,
        "window_id": 33554439,
        "title": "Desktop Icons 1",
        "is_on_screen": True,
        "z_index": 0,
    },
    {
        "app_name": "",
        "pid": 1816017,
        "window_id": 60817412,
        "title": "zcode",
        "is_on_screen": True,
        "z_index": 0,
    },
    {
        "app_name": "",
        "pid": 1877178,
        "window_id": 84043449,
        "title": "xr@10:~/hermes",
        "is_on_screen": True,
        "z_index": 0,
    },
    {
        "app_name": "",
        "pid": 1877178,
        "window_id": 84065715,
        "title": "HERMES-CU",
        "is_on_screen": True,
        "z_index": 0,
    },
]

# Linux metadata-quirk fixture from #54173 (null is_on_screen, GNOME Shell
# @!x,y;BDHF backdrop helper ahead of real app windows).
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


def _normalized_windows(raw=ISSUE_58026_WINDOWS):
    from tools.computer_use.cua_backend_parse import _ingest_windows

    return _ingest_windows(raw)


def test_parse_xprop_net_active_window_standard_output():
    from tools.computer_use.cua_backend_parse import _parse_xprop_net_active_window

    raw = "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x503000b\n"
    assert _parse_xprop_net_active_window(raw) == 0x503000b


@pytest.mark.linux_only
def test_default_capture_prefers_x11_active_window_when_z_index_tied():
    """The ``_NET_ACTIVE_WINDOW`` tie-break is a Linux/X11-only branch of
    ``_select_capture_target``; run it where ``sys.platform`` really is
    linux instead of patching the branch selector."""
    from tools.computer_use.cua_backend_capture import _select_capture_target

    windows = _normalized_windows()

    with patch(
        "tools.computer_use.cua_backend_capture._linux_x11_active_window_id",
        return_value=84043449,
    ):
        target = _select_capture_target(windows, app_requested=False)

    assert target["title"] == "xr@10:~/hermes"
    assert target["window_id"] == 84043449


@pytest.mark.linux_only
def test_default_capture_skips_desktop_helper_when_active_window_unknown():
    """Even without _NET_ACTIVE_WINDOW, ding/Desktop helpers must not win (#54173).

    Linux-only: the helper-skipping pool filter is inside the
    ``sys.platform == "linux"`` branch."""
    from tools.computer_use.cua_backend_capture import _select_capture_target

    windows = _normalized_windows()

    with patch(
        "tools.computer_use.cua_backend_capture._linux_x11_active_window_id",
        return_value=None,
    ):
        target = _select_capture_target(windows, app_requested=False)

    # "Desktop Icons 1" is a shell helper window that captures as empty; with
    # the active window unknown, the first REAL app window wins list order.
    assert target["window_id"] == 60817412
    assert target["title"] == "zcode"


def test_linux_null_is_on_screen_is_treated_as_unknown_not_offscreen():
    """cua-driver 0.6.x may return JSON null for Linux is_on_screen (#54173)."""
    windows = _normalized_windows(LINUX_LIST_WINDOWS)

    assert windows[0]["off_screen"] is False
    assert windows[1]["off_screen"] is False
    assert windows[2]["off_screen"] is True


def test_explicit_app_capture_preserves_filtered_target_order():
    """When the caller filters first, target selection should not skip the match."""
    from tools.computer_use.cua_backend_capture import _select_capture_target

    chrome = _normalized_windows(LINUX_LIST_WINDOWS)[1]

    assert _select_capture_target([chrome], app_requested=True) == chrome
