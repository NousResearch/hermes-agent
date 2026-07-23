"""Regression tests for Linux/X11 capture target selection (#58026)."""

from __future__ import annotations

from unittest.mock import patch

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


def _normalized_windows(raw=ISSUE_58026_WINDOWS):
    from tools.computer_use.cua_backend import _ingest_windows

    return _ingest_windows(raw)


def test_parse_xprop_net_active_window_standard_output():
    from tools.computer_use.cua_backend import _parse_xprop_net_active_window

    raw = "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x503000b\n"
    assert _parse_xprop_net_active_window(raw) == 0x503000b


def test_parse_xprop_net_active_window_bare_hex_fallback():
    from tools.computer_use.cua_backend import _parse_xprop_net_active_window

    assert _parse_xprop_net_active_window("active=0xABcdef01") == 0xABCDEF01


def test_parse_xprop_net_active_window_rejects_unparseable():
    from tools.computer_use.cua_backend import _parse_xprop_net_active_window

    assert _parse_xprop_net_active_window("") is None
    assert _parse_xprop_net_active_window("_NET_ACTIVE_WINDOW(WINDOW): none") is None
    assert _parse_xprop_net_active_window("window id # not-a-hex") is None


def test_default_capture_prefers_x11_active_window_when_z_index_tied():
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()

    with patch("tools.computer_use.cua_backend.sys.platform", "linux"), patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
        return_value=84043449,
    ):
        target = _select_capture_target(windows, app_requested=False)

    assert target["title"] == "xr@10:~/hermes"
    assert target["window_id"] == 84043449


def test_default_capture_falls_back_to_list_order_when_active_window_unknown():
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()

    with patch("tools.computer_use.cua_backend.sys.platform", "linux"), patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
        return_value=None,
    ):
        target = _select_capture_target(windows, app_requested=False)

    # Without informative z-order or active-window, keep list order (caller
    # sorts higher-z frontmost; here all tied so first on-screen wins).
    assert target["window_id"] == 33554439
    assert target["title"] == "Desktop Icons 1"


def test_default_capture_keeps_higher_z_index_when_ordering_informative():
    """Active-window fallback must not override a real frontmost z_index."""
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows(
        [
            {
                "app_name": "",
                "pid": 1,
                "window_id": 10,
                "title": "back",
                "is_on_screen": True,
                "z_index": 1,
            },
            {
                "app_name": "",
                "pid": 2,
                "window_id": 20,
                "title": "front",
                "is_on_screen": True,
                "z_index": 5,
            },
        ]
    )
    # Mirror _load_windows: higher z_index is frontmost.
    windows.sort(key=lambda w: w["z_index"], reverse=True)

    with patch("tools.computer_use.cua_backend.sys.platform", "linux"), patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
        return_value=10,
    ) as active:
        target = _select_capture_target(windows, app_requested=False)

    assert target["window_id"] == 20
    assert target["title"] == "front"
    active.assert_not_called()


def test_explicit_app_capture_skips_active_window_fallback():
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()

    with patch("tools.computer_use.cua_backend.sys.platform", "linux"), patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
        return_value=84043449,
    ) as active:
        target = _select_capture_target(windows, app_requested=True)

    assert target["window_id"] == 33554439
    active.assert_not_called()


def test_exact_target_selection_skips_active_window_fallback():
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()[:1]

    with patch("tools.computer_use.cua_backend.sys.platform", "linux"), patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
        return_value=84043449,
    ) as active:
        target = _select_capture_target(
            windows, app_requested=False, exact_target=True
        )

    assert target["window_id"] == 33554439
    active.assert_not_called()


def test_exact_pid_window_capture_does_not_probe_x11_active_window():
    """capture_after / exact pid+window_id must not pay for an xprop probe."""
    from unittest.mock import MagicMock

    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.call_tool.return_value = {
        "data": "✅ Chrome — 0 elements",
        "images": [],
        "structuredContent": {"elements": []},
        "isError": False,
    }
    backend._session = session

    with patch("tools.computer_use.cua_backend.sys.platform", "linux"), patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
        return_value=999,
    ) as active:
        backend.capture(mode="ax", pid=1816017, window_id=60817412)

    assert backend._active_pid == 1816017
    assert backend._active_window_id == 60817412
    active.assert_not_called()
    assert all(c.args[0] != "list_windows" for c in session.call_tool.call_args_list)
