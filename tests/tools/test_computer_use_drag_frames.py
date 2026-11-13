"""Element-index drag must send window-local screenshot pixels.

``get_window_state`` element frames are screen rectangles. The pinned cua-driver
0.21.0 ``drag`` tool reads ``from_x``/``from_y`` as offsets from the window
screenshot and adds the window origin itself. Sending the screen centre with
``pid`` and ``window_id`` misses by that origin.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_parse import _parse_key_combo


def _backend():
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {
        "data": "ok", "images": [], "image_mime_types": [],
        "structuredContent": None, "isError": False,
    }
    backend._session.supports_capability = lambda cap, tool=None: True
    backend._active_pid = 111
    backend._active_window_id = 222
    return backend


def _drag_args(backend):
    name, args = backend._session.call_tool.call_args.args
    assert name == "drag"
    return args


def test_window_at_screen_origin_uses_frame_centers():
    backend = _backend()
    backend._snapshot_bounds = {3: (10, 20, 80, 40), 7: (100, 50, 20, 20)}
    backend._snapshot_window_frame = (0, 0, 1000, 800)
    assert backend.drag(from_element=3, to_element=7).ok is True
    args = _drag_args(backend)
    assert "from_element" not in args and "to_element" not in args
    assert (args["from_x"], args["from_y"]) == (50, 40)
    assert (args["to_x"], args["to_y"]) == (110, 60)
    assert args["pid"] == 111 and args["window_id"] == 222


def test_screen_frame_centers_are_shifted_by_the_window_origin():
    """A window whose visible frame starts at screen (507, 300) must not be
    dragged at the raw screen centres (598, 381) and (838, 521)."""
    backend = _backend()
    backend._snapshot_bounds = {0: (528, 361, 140, 40), 1: (768, 501, 140, 40)}
    backend._snapshot_window_frame = (507, 300, 900, 700)
    assert backend.drag(from_element=0, to_element=1).ok is True
    args = _drag_args(backend)
    assert (args["from_x"], args["from_y"]) == (91, 81)
    assert (args["to_x"], args["to_y"]) == (331, 221)
    assert "scope" not in args


def test_delivered_screenshot_scale_is_applied_after_the_origin():
    backend = _backend()
    backend._snapshot_bounds = {4: (140, 220, 20, 20)}
    backend._snapshot_window_frame = (100, 200, 400, 300)
    backend._snapshot_screenshot_size = (800, 600)
    assert backend.drag(from_element=4, to_element=4).ok is True
    args = _drag_args(backend)
    assert (args["from_x"], args["from_y"]) == (100, 60)
    assert (args["to_x"], args["to_y"]) == (100, 60)


def test_element_drag_without_frames_does_not_call_the_driver():
    backend = _backend()
    backend._snapshot_bounds = {3: (0, 0, 0, 0), 7: (1, 2, 0, 8)}
    backend._snapshot_window_frame = (507, 300, 900, 700)
    result = backend.drag(from_element=3, to_element=7)
    assert result.ok is False
    backend._session.call_tool.assert_not_called()


def test_element_drag_without_a_window_frame_does_not_call_the_driver():
    backend = _backend()
    backend._snapshot_bounds = {0: (528, 361, 140, 40), 1: (768, 501, 140, 40)}
    backend._snapshot_window_frame = None
    result = backend.drag(from_element=0, to_element=1)
    assert result.ok is False
    backend._session.call_tool.assert_not_called()


def test_explicit_coordinates_skip_the_screen_frame_conversion():
    backend = _backend()
    backend._snapshot_bounds = {}
    backend._snapshot_window_frame = (507, 300, 900, 700)
    assert backend.drag(from_element=3, to_element=7, from_xy=(1, 2), to_xy=(3, 4)).ok is True
    args = _drag_args(backend)
    assert (args["from_x"], args["from_y"], args["to_x"], args["to_y"]) == (1, 2, 3, 4)


def _ok(structured=None):
    return {
        "data": "ok", "images": [], "image_mime_types": [],
        "structuredContent": structured, "isError": False,
    }


def test_capture_then_drag_uses_list_windows_bounds():
    backend = _backend()
    windows = {"windows": [{
        "app_name": "Demo", "pid": 9, "window_id": 1,
        "is_on_screen": True, "title": "Drag", "z_index": 0,
        "bounds": {"x": 507, "y": 300, "width": 900, "height": 700},
    }]}

    def fake_call_tool(name, args):
        if name == "list_windows":
            return _ok(windows)
        if name == "get_window_state":
            return _ok({"elements": [
                {"element_index": 0, "role": "Button", "label": "Drag source",
                 "frame": {"x": 528, "y": 361, "w": 140, "h": 40}},
                {"element_index": 1, "role": "Button", "label": "Drop target",
                 "frame": {"x": 768, "y": 501, "w": 140, "h": 40}},
            ]})
        return _ok()

    backend._session.call_tool.side_effect = fake_call_tool
    backend.capture(mode="ax")
    assert backend._snapshot_window_frame == (507, 300, 900, 700)
    assert backend.drag(from_element=0, to_element=1).ok is True
    args = _drag_args(backend)
    assert (args["from_x"], args["from_y"]) == (91, 81)
    assert (args["to_x"], args["to_y"]) == (331, 221)
    assert args["pid"] == 9 and args["window_id"] == 1


def test_capture_prefers_window_bounds_and_screenshot_scale():
    backend = _backend()
    windows = {"windows": [{
        "app_name": "Demo", "pid": 9, "window_id": 1,
        "is_on_screen": True, "title": "Drag", "z_index": 0,
        "bounds": {"x": 0, "y": 0, "width": 100, "height": 100},
    }]}

    def fake_call_tool(name, args):
        if name == "list_windows":
            return _ok(windows)
        if name == "get_window_state":
            return _ok({
                "window_bounds": {"x": 100.0, "y": 200.0, "width": 400.0, "height": 300.0},
                "screenshot_width": 800,
                "screenshot_height": 600,
                "elements": [
                    {"element_index": 4, "role": "AXButton", "label": "Thumb",
                     "frame": {"x": 140, "y": 220, "w": 20, "h": 20}},
                ],
            })
        return _ok()

    backend._session.call_tool.side_effect = fake_call_tool
    backend.capture(mode="ax")
    assert backend._snapshot_window_frame == (100, 200, 400, 300)
    assert backend._snapshot_screenshot_size == (800, 600)
    assert backend.drag(from_element=4, to_element=4).ok is True
    args = _drag_args(backend)
    assert (args["from_x"], args["from_y"]) == (100, 60)


def test_win_combo_is_a_hotkey_and_minus_is_a_key():
    assert _parse_key_combo("ctrl-alt-delete") == ("delete", ["ctrl", "option"])
    assert _parse_key_combo("cmd+-") == ("-", ["cmd"])
    assert _parse_key_combo("windows+d") == ("d", ["win"])
    assert _parse_key_combo("super+d") == ("d", ["win"])

    backend = _backend()
    assert backend.key("win+d").ok is True
    name, args = backend._session.call_tool.call_args.args
    assert name == "hotkey"
    assert args["keys"] == ["win", "d"]

    assert backend.key("cmd+-").ok is True
    name, args = backend._session.call_tool.call_args.args
    assert name == "hotkey"
    assert args["keys"] == ["cmd", "-"]

    backend._session.call_tool.reset_mock()
    assert backend.key("banana+d").ok is False
    backend._session.call_tool.assert_not_called()
