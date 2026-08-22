"""Regression tests for Linux/X11 capture target selection (#58026, #54173)."""

from __future__ import annotations

import base64
from io import BytesIO
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
    from tools.computer_use.cua_backend import _ingest_windows

    return _ingest_windows(raw)


def test_parse_xprop_net_active_window_standard_output():
    from tools.computer_use.cua_backend import _parse_xprop_net_active_window

    raw = "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x503000b\n"
    assert _parse_xprop_net_active_window(raw) == 0x503000b


@pytest.mark.linux_only
def test_default_capture_prefers_x11_active_window_when_z_index_tied():
    """The ``_NET_ACTIVE_WINDOW`` tie-break is a Linux/X11-only branch of
    ``_select_capture_target``; run it where ``sys.platform`` really is
    linux instead of patching the branch selector."""
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()

    with patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
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
    from tools.computer_use.cua_backend import _select_capture_target

    windows = _normalized_windows()

    with patch(
        "tools.computer_use.cua_backend._linux_x11_active_window_id",
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
    from tools.computer_use.cua_backend import _select_capture_target

    chrome = _normalized_windows(LINUX_LIST_WINDOWS)[1]

    assert _select_capture_target([chrome], app_requested=True) == chrome


def _desktop_png(width=300, height=150):
    from PIL import Image

    image = Image.new("RGB", (width, height), "navy")
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii")


class _GenericWaylandSession:
    capabilities_discovered = True

    def __init__(
        self,
        *,
        pixel_size=(300, 150),
        logical_size=(200, 100),
        screen_size_mode="ok",
        desktop_target=True,
        windows=None,
    ):
        self.png_b64 = _desktop_png(*pixel_size)
        self.logical_size = logical_size
        self.screen_size_mode = screen_size_mode
        self.desktop_target = desktop_target
        self.windows = windows or []
        self.calls = []

    def _has_tool(self, name):
        tools = {
            "get_desktop_state", "click", "drag", "scroll", "get_window_state",
        }
        if self.screen_size_mode != "missing":
            tools.add("get_screen_size")
        return name in tools

    def supports_input_property(self, tool, prop):
        if prop == "target":
            return self.desktop_target and tool in {"click", "drag", "scroll"}
        return (tool, prop) in {
            ("click", "count"),
            ("scroll", "x"),
            ("scroll", "y"),
        }

    def supports_capability(self, capability, tool=None):
        return False

    def _call_tool_via_cli(self, name, args, timeout):
        assert name == "list_windows"
        return self._result(structured={"windows": self.windows})

    @staticmethod
    def _result(*, structured=None, images=None, data="ok", is_error=False):
        images = images or []
        return {
            "data": data,
            "images": images,
            "image_mime_types": ["image/png"] if images else [],
            "structuredContent": structured or {},
            "isError": is_error,
        }

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, dict(args)))
        if name == "list_windows":
            return self._result(structured={"windows": self.windows})
        if name == "get_desktop_state":
            return self._result(images=[self.png_b64])
        if name == "get_screen_size":
            if self.screen_size_mode == "error":
                raise RuntimeError("screen size unavailable")
            if self.screen_size_mode == "malformed":
                return self._result(
                    structured={"width": float("nan"), "height": 100},
                )
            return self._result(structured={
                "width": self.logical_size[0],
                "height": self.logical_size[1],
            })
        if name == "get_window_state":
            return self._result(
                images=[self.png_b64],
                structured={"elements": [{
                    "element_index": 1,
                    "role": "AXButton",
                    "label": "Open",
                    "frame": {"x": 10, "y": 20, "w": 30, "h": 40},
                }]},
            )
        return self._result()


@pytest.mark.linux_only
def test_generic_wayland_desktop_capture_is_visual_only(monkeypatch):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession()
    backend = CuaDriverBackend()
    backend._session = session

    capture = backend.capture(mode="som", app="desktop")

    assert (capture.width, capture.height) == (300, 150)
    assert capture.png_b64 == session.png_b64
    assert capture.elements == []
    assert capture.app == "desktop"
    assert backend._active_desktop_geometry == {
        "pixel_left": 0.0,
        "pixel_top": 0.0,
        "pixel_width": 300.0,
        "pixel_height": 150.0,
        "logical_left": 0.0,
        "logical_top": 0.0,
        "logical_width": 200.0,
        "logical_height": 100.0,
        "scale_x": 1.5,
        "scale_y": 1.5,
    }
    assert [name for name, _ in session.calls] == [
        "get_desktop_state", "get_screen_size",
    ]


@pytest.mark.linux_only
@pytest.mark.parametrize(
    ("pixel_size", "logical_size", "point", "expected", "click_count"),
    [
        ((300, 150), (200, 100), (150, 75), (100, 50), 1),
        ((400, 200), (200, 100), (300, 100), (150, 50), 2),
    ],
)
def test_generic_wayland_desktop_scales_click_and_double_click(
    monkeypatch, pixel_size, logical_size, point, expected, click_count,
):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession(
        pixel_size=pixel_size, logical_size=logical_size,
    )
    backend = CuaDriverBackend()
    backend._session = session
    backend.capture(mode="som", app="screen")

    clicked = backend.click(x=point[0], y=point[1], click_count=click_count)

    assert clicked.ok is True
    action, args = session.calls[-1]
    assert action == "click"
    assert (args["x"], args["y"]) == expected
    assert args["target"] == {"kind": "desktop", "display_id": "primary"}
    assert args.get("count", 1) == click_count


@pytest.mark.linux_only
def test_generic_wayland_desktop_scales_drag_and_scroll(monkeypatch):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession()
    backend = CuaDriverBackend()
    backend._session = session
    backend.capture(mode="som", app="screen")

    dragged = backend.drag(from_xy=(15, 30), to_xy=(285, 135))
    scrolled = backend.scroll(direction="down", amount=4, x=75, y=45)

    assert dragged.ok is True
    assert scrolled.ok is True
    drag_args = session.calls[-2][1]
    assert {key: drag_args[key] for key in (
        "from_x", "from_y", "to_x", "to_y", "target",
    )} == {
        "from_x": 10,
        "from_y": 20,
        "to_x": 190,
        "to_y": 90,
        "target": {"kind": "desktop", "display_id": "primary"},
    }
    assert (session.calls[-1][1]["x"], session.calls[-1][1]["y"]) == (50, 30)


@pytest.mark.linux_only
@pytest.mark.parametrize("action", ["click", "drag", "scroll"])
def test_generic_wayland_desktop_rejects_out_of_bounds_pixels(monkeypatch, action):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession()
    backend = CuaDriverBackend()
    backend._session = session
    backend.capture(mode="som", app="screen")

    if action == "click":
        blocked = backend.click(x=300, y=75)
    elif action == "drag":
        blocked = backend.drag(from_xy=(0, 0), to_xy=(-1, 149))
    else:
        blocked = backend.scroll(direction="down", x=150, y=float("inf"))

    assert blocked.ok is False
    assert blocked.code == "desktop_coordinate_mapping_invalid"
    assert not any(name == action for name, _ in session.calls)


@pytest.mark.linux_only
@pytest.mark.parametrize("screen_size_mode", ["missing", "malformed", "error"])
def test_generic_wayland_desktop_invalid_screen_size_blocks_pointer_input(
    monkeypatch, screen_size_mode,
):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession(screen_size_mode=screen_size_mode)
    backend = CuaDriverBackend()
    backend._session = session
    capture = backend.capture(mode="som", app="screen")

    blocked = backend.click(x=150, y=75)

    assert (capture.width, capture.height) == (300, 150)
    assert blocked.ok is False
    assert blocked.code == "desktop_coordinate_mapping_invalid"
    assert "get_screen_size" in blocked.message
    assert not any(name == "click" for name, _ in session.calls)


@pytest.mark.linux_only
def test_generic_wayland_desktop_rejects_inconsistent_scale(monkeypatch):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession()
    backend = CuaDriverBackend()
    backend._session = session
    backend.capture(mode="som", app="screen")
    backend._active_desktop_geometry["scale_x"] = 2.0

    blocked = backend.click(x=150, y=75)

    assert blocked.ok is False
    assert blocked.code == "desktop_coordinate_mapping_invalid"
    assert not any(name == "click" for name, _ in session.calls)


@pytest.mark.linux_only
def test_generic_wayland_desktop_requires_advertised_input_target(monkeypatch):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    session = _GenericWaylandSession(desktop_target=False)
    backend = CuaDriverBackend()
    backend._session = session
    backend.capture(mode="som", app="screen")

    blocked = backend.click(x=150, y=75)

    assert blocked.ok is False
    assert blocked.code == "desktop_target_unsupported"
    assert "does not advertise desktop targets" in blocked.message
    assert not any(name == "click" for name, _ in session.calls)


@pytest.mark.linux_only
def test_generic_wayland_normal_app_keeps_semantic_window_route(monkeypatch):
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    window = {
        "app_name": "Terminal",
        "pid": 4242,
        "window_id": 99,
        "title": "Shell",
        "is_on_screen": True,
        "z_index": 1,
    }
    session = _GenericWaylandSession(windows=[window])
    backend = CuaDriverBackend()
    backend._session = session

    capture = backend.capture(mode="som", app="Terminal")
    clicked = backend.click(element=1)

    assert len(capture.elements) == 1
    assert capture.elements[0].label == "Open"
    assert clicked.ok is True
    assert backend._active_pid == 4242
    assert backend._active_window_id == 99
    assert not any(name == "get_desktop_state" for name, _ in session.calls)
    assert session.calls[-1][0] == "click"
    assert session.calls[-1][1]["element_index"] == 1
