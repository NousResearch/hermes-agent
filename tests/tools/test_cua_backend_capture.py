"""Regression tests for cua-driver capture routing.

Recent cua-driver releases no longer expose the old standalone ``screenshot``
MCP tool. Hermes must request screenshots through ``get_window_state`` with the
appropriate ``capture_mode`` instead. cua-driver 0.6.0 also returns element
geometry in structured ``elements`` records; Hermes should prefer those over the
back-compat markdown tree so SOM overlays and element-index actions stay useful.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
    "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _window_list() -> Dict[str, Any]:
    return {
        "data": "",
        "images": [],
        "structuredContent": {
            "windows": [
                {
                    "app_name": "Safari",
                    "pid": 1234,
                    "window_id": 5678,
                    "is_on_screen": True,
                    "title": "Test Page",
                    "z_index": 0,
                }
            ]
        },
        "isError": False,
    }


def _make_backend(get_window_state_response: Dict[str, Any]):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = MagicMock()

    responses = {
        "list_windows": _window_list(),
        "get_window_state": get_window_state_response,
    }

    def _call_tool(name, args):
        if name in responses:
            return responses[name]
        return {
            "data": f"Unknown tool: {name}",
            "images": [],
            "structuredContent": None,
            "isError": True,
        }

    backend._session.call_tool.side_effect = _call_tool
    return backend


def _tool_call_names(backend):
    return [call.args[0] for call in backend._session.call_tool.call_args_list]


def _get_window_state_args(backend):
    for call in backend._session.call_tool.call_args_list:
        if call.args[0] == "get_window_state":
            return call.args[1]
    raise AssertionError("get_window_state was not called")


class TestCuaVisionCapture:
    def test_vision_uses_get_window_state_not_screenshot(self):
        backend = _make_backend({
            "data": "",
            "images": [_PNG_B64],
            "structuredContent": {
                "screenshot_width": 1920,
                "screenshot_height": 1080,
            },
            "isError": False,
        })

        result = backend.capture(mode="vision")

        tool_names = _tool_call_names(backend)
        assert "screenshot" not in tool_names
        assert "get_window_state" in tool_names
        assert _get_window_state_args(backend)["capture_mode"] == "vision"
        assert result.png_b64 == _PNG_B64
        assert result.width == 1920
        assert result.height == 1080
        assert result.png_bytes_len > 0


class TestCuaSomCapture:
    def test_som_passes_capture_mode_and_returns_png_plus_elements(self):
        backend = _make_backend({
            "data": "✅ Safari — 2 elements\n[1] AXButton \"Go\"\n[2] AXTextField \"Search\"",
            "images": [_PNG_B64],
            "structuredContent": {
                "screenshot_width": 1280,
                "screenshot_height": 800,
            },
            "isError": False,
        })

        result = backend.capture(mode="som")

        assert _get_window_state_args(backend)["capture_mode"] == "som"
        assert result.png_b64 == _PNG_B64
        assert result.width == 1280
        assert result.height == 800
        assert [element.index for element in result.elements] == [1, 2]

    def test_som_prefers_structured_elements_with_bounds(self):
        backend = _make_backend({
            "data": "✅ Safari — 1 elements\n[7] AXButton \"Search\"",
            "images": [_PNG_B64],
            "structuredContent": {
                "screenshot_width": 1280,
                "screenshot_height": 800,
                "elements": [
                    {
                        "element_index": 7,
                        "role": "AXButton",
                        "label": "Search",
                        "frame": {"x": 101.2, "y": 202.6, "w": 88.0, "h": 34.0},
                    }
                ],
            },
            "isError": False,
        })

        result = backend.capture(mode="som")

        assert len(result.elements) == 1
        assert result.elements[0].index == 7
        assert result.elements[0].label == "Search"
        assert result.elements[0].bounds == (101, 203, 88, 34)


class TestCuaAxCapture:
    def test_ax_still_returns_elements_without_png(self):
        backend = _make_backend({
            "data": "✅ Safari — 2 elements\n[1] AXButton \"OK\"\n[2] AXStaticText \"Hello\"",
            "images": [],
            "structuredContent": None,
            "isError": False,
        })

        result = backend.capture(mode="ax")

        assert _get_window_state_args(backend)["capture_mode"] == "ax"
        assert result.png_b64 is None
        assert result.png_bytes_len == 0
        assert result.width == 0
        assert result.height == 0
        assert [element.label for element in result.elements] == ["OK", "Hello"]

    def test_ax_prefers_structured_elements_without_png(self):
        backend = _make_backend({
            "data": "✅ Safari — 1 elements\n[3] AXTextField \"Address\"",
            "images": [],
            "structuredContent": {
                "elements": [
                    {
                        "element_index": 3,
                        "role": "AXTextField",
                        "label": "Address",
                        "frame": {"x": 20, "y": 40, "w": 600, "h": 28},
                        "enabled": True,
                    }
                ],
            },
            "isError": False,
        })

        result = backend.capture(mode="ax")

        assert result.png_b64 is None
        assert len(result.elements) == 1
        assert result.elements[0].role == "AXTextField"
        assert result.elements[0].bounds == (20, 40, 600, 28)
        assert result.elements[0].attributes["enabled"] is True

    def test_missing_structured_content_keeps_dimensions_zero(self):
        backend = _make_backend({
            "data": "✅ Safari — 0 elements\n",
            "images": [],
            "structuredContent": None,
            "isError": False,
        })

        result = backend.capture(mode="ax")

        assert result.width == 0
        assert result.height == 0
