"""Regression tests for issue #39242 — cua-driver vision/SOM capture.

Three layers:
1. Vision mode calls non-existent 'screenshot' tool → should use get_window_state
2. SOM mode doesn't pass capture_mode → should pass capture_mode='som'
3. Width/height not populated from structuredContent → should extract screenshot_width/height

These tests mock the MCP session to simulate cua-driver 0.5.x responses.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, call

import pytest


# Minimal fake PNG base64 (1x1 transparent PNG)
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
    "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _make_cua_backend(
    list_windows_response: Dict[str, Any],
    get_window_state_response: Dict[str, Any] = None,
    screenshot_response: Dict[str, Any] = None,
):
    """Construct a CuaDriverBackend with a mocked MCP session.

    The session.call_tool mock uses side_effect to return different responses
    for 'list_windows' vs 'get_window_state' vs 'screenshot' calls.
    """
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = MagicMock()

    call_map = {
        "list_windows": list_windows_response,
    }
    if get_window_state_response is not None:
        call_map["get_window_state"] = get_window_state_response
    if screenshot_response is not None:
        call_map["screenshot"] = screenshot_response

    def _call_tool(name, args):
        if name in call_map:
            return call_map[name]
        # Simulate cua-driver returning error for unknown tools
        return {
            "data": f"Unknown tool: {name}",
            "images": [],
            "structuredContent": None,
            "isError": True,
        }

    backend._session.call_tool.side_effect = _call_tool
    return backend


def _window_list():
    """Standard window list for tests."""
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


# ---------------------------------------------------------------------------
# Layer 1: Vision mode must use get_window_state, not screenshot
# ---------------------------------------------------------------------------


class TestVisionModeCapture:
    """Vision mode should call get_window_state with capture_mode='vision',
    not the non-existent 'screenshot' tool."""

    def test_vision_calls_get_window_state_not_screenshot(self):
        """Vision mode must call get_window_state, not 'screenshot'."""
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1920,
                    "screenshot_height": 1080,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="vision")

        # Verify call_tool was called with 'get_window_state', NOT 'screenshot'
        calls = backend._session.call_tool.call_args_list
        tool_names = [c[0][0] for c in calls]
        assert "screenshot" not in tool_names, (
            f"Vision mode must NOT call 'screenshot' tool; got calls: {tool_names}"
        )
        assert "get_window_state" in tool_names, (
            f"Vision mode must call 'get_window_state'; got calls: {tool_names}"
        )

    def test_vision_passes_capture_mode_vision(self):
        """Vision mode must pass capture_mode='vision' to get_window_state."""
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1920,
                    "screenshot_height": 1080,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="vision")

        # Find the get_window_state call
        gws_call = None
        for c in backend._session.call_tool.call_args_list:
            if c[0][0] == "get_window_state":
                gws_call = c
                break

        assert gws_call is not None, "get_window_state was not called"
        args = gws_call[0][1]  # second positional arg is the args dict
        assert args.get("capture_mode") == "vision", (
            f"Expected capture_mode='vision', got args: {args}"
        )

    def test_vision_returns_png(self):
        """Vision mode must return the screenshot PNG from get_window_state."""
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1920,
                    "screenshot_height": 1080,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="vision")

        assert result.png_b64 == _PNG_B64, (
            "Vision mode must return the PNG from get_window_state images"
        )
        assert result.png_bytes_len > 0, (
            f"png_bytes_len must be > 0 when PNG is present; got {result.png_bytes_len}"
        )


# ---------------------------------------------------------------------------
# Layer 2: SOM mode must pass capture_mode='som'
# ---------------------------------------------------------------------------


class TestSOMModeCapture:
    """SOM mode should pass capture_mode='som' to get_window_state so it
    returns both AX tree AND screenshot."""

    def test_som_passes_capture_mode_som(self):
        """SOM mode must pass capture_mode='som' to get_window_state."""
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "✅ Safari — 42 elements, turn 1\n[0] AXButton \"Click me\"\n[1] AXTextField \"Search\"",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1280,
                    "screenshot_height": 800,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="som")

        gws_call = None
        for c in backend._session.call_tool.call_args_list:
            if c[0][0] == "get_window_state":
                gws_call = c
                break

        assert gws_call is not None
        args = gws_call[0][1]
        assert args.get("capture_mode") == "som", (
            f"Expected capture_mode='som', got args: {args}"
        )

    def test_som_returns_png_when_capture_mode_set(self):
        """SOM mode with capture_mode='som' returns both AX tree and PNG."""
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "✅ Safari — 2 elements\n[0] AXButton \"Go\"\n[1] AXStaticText \"Hello\"",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1280,
                    "screenshot_height": 800,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="som")

        assert result.png_b64 == _PNG_B64, "SOM mode must return PNG"
        assert len(result.elements) == 2, f"Expected 2 elements, got {len(result.elements)}"


# ---------------------------------------------------------------------------
# Layer 3: Width/height populated from structuredContent
# ---------------------------------------------------------------------------


class TestWidthHeightFromStructuredContent:
    """Width and height must be extracted from structuredContent when
    get_window_state returns screenshot_width/screenshot_height."""

    def test_vision_extracts_dimensions(self):
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1920,
                    "screenshot_height": 1080,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="vision")

        assert result.width == 1920, f"Expected width=1920, got {result.width}"
        assert result.height == 1080, f"Expected height=1080, got {result.height}"

    def test_som_extracts_dimensions(self):
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "✅ Safari — 0 elements\n",
                "images": [_PNG_B64],
                "structuredContent": {
                    "screenshot_width": 1280,
                    "screenshot_height": 800,
                },
                "isError": False,
            },
        )

        result = backend.capture(mode="som")

        assert result.width == 1280, f"Expected width=1280, got {result.width}"
        assert result.height == 800, f"Expected height=800, got {result.height}"

    def test_no_structured_content_leaves_dimensions_zero(self):
        """When structuredContent is missing, dimensions stay 0 (no crash)."""
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "✅ Safari — 0 elements\n",
                "images": [],
                "structuredContent": None,
                "isError": False,
            },
        )

        result = backend.capture(mode="ax")

        assert result.width == 0
        assert result.height == 0


# ---------------------------------------------------------------------------
# AX mode should be unchanged
# ---------------------------------------------------------------------------


class TestAXModeUnchanged:
    """AX mode should not pass capture_mode (or pass 'ax') and return
    no PNG — existing behavior preserved."""

    def test_ax_no_png(self):
        backend = _make_cua_backend(
            list_windows_response=_window_list(),
            get_window_state_response={
                "data": "✅ Safari — 5 elements\n[0] AXButton \"OK\"\n[1] AXStaticText \"Hello\"",
                "images": [],
                "structuredContent": None,
                "isError": False,
            },
        )

        result = backend.capture(mode="ax")

        assert result.png_b64 is None or result.png_b64 == ""
        assert result.png_bytes_len == 0
        assert len(result.elements) == 2
