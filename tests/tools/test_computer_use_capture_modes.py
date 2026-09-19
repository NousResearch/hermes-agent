"""Capture-mode projection onto cua-driver's get_window_state output selectors.

Regression tests for #112639 (Kevin's capture-first patch): the requested Hermes
capture mode is passed down to the driver so it skips producer work whose output
the mode discards.

- ``ax``     -> ``include_screenshot: false`` (tree only; skip the grab)
- ``vision`` -> ``include_accessibility_tree: false`` (image only; skip the AX walk)
- ``som``    -> both (unchanged)
- older driver (selector not advertised) -> the existing full request (unchanged)

Also: ``capture()`` records ``capture_duration_ms`` (local timing metadata only),
and an intentionally omitted image in ``ax`` mode is not a failed capture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

# 8x8 PNG (transparent) — minimal provider-acceptable bytes that decode cleanly.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)

_NEW_DRIVER_PROPERTIES = {
    "include_screenshot": {"type": "boolean"},
    "include_accessibility_tree": {"type": "boolean"},
    "pid": {"type": "integer"},
    "window_id": {"type": "integer"},
}


def _make_backend(gws_properties: Dict, gws_result: Dict):
    """CuaDriverBackend with a stubbed session; exact pid/window_id targeting skips discovery."""
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.call_tool.return_value = gws_result
    session.supports_input_property.side_effect = (
        lambda tool, prop: tool == "get_window_state" and prop in gws_properties
    )
    session._has_tool.return_value = False  # straight to the get_window_state fallback
    session.capabilities_discovered = True
    backend._session = session
    return backend, session


def _tree_result(**overrides) -> Dict:
    return {
        "data": 'summary line\nAXWindow "Terminal"\n  AXButton "OK"\n',
        "images": [],
        "structuredContent": {
            "elements": [
                {"element_index": 0, "role": "AXButton", "label": "OK",
                 "frame": {"x": 10, "y": 20, "w": 30, "h": 40}}
            ]
        },
        "isError": False,
        **overrides,
    }


def _gws_call_args(session) -> Dict:
    return session.call_tool.call_args[0][1]


class TestGwsArgsProjection:
    def test_ax_skips_screenshot_when_advertised(self):
        backend, _ = _make_backend(_NEW_DRIVER_PROPERTIES, _tree_result())
        args = backend._gws_args("ax")
        assert args["include_screenshot"] is False
        assert "include_accessibility_tree" not in args

    def test_vision_skips_tree_when_advertised(self):
        backend, _ = _make_backend(_NEW_DRIVER_PROPERTIES, _tree_result())
        args = backend._gws_args("vision")
        assert args["include_accessibility_tree"] is False
        assert "include_screenshot" not in args

    def test_som_keeps_both_when_advertised(self):
        backend, _ = _make_backend(_NEW_DRIVER_PROPERTIES, _tree_result())
        args = backend._gws_args("som")
        assert "include_screenshot" not in args
        assert "include_accessibility_tree" not in args

    @pytest.mark.parametrize("mode", ["ax", "vision", "som"])
    def test_old_driver_keeps_full_request(self, mode: str):
        backend, _ = _make_backend({}, _tree_result())
        args = backend._gws_args(mode)
        assert args == {"pid": None, "window_id": None, "session": backend._session_id}


class TestCaptureProjectionEndToEnd:
    def test_ax_tree_only_is_not_a_failure(self):
        backend, session = _make_backend(_NEW_DRIVER_PROPERTIES, _tree_result())
        result = backend.capture(mode="ax", pid=123, window_id=456)

        args = _gws_call_args(session)
        assert args["include_screenshot"] is False
        # Intentionally omitted image: valid elements, no second fetch, no failure.
        assert result.mode == "ax"
        assert result.png_b64 is None
        assert [e.index for e in result.elements] == [0]
        assert result.window_title == "Terminal"
        session._call_tool_via_cli.assert_not_called()

    def test_ax_old_driver_still_fetches_screenshot(self):
        backend, session = _make_backend({}, _tree_result(images=[_PNG_B64]))
        result = backend.capture(mode="ax", pid=123, window_id=456)

        args = _gws_call_args(session)
        assert "include_screenshot" not in args
        assert result.png_b64 == _PNG_B64

    def test_vision_image_only_keeps_title_from_metadata(self):
        backend, session = _make_backend(
            _NEW_DRIVER_PROPERTIES,
            {
                "data": "",
                "images": [_PNG_B64],
                "image_mime_types": ["image/png"],
                "structuredContent": {"window_title": "Preview"},
                "isError": False,
            },
        )
        result = backend.capture(mode="vision", pid=123, window_id=456)

        args = _gws_call_args(session)
        assert args["include_accessibility_tree"] is False
        assert "include_screenshot" not in args
        assert result.mode == "vision"
        assert result.png_b64 == _PNG_B64
        assert result.elements == []
        assert result.window_title == "Preview"

    def test_vision_old_driver_uses_tree_title(self):
        backend, session = _make_backend(
            {},
            {
                "data": 'summary\nAXWindow "Terminal"\n',
                "images": [_PNG_B64],
                "structuredContent": {},
                "isError": False,
            },
        )
        result = backend.capture(mode="vision", pid=123, window_id=456)

        assert "include_accessibility_tree" not in _gws_call_args(session)
        assert result.window_title == "Terminal"


class TestCaptureTiming:
    def test_capture_records_duration_ms(self, monkeypatch):
        import tools.computer_use.cua_backend_capture as cap_mod

        ticks = iter([100.0, 100.042]).__next__
        monkeypatch.setattr(cap_mod, "time", SimpleNamespace(monotonic=ticks))
        backend, _ = _make_backend(_NEW_DRIVER_PROPERTIES, _tree_result())

        result = backend.capture(mode="ax", pid=123, window_id=456)

        assert result.capture_duration_ms == pytest.approx(42.0)
        assert result.mode == "ax"

    def test_capture_records_duration_without_clock_patch(self):
        backend, _ = _make_backend(_NEW_DRIVER_PROPERTIES, _tree_result())
        result = backend.capture(mode="som", pid=123, window_id=456)
        assert result.capture_duration_ms >= 0.0
