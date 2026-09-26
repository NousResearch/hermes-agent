"""Capture mode should suppress producer work only when cua-driver advertises the selector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


def _backend(monkeypatch, *, properties=(), result=None, ax_max=0):
    from tools.computer_use import cua_backend as cb
    from tools.computer_use.cua_backend import CuaDriverBackend

    monkeypatch.setattr(cb, "_cua_configured_ax_max_elements", lambda: ax_max)
    backend = CuaDriverBackend()
    session = MagicMock()
    session.supports_input_property.side_effect = (
        lambda tool, prop: tool == "get_window_state" and prop in set(properties)
    )
    session._has_tool.return_value = False
    session.capabilities_discovered = True
    session.call_tool.return_value = result or {
        "data": 'summary\nAXWindow "Terminal"\n  AXButton "OK"\n',
        "images": [],
        "structuredContent": {
            "elements": [{
                "element_index": 0,
                "role": "AXButton",
                "label": "OK",
                "frame": {"x": 10, "y": 20, "w": 30, "h": 40},
            }]
        },
        "isError": False,
    }
    backend._session = session
    return backend, session


def _gws_args(session):
    calls = [call for call in session.call_tool.call_args_list if call.args and call.args[0] == "get_window_state"]
    assert calls
    return calls[-1].args[1]


def test_projection_preserves_current_ax_walk_bound(monkeypatch):
    backend, _ = _backend(
        monkeypatch,
        properties={"include_screenshot", "include_accessibility_tree"},
        ax_max=200,
    )

    ax = backend._gws_args("ax")
    vision = backend._gws_args("vision")
    som = backend._gws_args("som")

    assert ax["max_elements"] == vision["max_elements"] == som["max_elements"] == 200
    assert ax["include_screenshot"] is False
    assert vision["include_accessibility_tree"] is False
    assert "include_accessibility_tree" not in ax
    assert "include_screenshot" not in vision
    assert "include_screenshot" not in som and "include_accessibility_tree" not in som


@pytest.mark.parametrize("mode", ["ax", "vision", "som"])
def test_old_driver_keeps_full_request(monkeypatch, mode):
    backend, _ = _backend(monkeypatch, properties=(), ax_max=0)

    args = backend._gws_args(mode)

    assert args == {"pid": None, "window_id": None, "session": backend._session_id}


def test_ax_capture_can_return_tree_without_screenshot(monkeypatch):
    backend, session = _backend(
        monkeypatch,
        properties={"include_screenshot"},
        ax_max=200,
    )

    result = backend.capture(mode="ax", pid=123, window_id=456)

    args = _gws_args(session)
    assert args["include_screenshot"] is False
    assert args["max_elements"] == 200
    assert result.mode == "ax"
    assert result.png_b64 is None
    assert result.window_title == "Terminal"
    assert len(result.elements) == 1


def test_vision_capture_uses_structured_title_when_tree_is_omitted(monkeypatch):
    backend, session = _backend(
        monkeypatch,
        properties={"include_accessibility_tree"},
        result={
            "data": "",
            "images": [_PNG_B64],
            "image_mime_types": ["image/png"],
            "structuredContent": {"window_title": "Preview"},
            "isError": False,
        },
    )

    result = backend.capture(mode="vision", pid=123, window_id=456)

    assert _gws_args(session)["include_accessibility_tree"] is False
    assert result.png_b64 == _PNG_B64
    assert result.elements == []
    assert result.window_title == "Preview"
