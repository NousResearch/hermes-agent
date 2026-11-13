"""Coordinate-space hints for window captures, and the no-element-list refusal.

Live QA (cua-driver 0.21.0, macOS, Calculator): element bounds are desktop coordinates, but click
``coordinate=`` is window-local screenshot pixels. The old hint told the model to click in desktop space
("estimated scale ~16.7x") and every such click was refused as outside the window. After a vision capture
there is no element list, so ``click(element=N)`` reached the driver as a bare ``element_index``.
"""
import json
from unittest.mock import MagicMock

import pytest

from tools.computer_use.backend import ActionResult, CaptureResult, UIElement
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.tool import _capture_response, _classify_action_result

WINDOW = UIElement(index=0, role="AXWindow", label="Calculator", bounds=(2430, 539, 230, 408), app="Calculator")
BUTTON = UIElement(index=5, role="AXButton", label="7", bounds=(2464, 858, 40, 30), app="Calculator")


def _summary(cap):
    out = _capture_response(cap)
    return json.loads(out) if isinstance(out, str) else out


def test_window_capture_note_is_window_local_not_desktop_scale(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cap = CaptureResult(mode="ax", width=230, height=408, elements=[WINDOW, BUTTON], app="Calculator")
    out = _summary(cap)
    assert "window-local" in out["summary"]
    assert "2430" in out["summary"] and "539" in out["summary"]  # the origin to subtract
    assert "native desktop" not in out["summary"]
    assert "bounds_scale" not in out


def test_downscaled_window_capture_reports_window_scale(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    window = UIElement(index=0, role="AXWindow", label="W", bounds=(100, 50, 460, 816), app="A")
    cap = CaptureResult(mode="ax", width=230, height=408, elements=[window, BUTTON], app="A")
    out = _summary(cap)
    assert "window-local" in out["summary"]
    assert out["bounds_scale"] == pytest.approx(2.0)


def test_upscaled_window_capture_reports_window_scale(tmp_path, monkeypatch):
    # 2x Retina: a 460x816 image of a 230x408 window.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cap = CaptureResult(mode="ax", width=460, height=816, elements=[WINDOW, BUTTON], app="Calculator")
    out = _summary(cap)
    assert "window-local" in out["summary"]
    assert out["bounds_scale"] == pytest.approx(0.5)


def test_capture_without_window_element_keeps_desktop_scale(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    elems = [UIElement(index=1, role="AXButton", label="Close", bounds=(3730, 0, 69, 60), app="")]
    out = _summary(CaptureResult(mode="som", width=1455, height=791, elements=elems))
    assert "native desktop coordinates" in out["summary"]
    assert out["bounds_scale"] == pytest.approx(3799 / 1455, abs=0.01)


def _backend(call_tool):
    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = MagicMock()
    backend._session.call_tool.side_effect = call_tool
    backend._session.supports_input_property = lambda tool, prop: prop == "element_token"
    backend._session_id = "t"
    backend._snapshot_tokens = {}
    backend._set_active_target({"pid": 9, "window_id": 1})
    return backend


def _driver(elements):
    def call_tool(name, args):
        state = {"structuredContent": {"elements": elements}} if name == "get_window_state" else {}
        return {"data": "", "images": [], "image_mime_types": [], "structuredContent": None, "isError": False, **state}
    return call_tool


def test_element_click_without_element_list_is_refused_before_the_driver():
    backend = _backend(_driver([]))
    backend._snapshot_tokens = {}
    res = backend.click(element=3)
    assert res.ok is False and res.code == "no_elements"
    assert "mode='som'" in res.message and "coordinate" in res.message
    backend._session.call_tool.assert_not_called()
    assert _classify_action_result(res)["decision"] == "recapture"


def test_verdict_echoes_target_only_escalation():
    # cua-driver 0.21 sends escalation={"target", "reason"}; older drivers send "recommended".
    res = ActionResult(ok=True, action="type_text", effect="suspected_noop",
                       escalation={"target": "foreground", "reason": "delivery_failed"})
    assert _classify_action_result(res)["recommended"] == "foreground"
    old = ActionResult(ok=True, action="click", effect="suspected_noop", escalation={"recommended": "px"})
    assert _classify_action_result(old)["recommended"] == "px"


def test_element_click_after_capture_with_elements_still_reaches_the_driver():
    # Older drivers return elements without tokens and accept a bare element_index.
    backend = _backend(_driver([]))
    backend._snapshot_has_elements = True
    backend.click(element=3)
    assert backend._session.call_tool.call_args.args[1]["element_index"] == 3
