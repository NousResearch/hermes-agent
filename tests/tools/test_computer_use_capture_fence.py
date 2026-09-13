"""A frame captured across a human takeover never leaves the process: the lease fence runs as soon as the
backend hands the frame back, before it is persisted to the media cache, spilled, or routed to aux vision."""

from __future__ import annotations

import base64
import json

import pytest

from tools.bot_desktop import lease
from tools.computer_use.backend import ActionResult, CaptureResult


@pytest.fixture(autouse=True)
def _fresh_lease():
    lease._reset_for_tests()
    yield
    lease._reset_for_tests()


class _TakeoverBackend:
    """Driver whose capture returns while a take-over / hand-back cycle happened underneath it."""
    _last_target = None
    _last_app = None

    def capture(self, **kw):
        lease.acquire("human")
        lease.release("human")
        return CaptureResult(mode="som", width=64, height=64, png_b64=base64.b64encode(b"SECRETPNG").decode(), elements=[],
                             app="Bank", window_title="login")

    def click(self, **kw):
        return ActionResult(ok=True, action="click")


def _spy_sinks(monkeypatch, tool):
    leaked: list = []
    monkeypatch.setattr(tool, "_persist_capture_image", lambda cap: leaked.append(("persist", cap)))
    monkeypatch.setattr(tool, "_spill_elements_to_file", lambda cap: leaked.append(("spill", cap)))
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: leaked.append(("aux-decide",)) or True)
    monkeypatch.setattr(tool, "_route_capture_through_aux_vision", lambda cap, summary, **kw: leaked.append(("aux", cap)))
    return leaked


@pytest.mark.parametrize("args", [{"action": "capture"}, {"action": "click", "coordinate": [1, 1], "capture_after": True}])
def test_frame_captured_across_a_takeover_is_dropped_before_any_sink(monkeypatch, args):
    from tools.computer_use import tool

    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": _TakeoverBackend())
    monkeypatch.setattr(tool, "_request_approval", lambda *a, **k: None)
    leaked = _spy_sinks(monkeypatch, tool)
    res = json.loads(tool.handle_computer_use(args))
    assert res["code"] == "human_has_control", res
    assert leaked == [], f"the human's frame reached a sink: {leaked}"
    assert "SECRETPNG" not in json.dumps(res)
