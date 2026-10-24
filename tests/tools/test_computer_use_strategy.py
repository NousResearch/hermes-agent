"""strategy="auto" | "a11y" | "event": which addressing form an input action may use."""

import json

from tools.computer_use.backend import ActionResult
from tools.computer_use.tool import _dispatch


class FakeBackend:
    def __init__(self):
        self.clicked_args = []
        self.dragged_args = []
        self.scrolled_args = []
        self._last_app = None

    def click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="click", message="clicked")

    def double_click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="double_click", message="double_clicked")

    def right_click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="right_click", message="right_clicked")

    def middle_click(self, **kwargs):
        self.clicked_args.append(kwargs)
        return ActionResult(ok=True, action="middle_click", message="middle_clicked")

    def drag(self, **kwargs):
        self.dragged_args.append(kwargs)
        return ActionResult(ok=True, action="drag", message="dragged")

    def scroll(self, **kwargs):
        self.scrolled_args.append(kwargs)
        return ActionResult(ok=True, action="scroll", message="scrolled")

    def set_value(self, **kwargs):
        return ActionResult(ok=True, action="set_value", message="value_set")


def test_invalid_strategy_rejected():
    backend = FakeBackend()
    res = _dispatch(backend, "click", {"element": 1, "strategy": "invalid_strat"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "invalid_strategy"
    assert "invalid strategy" in parsed["error"]


def test_strategy_a11y_requires_element():
    backend = FakeBackend()

    # Reject coordinate-only click under strategy='a11y'
    res = _dispatch(backend, "click", {"coordinate": [100, 200], "strategy": "a11y"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_a11y_required"
    assert "strategy='a11y' requires targeting by `element`" in parsed["error"]

    # Allow element click under strategy='a11y' and strip coordinates
    res2 = _dispatch(backend, "click", {"element": 5, "coordinate": [100, 200], "strategy": "a11y"})
    assert json.loads(res2)["ok"] is True
    assert backend.clicked_args[-1]["element"] == 5
    assert backend.clicked_args[-1]["x"] is None
    assert backend.clicked_args[-1]["y"] is None


def test_strategy_a11y_drag_and_scroll():
    backend = FakeBackend()

    # Drag without element indices fails
    res = _dispatch(backend, "drag", {"from_coordinate": [0, 0], "to_coordinate": [10, 10], "strategy": "a11y"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_a11y_required"

    # Drag with element indices succeeds
    res2 = _dispatch(backend, "drag", {"from_element": 1, "to_element": 2, "strategy": "a11y"})
    assert json.loads(res2)["ok"] is True

    # Scroll with coordinate-only fails
    res3 = _dispatch(backend, "scroll", {"coordinate": [50, 50], "direction": "down", "strategy": "a11y"})
    parsed3 = json.loads(res3)
    assert parsed3["ok"] is False
    assert parsed3["code"] == "strategy_a11y_required"

    # Scroll with element succeeds
    res4 = _dispatch(backend, "scroll", {"element": 3, "direction": "down", "strategy": "a11y"})
    assert json.loads(res4)["ok"] is True


def test_strategy_event_requires_coordinates_and_strips_element():
    backend = FakeBackend()

    # Reject element-only click under strategy='event'
    res = _dispatch(backend, "click", {"element": 4, "strategy": "event"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_event_required"
    assert "strategy='event' requires `coordinate=[x, y]`" in parsed["error"]

    # When both are given, strip element and use coordinates
    res2 = _dispatch(backend, "click", {"element": 4, "coordinate": [150, 250], "strategy": "event"})
    assert json.loads(res2)["ok"] is True
    assert backend.clicked_args[-1]["element"] is None
    assert backend.clicked_args[-1]["x"] == 150
    assert backend.clicked_args[-1]["y"] == 250


def test_strategy_auto_permits_both():
    backend = FakeBackend()

    # Element targeting
    res1 = _dispatch(backend, "click", {"element": 2, "strategy": "auto"})
    assert json.loads(res1)["ok"] is True
    assert backend.clicked_args[-1]["element"] == 2

    # Coordinate targeting
    res2 = _dispatch(backend, "click", {"coordinate": [10, 20], "strategy": "auto"})
    assert json.loads(res2)["ok"] is True
    assert backend.clicked_args[-1]["x"] == 10
    assert backend.clicked_args[-1]["y"] == 20


def test_strategy_event_rejects_set_value():
    backend = FakeBackend()
    res = _dispatch(backend, "set_value", {"value": "text", "strategy": "event"})
    parsed = json.loads(res)
    assert parsed["ok"] is False
    assert parsed["code"] == "strategy_event_unsupported"
    assert "strategy='event' is not supported for set_value" in parsed["error"]
