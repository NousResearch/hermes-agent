"""Screenshot input units must not depend on AX geometry or desktop placement."""

import base64
import importlib
import io
import json

import pytest
from PIL import Image

from tools.computer_use import tool
from tools.computer_use.backend import ActionResult, CaptureResult, UIElement
from tools.registry import registry


@pytest.fixture
def backend(monkeypatch, grant_computer_use_approvals):
    image = io.BytesIO()
    Image.new("RGB", (800, 600)).save(image, format="PNG")
    png = base64.b64encode(image.getvalue()).decode("ascii")

    class RecordingBackend(tool._NoopBackend):
        def __init__(self):
            super().__init__()
            self.elements = []
            self.png = png
            self.size = (800, 600)
            self.inputs = []

        def capture(self, mode="som", app=None, pid=None, window_id=None):
            self._last_app = app or "Fixture app"
            self._last_target = {"pid": pid, "window_id": window_id}
            self.calls.append(("capture", dict(self._last_target)))
            return CaptureResult(
                mode=mode, width=self.size[0], height=self.size[1], png_b64=self.png,
                elements=self.elements, app=self._last_app,
            )

        def _input(self, action, kwargs):
            self.inputs.append((action, dict(self._last_target), kwargs))
            return ActionResult(ok=True, action=action)

        def click(self, **kwargs):
            return self._input("click", kwargs)

        def drag(self, **kwargs):
            return self._input("drag", kwargs)

        def scroll(self, **kwargs):
            return self._input("scroll", kwargs)

    value = RecordingBackend()
    tool.reset_backend_for_tests()
    monkeypatch.setattr(tool, "_new_backend", lambda mode: value)
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    importlib.import_module("tools.computer_use_tool")
    yield value
    tool.reset_backend_for_tests()


@pytest.mark.parametrize("bounds", [
    pytest.param((3440, 1900, 1720, 1302), id="positive-desktop-origin"),
    pytest.param((-3440, -1440, 1720, 1302), id="negative-desktop-origin"),
    pytest.param((10, 20, 30, 40), id="small-element"),
    pytest.param(None, id="no-ax-elements"),
])
@pytest.mark.parametrize("mode", ["ax", "som"])
def test_capture_keeps_screenshot_input_contract_independent_of_ax_bounds(backend, bounds, mode):
    if bounds is not None:
        backend.elements = [UIElement(index=1, role="AXButton", label="Target", bounds=bounds)]

    result = registry.dispatch("computer_use", {
        "action": "capture", "mode": mode, "pid": 42, "window_id": 87,
    })
    payload = json.loads(result) if isinstance(result, str) else result
    summary = payload.get("summary", payload.get("text_summary", ""))

    # This guidance is an input contract, not a guessed transformation derived
    # from where an AX element happens to sit on a multi-monitor desktop.
    assert "pass screenshot positions unchanged" in summary
    assert "do not infer a screenshot scale from element extents" in summary
    assert "estimated scale" not in summary
    assert payload.get("bounds_scale") is None
    assert payload.get("meta", {}).get("bounds_scale") is None


@pytest.mark.parametrize("action, coordinates, expected", [
    ("click", {"coordinate": [511, 287]}, {"x": 511, "y": 287}),
    ("drag", {"from_coordinate": [511, 287], "to_coordinate": [621, 333]},
     {"from_xy": (511, 287), "to_xy": (621, 333)}),
    ("scroll", {"coordinate": [511, 287], "direction": "down"}, {"x": 511, "y": 287}),
])
def test_screenshot_coordinates_reach_same_captured_target_unchanged(backend, action, coordinates, expected):
    backend.elements = [
        UIElement(index=1, role="AXButton", label="Target", bounds=(3440, 1900, 1720, 1302)),
    ]
    registry.dispatch("computer_use", {"action": "capture", "mode": "ax", "pid": 42, "window_id": 87})

    result = registry.dispatch("computer_use", {"action": action, "capture_after": True, **coordinates})
    payload = json.loads(result) if isinstance(result, str) else result

    assert payload.get("action_result", payload)["ok"] is True
    delivered_action, target, delivered = backend.inputs[-1]
    assert delivered_action == action
    assert target == {"pid": 42, "window_id": 87}
    assert {key: delivered[key] for key in expected} == expected
    assert backend.calls[-1] == ("capture", target)


def test_driver_screenshot_bounds_survive_capture_and_ground_pixel_input(backend):
    from tools.computer_use.cua_backend_parse import _parse_elements_from_structured

    backend.elements = _parse_elements_from_structured([{
        "element_index": 2, "role": "AXButton", "label": "Confirm",
        "frame": {"x": 3440, "y": 1900, "w": 160, "h": 60},
        "screenshot_frame": {"x": 511, "y": 287, "w": 80, "h": 30},
        "element_token": "snapshot:2",
    }])
    payload = json.loads(registry.dispatch("computer_use", {
        "action": "capture", "mode": "ax", "pid": 42, "window_id": 87,
    }))
    element = payload["elements"][0]
    assert element["bounds"] == [3440, 1900, 160, 60]
    assert element["screenshot_bounds"] == [511, 287, 80, 30]
    assert "screenshot pixels" in payload["summary"]
    x, y, width, height = element["screenshot_bounds"]
    registry.dispatch("computer_use", {
        "action": "click", "coordinate": [x + width // 2, y + height // 2],
    })
    assert backend.inputs[-1][1] == {"pid": 42, "window_id": 87}
    assert (backend.inputs[-1][2]["x"], backend.inputs[-1][2]["y"]) == (551, 302)


def test_aux_vision_coordinates_map_back_to_capture_pixels(backend, monkeypatch):
    from tools import vision_tools

    backend.size = (3200, 2000)
    image = io.BytesIO()
    Image.new("RGB", backend.size).save(image, format="PNG")
    backend.png = base64.b64encode(image.getvalue()).decode("ascii")
    seen = {}

    async def analyze(path, prompt):
        with Image.open(path) as shrunk:
            seen["size"] = shrunk.size
        seen["prompt"] = prompt
        return json.dumps({"analysis": "Synthetic coordinate analysis"})

    monkeypatch.setattr(vision_tools, "vision_analyze_tool", analyze)
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: True)
    result = registry.dispatch("computer_use", {"action": "capture", "mode": "som"})
    payload = json.loads(result)

    # Only the aux copy is resized. Reported points must return to the original
    # capture's pixel units before the driver applies its own window transform.
    assert max(seen["size"]) == tool._MAX_VISION_DIM
    assert payload["width"] == backend.size[0]
    assert payload["height"] == backend.size[1]
    assert "original capture screenshot pixels" in seen["prompt"]
    assert "real screen" not in seen["prompt"]
    assert "multiply any coordinates" in seen["prompt"]
    assert "pass screenshot positions unchanged" in payload["summary"]
    assert "bounds_scale" not in payload
