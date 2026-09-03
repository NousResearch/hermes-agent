"""Tests for the GUI-surface ``annotate_screen`` tool."""

import json

from tools import annotate_screen_tool as sa
from tools.registry import registry

CIRCLE = {"kind": "circle", "x": 320, "y": 540}


def _run(**kwargs):
    kwargs.setdefault("callback", lambda _payload: json.dumps({"success": True}))
    return json.loads(sa.annotate_screen_tool(**kwargs))


def _draw(**kwargs):
    kwargs.setdefault("action", "draw")
    kwargs.setdefault("shapes", [dict(CIRCLE)])
    kwargs.setdefault("frame_width", 1024)
    kwargs.setdefault("frame_height", 768)
    return _run(**kwargs)


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Scoped by toolset, not by the backend's env — see AGENTS.md."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("annotate_screen")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    assert "desktop" in json.loads(sa.annotate_screen_tool(action="draw", callback=None))["error"]


def test_rejects_unknown_action():
    assert "action must be one of" in _run(action="doodle")["error"]


def test_draw_validates_its_shapes():
    assert "non-empty shapes" in _draw(shapes=None)["error"]
    assert "non-empty shapes" in _draw(shapes=[])["error"]
    assert "shapes[0] must be an object" in _draw(shapes=["nope"])["error"]
    assert "kind must be one of" in _draw(shapes=[{"kind": "scribble"}])["error"]
    assert "needs numeric 'y'" in _draw(shapes=[{"kind": "circle", "x": 5}])["error"]
    assert "needs numeric 'to_x'" in _draw(
        shapes=[{"kind": "arrow", "from_x": 1, "from_y": 2, "to_y": 3}]
    )["error"]
    assert "needs numeric 'width'" in _draw(shapes=[{"kind": "rect", "x": 1, "y": 2, "height": 3}])["error"]
    assert "needs non-empty 'text'" in _draw(shapes=[{"kind": "label", "x": 1, "y": 2}])["error"]
    assert "at least two points" in _draw(shapes=[{"kind": "polyline", "points": [{"x": 1, "y": 2}]}])["error"]
    assert "numeric 'x' and 'y'" in _draw(
        shapes=[{"kind": "polyline", "points": [{"x": 1, "y": 2}, {"x": "right", "y": 3}]}]
    )["error"]
    assert "at most" in _draw(
        shapes=[{"kind": "polyline", "points": [{"x": i, "y": i} for i in range(sa.MAX_POLYLINE_POINTS + 1)]}]
    )["error"]
    assert "dashed must be a boolean" in _draw(shapes=[{**CIRCLE, "dashed": 1}])["error"]
    assert "step must be an integer" in _draw(shapes=[{**CIRCLE, "step": 0}])["error"]
    assert "step must be an integer" in _draw(shapes=[{**CIRCLE, "step": 13}])["error"]
    assert "step must be an integer" in _draw(shapes=[{**CIRCLE, "step": 1.5}])["error"]
    assert "step must be an integer" in _draw(shapes=[{**CIRCLE, "step": True}])["error"]
    assert "error" not in _draw(shapes=[{**CIRCLE, "step": 1}])
    assert "error" not in _draw(shapes=[{**CIRCLE, "step": 12}])
    assert "color must be one of" in _draw(shapes=[{**CIRCLE, "color": "chartreuse"}])["error"]


def test_draw_requires_the_frame_size():
    assert "frame_width and frame_height" in _draw(frame_width=None, frame_height=None)["error"]
    assert "frame_width and frame_height" in _draw(frame_width=0)["error"]
    # Bool is an int in Python and must not pass as a frame dimension.
    assert "frame_width and frame_height" in _draw(frame_width=True)["error"]


def test_ttl_must_be_a_positive_number():
    assert "positive" in _draw(ttl_seconds=-3)["error"]
    assert "positive" in _draw(ttl_seconds=0)["error"]
    assert "error" not in _draw(ttl_seconds=45)
    assert "error" not in _draw(ttl_seconds=180)


def test_draw_payload_carries_frame_shapes_and_target():
    seen = {}

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    sa.annotate_screen_tool(
        action="draw",
        target="Chess",
        frame_width=2048,
        frame_height=1280,
        shapes=[dict(CIRCLE)],
        ttl_seconds=45,
        callback=cb,
    )
    assert seen == {
        "action": "draw",
        "target": "Chess",
        "frame": {"width": 2048, "height": 1280},
        "shapes": [CIRCLE],
        "ttl_seconds": 45,
    }


def test_polyline_payload_keeps_points_and_dashed():
    seen = {}
    path = {
        "kind": "polyline",
        "dashed": True,
        "label": "trend",
        "color": "green",
        "points": [{"x": 10, "y": 80}, {"x": 40, "y": 50}, {"x": 90, "y": 20}],
    }

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    sa.annotate_screen_tool(
        action="draw",
        frame_width=100,
        frame_height=100,
        shapes=[path],
        ttl_seconds=180,
        callback=cb,
    )

    assert seen["shapes"] == [path]
    assert seen["ttl_seconds"] == 180


def test_step_payload_rides_through_on_the_shape():
    seen = {}
    mark = {**CIRCLE, "step": 2, "label": "Layer mask"}

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    sa.annotate_screen_tool(
        action="draw",
        frame_width=1024,
        frame_height=768,
        shapes=[mark],
        ttl_seconds=180,
        callback=cb,
    )

    assert seen["shapes"][0]["step"] == 2


def test_clear_payload_omits_the_draw_fields():
    seen = {}

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    sa.annotate_screen_tool(action="clear", callback=cb)
    assert seen == {"action": "clear"}


def test_unanswered_bridge_is_reported_rather_than_faked_as_success():
    assert "error" in _draw(callback=lambda _p: "")


def test_passes_renderer_json_through():
    payload = {"success": True, "shapes_drawn": 2, "expires_in_seconds": 30}
    assert _draw(callback=lambda _p: json.dumps(payload)) == payload


def test_wraps_non_json_text():
    assert _run(action="clear", callback=lambda _p: "cleared") == {"text": "cleared"}


def test_callback_failure_is_reported():
    def _boom(_payload):
        raise RuntimeError("renderer went away")

    assert "renderer went away" in _run(action="clear", callback=_boom)["error"]
