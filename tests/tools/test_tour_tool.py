"""Tests for the GUI-surface ``tour`` tool."""

import json

from tools import tour_tool as tt
from tools.registry import registry


def _run(**kwargs):
    kwargs.setdefault("callback", lambda _payload: json.dumps({"success": True}))
    return json.loads(tt.tour_tool(**kwargs))






def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    assert "desktop" in json.loads(tt.tour_tool(action="targets", callback=None))["error"]


def test_rejects_unknown_action_and_surface():
    assert "action must be one of" in _run(action="dance")["error"]
    assert "surface must be one of" in _run(action="targets", surface="hologram")["error"]
    assert "side must be one of" in _run(action="show", selector="#a", side="diagonal")["error"]


def test_show_needs_something_to_point_at_or_say():
    assert "show needs" in _run(action="show")["error"]
    assert "error" not in _run(action="show", text="just narration")


def test_start_validates_its_steps():
    # No steps is the app's built-in tour; it exists only on the app surface.
    assert "error" not in _run(action="start")
    assert "surface='app'" in _run(action="start", surface="preview")["error"]
    assert "non-empty steps" in _run(action="start", steps=[])["error"]
    assert "steps[1] must be an object" in _run(action="start", steps=[{"selector": "#a"}, "nope"])["error"]
    assert "steps[1] needs" in _run(action="start", steps=[{"selector": "#a"}, {}])["error"]


def test_preset_is_validated_and_reaches_the_request_through_the_handler():
    seen = {}

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    entry = registry.get_entry("gui_tour")
    assert entry is not None
    handler = entry.handler
    assert "preset must be one of" in json.loads(handler({"action": "start", "preset": "medium"}, callback=cb))["error"]
    assert not seen
    assert "not both" in json.loads(
        handler({"action": "start", "preset": "quick", "steps": [{"text": "hi"}]}, callback=cb))["error"]
    assert not seen
    assert "error" not in json.loads(handler({"action": "start", "preset": "quick"}, callback=cb))
    assert seen == {"action": "start", "surface": "app", "preset": "quick"}


def test_payload_omits_unset_fields_and_defaults_the_surface():
    seen = {}

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    tt.tour_tool(action="show", selector="#composer", title="Composer", callback=cb)
    assert seen == {
        "action": "show",
        "surface": "app",
        "selector": "#composer",
        "title": "Composer",
    }


def test_unanswered_bridge_is_reported_rather_than_faked_as_success():
    assert "error" in _run(action="targets", callback=lambda _p: "")


def test_passes_renderer_json_through():
    payload = {"success": True, "matched": True, "step": 2}
    assert _run(action="next", callback=lambda _p: json.dumps(payload)) == payload


def test_wraps_non_json_text():
    assert _run(action="stop", callback=lambda _p: "stopped") == {"text": "stopped"}


def test_callback_failure_is_reported():
    def _boom(_payload):
        raise RuntimeError("renderer went away")

    assert "renderer went away" in _run(action="stop", callback=_boom)["error"]
