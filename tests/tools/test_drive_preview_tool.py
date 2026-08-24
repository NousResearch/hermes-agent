"""Tests for the GUI-surface ``drive_preview`` tool."""

import json

from tools import drive_preview_tool as ap
from tools.registry import registry


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Mirrors read_preview: scoped by toolset, not by the backend's env."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("drive_preview")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    result = json.loads(ap.drive_preview_tool(action="elements", callback=None))
    assert "desktop" in result["error"]


def test_rejects_an_unknown_action():
    result = json.loads(ap.drive_preview_tool(action="teleport", callback=lambda _p: "{}"))
    assert "action must be one of" in result["error"]


def test_interaction_verbs_need_a_target():
    """A click with nowhere to land is a mistake worth naming before the bridge."""
    for verb in ("click", "type", "press"):
        result = json.loads(ap.drive_preview_tool(action=verb, text="x", key="Enter", callback=lambda _p: "{}"))
        assert "ref" in result["error"], verb


def test_type_needs_text_and_press_needs_a_key():
    calls = []

    def cb(payload):
        calls.append(payload)
        return json.dumps({"success": True})

    assert "text" in json.loads(ap.drive_preview_tool(action="type", ref="@e1", callback=cb))["error"]
    assert "key" in json.loads(ap.drive_preview_tool(action="press", ref="@e1", callback=cb))["error"]
    assert calls == []


def test_typing_an_empty_string_is_allowed():
    """Clearing a field is a real intent — only a missing `text` is an error."""
    seen = {}
    ap.drive_preview_tool(action="type", ref="@e1", text="", callback=lambda p: seen.update(p) or "{}")

    assert seen["text"] == ""


def test_scroll_needs_no_target_and_validates_its_destination():
    seen = {}

    def cb(payload):
        seen.clear()
        seen.update(payload)
        return json.dumps({"success": True})

    ap.drive_preview_tool(action="scroll", callback=cb)
    assert seen == {"action": "scroll"}

    result = json.loads(ap.drive_preview_tool(action="scroll", to="sideways", callback=cb))
    assert "to must be one of" in result["error"]


def test_payload_forwards_only_what_was_given():
    seen = {}
    ap.drive_preview_tool(
        action="type",
        ref="inp-password",
        text="hunter2",
        submit=True,
        callback=lambda p: seen.update(p) or json.dumps({"success": True}),
    )

    assert seen == {"action": "type", "ref": "inp-password", "text": "hunter2", "submit": True}


def test_full_asks_the_renderer_for_a_whole_inventory():
    seen = {}
    ap.drive_preview_tool(
        action="elements",
        full=True,
        callback=lambda p: seen.update(p) or json.dumps({"success": True}),
    )

    assert seen == {"action": "elements", "full": True}


def test_numeric_arguments_are_validated():
    result = json.loads(ap.drive_preview_tool(action="scroll", amount="lots", callback=lambda _p: "{}"))
    assert "integers" in result["error"]


def test_empty_answer_means_nothing_open():
    result = json.loads(ap.drive_preview_tool(action="elements", callback=lambda _p: ""))
    assert "open_preview" in result["error"]


def test_passes_the_renderer_answer_through():
    payload = {
        "success": True,
        "acted": 'clicked button "Sign in"',
        "url": "https://example.com/app",
        "elements": [{"ref": "@e1", "role": "button", "label": "Log out", "selector": "#out"}],
    }
    result = json.loads(ap.drive_preview_tool(action="click", ref="@e2", callback=lambda _p: json.dumps(payload)))

    assert result == payload


def test_wraps_non_json_text():
    result = json.loads(ap.drive_preview_tool(action="elements", callback=lambda _p: "plain words"))
    assert result == {"text": "plain words"}


def test_callback_failure_is_reported():
    def _boom(_payload):
        raise RuntimeError("renderer went away")

    result = json.loads(ap.drive_preview_tool(action="elements", callback=_boom))
    assert "renderer went away" in result["error"]
