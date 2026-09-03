"""Tests for the GUI-surface ``annotate_preview`` tool."""

import json

import pytest

from tools import annotate_preview_tool as an
from tools.registry import registry


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Scoped by toolset, not by the backend's env — same as its siblings."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("annotate_preview")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_requires_callback():
    result = json.loads(an.annotate_preview_tool(ref="@e1", callback=None))
    assert "desktop" in result["error"]


def test_rejects_an_unknown_action():
    result = json.loads(an.annotate_preview_tool(action="scribble", callback=lambda _p: "{}"))
    assert "action must be one of" in result["error"]


@pytest.mark.parametrize("verb", ["add", "remove"])
def test_marking_one_thing_needs_a_target(verb):
    """A mark with nothing to attach to is a mistake worth naming early."""
    result = json.loads(an.annotate_preview_tool(action=verb, callback=lambda _p: "{}"))
    assert "ref" in result["error"]


def test_speaks_the_renderer_s_verbs():
    """The overlay knows pin/unpin; the model gets words it can guess at."""
    sent = []

    def cb(payload):
        sent.append(payload)
        return json.dumps({"acted": "pinned Buy now", "success": True})

    an.annotate_preview_tool(action="add", ref="@e4", label="cheapest", callback=cb)
    an.annotate_preview_tool(action="remove", ref="@e4", callback=cb)

    assert sent[0] == {"action": "pin", "ref": "@e4", "text": "cheapest"}
    assert sent[1] == {"action": "unpin", "ref": "@e4"}


def test_clear_sends_no_target_so_the_overlay_drops_them_all():
    """`clear` IS an untargeted unpin — a ref left on it would take down one."""
    sent = []

    def cb(payload):
        sent.append(payload)
        return json.dumps({"acted": "cleared every pin", "success": True})

    an.annotate_preview_tool(action="clear", ref="@e4", selector="a", callback=cb)

    assert sent == [{"action": "unpin"}]


def test_defaults_to_adding():
    sent = []

    def cb(payload):
        sent.append(payload)
        return json.dumps({"success": True})

    an.annotate_preview_tool(ref="@e2", callback=cb)

    assert sent[0]["action"] == "pin"


def test_passes_the_renderer_s_answer_straight_through():
    out = json.loads(
        an.annotate_preview_tool(ref="@e1", callback=lambda _p: json.dumps({"acted": "pinned Save", "success": True}))
    )

    assert out == {"acted": "pinned Save", "success": True}


def test_reports_a_silent_bridge():
    result = json.loads(an.annotate_preview_tool(ref="@e1", callback=lambda _p: ""))
    assert "open_preview" in result["error"]
