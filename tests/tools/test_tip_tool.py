"""Tests for the GUI-surface ``tip`` tool."""

import json

import pytest

from tools import tip_tool as tt
from tools.registry import registry


@pytest.fixture
def emitted(monkeypatch):
    """Capture what the desktop bridge is asked to send, and say it landed."""
    sent = []

    def _emit(event, payload):
        sent.append((event, payload))
        return True

    monkeypatch.setattr(tt.desktop_ui, "emit", _emit)
    return sent


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Scoped by toolset, not by the backend's env — see AGENTS.md."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("show_tip")

    assert entry is not None
    assert entry.toolset == "desktop_ui"


def test_answers_to_the_appearance_switch():
    """Tips off has to mean the model never sees the tool. See
    tests/tools/test_display_toggles.py for the config end of it."""
    entry = registry.get_entry("show_tip")

    assert entry is not None
    assert entry.check_fn is tt.check_tips_enabled


def test_requires_the_desktop_bridge(monkeypatch):
    """Outside the desktop GUI there is no emitter — a clear error, no crash."""
    monkeypatch.setattr(tt.desktop_ui, "emit", lambda _event, _payload: False)

    result = json.loads(tt.tip_tool(text="Over here", selector="#composer"))

    assert "desktop" in result["error"]


def test_needs_both_something_to_say_and_somewhere_to_point(emitted):
    assert "tip needs text" in json.loads(tt.tip_tool(text="", selector="#a"))["error"]
    assert "tip needs a selector" in json.loads(tt.tip_tool(text="Hi", selector=" "))["error"]
    assert not emitted


def test_rejects_an_unknown_side(emitted):
    result = json.loads(tt.tip_tool(text="Hi", selector="#a", side="diagonal"))

    assert "side must be one of" in result["error"]
    assert not emitted


def test_emits_the_renderer_event_omitting_unset_fields(emitted):
    result = json.loads(tt.tip_tool(text="  The model name is a button  ", selector="  #model  "))

    assert result == {"success": True, "selector": "#model"}
    assert emitted == [
        ("tip.show", {
            "selector": "#model",
            "text": "The model name is a button",
            "tip_id": "agent:03905c0426064130",
        })
    ]


def test_carries_title_and_side_when_given(emitted):
    tt.tip_tool(text="Type here", selector="#composer", title="Composer", side="top")

    assert emitted[0][1] == {
        "selector": "#composer",
        "text": "Type here",
        "tip_id": "agent:cde95bdea4a2da29",
        "title": "Composer",
        "side": "top",
    }


def test_tip_id_is_a_stable_content_hash(emitted):
    """The renderer retires a ✕'d tip by id (#117216), and an agent tip has
    no catalog entry to key on — so the id must be the content: same
    selector+text, same id, whichever conversation sends it."""
    tt.tip_tool(text="Same words", selector="#a")
    tt.tip_tool(text="  Same words  ", selector=" #a ")

    assert emitted[0][1]["tip_id"] == emitted[1][1]["tip_id"]
    assert emitted[0][1]["tip_id"].startswith("agent:")


def test_tip_id_follows_the_content_not_the_call(emitted):
    """Different words are a different tip — the ✕ retired the old one, not
    the tool's right to speak."""
    tt.tip_tool(text="One thing", selector="#a")
    tt.tip_tool(text="Another thing", selector="#a")

    assert emitted[0][1]["tip_id"] != emitted[1][1]["tip_id"]


def test_bridge_failure_is_reported(monkeypatch):
    def _boom(_event, _payload):
        raise RuntimeError("renderer went away")

    monkeypatch.setattr(tt.desktop_ui, "emit", _boom)

    assert "renderer went away" in json.loads(tt.tip_tool(text="Hi", selector="#a"))["error"]
