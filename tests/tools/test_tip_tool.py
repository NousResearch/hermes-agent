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


def test_is_ungated_like_tour():
    """The Appearance switch governs the app's idle rotation, not this."""
    entry = registry.get_entry("show_tip")

    assert entry is not None
    assert entry.check_fn is None


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
    assert emitted == [("tip.show", {"selector": "#model", "text": "The model name is a button"})]


def test_carries_title_and_side_when_given(emitted):
    tt.tip_tool(text="Type here", selector="#composer", title="Composer", side="top")

    assert emitted[0][1] == {
        "selector": "#composer",
        "text": "Type here",
        "title": "Composer",
        "side": "top",
    }


def test_bridge_failure_is_reported(monkeypatch):
    def _boom(_event, _payload):
        raise RuntimeError("renderer went away")

    monkeypatch.setattr(tt.desktop_ui, "emit", _boom)

    assert "renderer went away" in json.loads(tt.tip_tool(text="Hi", selector="#a"))["error"]
