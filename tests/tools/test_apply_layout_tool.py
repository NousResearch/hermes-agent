"""Tests for the GUI-surface ``apply_layout`` tool."""

import json

import pytest

from tools import apply_layout_tool as al, desktop_ui
from tools.registry import registry


@pytest.fixture(autouse=True)
def _reset_emitter():
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Surface eligibility is the toolset's job, not a process env var — the
    desktop client can be driving a remote/cloud backend that never sees
    HERMES_DESKTOP."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("apply_layout")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_emits_layout_apply():
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(al.apply_layout_tool("  focus  "))

    assert out == {"success": True, "preset": "focus"}
    assert calls == [("layout.apply", {"preset": "focus"})]


def test_preset_ids_pass_through_unmapped():
    """Ids are free-form: plugin/user presets must not be filtered by an enum
    the backend can't know about."""
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(al.apply_layout_tool("user-research-cockpit"))

    assert out["success"] is True
    assert calls[0][1] == {"preset": "user-research-cockpit"}


def test_empty_preset_is_an_error():
    desktop_ui.set_emitter(lambda sid, event, payload: None)

    out = al.apply_layout_tool("   ")

    assert "preset is required" in out


def test_reports_desktop_only_without_emitter():
    out = al.apply_layout_tool("focus")

    assert "desktop app" in out
