"""Tests for the GUI-surface ``focus_pane`` tool."""

import json

import pytest

from tools import desktop_ui, focus_pane_tool as fp
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
    entry = registry.get_entry("focus_pane")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


@pytest.mark.parametrize("pane", fp.PANES)
def test_emits_pane_reveal(pane):
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(fp.focus_pane_tool(f"  {pane.upper()}  "))

    assert out == {"success": True, "pane": pane}
    assert calls == [("pane.reveal", {"pane": pane})]
