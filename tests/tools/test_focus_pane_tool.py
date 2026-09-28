"""Tests for the GUI-surface ``focus_pane`` tool."""

import json

import pytest

from tools import desktop_ui, focus_pane_tool as fp


@pytest.fixture(autouse=True)
def _reset_emitter():
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)




@pytest.mark.parametrize("pane", fp.PANES)
def test_emits_pane_reveal(pane):
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(fp.focus_pane_tool(f"  {pane.upper()}  "))

    assert out == {"success": True, "pane": pane}
    assert calls == [("pane.reveal", {"pane": pane})]
