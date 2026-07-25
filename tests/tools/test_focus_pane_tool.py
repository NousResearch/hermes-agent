"""Tests for the desktop-gated ``focus_pane`` tool."""

import json

import pytest

from tools import desktop_ui, focus_pane_tool as fp


@pytest.fixture(autouse=True)
def _reset_emitter():
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def test_gated_on_desktop(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert fp.check_focus_pane_requirements() is False

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert fp.check_focus_pane_requirements() is True


def test_rejects_unknown_pane():
    desktop_ui.set_emitter(lambda *a: None)
    assert json.loads(fp.focus_pane_tool("banana"))["error"]


def test_desktop_only_without_emitter():
    assert "desktop" in json.loads(fp.focus_pane_tool("terminal"))["error"].lower()


@pytest.mark.parametrize("pane", fp.PANES)
def test_emits_pane_reveal(pane):
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(fp.focus_pane_tool(f"  {pane.upper()}  "))

    assert out == {"success": True, "pane": pane}
    assert calls == [("pane.reveal", {"pane": pane})]
