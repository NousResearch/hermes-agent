"""Tests for the desktop-gated ``open_preview`` tool."""

import json

import pytest

from tools import desktop_ui, open_preview_tool as op


@pytest.fixture(autouse=True)
def _reset_emitter():
    """Each test controls the emitter; never leak one across tests."""
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def test_gated_on_desktop(monkeypatch):
    """Hidden unless HERMES_DESKTOP is set (mirrors read_terminal/close_terminal)."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert op.check_open_preview_requirements() is False

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert op.check_open_preview_requirements() is True


def test_emitter_failure_is_reported():
    def _boom(*_a):
        raise RuntimeError("no window")

    desktop_ui.set_emitter(_boom)
    assert "no window" in json.loads(op.open_preview_tool("https://x.example"))["error"]
