"""Tests for the GUI-surface ``close_preview`` tool."""

import json

import pytest

from tools import close_preview_tool as cp, desktop_ui


@pytest.fixture(autouse=True)
def _reset_emitter():
    """Each test controls the emitter; never leak one across tests."""
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)




def test_emits_preview_close_for_the_whole_pane():
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(cp.close_preview_tool())

    assert out == {"success": True, "url": ""}
    assert calls == [("preview.close", {"url": ""})]


def test_normalizes_a_bare_domain_like_open_does():
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(cp.close_preview_tool("www.cnn.com"))

    assert out == {"success": True, "url": "https://www.cnn.com"}
    assert calls == [("preview.close", {"url": "https://www.cnn.com"})]


def test_reports_desktop_only_without_emitter():
    out = cp.close_preview_tool()

    assert "desktop app" in out


def test_emitter_failure_is_reported():
    def _boom(*_a):
        raise RuntimeError("no window")

    desktop_ui.set_emitter(_boom)
    assert "no window" in json.loads(cp.close_preview_tool("https://x.example"))["error"]
