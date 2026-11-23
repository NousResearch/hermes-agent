"""``desktop_preview`` action=close: the live preview-close path (``tools/preview_tool.py``)."""

import json

import pytest

from tools import desktop_ui, preview_tool


@pytest.fixture(autouse=True)
def _reset_emitter():
    """Each test controls the emitter; never leak one across tests."""
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def _close(**args):
    return preview_tool._handle_preview({"action": "close", **args})


def test_emits_preview_close_for_the_whole_pane():
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    assert json.loads(_close()) == {"success": True, "closed": "all"}
    assert calls == [("preview.close", {"url": ""})]


def test_normalizes_a_bare_domain_like_open_does():
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    assert json.loads(_close(url="www.cnn.com")) == {"success": True, "closed": "https://www.cnn.com"}
    assert calls == [("preview.close", {"url": "https://www.cnn.com"})]


def test_reports_desktop_only_without_emitter():
    assert "desktop app" in json.loads(_close())["error"]


def test_emitter_failure_is_reported():
    def _boom(*_a):
        raise RuntimeError("no window")

    desktop_ui.set_emitter(_boom)
    assert "no window" in json.loads(_close(url="https://x.example"))["error"]
