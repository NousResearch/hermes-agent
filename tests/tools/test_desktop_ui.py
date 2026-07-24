"""Tests for the desktop-only renderer-event bridge."""

import pytest

from tools import desktop_ui


@pytest.fixture(autouse=True)
def _reset_emitter():
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def test_unavailable_without_emitter():
    assert desktop_ui.available() is False
    assert desktop_ui.emit("preview.open", {"url": "x"}) is False


def test_routes_event_to_owning_window(monkeypatch):
    monkeypatch.setattr(
        desktop_ui, "get_session_env",
        lambda name, default="": "win-7" if name == "HERMES_UI_SESSION_ID" else default,
    )
    seen = []
    desktop_ui.set_emitter(lambda sid, event, payload: seen.append((sid, event, payload)))

    assert desktop_ui.available() is True
    assert desktop_ui.emit("pane.reveal", {"pane": "terminal"}) is True
    assert seen == [("win-7", "pane.reveal", {"pane": "terminal"})]
