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


def test_session_scoped_event_uses_session_id(monkeypatch):
    """message.reaction (and future session-scoped events) route by
    HERMES_SESSION_ID, not HERMES_UI_SESSION_ID (#80678)."""
    monkeypatch.setattr(
        desktop_ui, "get_session_env",
        lambda name, default="": {
            "HERMES_UI_SESSION_ID": "win-3",
            "HERMES_SESSION_ID": "sess-9",
        }.get(name, default),
    )
    seen = []
    desktop_ui.set_emitter(lambda sid, event, payload: seen.append((sid, event, payload)))

    assert desktop_ui.emit("message.reaction", {"row_id": 42, "reactions": ["❤️"]}) is True
    assert seen == [("sess-9", "message.reaction", {"row_id": 42, "reactions": ["❤️"]})]


def test_session_scoped_falls_back_to_ui_session_id(monkeypatch):
    """When HERMES_SESSION_ID is empty, fall back to the window id
    so legacy sessions without a chat-session env var still work."""
    monkeypatch.setattr(
        desktop_ui, "get_session_env",
        lambda name, default="": "win-5" if name == "HERMES_UI_SESSION_ID" else default,
    )
    seen = []
    desktop_ui.set_emitter(lambda sid, event, payload: seen.append((sid, event, payload)))

    assert desktop_ui.emit("message.reaction", {"row_id": 1, "reactions": ["👍"]}) is True
    assert seen == [("win-5", "message.reaction", {"row_id": 1, "reactions": ["👍"]})]
