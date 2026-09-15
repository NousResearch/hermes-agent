"""Tests for display.show_reasoning gating on reasoning emissions in tui_gateway."""

import pytest
from tui_gateway import server


def test_show_reasoning_enabled_default(monkeypatch):
    monkeypatch.setattr(server, "_display_cfg", lambda: {"show_reasoning": True})
    assert server._show_reasoning_enabled("s1") is True

    monkeypatch.setattr(server, "_display_cfg", lambda: {"show_reasoning": False})
    assert server._show_reasoning_enabled("s1") is False


def test_show_reasoning_enabled_session_override(monkeypatch):
    monkeypatch.setattr(server, "_display_cfg", lambda: {"show_reasoning": True})
    monkeypatch.setattr(server, "_sessions", {"s1": {"show_reasoning": False}})

    assert server._show_reasoning_enabled("s1") is False


def test_reasoning_delta_and_available_gated(monkeypatch):
    emitted = []

    def mock_emit(event_type, sid, payload=None):
        emitted.append((event_type, sid, payload))

    monkeypatch.setattr(server, "_emit", mock_emit)
    monkeypatch.setattr(server, "_session_verbose", lambda sid: False)

    # Disabled
    monkeypatch.setattr(server, "_display_cfg", lambda: {"show_reasoning": False})
    monkeypatch.setattr(server, "_sessions", {})

    cbs = server._agent_cbs("s1")
    cbs["reasoning_callback"]("some reasoning text")
    server._progress_reasoning("s1", "reasoning", "some preview text", {})

    assert len(emitted) == 0

    # Enabled
    monkeypatch.setattr(server, "_display_cfg", lambda: {"show_reasoning": True})
    cbs["reasoning_callback"]("some reasoning text")
    server._progress_reasoning("s1", "reasoning", "some preview text", {})

    assert len(emitted) == 2
    assert emitted[0][0] == "reasoning.delta"
    assert emitted[1][0] == "reasoning.available"
