"""Tests for the ``voice.toggle`` TTS action in tui_gateway.

Verifies that TTS output can be toggled independently of voice mode
(microphone/STT), mirroring the classic CLI's ``_toggle_voice_tts``
which no longer gates on ``_voice_mode``.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = "sid-voice-test"
    session_key = "tui-voice-session-1"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


# ── voice.toggle tts — independent of voice mode ─────────────────────


class TestVoiceToggleTts:
    def test_tts_toggle_on_without_voice_mode(self, server, session, monkeypatch):
        """TTS can be enabled even when voice mode is OFF."""
        monkeypatch.setenv("HERMES_VOICE", "0")
        monkeypatch.setenv("HERMES_VOICE_TTS", "0")

        r = _call(server, "voice.toggle", action="tts")

        assert "error" not in r
        assert r["result"]["tts"] is True
        # Voice mode stays off — TTS is independent.
        assert r["result"]["enabled"] is False
        assert server._voice_tts_enabled() is True

    def test_tts_toggle_off_without_voice_mode(self, server, session, monkeypatch):
        """TTS can be disabled even when voice mode is OFF."""
        monkeypatch.setenv("HERMES_VOICE", "0")
        monkeypatch.setenv("HERMES_VOICE_TTS", "1")

        r = _call(server, "voice.toggle", action="tts")

        assert "error" not in r
        assert r["result"]["tts"] is False
        assert r["result"]["enabled"] is False
        assert server._voice_tts_enabled() is False

    def test_tts_toggle_with_voice_mode_on(self, server, session, monkeypatch):
        """TTS toggle still works when voice mode is ON."""
        monkeypatch.setenv("HERMES_VOICE", "1")
        monkeypatch.setenv("HERMES_VOICE_TTS", "0")

        r = _call(server, "voice.toggle", action="tts")

        assert "error" not in r
        assert r["result"]["tts"] is True
        assert r["result"]["enabled"] is True

    def test_tts_toggle_no_error_code(self, server, session, monkeypatch):
        """The old 4014 'enable voice mode first' error must not fire."""
        monkeypatch.setenv("HERMES_VOICE", "0")
        monkeypatch.setenv("HERMES_VOICE_TTS", "0")

        r = _call(server, "voice.toggle", action="tts")

        # Success, not an error envelope.
        assert "error" not in r
        assert r["result"]["tts"] is True
