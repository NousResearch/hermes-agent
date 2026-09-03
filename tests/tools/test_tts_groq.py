"""Offline tests for the Groq Orpheus TTS provider.

These tests never hit the network. An optional live smoke is skipped
unless GROQ_API_KEY is already present in the environment.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_GROQ_TTS_BASE_URL,
    DEFAULT_GROQ_TTS_MODEL,
    DEFAULT_GROQ_TTS_VOICE,
    _bounded_provider_error,
    _generate_groq_tts,
    _groq_tts_response_format,
)


def test_groq_is_builtin_tts_provider():
    assert "groq" in BUILTIN_TTS_PROVIDERS


def test_requirements_follow_explicit_groq_provider(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "groq", "groq": {}},
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert tts_tool.check_tts_requirements() is True


def test_requirements_fail_without_groq_key(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "groq", "groq": {}},
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **k: "")
    assert tts_tool.check_tts_requirements() is False


def test_unselected_groq_credentials_do_not_expose_edge_tool(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", MagicMock(side_effect=ImportError))
    monkeypatch.setenv("GROQ_API_KEY", "unselected-key")
    assert tts_tool.check_tts_requirements() is False


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.tts_tool._resolve_provider_key",
        lambda *a, **k: "",
    )
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _generate_groq_tts("hi", str(tmp_path / "out.mp3"), {})


def test_forwards_model_and_voice_exactly(monkeypatch, tmp_path):
    captured = {}

    class _Speech:
        def create(self, **kwargs):
            captured.update(kwargs)
            response = MagicMock()
            response.stream_to_file.side_effect = (
                lambda path: open(path, "wb").write(b"ID3fake")
            )
            return response

    class _Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.audio = MagicMock()
            self.audio.speech = _Speech()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a, **k: "gsk_live")
    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: _Client)

    out = tmp_path / "out.mp3"
    cfg = {
        "groq": {
            "model": "canopylabs/orpheus-arabic-saudi",
            "voice": "hannah",
            "base_url": "https://example.test/openai/v1/",
        }
    }
    assert _generate_groq_tts("hello", str(out), cfg) == str(out)
    assert captured["model"] == "canopylabs/orpheus-arabic-saudi"
    assert captured["voice"] == "hannah"
    assert captured["input"] == "hello"
    assert captured["response_format"] == "mp3"
    assert captured["client"]["api_key"] == "gsk_live"
    assert captured["client"]["base_url"] == "https://example.test/openai/v1"
    assert captured.get("closed") is True
    assert out.read_bytes() == b"ID3fake"


def test_defaults_when_section_missing(monkeypatch, tmp_path):
    captured = {}

    class _Speech:
        def create(self, **kwargs):
            captured.update(kwargs)
            response = MagicMock()
            response.stream_to_file.side_effect = (
                lambda path: open(path, "wb").write(b"RIFF")
            )
            return response

    class _Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.audio = MagicMock()
            self.audio.speech = _Speech()

        def close(self):
            pass

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a, **k: "gsk")
    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: _Client)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)

    out = tmp_path / "out.wav"
    _generate_groq_tts("hi", str(out), {"groq": None})
    assert captured["model"] == DEFAULT_GROQ_TTS_MODEL
    assert captured["voice"] == DEFAULT_GROQ_TTS_VOICE
    assert captured["response_format"] == "wav"
    assert captured["client"]["base_url"] == DEFAULT_GROQ_TTS_BASE_URL


def test_empty_body_fails_closed(monkeypatch, tmp_path):
    class _Speech:
        def create(self, **kwargs):
            response = MagicMock()
            response.stream_to_file = None
            response.content = b""
            return response

    class _Client:
        def __init__(self, **kwargs):
            self.audio = MagicMock()
            self.audio.speech = _Speech()

        def close(self):
            pass

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a, **k: "gsk")
    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: _Client)
    with pytest.raises(ValueError, match="empty audio body"):
        _generate_groq_tts("hi", str(tmp_path / "out.mp3"), {})


def test_provider_errors_are_redacted(monkeypatch, tmp_path):
    class _Speech:
        def create(self, **kwargs):
            raise RuntimeError(
                "POST https://api.groq.com/openai/v1/audio/speech failed "
                "Authorization: Bearer gsk_SUPERSECRETTOKEN status 401"
            )

    class _Client:
        def __init__(self, **kwargs):
            self.audio = MagicMock()
            self.audio.speech = _Speech()

        def close(self):
            pass

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a, **k: "gsk")
    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: _Client)
    with pytest.raises(ValueError) as excinfo:
        _generate_groq_tts("hi", str(tmp_path / "out.mp3"), {})
    msg = str(excinfo.value)
    assert "gsk_SUPERSECRETTOKEN" not in msg
    assert "https://api.groq.com" not in msg
    assert "<redacted>" in msg
    assert "<url>" in msg


def test_timeout_is_bounded(monkeypatch, tmp_path):
    class _Speech:
        def create(self, **kwargs):
            raise TimeoutError("timed out after 60s contacting https://api.groq.com/x")

    class _Client:
        def __init__(self, **kwargs):
            self.audio = MagicMock()
            self.audio.speech = _Speech()

        def close(self):
            pass

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a, **k: "gsk")
    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: _Client)
    with pytest.raises(ValueError, match="Groq TTS request failed"):
        _generate_groq_tts("hi", str(tmp_path / "out.mp3"), {})


def test_response_format_mapping():
    assert _groq_tts_response_format("a.mp3") == "mp3"
    assert _groq_tts_response_format("a.wav") == "wav"
    assert _groq_tts_response_format("a.ogg") == "ogg"
    assert _groq_tts_response_format("a.opus") == "ogg"
    assert _groq_tts_response_format("a.flac") == "flac"
    assert _groq_tts_response_format("a.bin") == "mp3"


def test_bounded_error_truncates():
    msg = _bounded_provider_error(RuntimeError("x" * 500), limit=40)
    assert len(msg) <= 40


def test_sabotage_builtin_name_must_stay():
    """Removal of groq from the built-in set is a regression."""
    from agent.tts_registry import _BUILTIN_NAMES

    assert "groq" in _BUILTIN_NAMES
    assert "groq" in BUILTIN_TTS_PROVIDERS


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="optional live smoke; skipped without existing GROQ_API_KEY",
)
def test_optional_live_smoke(tmp_path):
    out = tmp_path / "live.mp3"
    path = _generate_groq_tts(
        "Hermes Groq TTS smoke.",
        str(out),
        {"groq": {"model": DEFAULT_GROQ_TTS_MODEL, "voice": DEFAULT_GROQ_TTS_VOICE}},
    )
    assert os.path.getsize(path) > 0
