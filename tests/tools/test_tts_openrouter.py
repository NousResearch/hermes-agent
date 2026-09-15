"""Tests for the OpenRouter TTS provider.

``_generate_openrouter_tts`` is a thin shim that resolves ``OPENROUTER_API_KEY``
(the same key the chat provider uses) and delegates to ``_generate_openai_tts``
with OpenRouter's base URL — OpenRouter implements the OpenAI
``audio.speech.create`` shape at ``/api/v1/audio/speech``. These tests pin the
credential gating, the delegation shape (base_url + model + voice), the
dispatcher wiring and the reserved-name invariant.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _FakeSpeech:
    def __init__(self, captured):
        self._captured = captured

    def create(self, **kwargs):
        self._captured["create_kwargs"] = kwargs
        return MagicMock()


class _FakeOpenAIClient:
    def __init__(self, captured):
        self._captured = captured

    def __call__(self, api_key=None, base_url=None):
        self._captured["api_key"] = api_key
        self._captured["base_url"] = base_url
        speech = _FakeSpeech(self._captured)
        client = MagicMock()
        client.audio.speech = speech
        return client


def _patch_openai_sdk(captured):
    """Patch the lazy SDK importer with a recording fake (returns the OpenAI class)."""
    return patch("tools.tts_tool._import_openai_client", lambda: _FakeOpenAIClient(captured))


def test_raises_without_openrouter_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from tools.tts_tool_openai import _generate_openrouter_tts

    with patch("tools.tts_tool._resolve_provider_key", lambda *a, **kw: ""):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            _generate_openrouter_tts("hello", str(tmp_path / "out.mp3"), {})


def test_delegates_to_openai_handler_with_openrouter_creds(monkeypatch, tmp_path):
    """Happy path: the OpenAI SDK is pointed at OpenRouter with the key, slug and voice."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict = {}
    out = tmp_path / "out.mp3"

    with _patch_openai_sdk(captured):
        from tools.tts_tool import _generate_openrouter_tts
        result = _generate_openrouter_tts("hello there", str(out), {})

    assert result == str(out)
    assert captured["api_key"] == "test-key"
    assert captured["base_url"].rstrip("/").endswith("openrouter.ai/api/v1")
    kwargs = captured["create_kwargs"]
    assert kwargs["model"] == "deepgram/aura-2"
    assert kwargs["voice"] == "aura-2-thalia-en"
    assert kwargs["input"] == "hello there"
    assert kwargs["response_format"] == "mp3"


def test_config_overrides_model_and_voice(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict = {}

    with _patch_openai_sdk(captured):
        from tools.tts_tool import _generate_openrouter_tts
        _generate_openrouter_tts(
            "hi", str(tmp_path / "out.mp3"),
            {"provider": "openrouter", "openrouter": {"model": "minimax/speech-2.8-turbo",
                                                     "voice": "English_expressive_narrator"}},
        )

    assert captured["create_kwargs"]["model"] == "minimax/speech-2.8-turbo"
    assert captured["create_kwargs"]["voice"] == "English_expressive_narrator"


def test_synthesize_builtin_routes_to_openrouter(monkeypatch, tmp_path):
    """The dispatcher calls the openrouter generator (not the openai one)."""
    from tools import tts_tool

    calls: list = []
    monkeypatch.setattr(tts_tool, "_generate_openrouter_tts",
                        lambda text, path, cfg, **kw: calls.append((text, path, kw)))
    monkeypatch.setattr(tts_tool, "_generate_openai_tts",
                        lambda *a, **kw: calls.append(("openai", a, kw)))

    tts_tool._synthesize_builtin("openrouter", "hi", str(tmp_path / "o.mp3"), {}, "whisper-y")

    assert len(calls) == 1
    assert calls[0][0] == "hi"
    assert calls[0][2]["instructions"] == "whisper-y"


def test_requirements_gate_on_openrouter_key(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "openrouter", "openrouter": {}})
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
    monkeypatch.setattr(tts_tool, "_package_installed", lambda *_a, **_kw: True)

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **kw: "")
    assert tts_tool.check_tts_requirements() is False

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **kw: "test-key")
    assert tts_tool.check_tts_requirements() is True


def test_openrouter_is_a_reserved_builtin():
    """Plugins cannot claim the name, and the two built-in sets must not drift."""
    from agent.tts_registry import _BUILTIN_NAMES
    from tools.tts_command_provider import BUILTIN_TTS_PROVIDERS
    from tools.tts_tool import _BUILTIN_DISPATCH, _FFMPEG_OPUS_PROVIDERS

    assert "openrouter" in BUILTIN_TTS_PROVIDERS
    assert "openrouter" in _BUILTIN_NAMES
    assert "openrouter" in _BUILTIN_DISPATCH
    # OpenRouter returns MP3 (never Ogg/Opus), so voice-bubble platforms need the ffmpeg hop.
    assert "openrouter" in _FFMPEG_OPUS_PROVIDERS


def test_client_direct_voice_uses_the_openai_speech_wire(monkeypatch):
    """Desktop voice skips the relay hop: OpenRouter speaks the OpenAI speech wire."""
    from tools import voice_client_config as vcc
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config",
                        lambda: {"provider": "openrouter", "openrouter": {}})
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **kw: "test-key")

    resolved = vcc._resolve_tts_client_config()

    assert resolved["mode"] == "direct"
    assert resolved["wire"] == vcc.TTS_WIRE_OPENAI
    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"].rstrip("/").endswith("openrouter.ai/api/v1")
    assert resolved["model"] == "deepgram/aura-2"
    assert resolved["voice"] == "aura-2-thalia-en"
