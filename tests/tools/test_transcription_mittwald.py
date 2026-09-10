"""Tests for the mittwald AI Hosting STT provider.

``_transcribe_mittwald`` resolves credentials/base URL and then delegates to
``_transcribe_openai``. These pin the STT-specific gating (an unset
MITTWALD_LLM_API_KEY refuses dispatch), the delegation happy path, and the
base-URL override precedence; the shared OpenAI-SDK behaviour is covered by
``tests/tools/test_transcription.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_openai():
    """OpenAI SDK stub recording the client kwargs and the transcription call."""
    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            transcriptions = MagicMock()

            def _create(**kwargs):
                captured["create_kwargs"] = kwargs
                return MagicMock(text="ok")

            transcriptions.create = _create
            self.audio = MagicMock(transcriptions=transcriptions)

        def close(self):
            pass

    module = MagicMock()
    module.OpenAI = _FakeClient
    module.APIError = Exception
    module.APIConnectionError = ConnectionError
    module.APITimeoutError = TimeoutError
    module.BadRequestError = type("BadRequestError", (Exception,), {})
    return module, captured


def test_get_provider_gating_keys_on_mittwald_api_key(monkeypatch):
    """Explicit-provider gate: MITTWALD_LLM_API_KEY presence flips ``mittwald`` on/off."""
    monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
    from tools.transcription_tools import _get_provider
    assert _get_provider({"provider": "mittwald"}) == "none"
    monkeypatch.setenv("MITTWALD_LLM_API_KEY", "test-key")
    assert _get_provider({"provider": "mittwald"}) == "mittwald"


def test_missing_key_is_an_error_not_a_silent_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)
    with patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_tools import _transcribe_mittwald
        result = _transcribe_mittwald(str(audio), "whisper-large-v3-turbo")
    assert result["success"] is False
    assert "MITTWALD_LLM_API_KEY" in result["error"]


def test_delegates_to_openai_handler_with_mittwald_creds(monkeypatch, tmp_path, fake_openai):
    """Happy path: EU endpoint + key on the client, ``provider="mittwald"`` on the result."""
    monkeypatch.setenv("MITTWALD_LLM_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)
    module, captured = fake_openai

    with patch.dict("sys.modules", {"openai": module}), \
         patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_tools import _transcribe_mittwald
        result = _transcribe_mittwald(str(audio), "whisper-large-v3-turbo")

    assert result["success"] is True
    assert result["provider"] == "mittwald"
    assert captured["base_url"] == "https://llm.aihosting.mittwald.de/v1"
    assert captured["api_key"] == "test-key"
    # Only json/verbose_json are served, and the shared handler already picks json
    # for everything but whisper-1.
    assert captured["create_kwargs"]["response_format"] == "json"
    assert captured["create_kwargs"]["model"] == "whisper-large-v3-turbo"


def test_config_base_url_beats_the_default(monkeypatch, tmp_path, fake_openai):
    monkeypatch.setenv("MITTWALD_LLM_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)
    module, captured = fake_openai
    config = {"mittwald": {"base_url": "https://proxy.example/v1/"}}

    with patch.dict("sys.modules", {"openai": module}), \
         patch("tools.transcription_tools._load_stt_config", return_value=config):
        from tools.transcription_tools import _transcribe_mittwald
        _transcribe_mittwald(str(audio), "whisper-large-v3-turbo")

    assert captured["base_url"] == "https://proxy.example/v1"


def test_language_is_sent_as_iso_639_1(monkeypatch, tmp_path, fake_openai):
    """The transcription endpoint takes ISO codes (unlike mittwald's TTS side)."""
    monkeypatch.setenv("MITTWALD_LLM_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)
    module, captured = fake_openai

    with patch.dict("sys.modules", {"openai": module}), \
         patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_tools import _transcribe_mittwald
        _transcribe_mittwald(str(audio), "whisper-large-v3-turbo", language="de")

    assert captured["create_kwargs"]["language"] == "de"


def test_dispatcher_routes_the_builtin_name(monkeypatch, tmp_path):
    """``mittwald`` is a built-in name, so ``_dispatch_stt_provider`` resolves the handler
    by convention instead of falling through to the command/plugin layers."""
    from tools.transcription_common import BUILTIN_STT_PROVIDERS
    import tools.transcription_tools as tt

    assert "mittwald" in BUILTIN_STT_PROVIDERS
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)
    calls: dict = {}

    def _fake(file_path, model_name, *, language=None, prompt=None):
        calls.update(model=model_name, language=language)
        return {"success": True, "transcript": "ok", "provider": "mittwald"}

    monkeypatch.setattr(tt, "_transcribe_mittwald", _fake)
    result = tt._dispatch_stt_provider(str(audio), "mittwald", {"mittwald": {"language": "de"}})

    assert result["provider"] == "mittwald"
    # Default model comes from the built-in model table, not from the handler's own fallback.
    # ``language`` stays None without a pre_transcription hook by design — the backend does its
    # own stt.<provider>.language resolution (covered by test_language_is_sent_as_iso_639_1).
    assert calls == {"model": "whisper-large-v3-turbo", "language": None}


def test_prompt_is_capped_like_the_other_whisper_backends():
    """mittwald serves whisper-large-v3-turbo, which only conditions on the last ~224 tokens."""
    from tools.transcription_command import (
        _PROMPT_CHARS_PER_TOKEN, _WHISPER_PROMPT_TOKEN_CAP, _enforce_prompt_length_limit)

    max_chars = _WHISPER_PROMPT_TOKEN_CAP * _PROMPT_CHARS_PER_TOKEN
    prompt = "x" * (max_chars + 50)
    capped = _enforce_prompt_length_limit(prompt, "mittwald")
    assert capped is not None and len(capped) == max_chars
