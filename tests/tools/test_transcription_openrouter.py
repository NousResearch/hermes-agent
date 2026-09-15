"""Tests for the OpenRouter STT provider.

``_transcribe_openrouter`` is a thin shim that resolves ``OPENROUTER_API_KEY``
(the same key the chat provider uses), normalizes the model to a
vendor-prefixed catalog slug, then delegates to ``_transcribe_openai`` with
OpenRouter's base URL. These tests pin the credential gating, the delegation
happy path (base_url + provider label), and the native-model-name swap.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _fake_openai_module(captured: dict) -> MagicMock:
    """Minimal ``openai`` stub recording the client + create() kwargs."""

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            transcriptions = MagicMock()
            create = MagicMock(return_value=MagicMock(text="ok"))
            transcriptions.create = create
            captured["create"] = create
            self.audio = MagicMock(transcriptions=transcriptions)

        def close(self):
            pass

    fake_openai = MagicMock()
    fake_openai.OpenAI = _FakeClient
    fake_openai.APIError = Exception
    fake_openai.APIConnectionError = ConnectionError
    fake_openai.APITimeoutError = TimeoutError
    return fake_openai


def test_get_provider_gating_keys_on_openrouter_api_key(monkeypatch):
    """Explicit-provider gate: OPENROUTER_API_KEY presence flips ``openrouter`` on/off."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from tools.transcription_tools import _get_provider

    with patch("tools.transcription_tools._load_stt_config", return_value={}):
        assert _get_provider({"provider": "openrouter"}) == "none"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("tools.transcription_tools._load_stt_config", return_value={}):
        assert _get_provider({"provider": "openrouter"}) == "openrouter"


def test_delegates_to_openai_handler_with_openrouter_creds(monkeypatch, tmp_path):
    """Happy path: vendor slug → openai SDK called with OpenRouter base_url + key,
    and the response carries ``provider="openrouter"`` (not openai)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    captured: dict = {}
    fake_openai = _fake_openai_module(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}), \
         patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_tools import _transcribe_openrouter
        result = _transcribe_openrouter(str(audio), "openai/whisper-large-v3")

    assert result["success"] is True
    assert result["provider"] == "openrouter"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"].rstrip("/").endswith("openrouter.ai/api/v1")
    assert captured["create"].call_args.kwargs["model"] == "openai/whisper-large-v3"


def test_native_model_name_is_swapped_for_a_catalog_slug(monkeypatch, tmp_path):
    """``whisper-1`` (an OpenAI-native name) can't resolve on OpenRouter's
    vendor-prefixed catalog, so it falls back to the default slug."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    captured: dict = {}
    fake_openai = _fake_openai_module(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}), \
         patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_common import DEFAULT_OPENROUTER_STT_MODEL
        from tools.transcription_tools import _transcribe_openrouter
        result = _transcribe_openrouter(str(audio), "whisper-1")

    assert result["success"] is True
    assert captured["create"].call_args.kwargs["model"] == DEFAULT_OPENROUTER_STT_MODEL


def test_missing_key_refuses_before_any_request(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    with patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_tools import _transcribe_openrouter
        result = _transcribe_openrouter(str(audio), "openai/whisper-large-v3")

    assert result["success"] is False
    assert "OPENROUTER_API_KEY" in result["error"]


def test_default_model_comes_from_common():
    """The picker default and the runtime default are one value."""
    from tools.transcription_common import DEFAULT_OPENROUTER_STT_MODEL, OPENROUTER_STT_MODELS

    assert DEFAULT_OPENROUTER_STT_MODEL in OPENROUTER_STT_MODELS


@pytest.mark.parametrize("provider", ["openrouter"])
def test_openrouter_is_a_reserved_builtin(provider):
    """Plugins cannot claim the name — the built-in wins (see transcription_registry)."""
    from agent.transcription_registry import _BUILTIN_NAMES
    from tools.transcription_common import BUILTIN_STT_PROVIDERS

    assert provider in BUILTIN_STT_PROVIDERS
    assert provider in _BUILTIN_NAMES
