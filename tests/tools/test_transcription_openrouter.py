"""Unit tests for the OpenRouter STT provider (#24415).

Three test classes cover the three observable surfaces of the new
provider:

- ``TestGetProviderOpenRouter`` -- ``_get_provider()`` resolution
  for both explicit-config and auto-detect paths, with the
  precedence rules from the issue baked in as assertions.
- ``TestTranscribeOpenRouter`` -- the ``_transcribe_openrouter()``
  helper: env-var key resolution, ``stt.openrouter.api_key`` /
  ``base_url`` overrides, the OpenAI-SDK call shape, and every
  error path ``_transcribe_groq`` already covers.
- ``TestTranscribeAudioDispatchOpenRouter`` -- the public
  ``transcribe_audio()`` dispatch plus the model-default chain
  (``stt.openrouter.model`` > ``DEFAULT_OPENROUTER_STT_MODEL``).

All tests mock the OpenAI SDK; no real network traffic, runs on
every CI box.  Mirrors the conventions in
``tests/tools/test_transcription_tools.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Drop every API key the provider chain inspects so tests stay
    hermetic and don't pick up the developer's real credentials."""
    for var in (
        "OPENROUTER_API_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_API_KEY",
        "STT_OPENROUTER_BASE_URL",
        "STT_OPENROUTER_MODEL",
        "HERMES_LOCAL_STT_COMMAND",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sample_audio(tmp_path):
    """A minimal byte-stream audio file the helpers can ``open()``."""
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"fake-ogg-bytes")
    return str(audio_path)


def _patch_load_stt_config(stt_config: dict):
    """Patch ``_load_stt_config`` so the helper sees the test config."""
    return patch(
        "tools.transcription_tools._load_stt_config",
        return_value=stt_config,
    )


# ---------------------------------------------------------------------------
# _get_provider -- OpenRouter resolution
# ---------------------------------------------------------------------------


class TestGetProviderOpenRouter:
    def test_explicit_openrouter_with_key_returns_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openrouter"}) == "openrouter"

    def test_explicit_openrouter_no_key_returns_none(self, monkeypatch):
        """Explicit provider must NOT silently fall back to a different
        cloud provider just because another key is set -- mirrors the
        GH-1774 contract for groq / openai."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-real")
        with patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openrouter"}) == "none"

    def test_explicit_openrouter_without_openai_sdk_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openrouter"}) == "none"

    def test_auto_detect_picks_openrouter_when_only_or_key_set(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._has_openai_audio_backend", return_value=False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "openrouter"

    def test_auto_detect_prefers_groq_over_openrouter(self, monkeypatch):
        """Per the issue: openrouter is last in the cloud chain so a
        dedicated STT key wins -- users who set both want the
        dedicated provider's pricing & routing."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "groq"

    def test_auto_detect_prefers_openai_over_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "openai"

    def test_auto_detect_prefers_xai_over_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._has_openai_audio_backend", return_value=False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "xai"


# ---------------------------------------------------------------------------
# _transcribe_openrouter
# ---------------------------------------------------------------------------


class TestTranscribeOpenRouter:
    def test_no_key_returns_clear_error(self, monkeypatch, sample_audio):
        from tools.transcription_tools import _transcribe_openrouter
        with _patch_load_stt_config({}):
            result = _transcribe_openrouter(sample_audio, "openai/whisper-1")
        assert result["success"] is False
        assert "OPENROUTER_API_KEY" in result["error"]

    def test_openai_package_missing(self, monkeypatch, sample_audio):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        with _patch_load_stt_config({}), \
             patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_openrouter
            result = _transcribe_openrouter(sample_audio, "openai/whisper-1")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_happy_path_uses_env_key_and_default_base_url(
        self, monkeypatch, sample_audio,
    ):
        """The headline contract: ``OPENROUTER_API_KEY`` from env reaches
        the OpenAI SDK along with the OpenRouter base URL."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")

        fake_openai_module = MagicMock()
        fake_client = MagicMock()
        fake_openai_module.OpenAI.return_value = fake_client
        fake_openai_module.APIError = Exception
        fake_openai_module.APIConnectionError = Exception
        fake_openai_module.APITimeoutError = Exception
        fake_client.audio.transcriptions.create.return_value = "hello world"

        with _patch_load_stt_config({}), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch.dict("sys.modules", {"openai": fake_openai_module}):
            from tools.transcription_tools import (
                OPENROUTER_STT_BASE_URL,
                _transcribe_openrouter,
            )
            result = _transcribe_openrouter(sample_audio, "openai/whisper-1")

        assert result == {
            "success": True,
            "transcript": "hello world",
            "provider": "openrouter",
        }
        kwargs = fake_openai_module.OpenAI.call_args.kwargs
        assert kwargs["api_key"] == "sk-or-real"
        assert kwargs["base_url"] == OPENROUTER_STT_BASE_URL
        # response_format=text is the OpenAI/Groq Whisper default that
        # also works against OpenRouter -- pin it so a refactor can't
        # silently switch to JSON and break legacy whisper-1 routing.
        create_kwargs = fake_client.audio.transcriptions.create.call_args.kwargs
        assert create_kwargs["response_format"] == "text"
        assert create_kwargs["model"] == "openai/whisper-1"

    def test_config_api_key_wins_over_env(self, monkeypatch, sample_audio):
        """``stt.openrouter.api_key`` overrides the env var so users on
        managed deployments can pin per-deployment credentials."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI.return_value.audio.transcriptions.create.return_value = "ok"
        fake_openai_module.APIError = Exception
        fake_openai_module.APIConnectionError = Exception
        fake_openai_module.APITimeoutError = Exception

        cfg = {"openrouter": {"api_key": "config-key"}}
        with _patch_load_stt_config(cfg), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch.dict("sys.modules", {"openai": fake_openai_module}):
            from tools.transcription_tools import _transcribe_openrouter
            _transcribe_openrouter(sample_audio, "openai/whisper-1")

        kwargs = fake_openai_module.OpenAI.call_args.kwargs
        assert kwargs["api_key"] == "config-key", (
            "stt.openrouter.api_key in config must win over the env var"
        )

    def test_config_base_url_override_is_respected(
        self, monkeypatch, sample_audio,
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI.return_value.audio.transcriptions.create.return_value = "ok"
        fake_openai_module.APIError = Exception
        fake_openai_module.APIConnectionError = Exception
        fake_openai_module.APITimeoutError = Exception

        cfg = {"openrouter": {"base_url": "https://proxy.example.com/v1/"}}
        with _patch_load_stt_config(cfg), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch.dict("sys.modules", {"openai": fake_openai_module}):
            from tools.transcription_tools import _transcribe_openrouter
            _transcribe_openrouter(sample_audio, "openai/whisper-1")

        kwargs = fake_openai_module.OpenAI.call_args.kwargs
        # Trailing slash stripped to match how the OpenAI SDK
        # builds endpoint URLs internally.
        assert kwargs["base_url"] == "https://proxy.example.com/v1"

    def test_api_error_wrapped(self, monkeypatch, sample_audio):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

        # Three distinct subclasses so the except-chain in
        # _transcribe_openrouter routes APIError to the right handler
        # rather than matching the broader Exception base first.
        class _APIError(Exception):
            pass

        class _APIConnectionError(Exception):
            pass

        class _APITimeoutError(Exception):
            pass

        fake_openai_module = MagicMock()
        fake_client = MagicMock()
        fake_openai_module.OpenAI.return_value = fake_client
        fake_openai_module.APIError = _APIError
        fake_openai_module.APIConnectionError = _APIConnectionError
        fake_openai_module.APITimeoutError = _APITimeoutError
        fake_client.audio.transcriptions.create.side_effect = _APIError("rate limited")

        with _patch_load_stt_config({}), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch.dict("sys.modules", {"openai": fake_openai_module}):
            from tools.transcription_tools import _transcribe_openrouter
            result = _transcribe_openrouter(sample_audio, "openai/whisper-1")

        assert result["success"] is False
        assert "API error" in result["error"]
        assert "rate limited" in result["error"]

    def test_permission_error_returns_clear_message(
        self, monkeypatch, sample_audio,
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI.side_effect = PermissionError("no read")
        fake_openai_module.APIError = Exception
        fake_openai_module.APIConnectionError = Exception
        fake_openai_module.APITimeoutError = Exception

        with _patch_load_stt_config({}), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch.dict("sys.modules", {"openai": fake_openai_module}):
            from tools.transcription_tools import _transcribe_openrouter
            result = _transcribe_openrouter(sample_audio, "openai/whisper-1")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# transcribe_audio() -- end-to-end dispatch + model defaults
# ---------------------------------------------------------------------------


class TestTranscribeAudioDispatchOpenRouter:
    def test_dispatch_routes_to_openrouter_with_default_model(
        self, monkeypatch, sample_audio,
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

        cfg = {"enabled": True, "provider": "openrouter"}
        with _patch_load_stt_config(cfg), \
             patch(
                 "tools.transcription_tools._transcribe_openrouter",
                 return_value={"success": True, "transcript": "hi", "provider": "openrouter"},
             ) as mock_transcribe, \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import (
                DEFAULT_OPENROUTER_STT_MODEL,
                transcribe_audio,
            )
            result = transcribe_audio(sample_audio)

        assert result["success"] is True
        assert result["provider"] == "openrouter"
        # Default model wired through
        called_model = mock_transcribe.call_args.args[1]
        assert called_model == DEFAULT_OPENROUTER_STT_MODEL == "openai/whisper-1"

    def test_dispatch_uses_config_model_override(
        self, monkeypatch, sample_audio,
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        cfg = {
            "enabled": True,
            "provider": "openrouter",
            "openrouter": {"model": "openai/gpt-4o-mini-transcribe"},
        }
        with _patch_load_stt_config(cfg), \
             patch(
                 "tools.transcription_tools._transcribe_openrouter",
                 return_value={"success": True, "transcript": "", "provider": "openrouter"},
             ) as mock_transcribe, \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_audio)

        assert mock_transcribe.call_args.args[1] == "openai/gpt-4o-mini-transcribe"

    def test_dispatch_explicit_model_argument_wins(
        self, monkeypatch, sample_audio,
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        cfg = {
            "enabled": True,
            "provider": "openrouter",
            "openrouter": {"model": "openai/whisper-1"},
        }
        with _patch_load_stt_config(cfg), \
             patch(
                 "tools.transcription_tools._transcribe_openrouter",
                 return_value={"success": True, "transcript": "", "provider": "openrouter"},
             ) as mock_transcribe, \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_audio, model="openai/gpt-4o-transcribe")

        assert mock_transcribe.call_args.args[1] == "openai/gpt-4o-transcribe"

    def test_no_provider_error_message_mentions_openrouter(
        self, monkeypatch, sample_audio,
    ):
        """The fallback hint must list every supported provider so
        users discover OpenRouter without spelunking the source."""
        with _patch_load_stt_config({"enabled": True}), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._has_openai_audio_backend", return_value=False):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_audio)

        assert result["success"] is False
        assert "OPENROUTER_API_KEY" in result["error"]
        assert "OpenRouter" in result["error"]
