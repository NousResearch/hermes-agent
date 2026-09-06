"""tts.openai.api_key / base_url from config.yaml drive the OpenAI audio client.

Regression coverage for issue #26175: the resolver was env-only
(VOICE_TOOLS_OPENAI_KEY / OPENAI_API_KEY) and always returned the default
OpenAI base URL, ignoring the tts.openai config block. Resolution order now
mirrors the STT resolver: config -> env (still honoring config base_url) ->
managed gateway.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import tts_tool, tts_tool_openai


class TestResolveOpenaiAudioClientConfig:
    def test_prefers_tts_config_credentials_and_base_url(self):
        config = {
            "provider": "openai",
            "openai": {
                "api_key": "cfg-key",
                "base_url": "http://localhost:4003/v1",
            },
        }

        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool_openai, "read_selection", return_value="openai"), \
             patch.object(tts_tool_openai, "resolve_openai_audio_api_key", return_value="env-key"), \
             patch.object(tts_tool_openai, "resolve_managed_tool_gateway", return_value=None):
            assert tts_tool_openai._resolve_openai_audio_client_config() == (
                "cfg-key",
                "http://localhost:4003/v1",
                False,
            )

    def test_config_without_base_url_falls_back_to_default_openai_base(self):
        config = {"openai": {"api_key": "cfg-key"}}

        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool_openai, "read_selection", return_value=None):
            assert tts_tool_openai._resolve_openai_audio_client_config() == (
                "cfg-key",
                tts_tool_openai.DEFAULT_OPENAI_BASE_URL,
                False,
            )


    def test_nous_selection_overrides_config_credentials(self):
        """A stored 'nous' selection (or legacy use_gateway: true) routes
        managed even when direct credentials are present."""
        config = {"openai": {"api_key": "cfg-key", "base_url": "http://localhost:4003/v1"}}
        managed = SimpleNamespace(
            nous_user_token="managed-token",
            gateway_origin="https://openai-audio-gateway.nousresearch.com",
        )

        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool_openai, "read_selection", return_value="nous"), \
             patch.object(tts_tool_openai, "resolve_openai_audio_api_key", return_value="env-key"), \
             patch.object(tts_tool_openai, "resolve_managed_tool_gateway", return_value=managed):
            assert tts_tool_openai._resolve_openai_audio_client_config() == (
                "managed-token",
                "https://openai-audio-gateway.nousresearch.com/v1",
                True,
            )

    def test_nous_selection_unentitled_raises_selection_error(self):
        """Selected managed route + unavailable gateway = honest error naming
        the selection, never a silent fall back to direct credentials."""
        config = {"openai": {"api_key": "cfg-key"}}
        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool_openai, "read_selection", return_value="nous"), \
             patch.object(tts_tool_openai, "resolve_openai_audio_api_key", return_value="env-key"), \
             patch.object(tts_tool_openai, "resolve_managed_tool_gateway", return_value=None):
            with pytest.raises(ValueError) as exc:
                tts_tool_openai._resolve_openai_audio_client_config()
        assert "nous" in str(exc.value)
        assert "hermes tools" in str(exc.value)

    def test_vendor_selection_missing_key_raises_selection_error(self):
        """A stored vendor selection with no credentials errors by name —
        NO managed gateway call is attempted."""
        with patch.object(tts_tool, "_load_tts_config", return_value={"provider": "openai"}), \
             patch.object(tts_tool_openai, "read_selection", return_value="openai"), \
             patch.object(tts_tool_openai, "resolve_openai_audio_api_key", return_value=""), \
             patch.object(tts_tool_openai, "resolve_managed_tool_gateway") as gateway_mock:
            with pytest.raises(ValueError) as exc:
                tts_tool_openai._resolve_openai_audio_client_config()
        gateway_mock.assert_not_called()
        assert "openai" in str(exc.value)
        assert "hermes tools" in str(exc.value)

    def test_missing_config_and_env_raises_updated_error(self):
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool_openai, "read_selection", return_value=None), \
             patch.object(tts_tool_openai, "resolve_openai_audio_api_key", return_value=""), \
             patch.object(tts_tool_openai, "resolve_managed_tool_gateway", return_value=None), \
             patch.object(tts_tool_openai, "managed_nous_tools_enabled", return_value=False):
            with pytest.raises(ValueError) as exc:
                tts_tool_openai._resolve_openai_audio_client_config()

        assert (
            str(exc.value)
            == "Neither tts.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        )

    def test_config_api_key_counts_as_available_backend(self):
        config = {"openai": {"api_key": "cfg-key"}}
        with patch.object(tts_tool, "_load_tts_config", return_value=config):
            assert tts_tool_openai._has_openai_audio_backend() is True
