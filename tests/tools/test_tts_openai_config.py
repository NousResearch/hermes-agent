from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import tts_tool


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
             patch.object(tts_tool, "prefers_gateway", return_value=False), \
             patch.object(tts_tool, "resolve_openai_audio_api_key", return_value="env-key"), \
             patch.object(tts_tool, "resolve_managed_tool_gateway", return_value=None):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "cfg-key",
                "http://localhost:4003/v1",
            )

    def test_config_without_base_url_falls_back_to_default_openai_base(self):
        config = {"openai": {"api_key": "cfg-key"}}

        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool, "prefers_gateway", return_value=False):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "cfg-key",
                tts_tool.DEFAULT_OPENAI_BASE_URL,
            )

    def test_use_gateway_overrides_config_credentials(self):
        config = {"openai": {"api_key": "cfg-key", "base_url": "http://localhost:4003/v1"}}
        managed = SimpleNamespace(
            nous_user_token="managed-token",
            gateway_origin="https://openai-audio-gateway.nousresearch.com",
        )

        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool, "prefers_gateway", return_value=True), \
             patch.object(tts_tool, "resolve_openai_audio_api_key", return_value="env-key"), \
             patch.object(tts_tool, "resolve_managed_tool_gateway", return_value=managed):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "managed-token",
                "https://openai-audio-gateway.nousresearch.com/v1",
            )

    def test_missing_config_and_env_raises_updated_error(self):
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "prefers_gateway", return_value=False), \
             patch.object(tts_tool, "resolve_openai_audio_api_key", return_value=""), \
             patch.object(tts_tool, "resolve_managed_tool_gateway", return_value=None), \
             patch.object(tts_tool, "managed_nous_tools_enabled", return_value=False):
            with pytest.raises(ValueError) as exc:
                tts_tool._resolve_openai_audio_client_config()

        assert (
            str(exc.value)
            == "Neither tts.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        )
