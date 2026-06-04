"""Tests for the unified ``audio_generate`` tool dispatch surface."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from agent import audio_gen_registry
from agent.audio_gen_provider import AudioGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    audio_gen_registry._reset_for_tests()
    yield
    audio_gen_registry._reset_for_tests()


class _RecordingProvider(AudioGenProvider):
    """Captures the kwargs the tool layer hands it."""

    def __init__(self, name: str = "fake"):
        self._name = name
        self.last_kwargs: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": "model-a"}]

    def default_model(self) -> Optional[str]:
        return "model-a"

    def generate(self, prompt, **kwargs):
        self.last_kwargs = {"prompt": prompt, **kwargs}
        return {
            "success": True,
            "audio": "/tmp/out.mp3",
            "model": kwargs.get("model") or "model-a",
            "prompt": prompt,
            "format": kwargs.get("audio_format", "mp3"),
            "duration": kwargs.get("duration") or 0,
            "provider": self._name,
        }


class _RaisingProvider(AudioGenProvider):
    @property
    def name(self) -> str:
        return "raises"

    def generate(self, prompt, **kwargs):
        raise RuntimeError("boom")


class TestUnifiedDispatch:
    def _run(self, args: Dict[str, Any], *, configured: Optional[str] = None) -> Dict[str, Any]:
        from tools import audio_generation_tool
        import hermes_cli.plugins as plugins_module

        saved = audio_generation_tool._read_configured_audio_provider
        audio_generation_tool._read_configured_audio_provider = lambda: configured  # type: ignore
        saved_discover = plugins_module._ensure_plugins_discovered
        plugins_module._ensure_plugins_discovered = lambda *_a, **_k: None  # type: ignore
        try:
            raw = audio_generation_tool._handle_audio_generate(args)
        finally:
            audio_generation_tool._read_configured_audio_provider = saved  # type: ignore
            plugins_module._ensure_plugins_discovered = saved_discover  # type: ignore
        return json.loads(raw)

    def test_no_provider_returns_clear_error(self):
        result = self._run({"prompt": "a jingle"})
        assert result["success"] is False
        assert result["error_type"] == "no_provider_configured"

    def test_unknown_provider_returns_clear_error(self):
        result = self._run({"prompt": "a jingle"}, configured="ghost")
        assert result["success"] is False
        assert result["error_type"] == "provider_not_registered"

    def test_basic_generate_routes(self):
        provider = _RecordingProvider("rec")
        audio_gen_registry.register_provider(provider)
        result = self._run({"prompt": "lofi beat", "duration": 30, "audio_format": "wav"})
        assert result["success"] is True
        assert provider.last_kwargs["duration"] == 30
        assert provider.last_kwargs["audio_format"] == "wav"

    def test_lyrics_and_negative_prompt_passed(self):
        provider = _RecordingProvider("rec")
        audio_gen_registry.register_provider(provider)
        self._run({
            "prompt": "a ballad",
            "lyrics": "hello world",
            "negative_prompt": "no drums",
        })
        assert provider.last_kwargs["lyrics"] == "hello world"
        assert provider.last_kwargs["negative_prompt"] == "no drums"

    def test_model_override_wins(self):
        provider = _RecordingProvider("rec")
        audio_gen_registry.register_provider(provider)
        result = self._run({"prompt": "x", "model": "custom/model"})
        assert result["model"] == "custom/model"
        assert provider.last_kwargs["model"] == "custom/model"

    def test_prompt_required(self):
        provider = _RecordingProvider("rec")
        audio_gen_registry.register_provider(provider)
        result = self._run({"prompt": "   "})
        assert "error" in result
        assert "prompt" in result["error"].lower()

    def test_provider_exception_caught(self):
        audio_gen_registry.register_provider(_RaisingProvider())
        result = self._run({"prompt": "x"})
        assert result["success"] is False
        assert result["error_type"] == "provider_exception"

    def test_schema_required_prompt(self):
        from tools.audio_generation_tool import AUDIO_GENERATE_SCHEMA
        assert AUDIO_GENERATE_SCHEMA["parameters"]["required"] == ["prompt"]
        props = AUDIO_GENERATE_SCHEMA["parameters"]["properties"]
        assert "lyrics" in props
        assert "audio_format" in props

    def test_non_string_prompt_is_clean_error(self):
        """A non-string prompt (tool args are advisory) must return a clean
        tool_error, not raise AttributeError."""
        provider = _RecordingProvider("rec")
        audio_gen_registry.register_provider(provider)
        result = self._run({"prompt": 123})
        assert "error" in result
        assert "prompt" in result["error"].lower()

    def test_non_string_optional_args_coerced(self):
        """Non-string optional args don't crash; they're treated as absent."""
        provider = _RecordingProvider("rec")
        audio_gen_registry.register_provider(provider)
        result = self._run({"prompt": "a song", "lyrics": 5, "model": 7})
        assert result["success"] is True
        # bogus non-string overrides were dropped, not forwarded
        assert "lyrics" not in provider.last_kwargs
        assert provider.last_kwargs.get("model") == "model-a"
