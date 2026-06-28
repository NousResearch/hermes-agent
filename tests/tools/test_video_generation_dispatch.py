"""Tests for the unified ``video_generate`` tool dispatch surface."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from agent import video_gen_registry
from agent.video_gen_provider import VideoGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


class _RecordingProvider(VideoGenProvider):
    """Captures the kwargs the tool layer hands it."""

    def __init__(self, name: str = "fake", capabilities: Optional[Dict[str, Any]] = None):
        self._name = name
        self._capabilities = capabilities or {"modalities": ["text", "image"]}
        self.last_kwargs: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": "model-a"}]

    def default_model(self) -> Optional[str]:
        return "model-a"

    def capabilities(self) -> Dict[str, Any]:
        return self._capabilities

    def generate(self, prompt, **kwargs):
        self.last_kwargs = {"prompt": prompt, **kwargs}
        modality = "image" if kwargs.get("image_url") else "text"
        return {
            "success": True,
            "video": "https://example.com/v.mp4",
            "model": kwargs.get("model") or "model-a",
            "prompt": prompt,
            "modality": modality,
            "aspect_ratio": kwargs.get("aspect_ratio", ""),
            "duration": kwargs.get("duration") or 0,
            "provider": self._name,
        }


class _RaisingProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "raises"

    def generate(self, prompt, **kwargs):
        raise RuntimeError("boom")


class TestUnifiedDispatch:
    def _run(self, args: Dict[str, Any], *, configured: Optional[str] = None) -> Dict[str, Any]:
        from tools import video_generation_tool
        import hermes_cli.plugins as plugins_module

        saved = video_generation_tool._read_configured_video_provider
        video_generation_tool._read_configured_video_provider = lambda: configured  # type: ignore
        saved_discover = plugins_module._ensure_plugins_discovered
        plugins_module._ensure_plugins_discovered = lambda *_a, **_k: None  # type: ignore
        try:
            raw = video_generation_tool._handle_video_generate(args)
        finally:
            video_generation_tool._read_configured_video_provider = saved  # type: ignore
            plugins_module._ensure_plugins_discovered = saved_discover  # type: ignore
        return json.loads(raw)

    def test_no_provider_returns_clear_error(self):
        result = self._run({"prompt": "a dog"})
        assert result["success"] is False
        assert result["error_type"] == "no_provider_configured"

    def test_unknown_provider_returns_clear_error(self):
        result = self._run({"prompt": "a dog"}, configured="ghost")
        assert result["success"] is False
        assert result["error_type"] == "provider_not_registered"

    def test_text_to_video_routes_without_image_url(self):
        provider = _RecordingProvider("rec")
        video_gen_registry.register_provider(provider)
        result = self._run({"prompt": "a happy dog"})
        assert result["success"] is True
        assert result["modality"] == "text"
        assert "image_url" not in provider.last_kwargs
        assert provider.last_kwargs["aspect_ratio"] == "16:9"
        assert provider.last_kwargs["resolution"] == "720p"

    def test_image_to_video_routes_with_image_url(self):
        provider = _RecordingProvider("rec")
        video_gen_registry.register_provider(provider)
        result = self._run({
            "prompt": "animate this",
            "image_url": "https://example.com/img.png",
        })
        assert result["success"] is True
        assert result["modality"] == "image"
        assert provider.last_kwargs["image_url"] == "https://example.com/img.png"

    def test_reference_media_inputs_reach_provider(self):
        provider = _RecordingProvider(
            "rec",
            {
                "modalities": ["text", "image"],
                "max_reference_videos": 3,
                "max_reference_audios": 3,
                "supports_first_frame": True,
                "supports_last_frame": True,
            },
        )
        video_gen_registry.register_provider(provider)
        result = self._run({
            "prompt": "match the references",
            "reference_video_urls": [" https://example.com/ref.mp4 ", ""],
            "reference_audio_urls": "https://example.com/ref.wav",
            "first_frame_url": " https://example.com/first.png ",
            "last_frame_url": "https://example.com/last.png",
        })

        assert result["success"] is True
        assert provider.last_kwargs["reference_video_urls"] == ["https://example.com/ref.mp4"]
        assert provider.last_kwargs["reference_audio_urls"] == ["https://example.com/ref.wav"]
        assert provider.last_kwargs["first_frame_url"] == "https://example.com/first.png"
        assert provider.last_kwargs["last_frame_url"] == "https://example.com/last.png"

    def test_reference_media_inputs_rejected_when_model_lacks_capability(self):
        provider = _RecordingProvider("rec")
        video_gen_registry.register_provider(provider)

        result = self._run({
            "prompt": "match the reference",
            "reference_video_urls": ["https://example.com/ref.mp4"],
        })

        assert "error" in result
        assert "does not support" in result["error"]

    def test_prompt_required(self):
        provider = _RecordingProvider("rec")
        video_gen_registry.register_provider(provider)
        result = self._run({"prompt": "", "image_url": "https://example.com/i.png"})
        assert "error" in result
        assert "prompt" in result["error"].lower()

    def test_edit_extend_args_are_rejected_by_generate_tool(self):
        provider = _RecordingProvider("rec")
        video_gen_registry.register_provider(provider)
        result = self._run({
            "prompt": "make it rain",
            "operation": "edit",
            "video_url": "https://example.com/in.mp4",
        })
        assert "error" in result
        assert "provider-specific tool" in result["error"]

    def test_provider_exception_caught(self):
        video_gen_registry.register_provider(_RaisingProvider())
        result = self._run({"prompt": "x"})
        assert result["success"] is False
        assert result["error_type"] == "provider_exception"

    def test_edit_extend_fields_not_in_schema(self):
        from tools.video_generation_tool import VIDEO_GENERATE_SCHEMA
        properties = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
        assert "operation" not in properties
        assert "video_url" not in properties

    def test_reference_media_fields_are_capability_gated(self, monkeypatch):
        from tools import video_generation_tool

        provider = _RecordingProvider(
            "rec",
            {
                "modalities": ["text", "image"],
                "max_reference_videos": 2,
                "max_reference_audios": 1,
                "supports_first_frame": True,
                "supports_last_frame": True,
            },
        )
        video_gen_registry.register_provider(provider)
        monkeypatch.setattr(
            video_generation_tool,
            "_read_configured_video_provider",
            lambda: "rec",
        )
        monkeypatch.setattr(
            video_generation_tool,
            "_read_configured_video_model",
            lambda: "model-a",
        )

        properties = video_generation_tool._build_dynamic_video_schema()[
            "parameters"
        ]["properties"]
        assert properties["reference_video_urls"]["items"]["type"] == "string"
        assert properties["reference_video_urls"]["maxItems"] == 2
        assert properties["reference_audio_urls"]["items"]["type"] == "string"
        assert properties["first_frame_url"]["type"] == "string"
        assert properties["last_frame_url"]["type"] == "string"

    def test_reference_media_fields_hidden_for_unsupported_model(self, monkeypatch):
        from tools import video_generation_tool

        provider = _RecordingProvider("rec")
        video_gen_registry.register_provider(provider)
        monkeypatch.setattr(
            video_generation_tool,
            "_read_configured_video_provider",
            lambda: "rec",
        )
        monkeypatch.setattr(
            video_generation_tool,
            "_read_configured_video_model",
            lambda: "model-a",
        )

        properties = video_generation_tool._build_dynamic_video_schema()[
            "parameters"
        ]["properties"]
        for field in (
            "reference_video_urls",
            "reference_audio_urls",
            "first_frame_url",
            "last_frame_url",
        ):
            assert field not in properties
