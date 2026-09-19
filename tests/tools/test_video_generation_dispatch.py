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

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"]}

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


    def test_edit_extend_fields_not_in_schema(self):
        from tools.video_generation_tool import VIDEO_GENERATE_SCHEMA
        props = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
        assert "operation" not in props
        assert "video_url" not in props


    def test_upscale_in_schema_and_forwarded(self):
        """`upscale` is advertised per-capability by the dynamic builder
        (#95681 diet — static schema no longer carries it) and forwarded
        to providers when set, omitted (not None) when unset."""
        from tools.video_generation_tool import (
            VIDEO_GENERATE_SCHEMA,
            _build_dynamic_video_schema,
        )
        # Static placeholder: capability args live in the dynamic override.
        props = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
        assert "upscale" not in props

        provider = _RecordingProvider()
        video_gen_registry.register_provider(provider)
        result = self._run({"prompt": "a dog", "upscale": True}, configured="fake")
        assert result["success"] is True
        assert provider.last_kwargs["upscale"] is True

        self._run({"prompt": "a dog"}, configured="fake")
        assert "upscale" not in provider.last_kwargs

    def test_new_capability_param_is_forwarded_not_dropped(self, monkeypatch):
        """A parameter added to _CAPABILITY_PARAMS reaches provider.generate().

        The dynamic schema advertises every entry in that table whose capability
        flag the active provider sets. The handler used to rebuild its forwarded
        set by hand, so a newly declared parameter was shown to the model and
        then silently dropped before dispatch — the schema and the handler could
        disagree with no error anywhere.
        """
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(
            vgt, "_CAPABILITY_PARAMS",
            vgt._CAPABILITY_PARAMS + (
                ("supports_end_image", "end_image_url", {"type": "string"}),
                ("supports_widget_count", "widget_count", {"type": "integer"}),
                ("supports_widget_mode", "widget_mode", {"type": "boolean"}),
            ),
        )

        provider = _RecordingProvider()
        video_gen_registry.register_provider(provider)

        result = self._run(
            {"prompt": "a dog", "end_image_url": "  data:image/png;base64,AAAA  ",
             "widget_count": "3", "widget_mode": True},
            configured="fake")
        assert result["success"] is True
        # String values are stripped, ints and bools coerced, as for the built-ins.
        assert provider.last_kwargs["end_image_url"] == "data:image/png;base64,AAAA"
        assert provider.last_kwargs["widget_count"] == 3
        assert provider.last_kwargs["widget_mode"] is True

        # Unset stays omitted rather than arriving as None.
        self._run({"prompt": "a dog"}, configured="fake")
        assert "end_image_url" not in provider.last_kwargs
        assert "widget_count" not in provider.last_kwargs
        assert "widget_mode" not in provider.last_kwargs
