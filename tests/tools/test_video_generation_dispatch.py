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
        self.last_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "model-a"}]

    def default_model(self) -> Optional[str]:
        return "model-a"

    def capabilities(self) -> dict[str, Any]:
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
    def _run(self, args: dict[str, Any], *, configured: Optional[str] = None) -> dict[str, Any]:
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




    def test_upscale_in_schema_and_forwarded(self):
        """`upscale` is advertised per-capability by the dynamic builder
        (#95681 diet — static schema no longer carries it) and forwarded
        to providers when set, omitted (not None) when unset."""
        provider = _RecordingProvider()
        video_gen_registry.register_provider(provider)
        result = self._run({"prompt": "a dog", "upscale": True}, configured="fake")
        assert result["success"] is True
        assert provider.last_kwargs["upscale"] is True

        self._run({"prompt": "a dog"}, configured="fake")
        assert "upscale" not in provider.last_kwargs

    def test_local_image_path_reaches_provider_as_data_url(self, tmp_path):
        """Video APIs fetch sources themselves, so a local path must arrive inlined, not as a path."""
        from PIL import Image
        src = tmp_path / "still.png"
        Image.new("RGB", (8, 8), "red").save(src)
        provider = _RecordingProvider()
        video_gen_registry.register_provider(provider)
        result = self._run({"prompt": "a dog", "image_url": str(src),
                            "reference_image_urls": ["https://example.com/ref.png"]}, configured="fake")
        assert result["success"] is True
        assert provider.last_kwargs["image_url"].startswith("data:image/png;base64,")
        assert provider.last_kwargs["reference_image_urls"] == ["https://example.com/ref.png"]

    @pytest.mark.parametrize("budget", [None, 200_000])
    def test_inline_images_shrink_only_under_a_declared_budget(self, budget):
        """A provider without ``max_inline_image_chars`` gets the caller's data: URL byte-for-byte."""
        import base64
        import io
        import os
        from PIL import Image
        buf = io.BytesIO()
        Image.frombytes("RGB", (1200, 1200), os.urandom(1200 * 1200 * 3)).save(buf, "PNG")  # past FAL's 3.5 MB
        big = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        provider = _RecordingProvider()
        if budget:
            provider.capabilities = lambda: {"modalities": ["text", "image"], "max_inline_image_chars": budget}
        video_gen_registry.register_provider(provider)
        assert self._run({"prompt": "a dog", "image_url": big}, configured="fake")["success"] is True
        sent = provider.last_kwargs["image_url"]
        assert (sent == big) if budget is None else (sent.startswith("data:image/") and len(sent) <= budget)
