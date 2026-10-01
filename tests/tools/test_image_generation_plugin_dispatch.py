from __future__ import annotations

import json

import pytest

from agent import image_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class TestPluginDispatch:


    def test_handler_forwards_only_the_creative_controls_the_plugin_declares(self, monkeypatch, tmp_path):
        """Declared controls reach generate(); an undeclared one the model sent anyway is dropped, so a
        plugin whose generate() lacks **kwargs never sees it."""
        from agent.image_gen_provider import ImageGenProvider
        from hermes_cli import plugins as plugins_module
        from tools import image_generation_tool

        seen = {}

        class _Recorder(ImageGenProvider):
            @property
            def name(self):
                return "recorder"

            def capabilities(self):
                return {"modalities": ["text"], "creative_controls": ["intensity"]}

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                seen.update(kwargs)
                return {"success": True, "image": "/tmp/recorder.png", "model": "m", "prompt": prompt,
                        "aspect_ratio": aspect_ratio, "provider": "recorder"}

            def list_models(self):
                return []

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        image_gen_registry.register_provider(_Recorder())
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recorder")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda **kwargs: None)

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "draw cat", "aspect_ratio": "square", "intensity": 80, "creativity": "raw"}))

        assert result["success"] is True
        assert seen["intensity"] == 80
        assert "creativity" not in seen

    def test_deepinfra_key_alone_does_not_select_image_backend(self, monkeypatch):
        """DeepInfra chat credentials do not imply consent to image billing."""
        from tools import image_generation_tool

        monkeypatch.setenv("DEEPINFRA_API_KEY", "«redacted:sk-…»")
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: None)
        assert image_generation_tool._dispatch_to_plugin_provider("a cat", "square") is None

    def test_requirements_ignore_unselected_paid_plugin(self, monkeypatch):
        from tools import image_generation_tool

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: None
        )
        assert image_generation_tool.check_image_generation_requirements() is False
