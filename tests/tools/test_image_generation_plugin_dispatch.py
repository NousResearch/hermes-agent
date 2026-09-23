from __future__ import annotations

import base64
import json

import pytest

from agent import image_gen_registry

# 1x1 transparent PNG — real bytes, so magic-byte sniffing and save_b64_image both succeed.
_B64_PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)).decode()


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class TestPluginDispatch:


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

    def test_foreign_top_level_model_does_not_reach_the_gemini_wire(self, monkeypatch, tmp_path):
        """``image_gen.model`` is provider-agnostic, so it can hold another backend's id.

        The tool forwards it to whichever backend is selected, so exercising the real dispatch path
        is the only way to see this: a plugin unit test that calls ``_resolve_model()`` with no
        argument passes while production POSTs ``/models/gpt-image-2-medium:generateContent``.
        """
        from unittest.mock import MagicMock, patch

        import plugins.image_gen.gemini as gemini_plugin
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-key")
        for var in ("GEMINI_IMAGE_MODEL", "GEMINI_BASE_URL", "GEMINI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  provider: gemini\n  model: gpt-image-2-medium\n")

        provider = gemini_plugin.GeminiImageGenProvider()
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "gemini")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(
            registry_module, "get_provider", lambda name: provider if name == "gemini" else None)

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"candidates": [{
            "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": _B64_PNG}}]},
            "finishReason": "STOP"}]}

        with patch("requests.post", return_value=resp) as mock_post:
            payload = json.loads(
                image_generation_tool._dispatch_to_plugin_provider("draw cat", "square"))

        assert payload["success"] is True
        called_url = mock_post.call_args.args[0]
        assert "gpt-image-2-medium" not in called_url
        assert called_url.endswith(f"/models/{gemini_plugin.DEFAULT_MODEL}:generateContent")
