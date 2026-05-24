from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


class _FakeStream:
    def __init__(self, events, final_response):
        self._events = list(events)
        self._final = final_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_response(self):
        return self._final


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class _FakeCodexProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {
            "success": True,
            "image": "/tmp/codex-test.png",
            "model": "gpt-5.2-codex",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "codex",
        }


class TestPluginDispatch:
    def test_dispatch_routes_to_codex_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        image_gen_registry.register_provider(_FakeCodexProvider())

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: _FakeCodexProvider() if name == "codex" else None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["image"] == "/tmp/codex-test.png"
        assert payload["aspect_ratio"] == "square"

    def test_dispatch_reports_missing_registered_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: missing-codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "missing-codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape")
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert "image_gen.provider='missing-codex'" in payload["error"]

    def test_dispatch_force_refreshes_plugins_when_provider_initially_missing(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")

        calls = []
        provider_state = {"provider": None}

        def fake_ensure_plugins_discovered(force=False):
            calls.append(force)
            if force:
                provider_state["provider"] = _FakeCodexProvider()

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", fake_ensure_plugins_discovered)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider_state["provider"])

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw hammy", "portrait")
        payload = json.loads(dispatched)

        assert calls == [False, True]
        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["aspect_ratio"] == "portrait"

    def test_dispatch_routes_custom_provider_to_dynamic_backend(self, monkeypatch):
        from tools import image_generation_tool

        calls = []

        def fake_custom(provider_name, prompt, aspect_ratio, model):
            calls.append((provider_name, prompt, aspect_ratio, model))
            return {
                "success": True,
                "image": "/tmp/yuna-image.png",
                "provider": provider_name,
                "model": model,
            }

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "custom:yuna")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: "gpt-image-2")
        monkeypatch.setattr(image_generation_tool, "_generate_custom_provider_image", fake_custom, raising=False)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
        payload = json.loads(dispatched)

        assert calls == [("custom:yuna", "draw cat", "square", "gpt-image-2")]
        assert payload["success"] is True
        assert payload["provider"] == "custom:yuna"
        assert payload["model"] == "gpt-image-2"

    def test_custom_provider_host_model_uses_main_model_default(self, monkeypatch, tmp_path):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  provider: custom:yuna\n"
            "  default: configured-host-model\n"
            "custom_providers:\n"
            "  - name: yuna\n"
            "    base_url: https://example.test/codex/v1\n"
            "    api_key: sk-test\n"
            "    api_mode: codex_responses\n"
        )

        entry = image_generation_tool._resolve_custom_image_provider_config("custom:yuna")

        assert image_generation_tool._resolve_custom_image_host_model("custom:yuna", entry) == "configured-host-model"

    def test_custom_provider_uses_named_custom_responses_image_generation(self, monkeypatch, tmp_path):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            image_generation_tool,
            "_resolve_custom_image_provider_config",
            lambda provider_name: {
                "name": "yuna",
                "base_url": "https://example.test/v1",
                "api_key": "sk-test",
                "api_mode": "codex_responses",
            },
            raising=False,
        )
        monkeypatch.setattr(
            image_generation_tool,
            "_resolve_custom_image_host_model",
            lambda provider_name, entry: "gpt-5.5",
            raising=False,
        )

        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(
                type="image_generation_call",
                result=_b64_png(),
            )
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.responses = SimpleNamespace(stream=_stream)

        monkeypatch.setattr(image_generation_tool, "_openai_client_class", lambda: _FakeOpenAI, raising=False)

        result = image_generation_tool._generate_custom_provider_image(
            "custom:yuna",
            "a cat",
            "portrait",
            "gpt-image-2",
        )

        assert result["success"] is True
        assert result["provider"] == "custom:yuna"
        assert result["model"] == "gpt-image-2"
        assert result["quality"] == "medium"
        assert Path(result["image"]).exists()
        assert captured["client"]["api_key"] == "sk-test"
        assert captured["client"]["base_url"] == "https://example.test/v1"
        assert captured["model"] == "gpt-5.5"
        assert captured["tools"][0]["type"] == "image_generation"
        assert captured["tools"][0]["model"] == "gpt-image-2"
        assert captured["tools"][0]["size"] == "1024x1536"

    def test_custom_provider_sanitizes_model_name_in_cache_filename(self, monkeypatch, tmp_path):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            image_generation_tool,
            "_resolve_custom_image_provider_config",
            lambda provider_name: {
                "name": "yuna",
                "base_url": "https://example.test/v1",
                "api_key": "sk-test",
                "api_mode": "codex_responses",
            },
            raising=False,
        )
        monkeypatch.setattr(
            image_generation_tool,
            "_resolve_custom_image_host_model",
            lambda provider_name, entry: "gpt-5.5",
            raising=False,
        )

        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(
                type="image_generation_call",
                result=_b64_png(),
            )
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = SimpleNamespace(stream=_stream)

        monkeypatch.setattr(image_generation_tool, "_openai_client_class", lambda: _FakeOpenAI, raising=False)

        result = image_generation_tool._generate_custom_provider_image(
            "custom:yuna",
            "a cat",
            "square",
            "vendor/gpt-image-2-high",
        )

        assert result["success"] is True
        assert result["model"] == "vendor/gpt-image-2-high"
        assert result["quality"] == "high"
        assert captured["tools"][0]["model"] == "vendor/gpt-image-2"
        image_path = Path(result["image"])
        assert image_path.exists()
        assert image_path.parent == tmp_path / "cache" / "images"
        assert image_path.name.startswith("custom_custom_yuna_vendor_gpt-image-2_")
