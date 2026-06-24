from __future__ import annotations

import json
import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


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


class _RecordingProvider(ImageGenProvider):
    def __init__(self):
        self.calls = []

    @property
    def name(self) -> str:
        return "recording"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        self.calls.append({
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "kwargs": kwargs,
        })
        return {
            "success": True,
            "image": "/tmp/recording-test.png",
            "model": "recording-model",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "recording",
            "image_url": kwargs.get("image_url"),
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

    def test_schema_exposes_optional_reference_image_url(self):
        from tools import image_generation_tool

        props = image_generation_tool.IMAGE_GENERATE_SCHEMA["parameters"]["properties"]

        assert "image_url" in props
        assert props["image_url"]["type"] == "string"
        assert "reference" in props["image_url"]["description"].lower()
        assert "image_url" not in image_generation_tool.IMAGE_GENERATE_SCHEMA["parameters"].get("required", [])

    def test_handle_forwards_reference_image_url_to_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        image_path = tmp_path / "reference.png"
        image_path.write_bytes(b"fake-png")
        provider = _RecordingProvider()

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recording")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "recording" else None)

        result = image_generation_tool._handle_image_generate({
            "prompt": "use this reference",
            "aspect_ratio": "square",
            "image_url": str(image_path),
        })
        payload = json.loads(result)

        assert payload["success"] is True
        assert payload["image_url"] == str(image_path)
        assert provider.calls == [{
            "prompt": "use this reference",
            "aspect_ratio": "square",
            "kwargs": {"image_url": str(image_path)},
        }]

    def test_dispatch_rejects_missing_reference_image_before_provider_call(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _RecordingProvider()
        missing_path = tmp_path / "missing.png"

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recording")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "recording" else None)

        result = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square", image_url=str(missing_path))
        payload = json.loads(result)

        assert payload["success"] is False
        assert payload["error_type"] == "invalid_reference_image"
        assert str(missing_path) in payload["error"]
        assert provider.calls == []

    def test_dispatch_rejects_http_reference_image_before_provider_call(self, monkeypatch):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _RecordingProvider()

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recording")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "recording" else None)

        result = image_generation_tool._dispatch_to_plugin_provider(
            "draw cat",
            "square",
            image_url="https://example.com/reference.png",
        )
        payload = json.loads(result)

        assert payload["success"] is False
        assert payload["error_type"] == "invalid_reference_image"
        assert "local file path" in payload["error"]
        assert provider.calls == []
