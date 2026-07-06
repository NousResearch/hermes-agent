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
            "model": kwargs.get("model") or "gpt-5.2-codex",
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

    def test_dispatch_honors_explicit_provider_and_model_override(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  provider: fal\n  model: fal-ai/flux-2/klein/9b\n"
        )

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
        monkeypatch.setattr(
            registry_module,
            "get_provider",
            lambda name: _FakeCodexProvider() if name == "codex" else None,
        )

        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "draw cat",
            "square",
            provider_override="codex",
            model_override="gpt-image-2-high",
        )
        assert dispatched is not None
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["model"] == "gpt-image-2-high"

    def test_dispatch_threads_explicit_model_override_to_fal_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module
        from plugins.image_gen.fal import FalImageGenProvider

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  provider: fal\n  model: fal-ai/flux-2/klein/9b\n"
        )

        captured = {}

        def fake_image_generate_tool(**kwargs):
            captured.update(kwargs)
            return json.dumps({
                "success": True,
                "image": "/tmp/fal-test.png",
                "model": kwargs.get("model") or "fal-ai/flux-2/klein/9b",
                "prompt": kwargs["prompt"],
                "aspect_ratio": kwargs["aspect_ratio"],
                "provider": "fal",
            })

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
        monkeypatch.setattr(
            registry_module,
            "get_provider",
            lambda name: FalImageGenProvider() if name == "fal" else None,
        )
        monkeypatch.setattr(image_generation_tool, "image_generate_tool", fake_image_generate_tool)

        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "draw cat",
            "square",
            provider_override="fal",
            model_override="fal-ai/nano-banana-pro",
        )
        assert dispatched is not None
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "fal"
        assert payload["model"] == "fal-ai/nano-banana-pro"
        assert captured["model"] == "fal-ai/nano-banana-pro"

    def test_handle_image_generate_threads_explicit_provider_and_model(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  provider: fal\n  model: fal-ai/flux-2/klein/9b\n"
        )

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
        monkeypatch.setattr(
            registry_module,
            "get_provider",
            lambda name: _FakeCodexProvider() if name == "codex" else None,
        )

        raw = image_generation_tool._handle_image_generate(
            {
                "prompt": "draw cat",
                "aspect_ratio": "portrait",
                "provider": "codex",
                "model": "gpt-image-2-high",
            }
        )
        payload = json.loads(raw)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["model"] == "gpt-image-2-high"
        assert payload["aspect_ratio"] == "portrait"
