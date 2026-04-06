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

    def test_auto_dispatches_to_matching_provider_when_image_gen_unset(self, monkeypatch):
        """``image_gen.provider`` unset → dispatch to the registry's active
        provider (resolved via _resolve_active_image_provider), else fall
        through (None)."""
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(image_generation_tool, "_resolve_active_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **kw: None)
        image_gen_registry.register_provider(_FakeCodexProvider())
        monkeypatch.setattr(
            registry_module, "get_provider",
            lambda name: _FakeCodexProvider() if name == "codex" else None,
        )

        # Active provider resolved → auto-dispatch.
        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape")
        assert dispatched is not None
        assert json.loads(dispatched)["provider"] == "codex"

        # Nothing available (or legacy FAL) → returns None (caller drops to
        # the in-tree FAL pipeline).
        monkeypatch.setattr(image_generation_tool, "_resolve_active_image_provider", lambda: None)
        assert image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape") is None
        monkeypatch.setattr(image_generation_tool, "_resolve_active_image_provider", lambda: "fal")
        assert image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape") is None

    def test_deepinfra_bootstrap_no_config_changes_needed(self, monkeypatch):
        """Bootstrap regression: with ``DEEPINFRA_API_KEY`` set and no FAL
        credentials, the dispatcher must route to the bundled DeepInfra plugin
        without any ``image_gen.provider`` entry — i.e. the user never sees the
        FAL ``FAL_KEY isn't set`` fallback. The unset-config path now goes
        through the availability-filtered registry (get_active_provider), so
        the single credentialled backend (DeepInfra) is selected automatically."""
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from plugins.image_gen import deepinfra as deepinfra_plugin
        from plugins.image_gen.deepinfra import DeepInfraImageGenProvider

        # Simulate: DEEPINFRA_API_KEY set, no FAL_KEY, fresh-out-of-box config.
        monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-test-bootstrap")
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: None)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **kw: None)

        # Only DeepInfra is registered (autouse fixture reset the registry), so
        # the registry's single-available fallback selects it — no bespoke
        # model.provider inference needed.
        image_gen_registry.register_provider(DeepInfraImageGenProvider())

        # Stub the live catalog so DeepInfra has at least one model to pick.
        from hermes_cli import models as models_mod
        monkeypatch.setattr(
            models_mod, "_fetch_deepinfra_models_by_tag",
            lambda tag, **kw: (
                [{"id": "black-forest-labs/FLUX.1-dev", "metadata": {}}]
                if tag == "image-gen" else []
            ),
        )
        # Avoid a real network fetch when caching the (stubbed) delivery URL.
        monkeypatch.setattr(
            deepinfra_plugin, "save_url_image",
            lambda url, **kw: __import__("pathlib").Path("/tmp/deepinfra_test.png"),
        )

        # Stub openai so we don't hit the network.
        import openai
        class _Images:
            def generate(self, **kw):
                class _Resp:
                    class _Data:
                        b64_json = None
                        url = "https://example.com/img.png"
                    data = [_Data()]
                return _Resp()
        class _Client:
            def __init__(self, **kw):
                self.images = _Images()
        monkeypatch.setattr(openai, "OpenAI", _Client)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("a cat", "square")
        assert dispatched is not None, "auto-resolution must dispatch to DeepInfra — falling through to FAL is the bug"
        payload = json.loads(dispatched)
        assert payload["provider"] == "deepinfra"
        assert payload["model"] == "black-forest-labs/FLUX.1-dev"
