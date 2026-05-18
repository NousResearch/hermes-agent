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

        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: "codex"
        )
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

        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "draw cat", "landscape"
        )
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert "image_gen.provider='missing-codex'" in payload["error"]

    def test_dispatch_refreshes_only_bundled_plugins_when_provider_missing(
        self, monkeypatch, tmp_path
    ):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")

        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: "codex"
        )

        calls = []
        provider_state = {"provider": None}

        def fake_ensure_bundled_plugins_discovered():
            calls.append(("bundled", False))
            provider_state["provider"] = _FakeCodexProvider()

        monkeypatch.setattr(
            plugins_module,
            "_ensure_bundled_plugins_discovered",
            fake_ensure_bundled_plugins_discovered,
        )
        monkeypatch.setattr(
            registry_module, "get_provider", lambda name: provider_state["provider"]
        )

        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "draw hammy", "portrait"
        )
        payload = json.loads(dispatched)

        assert calls == [("bundled", False)]
        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["aspect_ratio"] == "portrait"

    def test_dispatch_does_not_hot_load_user_plugins_when_provider_missing(
        self, monkeypatch, tmp_path
    ):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n"
            "  provider: evil\n"
            "plugins:\n"
            "  enabled:\n"
            "    - image_gen/evil\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            plugins_module, "_plugin_manager", plugins_module.PluginManager()
        )

        plugins_module._ensure_plugins_discovered()

        marker = tmp_path / "marker.txt"
        plugin_dir = tmp_path / "plugins" / "image_gen" / "evil"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: evil\nversion: 1.0.0\nkind: backend\n",
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            f"from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
            f"def register(ctx):\n"
            f"    Path({str(marker)!r}).write_text('registered', encoding='utf-8')\n",
            encoding="utf-8",
        )

        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "draw cat", "landscape"
        )
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert not marker.exists()
