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


class _FakeProvider(ImageGenProvider):
    def __init__(self, name: str, result=None, exc: Exception | None = None):
        self._name = name
        self.result = result
        self.exc = exc
        self.calls = []

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        self.calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
        if self.exc is not None:
            raise self.exc
        return self.result


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

    def test_dispatch_falls_back_to_configured_provider_when_primary_returns_error(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n"
            "  provider: primary\n"
            "  fallback_providers:\n"
            "    - backup\n"
        )
        primary = _FakeProvider(
            "primary",
            {
                "success": False,
                "image": None,
                "error": "quota exceeded",
                "error_type": "quota",
                "provider": "primary",
            },
        )
        backup = _FakeProvider(
            "backup",
            {
                "success": True,
                "image": "/tmp/backup.png",
                "model": "backup-model",
                "provider": "backup",
            },
        )
        image_gen_registry.register_provider(primary)
        image_gen_registry.register_provider(backup)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw fox", "portrait")
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "backup"
        assert payload["image"] == "/tmp/backup.png"
        assert [call["prompt"] for call in primary.calls] == ["draw fox"]
        assert [call["aspect_ratio"] for call in backup.calls] == ["portrait"]

    def test_dispatch_passes_explicit_fallback_model(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n"
            "  provider: primary\n"
            "  fallback_providers:\n"
            "    - provider: backup\n"
            "      model: backup-model-explicit\n"
        )
        primary = _FakeProvider(
            "primary",
            {"success": False, "image": None, "error": "primary quota", "provider": "primary"},
        )
        backup = _FakeProvider(
            "backup",
            {"success": True, "image": "/tmp/backup.png", "provider": "backup"},
        )
        image_gen_registry.register_provider(primary)
        image_gen_registry.register_provider(backup)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw fox", "square")
        assert dispatched is not None
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert backup.calls == [{"prompt": "draw fox", "aspect_ratio": "square", "model": "backup-model-explicit"}]

    def test_dispatch_aggregates_errors_when_all_fallback_providers_fail(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n"
            "  provider: primary\n"
            "  fallback_providers: [backup]\n"
        )
        primary = _FakeProvider("primary", exc=RuntimeError("primary down"))
        backup = _FakeProvider(
            "backup",
            {
                "success": False,
                "image": None,
                "error": "backup quota",
                "error_type": "quota",
                "provider": "backup",
            },
        )
        image_gen_registry.register_provider(primary)
        image_gen_registry.register_provider(backup)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw fox", "square")
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_fallback_failed"
        assert "primary" in payload["error"]
        assert "primary down" in payload["error"]
        assert "backup" in payload["error"]
        assert "backup quota" in payload["error"]

    def test_dispatch_does_not_fallback_on_non_retryable_policy_failure(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n"
            "  provider: primary\n"
            "  fallback_providers: [backup]\n"
        )
        primary = _FakeProvider(
            "primary",
            {
                "success": False,
                "image": None,
                "error": "content policy rejected prompt",
                "error_type": "content_policy",
                "provider": "primary",
            },
        )
        backup = _FakeProvider(
            "backup",
            {"success": True, "image": "/tmp/backup.png", "provider": "backup"},
        )
        image_gen_registry.register_provider(primary)
        image_gen_registry.register_provider(backup)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw fox", "square")
        assert dispatched is not None
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["provider"] == "primary"
        assert payload["error_type"] == "content_policy"
        assert backup.calls == []

    def test_dispatch_allows_same_provider_with_distinct_fallback_model(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n"
            "  provider: primary\n"
            "  model: fast-model\n"
            "  fallback_providers:\n"
            "    - provider: primary\n"
            "      model: quality-model\n"
        )
        provider = _FakeProvider(
            "primary",
            {"success": False, "image": None, "error": "quota exceeded", "error_type": "quota", "provider": "primary"},
        )
        image_gen_registry.register_provider(provider)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw fox", "square")
        assert dispatched is not None
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_fallback_failed"
        assert provider.calls == [
            {"prompt": "draw fox", "aspect_ratio": "square", "model": "fast-model"},
            {"prompt": "draw fox", "aspect_ratio": "square", "model": "quality-model"},
        ]
