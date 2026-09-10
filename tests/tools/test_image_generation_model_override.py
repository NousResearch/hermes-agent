"""Per-call ``model`` override on ``image_generate`` (#45278).

The tool serves one user-configured model for every call; the calling agent
had no way to route a specific request to a different image model (e.g. a
text-rendering prompt to ``gpt-image-2`` while ``nano-banana-pro`` is the
configured default). ``video_generate`` already exposes this exact surface:
an optional ``model`` override, defaulting to the configured model, validated
against the provider's own catalog. These tests pin the same contract:

- Resolution precedence: explicit arg > config > catalog default
- Unknown models are rejected with a catalog-listing error
- The chosen model is reported back in the success envelope
- The dynamic schema advertises the override (all models honor it)
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def image_tool():
    """Fresh import of tools.image_generation_tool per test."""
    import tools.image_generation_tool as mod
    return importlib.reload(mod)


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_cfg(home, cfg: dict):
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _patch_route(monkeypatch, image_tool):
    """Make every guard/route in _handle_image_generate resolvable and the
    plugin/Krea routes fall through to the in-tree FAL path."""
    monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
    monkeypatch.setattr(image_tool, "_maybe_route_managed_krea", lambda *a, **k: None)
    monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
    monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)
    monkeypatch.setattr(
        image_tool, "_load_fal_client",
        lambda: SimpleNamespace(submit=lambda *a, **k: {}))


class TestFalModelOverrideRouting:
    """Explicit ``model`` overrides ``image_gen.model`` in the in-tree FAL path."""

    def _generate(self, image_tool, monkeypatch, args):
        _patch_route(monkeypatch, image_tool)
        submitted = {}
        captured = {}

        def _fake_submit(endpoint, arguments=None):
            submitted["endpoint"] = endpoint
            submitted["arguments"] = dict(arguments or {})
            return SimpleNamespace(get=lambda: {"images": [
                {"url": "https://example.com/img.png"}]})

        monkeypatch.setattr(image_tool, "_submit_fal_request", _fake_submit)
        raw = image_tool._handle_image_generate(dict(args, task_id=None))
        captured.update(submitted)
        return json.loads(raw), captured

    def test_explicit_model_beats_config(self, image_tool, cfg_home, monkeypatch):
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        result, captured = self._generate(image_tool, monkeypatch, {
            "prompt": "a poster with readable text",
            "model": "fal-ai/gpt-image-2",
        })
        assert result["success"] is True
        assert captured["endpoint"] == "fal-ai/gpt-image-2"
        assert result["model"] == "fal-ai/gpt-image-2"

    def test_unknown_model_rejected_with_catalog_hint(self, image_tool, cfg_home, monkeypatch):
        result, captured = self._generate(image_tool, monkeypatch, {
            "prompt": "a cat", "model": "fal-ai/nonexistent"})
        assert result["success"] is False
        assert captured == {}
        assert "fal-ai/gpt-image-2" in result["error"]
        assert "image_gen.model" in result["error"]

    def test_no_override_keeps_configured_model(self, image_tool, cfg_home, monkeypatch):
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        result, captured = self._generate(image_tool, monkeypatch, {"prompt": "a cat"})
        assert result["success"] is True
        assert captured["endpoint"] == "fal-ai/nano-banana-pro"
        assert result["model"] == "fal-ai/nano-banana-pro"

    def test_whitespace_model_treated_as_unset(self, image_tool, cfg_home, monkeypatch):
        result, captured = self._generate(image_tool, monkeypatch, {
            "prompt": "a cat", "model": "   "})
        assert result["success"] is True
        assert captured["endpoint"] == "fal-ai/flux-2/klein/9b"


class TestManagedKreaModelOverride:
    """A per-call ``model`` naming a native ``krea-2-*`` id routes through the
    managed Krea gateway even when the configured model is not Krea."""

    def _krea_env(self, monkeypatch, image_tool):
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(
            image_tool, "_read_configured_image_model", lambda: "fal-ai/flux-2/klein/9b")
        import plugins.image_gen.krea as krea_mod
        monkeypatch.setattr(
            krea_mod,
            "_resolve_managed_krea_gateway",
            lambda: SimpleNamespace(
                vendor="krea", gateway_origin="https://krea-gateway.example.com",
                nous_user_token="tok", managed_mode=True))
        fake_provider = MagicMock()
        fake_provider.generate.return_value = {"success": True, "image": "/tmp/x.png"}
        monkeypatch.setattr(
            "agent.image_gen_registry.get_provider", lambda name: fake_provider)
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None)
        return fake_provider

    def test_override_krea_model_routes_to_managed_krea(self, image_tool, cfg_home, monkeypatch):
        fake_provider = self._krea_env(monkeypatch, image_tool)
        raw = image_tool._handle_image_generate({
            "prompt": "a cat", "model": "krea-2-large", "task_id": None})
        assert json.loads(raw)["success"] is True
        assert fake_provider.generate.call_args.kwargs["model"] == "krea-2-large"

    def test_non_krea_override_does_not_intercept(self, image_tool, cfg_home, monkeypatch):
        self._krea_env(monkeypatch, image_tool)
        submitted = {}

        def _fake_submit(endpoint, arguments=None):
            submitted["endpoint"] = endpoint
            return SimpleNamespace(get=lambda: {"images": [{"url": "https://x/y.png"}]})

        monkeypatch.setattr(image_tool, "_submit_fal_request", _fake_submit)
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)
        monkeypatch.setattr(image_tool, "_load_fal_client", lambda: SimpleNamespace())
        result = json.loads(image_tool._handle_image_generate({
            "prompt": "a cat", "model": "fal-ai/gpt-image-2", "task_id": None}))
        assert result["success"] is True
        assert submitted["endpoint"] == "fal-ai/gpt-image-2"

    def test_configured_krea_with_fal_override_routes_to_fal(self, image_tool, cfg_home, monkeypatch):
        # Regression (review finding): the configured Krea model must NOT intercept
        # a per-call override that names a FAL model.
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(
            image_tool, "_read_configured_image_model", lambda: "krea-2-large")
        monkeypatch.setattr(
            "plugins.image_gen.krea._resolve_managed_krea_gateway",
            lambda: SimpleNamespace(managed_mode=True))
        monkeypatch.setattr(
            "agent.image_gen_registry.get_provider", lambda name: MagicMock())
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None)
        submitted = {}

        def _fake_submit(endpoint, arguments=None):
            submitted["endpoint"] = endpoint
            return SimpleNamespace(get=lambda: {"images": [{"url": "https://x/y.png"}]})

        monkeypatch.setattr(image_tool, "_submit_fal_request", _fake_submit)
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)
        monkeypatch.setattr(image_tool, "_load_fal_client", lambda: SimpleNamespace())
        result = json.loads(image_tool._handle_image_generate({
            "prompt": "a cat", "model": "fal-ai/gpt-image-2", "task_id": None}))
        assert result["success"] is True
        assert submitted["endpoint"] == "fal-ai/gpt-image-2"


class TestPluginDispatchModelOverride:
    """The per-call model reaches plugin providers as a ``model`` kwarg."""

    def _plugin_env(self, monkeypatch, image_tool):
        monkeypatch.setattr(image_tool, "_maybe_route_managed_krea", lambda *a, **k: None)
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "openai")
        monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: "gpt-image-2")
        fake_provider = MagicMock()
        fake_provider.generate.return_value = {"success": True, "image": "/tmp/x.png"}
        monkeypatch.setattr(
            "agent.image_gen_registry.get_provider", lambda name: fake_provider)
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None)
        return fake_provider

    def test_explicit_model_beats_configured(self, image_tool, cfg_home, monkeypatch):
        fake_provider = self._plugin_env(monkeypatch, image_tool)
        raw = image_tool._handle_image_generate({
            "prompt": "a cat", "model": "gpt-image-1-mini", "task_id": None})
        result = json.loads(raw)
        assert result["success"] is True
        kwargs = fake_provider.generate.call_args.kwargs
        assert kwargs["model"] == "gpt-image-1-mini"

    def test_no_override_keeps_configured_model_kwarg(self, image_tool, cfg_home, monkeypatch):
        fake_provider = self._plugin_env(monkeypatch, image_tool)
        raw = image_tool._handle_image_generate({"prompt": "a cat", "task_id": None})
        result = json.loads(raw)
        assert result["success"] is True
        kwargs = fake_provider.generate.call_args.kwargs
        assert kwargs["model"] == "gpt-image-2"


class TestSchemaAdvertisesOverride:
    """The dynamic schema surfaces the override + the active model identity."""

    def _schema(self, monkeypatch, image_tool, model_id):
        monkeypatch.setattr(
            image_tool, "_resolve_fal_model",
            lambda: (model_id, image_tool.FAL_MODELS[model_id]))
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
        return image_tool._build_dynamic_image_schema()

    def test_model_param_present_with_active_model_name(self, image_tool, monkeypatch):
        schema = self._schema(monkeypatch, image_tool, "fal-ai/flux-2/klein/9b")
        props = schema["parameters"]["properties"]
        assert "model" in props
        assert props["model"]["type"] == "string"
        assert props["model"]["default"] == "fal-ai/flux-2/klein/9b"
        # Exact catalog id leads (agents copy from the description); display name in parens.
        assert "fal-ai/flux-2/klein/9b" in schema["description"]
        assert "FLUX 2 Klein 9B" in schema["description"]
        assert "override" in props["model"]["description"].lower()

    def test_no_enum_on_model_param(self, image_tool, monkeypatch):
        # video_generate precedent: free string. An enum + configured-default-outside-
        # catalog combo is invalid JSON Schema on strict API validators.
        schema = self._schema(monkeypatch, image_tool, "fal-ai/flux-2/klein/9b")
        assert "enum" not in schema["parameters"]["properties"]["model"]

    def test_text_rendering_strength_surfaced(self, image_tool, monkeypatch):
        schema = self._schema(monkeypatch, image_tool, "fal-ai/flux-2/klein/9b")
        assert "text" in schema["description"].lower()

    def test_text_only_active_model_keeps_edit_args_advertised(self, image_tool, monkeypatch):
        # #45278 core ask: editing must stay reachable via a model override even when
        # the CONFIGURED model is text-only.
        text_only = next(
            m for m, meta in image_tool.FAL_MODELS.items() if not meta.get("edit_endpoint"))
        schema = self._schema(monkeypatch, image_tool, text_only)
        props = schema["parameters"]["properties"]
        assert "image_url" in props
        assert "model override" in schema["description"]
