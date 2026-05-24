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
            "kwargs": dict(kwargs),
        })
        return {
            "success": True,
            "image": "/tmp/recording-test.png",
            "model": kwargs.get("model", "test-model"),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "recording",
        }


class _ScriptedProvider(ImageGenProvider):
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    @property
    def name(self) -> str:
        return "scripted"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        self.calls.append({
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "kwargs": dict(kwargs),
        })
        index = min(len(self.calls) - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)


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

    def test_handle_forwards_generation_and_edit_parameters_to_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _RecordingProvider()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recording")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "recording" else None)

        args = {
            "prompt": "change the jacket to red",
            "aspect_ratio": "portrait",
            "size": "1536x1024",
            "quality": "high",
            "n": 2,
            "background": "opaque",
            "output_format": "webp",
            "output_compression": 82,
            "moderation": "low",
            "seed": 123,
            "image": "/tmp/input.png",
            "mask": "/tmp/mask.png",
            "input_fidelity": "high",
        }

        payload = json.loads(image_generation_tool._handle_image_generate(args))

        assert payload["success"] is True
        assert provider.calls == [{
            "prompt": "change the jacket to red",
            "aspect_ratio": "portrait",
            "kwargs": {
                "size": "1536x1024",
                "quality": "high",
                "n": 2,
                "background": "opaque",
                "output_format": "webp",
                "output_compression": 82,
                "moderation": "low",
                "seed": 123,
                "image": "/tmp/input.png",
                "mask": "/tmp/mask.png",
                "input_fidelity": "high",
            },
        }]

    def test_handle_drops_edit_parameters_for_text_to_image(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _RecordingProvider()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recording")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "recording" else None)

        args = {
            "prompt": "draw a cat",
            "aspect_ratio": "square",
            "image": "",
            "mask": "",
            "input_fidelity": "low",
        }

        payload = json.loads(image_generation_tool._handle_image_generate(args))

        assert payload["success"] is True
        assert provider.calls == [{
            "prompt": "draw a cat",
            "aspect_ratio": "square",
            "kwargs": {},
        }]

    def test_handle_drops_empty_mask_for_image_to_image(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _RecordingProvider()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recording")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "recording" else None)

        args = {
            "prompt": "make it cyberpunk",
            "aspect_ratio": "square",
            "image": "/tmp/input.png",
            "mask": "",
            "input_fidelity": "high",
        }

        payload = json.loads(image_generation_tool._handle_image_generate(args))

        assert payload["success"] is True
        assert provider.calls == [{
            "prompt": "make it cyberpunk",
            "aspect_ratio": "square",
            "kwargs": {
                "image": "/tmp/input.png",
                "input_fidelity": "high",
            },
        }]

    def test_dispatch_retries_upstream_eof_with_identical_parameters_five_times(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _ScriptedProvider([
            {
                "success": False,
                "image": None,
                "provider": "scripted",
                "model": "gpt-image-2",
                "prompt": "draw cat",
                "aspect_ratio": "square",
                "error": "OpenAI image generation failed: EOF while reading upstream response",
                "error_type": "api_error",
            }
        ])
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "scripted")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: "gpt-image-2")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "scripted" else None)

        payload = json.loads(image_generation_tool._dispatch_to_plugin_provider(
            "draw cat",
            "square",
            {"size": "1024x1024", "quality": "high", "output_format": "png"},
        ))

        expected_call = {
            "prompt": "draw cat",
            "aspect_ratio": "square",
            "kwargs": {
                "size": "1024x1024",
                "quality": "high",
                "output_format": "png",
                "model": "gpt-image-2",
            },
        }
        assert provider.calls == [expected_call] * 5
        assert payload["success"] is False
        assert payload["error_type"] == "upstream_transport_error"
        assert "5 identical attempts" in payload["error"]
        assert payload["retry_policy"] == {"max_attempts": 5, "same_parameters": True}
        assert len(payload["retry_attempts"]) == 5
        assert payload["parameters"] == {
            "prompt": "draw cat",
            "aspect_ratio": "square",
            "size": "1024x1024",
            "quality": "high",
            "output_format": "png",
            "model": "gpt-image-2",
        }

    def test_dispatch_stops_retrying_after_transient_error_then_success(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _ScriptedProvider([
            ConnectionError("Connection reset by peer"),
            {
                "success": True,
                "image": "/tmp/recovered.png",
                "provider": "scripted",
                "model": "gpt-image-2",
                "prompt": "draw cat",
                "aspect_ratio": "square",
            },
        ])
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "scripted")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: "gpt-image-2")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "scripted" else None)

        payload = json.loads(image_generation_tool._dispatch_to_plugin_provider(
            "draw cat",
            "square",
            {"size": "1024x1024", "quality": "high"},
        ))

        assert len(provider.calls) == 2
        assert provider.calls[0] == provider.calls[1]
        assert payload["success"] is True
        assert payload["image"] == "/tmp/recovered.png"
        assert payload["retry_policy"] == {"max_attempts": 5, "same_parameters": True}
        assert len(payload["retry_attempts"]) == 1

    def test_dispatch_does_not_retry_non_transport_provider_errors(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        provider = _ScriptedProvider([
            {
                "success": False,
                "image": None,
                "provider": "scripted",
                "model": "gpt-image-2",
                "prompt": "draw cat",
                "aspect_ratio": "square",
                "error": "Invalid size parameter",
                "error_type": "invalid_argument",
            }
        ])
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "scripted")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: "gpt-image-2")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "scripted" else None)

        payload = json.loads(image_generation_tool._dispatch_to_plugin_provider("draw cat", "square"))

        assert len(provider.calls) == 1
        assert payload["success"] is False
        assert payload["error_type"] == "invalid_argument"
        assert "retry_attempts" not in payload
