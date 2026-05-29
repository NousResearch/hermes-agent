from __future__ import annotations

import json

import pytest

from agent.image_gen_provider import ImageGenProvider


class _EditProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "edit-capable"

    def is_available(self) -> bool:
        return True

    def supports_edit(self) -> bool:
        return True

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):  # pragma: no cover - not used
        raise AssertionError("generate should not be called")

    def edit(self, prompt, images=None, *, image=None, aspect_ratio=None, **kwargs):
        return {
            "success": True,
            "image": "/tmp/edited.png",
            "model": kwargs.get("model", "test-model"),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": self.name,
            "source_image": image,
        }


class _GenerateOnlyProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "generate-only"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):  # pragma: no cover - not used
        return {}


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture(autouse=True)
def _reset_registry():
    from agent import image_gen_registry

    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


def test_image_edit_is_in_image_gen_toolset_but_not_default_core():
    import toolsets
    import tools.image_edit_tool  # noqa: F401 - ensure registry side effect
    from tools.registry import registry

    assert "image_edit" in toolsets.resolve_toolset("image_gen")
    assert "image_edit" not in toolsets.resolve_toolset("hermes")
    assert registry.get_entry("image_edit") is not None


def test_image_edit_dispatches_to_configured_edit_provider(monkeypatch, tmp_path):
    from agent import image_gen_registry as registry_module
    from hermes_cli import plugins as plugins_module
    from tools import image_edit_tool

    home = tmp_path / "hermes-home"
    image_dir = home / "cache" / "images"
    image_dir.mkdir(parents=True)
    source = image_dir / "source.png"
    source.write_bytes(_png_bytes())

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "edit-capable")
    monkeypatch.setattr(image_edit_tool, "_read_configured_image_model", lambda: "configured-model")
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    monkeypatch.setattr(registry_module, "get_provider", lambda name: _EditProvider() if name == "edit-capable" else None)

    payload = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "make the background blue",
        "image": str(source),
        "aspect_ratio": "square",
    }))

    assert payload["success"] is True
    assert payload["provider"] == "edit-capable"
    assert payload["model"] == "configured-model"
    assert payload["source_image"] == str(source.resolve())
    assert payload["aspect_ratio"] == "square"


def test_image_edit_rejects_unsupported_provider(monkeypatch):
    from agent import image_gen_registry as registry_module
    from hermes_cli import plugins as plugins_module
    from tools import image_edit_tool

    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "generate-only")
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    monkeypatch.setattr(registry_module, "get_provider", lambda name: _GenerateOnlyProvider())

    payload = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "edit it",
        "image": "https://example.com/source.png",
    }))

    assert payload["success"] is False
    assert payload["error_type"] == "unsupported"


def test_image_edit_validates_reference_before_provider(monkeypatch):
    from agent import image_gen_registry as registry_module
    from hermes_cli import plugins as plugins_module
    from tools import image_edit_tool

    monkeypatch.setattr(image_edit_tool, "_read_configured_image_provider", lambda: "edit-capable")
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    monkeypatch.setattr(registry_module, "get_provider", lambda name: _EditProvider())

    payload = json.loads(image_edit_tool._handle_image_edit({
        "prompt": "edit it",
        "image": "file:///etc/passwd",
    }))

    assert payload["success"] is False
    assert payload["error_type"] == "invalid_argument"
    assert "Unsupported image reference scheme" in payload["error"]
