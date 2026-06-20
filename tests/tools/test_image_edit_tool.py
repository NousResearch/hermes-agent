"""Tests for the ``image_edit`` tool surface (tools/image_edit_tool.py).

The tool is a thin dispatcher: it validates inputs, resolves the active
image-gen provider via the existing ``image_gen.provider`` logic, and either
calls ``provider.edit(...)`` or returns a clear ``unsupported_capability``
result. xAI-specific behavior lives in the provider; these tests cover the
tool contract only.
"""

from __future__ import annotations

import json

import pytest

from agent.image_gen_provider import ImageGenProvider


class _FakeEditProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "fakeedit"

    def supports_edit(self) -> bool:
        return True

    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": "/tmp/gen.png", "provider": "fakeedit"}

    def edit(self, prompt, image=None, aspect_ratio="landscape", *, images=None, **kw):
        return {
            "success": True,
            "image": "/tmp/edited.png",
            "model": "fake-edit",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "fakeedit",
            "_received_image": image,
            "_received_images": images,
        }


class _FakeGenOnlyProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "genonly"

    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": "/tmp/gen.png", "provider": "genonly"}


def _patch_active(monkeypatch, provider):
    from agent import image_gen_registry
    from hermes_cli import plugins as plugins_module
    from tools import image_edit_tool  # noqa: F401 — ensure module imported

    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
    monkeypatch.setattr(image_gen_registry, "get_active_provider", lambda: provider)


class TestSchema:
    def test_schema_shape(self):
        from tools.image_edit_tool import IMAGE_EDIT_SCHEMA

        assert IMAGE_EDIT_SCHEMA["name"] == "image_edit"
        props = IMAGE_EDIT_SCHEMA["parameters"]["properties"]
        assert "prompt" in props
        assert "image" in props
        assert "aspect_ratio" in props
        required = IMAGE_EDIT_SCHEMA["parameters"]["required"]
        assert "prompt" in required
        assert "image" in required
        # Mask is intentionally omitted until xAI edit-mask semantics are clear.
        assert "mask" not in props


class TestRegistration:
    def test_registered_under_image_gen_toolset(self):
        import tools.image_edit_tool  # noqa: F401 — triggers registration
        from tools.registry import registry

        entry = registry.get_entry("image_edit")
        assert entry is not None
        assert entry.toolset == "image_gen"

    def test_toolset_check_still_anchored_to_generation(self):
        # Importing image_edit_tool must not hijack the image_gen toolset's
        # availability check away from the generation tool.
        import tools.image_edit_tool as edit_tool
        import tools.image_generation_tool as gen_tool
        from tools.registry import registry

        check = registry._toolset_checks.get("image_gen")
        assert check is not None

        # Identity (``is``) is too brittle here: in the broader suite a peer
        # test may reload ``tools.image_generation_tool``, rebinding
        # ``check_image_generation_requirements`` to a fresh function object
        # while the registry still holds the original (equally valid)
        # generation check. Anchor on module + qualified name instead, which is
        # stable across reloads. The real invariant is that the image_gen
        # toolset check is still the *generation* availability check...
        def _ident(fn):
            return (fn.__module__, fn.__qualname__)

        assert _ident(check) == _ident(gen_tool.check_image_generation_requirements)
        # ...and is NOT the *edit* availability check (image_edit must not
        # hijack the toolset-level gate).
        assert _ident(check) != _ident(edit_tool.check_image_edit_requirements)


class TestHandler:
    def test_missing_prompt(self, monkeypatch):
        from tools.image_edit_tool import _handle_image_edit

        result = json.loads(_handle_image_edit({"image": "https://x/a.png"}))
        assert "error" in result

    def test_missing_image(self, monkeypatch):
        from tools.image_edit_tool import _handle_image_edit

        result = json.loads(_handle_image_edit({"prompt": "make it blue"}))
        assert "error" in result

    def test_dispatches_to_edit_provider(self, monkeypatch):
        _patch_active(monkeypatch, _FakeEditProvider())
        from tools.image_edit_tool import _handle_image_edit

        result = json.loads(
            _handle_image_edit({"prompt": "make it blue", "image": "https://x/a.png", "aspect_ratio": "square"})
        )
        assert result["success"] is True
        assert result["provider"] == "fakeedit"
        assert result["image"] == "/tmp/edited.png"
        assert result["aspect_ratio"] == "square"
        assert result["_received_image"] == "https://x/a.png"

    def test_unsupported_capability_when_provider_cannot_edit(self, monkeypatch):
        _patch_active(monkeypatch, _FakeGenOnlyProvider())
        from tools.image_edit_tool import _handle_image_edit

        result = json.loads(
            _handle_image_edit({"prompt": "make it blue", "image": "https://x/a.png"})
        )
        assert result["success"] is False
        assert result["image"] is None
        assert result["error_type"] == "unsupported_capability"
        assert result["provider"] == "genonly"

    def test_no_provider_configured(self, monkeypatch):
        _patch_active(monkeypatch, None)
        from tools.image_edit_tool import _handle_image_edit

        result = json.loads(
            _handle_image_edit({"prompt": "make it blue", "image": "https://x/a.png"})
        )
        assert result["success"] is False
        assert result["image"] is None
        assert result["error_type"] == "no_provider"
