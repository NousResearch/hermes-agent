"""Tests for the image_edit tool schema, handler, and registry integration."""

from __future__ import annotations

import json
from typing import Any, Dict, cast

import pytest


# ── Import the tool so side-effects fire (registry registration). ──
import tools.image_edit_tool as edit_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestImageEditSchema:
    def test_name_is_image_edit(self):
        assert edit_tool.IMAGE_EDIT_SCHEMA["name"] == "image_edit"

    def test_required_fields_are_prompt_and_image(self):
        params = cast(dict, edit_tool.IMAGE_EDIT_SCHEMA["parameters"])
        assert set(params["required"]) == {"prompt", "image"}

    def test_prompt_is_string(self):
        params = cast(dict, edit_tool.IMAGE_EDIT_SCHEMA["parameters"])
        assert params["properties"]["prompt"]["type"] == "string"

    def test_image_is_string(self):
        params = cast(dict, edit_tool.IMAGE_EDIT_SCHEMA["parameters"])
        assert params["properties"]["image"]["type"] == "string"

    def test_aspect_ratio_has_valid_default(self):
        params = cast(dict, edit_tool.IMAGE_EDIT_SCHEMA["parameters"])
        ar = params["properties"]["aspect_ratio"]
        assert ar["default"] == "landscape"
        assert "landscape" in ar["enum"]
        assert "square" in ar["enum"]
        assert "portrait" in ar["enum"]


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


class TestImageEditHandler:
    def test_missing_prompt_returns_error(self):
        result = json.loads(edit_tool._handle_image_edit(
            {"image": "/tmp/test.png"}
        ))
        assert result["success"] is False
        assert "prompt" in result["error"].lower()

    def test_missing_image_returns_error(self):
        result = json.loads(edit_tool._handle_image_edit(
            {"prompt": "restore"}
        ))
        assert result["success"] is False
        assert "image" in result["error"].lower()

    def test_both_missing_returns_error(self):
        result = json.loads(edit_tool._handle_image_edit({}))
        assert result["success"] is False

    def test_falls_back_to_unsupported_when_no_plugin_provider(self, monkeypatch):
        monkeypatch.setattr(edit_tool, "_read_configured_image_provider", lambda: None)
        result = json.loads(edit_tool._handle_image_edit(
            {"prompt": "restore", "image": "/tmp/test.png"}
        ))
        assert result["success"] is False
        assert result["error_type"] == "unsupported_operation"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_image_edit_registered(self):
        from tools.registry import registry
        entry = registry._tools.get("image_edit")
        assert entry is not None, "image_edit should be in the tool registry"
        assert entry.schema["name"] == "image_edit"

    def test_image_edit_is_in_image_gen_toolset(self):
        from tools.registry import registry
        entry = registry._tools.get("image_edit")
        assert entry is not None
        assert entry.toolset == "image_gen"

    def test_image_generate_still_registered(self):
        """Sanity check — image_generate must not be broken."""
        import tools.image_generation_tool  # noqa: F401 — ensure it registers
        from tools.registry import registry
        entry = registry._tools.get("image_generate")
        assert entry is not None


# ---------------------------------------------------------------------------
# Dispatch tests (with fake provider)
# ---------------------------------------------------------------------------


class _FakeEditProvider:
    """Minimal provider stub that supports editing."""
    name = "fake-edit"
    last_edit: dict | None = None

    def supports_edit(self) -> bool:
        return True

    def edit(self, prompt, image, aspect_ratio="landscape", **kwargs):
        self.last_edit = {
            "prompt": prompt,
            "image": image,
            "aspect_ratio": aspect_ratio,
        }
        return {
            "success": True,
            "image": "/tmp/edit.png",
            "model": "test-model",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "fake-edit",
        }


class _FakeNoEditProvider:
    """Provider that does not support editing."""
    name = "fake-noedit"

    def supports_edit(self) -> bool:
        return False

    def generate(self, prompt, **kwargs):
        return {
            "success": True, "image": "/tmp/gen.png", "model": "t",
            "prompt": prompt, "aspect_ratio": "landscape", "provider": "fake-noedit",
        }


class TestDispatch:
    def test_calls_edit_on_supporting_provider(self, monkeypatch):
        fake = _FakeEditProvider()
        monkeypatch.setattr(
            edit_tool, "_read_configured_image_provider", lambda: "fake-edit"
        )
        # _dispatch_edit imports from hermes_cli.plugins — patch there
        try:
            import hermes_cli.plugins as plugin_mod
            monkeypatch.setattr(plugin_mod, "_ensure_plugins_discovered", lambda force=None: None)
        except ImportError:
            pass

        import agent.image_gen_registry as reg
        monkeypatch.setattr(
            reg, "get_provider",
            lambda name: fake if name == "fake-edit" else None,
        )

        result = json.loads(edit_tool._handle_image_edit(
            {"prompt": "restore", "image": "/tmp/source.png"}
        ))
        assert result["success"] is True
        assert result["image"] == "/tmp/edit.png"
        assert fake.last_edit["prompt"] == "restore"
        assert fake.last_edit["image"] == "/tmp/source.png"

    def test_returns_unsupported_for_noedit_provider(self, monkeypatch):
        fake = _FakeNoEditProvider()
        monkeypatch.setattr(
            edit_tool, "_read_configured_image_provider", lambda: "fake-noedit"
        )
        try:
            import hermes_cli.plugins as plugin_mod
            monkeypatch.setattr(plugin_mod, "_ensure_plugins_discovered", lambda force=None: None)
        except ImportError:
            pass

        import agent.image_gen_registry as reg
        monkeypatch.setattr(
            reg, "get_provider",
            lambda name: fake if name == "fake-noedit" else None,
        )

        result = json.loads(edit_tool._handle_image_edit(
            {"prompt": "restore", "image": "/tmp/source.png"}
        ))
        assert result["success"] is False
        assert result["error_type"] == "unsupported_operation"
