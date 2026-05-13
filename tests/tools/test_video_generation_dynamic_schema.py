"""Tests for the dynamic schema builder under the simplified surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
import yaml

from agent import video_gen_registry
from agent.video_gen_provider import VideoGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_cfg(home, cfg: dict):
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


class _BothModalitiesProvider(VideoGenProvider):
    """Supports both text-to-video AND image-to-video (the common case)."""

    @property
    def name(self) -> str:
        return "both"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": "family-a", "modalities": ["text", "image"]}]

    def default_model(self) -> Optional[str]:
        return "family-a"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16"],
            "resolutions": ["720p", "1080p"],
            "min_duration": 1,
            "max_duration": 15,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def generate(self, prompt, **kwargs):
        return {"success": True}


class _ImageOnlyProvider(VideoGenProvider):
    """Backend with only image-to-video support (rare but possible)."""

    @property
    def name(self) -> str:
        return "img-only"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": "img-only-v1", "modalities": ["image"]}]

    def default_model(self) -> Optional[str]:
        return "img-only-v1"

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["image"], "min_duration": 1, "max_duration": 10}

    def generate(self, prompt, **kwargs):
        return {"success": True}


class TestDynamicSchemaBuilder:
    def test_no_config_says_so(self, cfg_home):
        from tools.video_generation_tool import _build_dynamic_video_schema

        desc = _build_dynamic_video_schema()["description"]
        assert "No video backend is configured" in desc
        assert "hermes tools" in desc

    def test_does_not_mention_edit_or_extend(self, cfg_home):
        """The simplified surface only does text→video and image→video.
        The description must not mention edit/extend anywhere."""
        from tools.video_generation_tool import _build_dynamic_video_schema, _GENERIC_DESCRIPTION

        desc = _build_dynamic_video_schema()["description"]
        # Block words that would suggest functionality we removed
        assert "edit" not in desc.lower() or "audio" in desc.lower()  # 'audio' contains 'audi' not 'edit'
        # Stronger: no occurrence of the words "edit" or "extend" as standalone
        for forbidden in (" edit ", " edits ", " extend ", " extends "):
            assert forbidden not in desc.lower(), f"description leaks '{forbidden.strip()}'"
        # Sanity: the generic blurb itself is also clean
        for forbidden in ("edit", "extend"):
            assert forbidden not in _GENERIC_DESCRIPTION.lower()

    def test_both_modalities_advertises_auto_routing(self, cfg_home):
        from tools.video_generation_tool import _build_dynamic_video_schema

        _write_cfg(cfg_home, {"video_gen": {"provider": "both"}})
        video_gen_registry.register_provider(_BothModalitiesProvider())

        import hermes_cli.plugins as plugins_module
        saved = plugins_module._ensure_plugins_discovered
        plugins_module._ensure_plugins_discovered = lambda *a, **k: None
        try:
            desc = _build_dynamic_video_schema()["description"]
        finally:
            plugins_module._ensure_plugins_discovered = saved

        assert "Active backend: Both" in desc
        assert "text-to-video" in desc and "image-to-video" in desc
        assert "routes automatically" in desc
        # operations bullet is gone
        assert "operations supported" not in desc

    def test_image_only_model_warns_about_required_image_url(self, cfg_home):
        from tools.video_generation_tool import _build_dynamic_video_schema

        _write_cfg(cfg_home, {"video_gen": {"provider": "img-only"}})
        video_gen_registry.register_provider(_ImageOnlyProvider())

        import hermes_cli.plugins as plugins_module
        saved = plugins_module._ensure_plugins_discovered
        plugins_module._ensure_plugins_discovered = lambda *a, **k: None
        try:
            desc = _build_dynamic_video_schema()["description"]
        finally:
            plugins_module._ensure_plugins_discovered = saved

        assert "image-to-video only" in desc
        assert "image_url is REQUIRED" in desc

    def test_builder_wired_into_registry(self):
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        entry = registry._tools["video_generate"]
        assert entry.dynamic_schema_overrides is not None
        out = entry.dynamic_schema_overrides()
        assert "description" in out
