"""Tests for the dynamic schema builder."""

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
        # No provider configured AND none available → description says so. The
        # wording reflects the *resolved* active provider (mirrors execution),
        # so it reads "available" rather than "configured".
        assert "No video backend is available" in desc
        assert "hermes tools" in desc


    def test_builder_wired_into_registry(self):
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        entry = registry._tools["video_generate"]
        assert entry.dynamic_schema_overrides is not None
        out = entry.dynamic_schema_overrides()
        assert "description" in out

    def test_both_modalities_model_claims_both(self, cfg_home):
        from tools.video_generation_tool import _build_dynamic_video_schema

        video_gen_registry.register_provider(_BothModalitiesProvider())
        _write_cfg(cfg_home, {"video_gen": {"provider": "both", "model": "family-a"}})

        desc = _build_dynamic_video_schema()["description"]
        assert "supports both text-to-video" in desc
        assert "duration range: 1-15s" in desc

    def test_i2v_only_model_does_not_claim_text_to_video(self, cfg_home):
        """A dual-modality backend with an i2v-only active model must not
        contradict the model caveat with a 'supports both' line."""
        from tools.video_generation_tool import _build_dynamic_video_schema

        class _DualBackendI2VModel(VideoGenProvider):
            @property
            def name(self) -> str:
                return "dual-i2v"

            def is_available(self) -> bool:
                return True

            def list_models(self):
                return [{
                    "id": "gemini-like",
                    "modalities": ["image"],
                    "min_duration": 3,
                    "max_duration": 10,
                }]

            def default_model(self):
                return "gemini-like"

            def capabilities(self):
                return {
                    "modalities": ["text", "image"],
                    "min_duration": 1,
                    "max_duration": 30,
                }

            def generate(self, prompt, **kwargs):
                return {"success": True}

        video_gen_registry.register_provider(_DualBackendI2VModel())
        _write_cfg(
            cfg_home,
            {"video_gen": {"provider": "dual-i2v", "model": "gemini-like"}},
        )

        desc = _build_dynamic_video_schema()["description"]
        assert "image-to-video only" in desc
        assert "supports both text-to-video" not in desc
        # Prefer the active model's duration window over the backend union.
        assert "duration range: 3-10s" in desc
