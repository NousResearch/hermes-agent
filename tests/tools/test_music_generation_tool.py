"""Tests for tools/music_generation_tool.py."""

import json

from model_tools import TOOL_TO_TOOLSET_MAP, get_all_tool_names


class TestMusicToolRegistration:
    def test_model_tools_discovers_music_generate(self):
        assert "music_generate" in get_all_tool_names()
        assert TOOL_TO_TOOLSET_MAP["music_generate"] == "music_gen"


class TestMusicGenerationTool:
    def test_successful_generation(self, monkeypatch):
        from tools import music_generation_tool

        monkeypatch.setattr(
            music_generation_tool.image_generation_tool,
            "check_image_generation_requirements",
            lambda: True,
        )

        captured = {}

        class FakeHandle:
            def get(self):
                return {
                    "audio": {
                        "url": "https://cdn.example.com/track.mp3",
                        "content_type": "audio/mpeg",
                        "file_name": "track.mp3",
                    }
                }

        def fake_submit(model, arguments):
            captured["model"] = model
            captured["arguments"] = arguments
            return FakeHandle()

        monkeypatch.setattr(
            music_generation_tool.image_generation_tool,
            "_submit_fal_request",
            fake_submit,
        )

        result = json.loads(
            music_generation_tool.music_generate_tool(
                prompt="cinematic ambient piano with soft strings",
                duration_seconds=40,
                instrumental=False,
            )
        )

        assert result["success"] is True
        assert result["audio"] == "https://cdn.example.com/track.mp3"
        assert result["file_name"] == "track.mp3"
        assert captured["model"] == music_generation_tool.DEFAULT_MODEL
        assert captured["arguments"] == {
            "prompt": "cinematic ambient piano with soft strings",
            "music_length_ms": 40000,
        }

    def test_returns_error_when_requirements_missing(self, monkeypatch):
        from tools import music_generation_tool

        monkeypatch.setattr(
            music_generation_tool.image_generation_tool,
            "check_image_generation_requirements",
            lambda: False,
        )

        result = json.loads(music_generation_tool.music_generate_tool("test prompt"))

        assert result["success"] is False
        assert "FAL music generation is unavailable" in result["error"]

    def test_returns_error_for_invalid_duration(self, monkeypatch):
        from tools import music_generation_tool

        monkeypatch.setattr(
            music_generation_tool.image_generation_tool,
            "check_image_generation_requirements",
            lambda: True,
        )

        result = json.loads(
            music_generation_tool.music_generate_tool("test prompt", duration_seconds=5)
        )

        assert result["success"] is False
        assert "duration_seconds must be between" in result["error"]

    def test_handle_requires_prompt(self):
        from tools.music_generation_tool import _handle_music_generate

        result = json.loads(_handle_music_generate({}))
        assert result["error"] == "prompt is required for music generation"
