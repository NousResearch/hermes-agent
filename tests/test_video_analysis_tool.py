"""Tests for tools/video_analysis_tool.py — frame interval calculation, helpers, and tool registration."""

import json
import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from tools.video_analysis_tool import (
    VIDEO_ANALYZE_SCHEMA,
    _calculate_frame_interval,
    _handle_video_analyze,
    check_video_requirements,
    video_analyze_tool,
)


# ---------------------------------------------------------------------------
# _calculate_frame_interval
# ---------------------------------------------------------------------------

class TestCalculateFrameInterval:
    """Pure function — no mocking needed."""

    def test_short_video_uses_1s_tier(self):
        # < 60s => 1s base interval (capped at 30 frames)
        assert _calculate_frame_interval(10) == 1.0
        assert _calculate_frame_interval(29) == 1.0

    def test_medium_video_uses_5s_tier(self):
        # 60-300s => 5s base interval (capped at 30 frames)
        assert _calculate_frame_interval(60) == 5.0
        assert _calculate_frame_interval(120) == 5.0

    def test_long_video_uses_10s_tier(self):
        # 300s+ => 10s base interval (capped at 30 frames)
        assert _calculate_frame_interval(300) == 10.0

    def test_max_frames_cap_enforced(self):
        # Any duration should produce at most 30 frames
        for duration in [50, 200, 600, 1800, 7200]:
            interval = _calculate_frame_interval(duration)
            assert duration / interval <= 30, f"duration={duration} produced too many frames"

    def test_very_long_video_scales_interval(self):
        # 1800s (30 min) at 10s tier = 180 frames, way over cap
        interval = _calculate_frame_interval(1800)
        assert interval == 1800 / 30  # 60s

    def test_cap_kicks_in_for_short_video(self):
        # 59s / 1s = 59 frames > 30 => interval should be 59/30
        interval = _calculate_frame_interval(59)
        assert interval == pytest.approx(59 / 30)
        assert 59 / interval <= 30

    def test_cap_kicks_in_for_medium_video(self):
        # 299s / 5s ≈ 60 frames > 30 => interval should be 299/30
        interval = _calculate_frame_interval(299)
        assert interval == pytest.approx(299 / 30)

    def test_boundary_60s(self):
        # Exactly 60s uses medium tier: 60/5 = 12 frames, under cap
        assert _calculate_frame_interval(60) == 5.0

    def test_boundary_300s(self):
        # Exactly 300s uses long tier: 300/10 = 30 frames, exactly at cap
        assert _calculate_frame_interval(300) == 10.0

    def test_very_short_video(self):
        interval = _calculate_frame_interval(2)
        assert interval == 1.0


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe detection
# ---------------------------------------------------------------------------

class TestHasFFmpeg:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_ffmpeg_found(self, mock_which):
        from tools.video_analysis_tool import _has_ffmpeg
        assert _has_ffmpeg() is True

    @patch("shutil.which", return_value=None)
    def test_ffmpeg_not_found(self, mock_which):
        from tools.video_analysis_tool import _has_ffmpeg
        assert _has_ffmpeg() is False


class TestHasFFprobe:
    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    def test_ffprobe_found(self, mock_which):
        from tools.video_analysis_tool import _has_ffprobe
        assert _has_ffprobe() is True

    @patch("shutil.which", return_value=None)
    def test_ffprobe_not_found(self, mock_which):
        from tools.video_analysis_tool import _has_ffprobe
        assert _has_ffprobe() is False


# ---------------------------------------------------------------------------
# video_analyze_tool — async tests
# ---------------------------------------------------------------------------

def _make_video(tmp_path):
    """Create a dummy video file and return its path."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"\x00" * 100)
    return str(video)


class TestVideoAnalyzeTool:
    @pytest.mark.asyncio
    async def test_file_not_found(self):
        result = json.loads(await video_analyze_tool("/nonexistent/video.mp4", "what happens?"))
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=False)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=True)
    async def test_missing_ffmpeg(self, _probe, _ff, tmp_path):
        path = _make_video(tmp_path)
        result = json.loads(await video_analyze_tool(path, "describe"))
        assert result["success"] is False
        assert "ffmpeg" in result["error"].lower()

    @pytest.mark.asyncio
    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=True)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=False)
    async def test_missing_ffprobe(self, _probe, _ff, tmp_path):
        path = _make_video(tmp_path)
        result = json.loads(await video_analyze_tool(path, "describe"))
        assert result["success"] is False
        assert "ffmpeg" in result["error"].lower() or "ffprobe" in result["error"].lower()

    @pytest.mark.asyncio
    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=True)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=True)
    @patch("tools.video_analysis_tool._get_video_duration", return_value=10.0)
    @patch("tools.video_analysis_tool._extract_frames")
    @patch("tools.video_analysis_tool.async_call_llm")
    async def test_successful_analysis(self, mock_llm, mock_extract, _dur, _probe, _ff, tmp_path):
        video_path = _make_video(tmp_path)

        # Create fake frame files in tmp_path (auto-cleaned by pytest)
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        frame_paths = []
        for i in range(3):
            fp = frame_dir / f"frame_{i:04d}.jpg"
            fp.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
            frame_paths.append(str(fp))

        mock_extract.return_value = frame_paths

        mock_choice = MagicMock()
        mock_choice.message.content = "The video shows a cat playing."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_llm.return_value = mock_response

        result = json.loads(await video_analyze_tool(video_path, "what happens?"))
        assert result["success"] is True
        assert result["analysis"] == "The video shows a cat playing."
        assert result["duration"] == 10.0
        assert result["frames_analyzed"] == 3
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=True)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=True)
    @patch("tools.video_analysis_tool._get_video_duration", return_value=10.0)
    @patch("tools.video_analysis_tool._extract_frames", return_value=[])
    async def test_no_frames_extracted(self, _ext, _dur, _probe, _ff, tmp_path):
        path = _make_video(tmp_path)
        result = json.loads(await video_analyze_tool(path, "describe"))
        assert result["success"] is False
        assert "no frames" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_interrupted(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            result = json.loads(await video_analyze_tool("/any/path.mp4", "q"))
            assert result["success"] is False
            assert "interrupt" in result["error"].lower()


# ---------------------------------------------------------------------------
# _handle_video_analyze
# ---------------------------------------------------------------------------

class TestHandleVideoAnalyze:
    @patch("tools.video_analysis_tool.video_analyze_tool")
    def test_builds_correct_prompt(self, mock_tool):
        mock_tool.return_value = "mock_coroutine"
        args = {"video_path": "/tmp/test.mp4", "question": "What color is the car?"}
        _handle_video_analyze(args)
        call_args = mock_tool.call_args
        prompt = call_args[0][1]
        assert "What color is the car?" in prompt
        assert "video_path" not in prompt
        assert call_args[0][0] == "/tmp/test.mp4"

    @patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "gpt-4o"})
    @patch("tools.video_analysis_tool.video_analyze_tool")
    def test_respects_env_model_override(self, mock_tool):
        mock_tool.return_value = "mock"
        _handle_video_analyze({"video_path": "/tmp/v.mp4", "question": "q"})
        call_args = mock_tool.call_args
        assert call_args[0][2] == "gpt-4o"  # model arg


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestVideoAnalyzeSchema:
    def test_schema_has_required_fields(self):
        assert VIDEO_ANALYZE_SCHEMA["name"] == "video_analyze"
        assert "parameters" in VIDEO_ANALYZE_SCHEMA
        props = VIDEO_ANALYZE_SCHEMA["parameters"]["properties"]
        assert "video_path" in props
        assert "question" in props

    def test_both_params_required(self):
        assert set(VIDEO_ANALYZE_SCHEMA["parameters"]["required"]) == {"video_path", "question"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_video_analyze_registered(self):
        from tools.registry import registry
        assert "video_analyze" in registry._tools
        entry = registry._tools["video_analyze"]
        assert entry.toolset == "vision"
        assert entry.is_async is True

    def test_video_analyze_in_vision_toolset(self):
        from toolsets import resolve_toolset
        tools = resolve_toolset("vision")
        assert "video_analyze" in tools


# ---------------------------------------------------------------------------
# check_video_requirements
# ---------------------------------------------------------------------------

class TestCheckVideoRequirements:
    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=False)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=True)
    def test_fails_without_ffmpeg(self, _probe, _ff):
        assert check_video_requirements() is False

    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=True)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=False)
    def test_fails_without_ffprobe(self, _probe, _ff):
        assert check_video_requirements() is False

    @patch("tools.video_analysis_tool._has_ffmpeg", return_value=True)
    @patch("tools.video_analysis_tool._has_ffprobe", return_value=True)
    def test_fails_without_vision_provider(self, _probe, _ff):
        with patch("agent.auxiliary_client.resolve_vision_provider_client", side_effect=Exception("no provider")):
            assert check_video_requirements() is False
