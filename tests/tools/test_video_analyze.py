"""Tests for video_analyze tool in tools/vision_tools.py."""

import asyncio
import json
import os
from pathlib import Path
from typing import Awaitable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    _detect_video_mime_type,
    _video_to_base64_data_url,
    _handle_video_analyze,
    _MAX_VIDEO_BASE64_BYTES,
    _VIDEO_MIME_TYPES,
    _VIDEO_SIZE_WARN_BYTES,
    video_analyze_tool,
    VIDEO_ANALYZE_SCHEMA,
)


# ---------------------------------------------------------------------------
# _detect_video_mime_type
# ---------------------------------------------------------------------------


class TestDetectVideoMimeType:
    """Extension-based MIME detection for video files."""

    def test_mp4(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_webm(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/webm"

    def test_mov(self, tmp_path):
        p = tmp_path / "clip.mov"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mov"

    def test_avi_fallback_mp4(self, tmp_path):
        p = tmp_path / "clip.avi"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_mkv_fallback_mp4(self, tmp_path):
        p = tmp_path / "clip.mkv"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_mpeg(self, tmp_path):
        p = tmp_path / "clip.mpeg"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mpeg"

    def test_mpg(self, tmp_path):
        p = tmp_path / "clip.mpg"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mpeg"

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "clip.flv"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) is None

    def test_case_insensitive(self, tmp_path):
        p = tmp_path / "clip.MP4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"


# ---------------------------------------------------------------------------
# _video_to_base64_data_url
# ---------------------------------------------------------------------------


class TestVideoToBase64DataUrl:
    """Base64 encoding of video files."""

    def test_produces_data_url(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        assert result.startswith("data:video/mp4;base64,")

    def test_custom_mime_type(self, tmp_path):
        p = tmp_path / "test.webm"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p, mime_type="video/webm")
        assert result.startswith("data:video/webm;base64,")

    def test_default_mime_for_unknown_ext(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        # Falls back to video/mp4
        assert result.startswith("data:video/mp4;base64,")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestVideoAnalyzeSchema:
    """Schema structure is correct."""

    def test_schema_name(self):
        assert VIDEO_ANALYZE_SCHEMA["name"] == "video_analyze"

    def test_schema_has_required_fields(self):
        params = VIDEO_ANALYZE_SCHEMA["parameters"]
        assert "video_url" in params["properties"]
        assert "question" in params["properties"]
        assert params["required"] == ["video_url", "question"]

    def test_schema_description_mentions_video(self):
        assert "video" in VIDEO_ANALYZE_SCHEMA["description"].lower()


# ---------------------------------------------------------------------------
# _handle_video_analyze handler
# ---------------------------------------------------------------------------


class TestHandleVideoAnalyze:
    """Tests for the registry handler wrapper."""

    def test_returns_awaitable(self, tmp_path, monkeypatch):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00" * 100)
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "test"})
            result = _handle_video_analyze({"video_url": str(video_file), "question": "what is this?"})
            # Should return an awaitable (coroutine)
            assert asyncio.iscoroutine(result)
            # Clean up the unawaited coroutine
            result.close()

    def test_uses_auxiliary_video_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "google/gemini-2.5-flash")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "other-model")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-2.5-flash"

    def test_falls_back_to_vision_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "google/gemini-flash")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-flash"


# ---------------------------------------------------------------------------
# video_analyze_tool — integration-style tests with mocked LLM
# ---------------------------------------------------------------------------


class TestVideoAnalyzeTool:
    """Core video analysis function tests."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_local_file_success(self, tmp_path, monkeypatch):
        """Analyze a local video file — happy path."""
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A short video showing a demo."

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="A short video showing a demo."):
                result = self._run(video_analyze_tool(str(video), "What is this?"))

        data = json.loads(result)
        assert data["success"] is True
        assert "demo" in data["analysis"].lower()

    def test_local_file_not_found(self, tmp_path):
        """Non-existent file raises appropriate error."""
        result = self._run(video_analyze_tool("/nonexistent/video.mp4", "What?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "invalid video source" in data["analysis"].lower()

    def test_unsupported_format(self, tmp_path):
        """Unsupported extension raises error."""
        video = tmp_path / "clip.flv"
        video.write_bytes(b"\x00" * 100)

        result = self._run(video_analyze_tool(str(video), "What is this?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "unsupported video format" in data["analysis"].lower()

    def test_video_too_large(self, tmp_path, monkeypatch):
        """Video exceeding max size is rejected."""
        video = tmp_path / "huge.mp4"
        # Don't actually write 50MB — mock the stat
        video.write_bytes(b"\x00" * 100)

        # Patch the base64 encoding to return something huge
        with patch("tools.vision_tools._video_to_base64_data_url") as mock_encode:
            mock_encode.return_value = "data:video/mp4;base64," + "A" * (_MAX_VIDEO_BASE64_BYTES + 1)
            result = self._run(video_analyze_tool(str(video), "What?"))

        data = json.loads(result)
        assert data["success"] is False
        assert "too large" in data["analysis"].lower()

    def test_interrupt_check(self, tmp_path):
        """Tool respects interrupt flag."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        with patch("tools.interrupt.is_interrupted", return_value=True):
            result = self._run(video_analyze_tool(str(video), "What?"))

        data = json.loads(result)
        assert data["success"] is False

    def test_empty_response_retries(self, tmp_path):
        """Retries once on empty model response."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        call_count = 0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Video analysis result."

        async def fake_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch("tools.vision_tools.async_call_llm", side_effect=fake_llm):
            with patch("tools.vision_tools.extract_content_or_reasoning", side_effect=["", "Video analysis result."]):
                result = self._run(video_analyze_tool(str(video), "What?"))

        data = json.loads(result)
        assert data["success"] is True
        assert call_count == 2  # Initial call + retry

    def test_file_scheme_stripped(self, tmp_path):
        """file:// prefix is stripped correctly."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
                result = self._run(video_analyze_tool(f"file://{video}", "What?"))

        data = json.loads(result)
        assert data["success"] is True

    def test_api_message_format(self, tmp_path):
        """Verify the message sent to LLM uses video_url content type."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "OK"
            return mock_response

        with patch("tools.vision_tools.async_call_llm", side_effect=capture_llm):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
                self._run(video_analyze_tool(str(video), "Describe this"))

        messages = captured_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "video_url"
        assert "video_url" in content[1]
        assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")


# ---------------------------------------------------------------------------
# Toolset registration
# ---------------------------------------------------------------------------


class TestVideoToolsetRegistration:
    """Verify the tool is registered correctly."""

    def test_registered_in_video_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("video_analyze")
        assert entry is not None
        assert entry.toolset == "video"
        assert entry.is_async is True
        assert entry.emoji == "🎬"

    def test_not_in_core_tools(self):
        """video_analyze should NOT be in _HERMES_CORE_TOOLS (default disabled)."""
        from toolsets import _HERMES_CORE_TOOLS
        assert "video_analyze" not in _HERMES_CORE_TOOLS

    def test_in_video_toolset_definition(self):
        """Toolset 'video' should contain video_analyze."""
        from toolsets import TOOLSETS
        assert "video" in TOOLSETS
        assert "video_analyze" in TOOLSETS["video"]["tools"]
