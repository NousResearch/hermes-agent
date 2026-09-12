"""Tests for video_analyze tool in tools/vision_tools.py."""

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from tools.vision_tools import (
    _detect_video_mime_type,
    _video_to_base64_data_url,
    _handle_video_analyze,
    _MAX_VIDEO_BASE64_BYTES,
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

    def test_local_file_read_guard_blocks_env_via_video_extension(self, tmp_path):
        """A .env file symlinked with a video extension must still be blocked.

        _detect_video_mime_type only checks the file extension, not file
        content, so without a read guard a model could point video_url at
        any credential-store file (renamed/symlinked to look like a video)
        and have its raw bytes base64-encoded and sent to the vision
        provider. Regression for the shared agent.file_safety chokepoint
        added to video_analyze_tool's local-file branch.
        """
        secret = tmp_path / ".env"
        secret.write_text("OPENAI_API_KEY=sk-super-secret\n", encoding="utf-8")
        disguised = tmp_path / "video.mp4"
        disguised.symlink_to(secret)

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(str(disguised), "What is this?"))

        data = json.loads(result)
        assert data["success"] is False
        assert "secret-bearing environment file" in data["error"]
        mock_llm.assert_not_awaited()


    def test_unsupported_format(self, tmp_path):
        """Unsupported extension raises error."""
        video = tmp_path / "clip.flv"
        video.write_bytes(b"\x00" * 100)

        result = self._run(video_analyze_tool(str(video), "What is this?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "unsupported video format" in data["analysis"].lower()


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
        # No hardcoded output cap — the aux client omits max_tokens so the
        # provider uses its full output budget (max-tokens-knob policy).
        assert "max_tokens" not in captured_kwargs

    def test_non_local_backend_reads_video_from_terminal_backend(self, tmp_path, monkeypatch):
        """Non-local terminal backends must not read local host video paths.

        The read routes through the shared media resolver
        (tools.image_source, ``permitted=("video",)``) which exec-reads the
        bytes inside the sandbox — so the analyzed video is the container's
        file, never the host's.
        """
        host_video = tmp_path / "clip.mp4"
        host_video.write_bytes(b"HOST-VIDEO")
        remote_bytes = b"REMOTE-SANDBOX-VIDEO"
        remote_b64 = base64.b64encode(remote_bytes).decode("ascii")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        import tools.image_source as isrc
        import tools.terminal_tool as tt

        env_lookups = []

        def fake_get_active(task_id):
            env_lookups.append(task_id)
            return SimpleNamespace(
                execute=lambda cmd, **kw: {"returncode": 0, "output": remote_b64}
            )

        monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)
        monkeypatch.setattr(isrc, "_get_active_env", fake_get_active)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "sandbox video"
            return mock_response

        with (
            patch("tools.vision_tools.async_call_llm", side_effect=capture_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="sandbox video"),
        ):
            result = self._run(
                video_analyze_tool(str(host_video), "Describe this", task_id="task-123")
            )

        data = json.loads(result)
        assert data["success"] is True
        assert env_lookups == ["task-123"]
        video_url = captured_kwargs["messages"][0]["content"][1]["video_url"]["url"]
        uploaded_bytes = base64.b64decode(video_url.split(",", 1)[1])
        assert uploaded_bytes == remote_bytes
        assert uploaded_bytes != host_video.read_bytes()


# ---------------------------------------------------------------------------
# Local VLM endpoint detection + frame extraction
# ---------------------------------------------------------------------------


class TestLocalVisionEndpoint:
    """``_is_local_vision_endpoint`` decides when video_analyze should
    switch from whole-video base64 to frame extraction."""

    def test_recognizes_localhost_base_url(self):
        from tools.vision_tools import _is_local_vision_endpoint

        cfg = {
            "auxiliary": {
                "vision": {"base_url": "http://127.0.0.1:8123/v1"},
            }
        }
        assert _is_local_vision_endpoint(cfg) is True

    def test_recognizes_localhost_name(self):
        from tools.vision_tools import _is_local_vision_endpoint

        cfg = {
            "auxiliary": {
                "vision": {"base_url": "http://localhost:11434/v1"},
            }
        }
        assert _is_local_vision_endpoint(cfg) is True

    def test_rejects_cloud_endpoint(self):
        from tools.vision_tools import _is_local_vision_endpoint

        cfg = {
            "auxiliary": {
                "vision": {"base_url": "https://api.openai.com/v1"},
            }
        }
        assert _is_local_vision_endpoint(cfg) is False

    def test_rejects_missing_vision_block(self):
        from tools.vision_tools import _is_local_vision_endpoint

        assert _is_local_vision_endpoint({"auxiliary": {}}) is False
        assert _is_local_vision_endpoint(None) in (True, False)  # never raises


class TestVideoToFrameDataUrls:
    """``_video_to_frame_data_urls`` extracts evenly-spaced JPEG frames."""

    def test_returns_none_without_ffmpeg(self, tmp_path, monkeypatch):
        from tools.vision_tools import _video_to_frame_data_urls

        monkeypatch.setattr("shutil.which", lambda name: None)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)
        assert _video_to_frame_data_urls(video) is None

    def test_extracts_frames_when_ffmpeg_available(self, tmp_path, monkeypatch):
        from tools.vision_tools import _video_to_frame_data_urls

        fake_urls = [
            "data:image/jpeg;base64,AAAA",
            "data:image/jpeg;base64,BBBB",
        ]

        real_run = __import__("subprocess").run
        frames_written = []

        def fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "ffprobe":
                return __import__("subprocess").CompletedProcess(
                    cmd, 0,
                    stdout='{"streams": [{"nb_frames": "240", "duration": "10.0"}]}',
                    stderr="",
                )
            if cmd and cmd[0] == "ffmpeg":
                # Simulate frame extraction: write the destination file.
                for i, arg in enumerate(cmd):
                    if arg == "-y" and i + 1 < len(cmd):
                        out_path = cmd[-1]
                        Path(out_path).write_bytes(b"JPEG-FRAME")
                        frames_written.append(out_path)
                return __import__("subprocess").CompletedProcess(cmd, 0, stdout="", stderr="")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else "/usr/bin/ffprobe")
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(
            "tools.vision_tools._image_to_base64_data_url",
            lambda path, mime: fake_urls.pop(0) if fake_urls else "data:image/jpeg;base64,ZZZZ",
        )

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)
        urls = _video_to_frame_data_urls(video, num_frames=2)
        assert urls is not None
        assert len(urls) == 2
        assert frames_written  # ffmpeg actually produced frame files
        assert all(u.startswith("data:image/jpeg;base64,") for u in urls)


class TestVideoAnalyzeLocalEndpoint:
    """video_analyze_tool uses ordered frames for local VLM endpoints."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_local_endpoint_sends_frames_not_video_base64(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "two frames of a soccer match"
            return mock_response

        local_cfg = {
            "auxiliary": {
                "vision": {"base_url": "http://127.0.0.1:8123/v1"},
            }
        }

        fake_urls = [
            "data:image/jpeg;base64,FRAME1",
            "data:image/jpeg;base64,FRAME2",
        ]

        with (
            patch("tools.vision_tools.async_call_llm", side_effect=capture_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="two frames of a soccer match"),
            patch("tools.vision_tools._is_local_vision_endpoint", return_value=True),
            patch(
                "tools.vision_tools._video_to_frame_data_urls",
                return_value=fake_urls,
            ),
        ):
            result = self._run(video_analyze_tool(str(video), "Describe this"))

        data = json.loads(result)
        assert data["success"] is True

        messages = captured_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        # 2 image frames + 1 text prompt
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"
        assert "frames in chronological order" in content[2]["text"]
        # No whole-video base64 was sent
        assert not any(c.get("type") == "video_url" for c in content)

    def test_frame_extraction_failure_falls_back_to_video_base64(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "ok"
            return mock_response

        with (
            patch("tools.vision_tools.async_call_llm", side_effect=capture_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
            patch("tools.vision_tools._is_local_vision_endpoint", return_value=True),
            patch("tools.vision_tools._video_to_frame_data_urls", return_value=None),
        ):
            result = self._run(video_analyze_tool(str(video), "Describe this"))

        data = json.loads(result)
        assert data["success"] is True

        messages = captured_kwargs["messages"]
        content = messages[0]["content"]
        assert content[1]["type"] == "video_url"
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


    def test_in_video_toolset_definition(self):
        """Toolset 'video' should contain video_analyze."""
        from toolsets import TOOLSETS
        assert "video" in TOOLSETS
        assert "video_analyze" in TOOLSETS["video"]["tools"]
