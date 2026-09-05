"""Regression coverage for provider video-transport negotiation."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import video_analysis
from tools.video_analysis import (
    VideoFrame,
    VideoFrameExtractionError,
    video_analyze_tool,
)
from tools.video_analysis.frames import extract_video_frames


class ProviderError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _response(text: str = "ok") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def _run(coro):
    return asyncio.run(coro)


def test_llama_cpp_rejection_retries_documented_input_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video_bytes = b"native-video-bytes"
    video.write_bytes(video_bytes)
    calls = []

    async def call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderError(
                400,
                "HTTP 400: unsupported content[].type: video_url",
            )
        return _response("native llama.cpp video")

    with (
        patch("tools.vision_tools.async_call_llm", side_effect=call_llm),
        patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="native llama.cpp video",
        ),
        patch(
            "tools.video_analysis.core._extract_video_frames_async",
            new_callable=AsyncMock,
        ) as extract_frames,
    ):
        result = _run(
            video_analyze_tool(str(video), "Describe the timeline")
        )

    data = json.loads(result)
    assert data["success"] is True
    assert data["transport"] == "input_video"
    assert len(calls) == 2
    first_content = calls[0]["messages"][0]["content"]
    second_content = calls[1]["messages"][0]["content"]
    assert first_content[1]["type"] == "video_url"
    assert second_content[1]["type"] == "input_video"
    raw_payload = second_content[1]["input_video"]["data"]
    assert not raw_payload.startswith("data:")
    assert base64.b64decode(raw_payload) == video_bytes
    extract_frames.assert_not_awaited()


def test_bare_400_negotiates_input_video_before_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls = []

    async def call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderError(400, "Bad Request")
        return _response("accepted")

    with (
        patch("tools.vision_tools.async_call_llm", side_effect=call_llm),
        patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="accepted",
        ),
        patch(
            "tools.video_analysis.core._extract_video_frames_async",
            new_callable=AsyncMock,
        ) as extract_frames,
    ):
        result = _run(video_analyze_tool(str(video), "Describe this"))

    data = json.loads(result)
    assert data["success"] is True
    assert data["transport"] == "input_video"
    assert [
        call["messages"][0]["content"][1]["type"] for call in calls
    ] == ["video_url", "input_video"]
    extract_frames.assert_not_awaited()


def test_explicit_image_only_rejection_skips_input_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    frames = [VideoFrame(0.5, "data:image/jpeg;base64,RlJBTUU=")]
    calls = []

    async def call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderError(400, "model does not support video input")
        return _response("visible frame")

    with (
        patch("tools.vision_tools.async_call_llm", side_effect=call_llm),
        patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="visible frame",
        ),
        patch(
            "tools.video_analysis.core._extract_video_frames_async",
            new_callable=AsyncMock,
            return_value=frames,
        ) as extract_frames,
    ):
        result = _run(video_analyze_tool(str(video), "Describe this"))

    data = json.loads(result)
    assert data["success"] is True
    assert data["transport"] == "image_frames"
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"][1]["type"] == "video_url"
    assert [
        part["type"] for part in calls[1]["messages"][0]["content"]
    ] == ["text", "image_url"]
    extract_frames.assert_awaited_once_with(video)


def test_image_fallback_sends_real_ordered_jpeg_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls = []
    frames = [
        VideoFrame(0.5, "data:image/jpeg;base64,RlJBTUUx"),
        VideoFrame(2.5, "data:image/jpeg;base64,RlJBTUUy"),
    ]

    async def call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderError(400, "unsupported content[].type video_url")
        if len(calls) == 2:
            raise ProviderError(400, "unsupported content[].type input_video")
        return _response("frame analysis")

    with (
        patch("tools.vision_tools.async_call_llm", side_effect=call_llm),
        patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="frame analysis",
        ),
        patch(
            "tools.video_analysis.core._extract_video_frames_async",
            new_callable=AsyncMock,
            return_value=frames,
        ) as extract_frames,
    ):
        result = _run(
            video_analyze_tool(str(video), "Describe the timeline")
        )

    data = json.loads(result)
    assert data["success"] is True
    assert data["transport"] == "image_frames"
    assert data["frame_count"] == 2
    extract_frames.assert_awaited_once_with(video)

    frame_content = calls[2]["messages"][0]["content"]
    assert [part["type"] for part in frame_content] == [
        "text",
        "image_url",
        "image_url",
    ]
    assert [
        part["image_url"]["url"] for part in frame_content[1:]
    ] == [frame.data_url for frame in frames]
    evidence_note = frame_content[0]["text"]
    assert "0.50s, 2.50s" in evidence_note
    assert "Do not claim audio content" in evidence_note


def test_non_negotiable_error_does_not_retry_or_extract(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    call_llm = AsyncMock(
        side_effect=ProviderError(401, "HTTP 401 unauthorized")
    )

    with (
        patch("tools.vision_tools.async_call_llm", call_llm),
        patch(
            "tools.video_analysis.core._extract_video_frames_async",
            new_callable=AsyncMock,
        ) as extract_frames,
    ):
        result = _run(video_analyze_tool(str(video), "Describe this"))

    data = json.loads(result)
    assert data["success"] is False
    assert "401" in data["error"]
    assert call_llm.await_count == 1
    extract_frames.assert_not_awaited()


def test_extraction_failure_reports_each_native_attempt(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    call_llm = AsyncMock(
        side_effect=[
            ProviderError(400, "unsupported content[].type video_url"),
            ProviderError(400, "unsupported content[].type input_video"),
        ]
    )

    with (
        patch("tools.vision_tools.async_call_llm", call_llm),
        patch(
            "tools.video_analysis.core._extract_video_frames_async",
            new_callable=AsyncMock,
            side_effect=VideoFrameExtractionError(
                "Frame fallback requires ffmpeg and ffprobe on PATH"
            ),
        ),
    ):
        result = _run(video_analyze_tool(str(video), "Describe this"))

    data = json.loads(result)
    assert data["success"] is False
    assert "ffmpeg and ffprobe" in data["analysis"]
    assert "video_url" in data["analysis"]
    assert "input_video" in data["analysis"]
    assert "does not support video analysis" not in data["analysis"]
    assert call_llm.await_count == 2


def test_frame_extraction_uses_bounded_argv_and_jpeg_payloads(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_which(name):
        return f"/usr/bin/{name}"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0].endswith("ffprobe"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="4.0\n",
                stderr="",
            )
        Path(command[-1]).write_bytes(b"\xff\xd8\xffJPEG")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(
        "tools.video_analysis.frames.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "tools.video_analysis.frames.subprocess.run",
        fake_run,
    )

    frames = extract_video_frames(video, max_frames=2, max_side=320)

    assert [frame.timestamp_seconds for frame in frames] == pytest.approx(
        [1.0, 3.0]
    )
    assert all(
        frame.data_url.startswith("data:image/jpeg;base64,")
        for frame in frames
    )
    assert len(calls) == 3
    for command, kwargs in calls:
        assert isinstance(command, list)
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("timeout") in {30, 45}
        assert kwargs.get("shell", False) is False
    ffmpeg_commands = [
        command for command, _ in calls if command[0].endswith("ffmpeg")
    ]
    assert all("-threads" in command for command in ffmpeg_commands)
    assert all(
        (
            "scale=320:320:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2"
        ) in command
        for command in ffmpeg_commands
    )


def test_frame_extraction_requires_ffmpeg_and_ffprobe(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "tools.video_analysis.frames.shutil.which",
        lambda _name: None,
    )

    with pytest.raises(VideoFrameExtractionError) as exc_info:
        extract_video_frames(video)

    assert "ffmpeg" in str(exc_info.value)
    assert "ffprobe" in str(exc_info.value)


def test_frame_extraction_with_real_ffmpeg(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg and ffprobe are not installed")

    image_module = pytest.importorskip("PIL.Image")
    video = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=4",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )

    frames = extract_video_frames(video, max_frames=2, max_side=160)

    assert len(frames) == 2
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx(
        [0.5, 1.5],
        abs=0.15,
    )
    for frame in frames:
        payload = base64.b64decode(frame.data_url.split(",", 1)[1])
        with image_module.open(BytesIO(payload)) as image:
            assert image.format == "JPEG"
            assert max(image.size) <= 160


def test_discovery_bridge_settles_registry_and_legacy_api():
    from tools import vision_tools
    from tools import vision_video_analysis  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("video_analyze")
    assert entry is not None
    assert entry.handler is video_analysis._handle_video_analyze
    assert entry.toolset == "video"
    assert entry.is_async is True
    assert vision_tools.video_analyze_tool is video_analysis.video_analyze_tool
    assert vision_tools.VIDEO_ANALYZE_SCHEMA is video_analysis.VIDEO_ANALYZE_SCHEMA
