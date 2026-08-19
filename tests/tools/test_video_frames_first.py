"""Regression tests for frames-first local video analysis routing."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import vision_tools


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("/tmp/whatsapp-clip.mp4", True),
        ("file:///tmp/whatsapp-clip.mp4", True),
        ("https://example.com/clip.mp4", False),
    ],
)
def test_should_use_frames_first_by_source(source, expected):
    assert vision_tools._should_use_frames_first(source) is expected


def test_local_video_uses_frames_without_native_video_url(tmp_path):
    video = tmp_path / "local.mp4"
    video.write_bytes(b"dummy video bytes")
    frames_result = ("Grounded frame analysis", ["/tmp/frame-001.jpg"])

    with (
        patch(
            "tools.vision_tools._analyze_video_via_frames",
            new_callable=AsyncMock,
            return_value=frames_result,
        ) as mock_frames,
        patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm,
    ):
        result = _run(vision_tools.video_analyze_tool(str(video), "Describe it"))

    data = json.loads(result)
    assert data["success"] is True
    assert data["method"] == "ffmpeg_frames+vision"
    assert data["analysis"] == "Grounded frame analysis"
    mock_frames.assert_awaited_once()
    mock_llm.assert_not_awaited()


def test_local_video_falls_back_to_native_when_frames_fail(tmp_path):
    video = tmp_path / "local.mp4"
    video.write_bytes(b"dummy video bytes")
    native_response = MagicMock()

    with (
        patch(
            "tools.vision_tools._analyze_video_via_frames",
            new_callable=AsyncMock,
            side_effect=RuntimeError("ffmpeg produced zero frames"),
        ) as mock_frames,
        patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=native_response,
        ) as mock_llm,
        patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="Grounded native fallback analysis",
        ),
    ):
        result = _run(vision_tools.video_analyze_tool(str(video), "Describe it"))

    data = json.loads(result)
    assert data["success"] is True
    assert data["method"] == "video_url"
    mock_frames.assert_awaited_once()
    mock_llm.assert_awaited_once()
    content = mock_llm.await_args.kwargs["messages"][0]["content"]
    assert content[1]["type"] == "video_url"


def test_remote_url_prefers_native_video_url(tmp_path):
    native_response = MagicMock()

    async def fake_download(_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"downloaded dummy video bytes")
        return destination

    with (
        patch(
            "tools.vision_tools._validate_image_url_async",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "tools.vision_tools._download_video",
            new_callable=AsyncMock,
            side_effect=fake_download,
        ),
        patch(
            "tools.vision_tools._analyze_video_via_frames",
            new_callable=AsyncMock,
        ) as mock_frames,
        patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=native_response,
        ) as mock_llm,
        patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="Grounded remote native analysis",
        ),
    ):
        result = _run(
            vision_tools.video_analyze_tool(
                "https://example.com/clip.mp4", "Describe it"
            )
        )

    data = json.loads(result)
    assert data["success"] is True
    assert data["method"] == "video_url"
    mock_llm.assert_awaited_once()
    mock_frames.assert_not_awaited()


def test_frame_timestamps_span_full_video_duration():
    timestamps = vision_tools._video_frame_timestamps(20.0, max_frames=16)

    assert len(timestamps) == 16
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps[-1] == pytest.approx(20.0)
    assert any(timestamp > 8.0 for timestamp in timestamps)
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    assert max(gaps) == pytest.approx(min(gaps))