import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.video_tools import SampledFrame, _format_timestamp, _sample_timestamps, analyze_video_file


def test_format_timestamp_handles_minutes_and_hours():
    assert _format_timestamp(5) == "00:05"
    assert _format_timestamp(65) == "01:05"
    assert _format_timestamp(3661) == "01:01:01"


def test_sample_timestamps_is_bounded_and_spread_out():
    timestamps = _sample_timestamps(
        60,
        max_duration_seconds=30,
        max_frames=4,
        seconds_per_frame=5,
    )

    assert len(timestamps) == 4
    assert timestamps[0] == pytest.approx(0.5)
    assert timestamps[-1] == pytest.approx(29.5)
    assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
async def test_analyze_video_file_reports_missing_ffmpeg(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")

    with patch("tools.video_tools._find_binary", return_value=None):
        result = await analyze_video_file(str(video_path))

    assert result["success"] is False
    assert result["error"] == "ffmpeg_missing"
    assert "ffmpeg" in result["analysis"]


@pytest.mark.asyncio
async def test_analyze_video_file_combines_frames_and_audio(tmp_path):
    video_path = tmp_path / "clip.mp4"
    frame_path = tmp_path / "frame.jpg"
    video_path.write_bytes(b"fake video")
    frame_path.write_bytes(b"fake frame")

    with (
        patch("tools.video_tools._find_binary", return_value="/usr/bin/fake"),
        patch("tools.video_tools._probe_duration", return_value=12.0),
        patch(
            "tools.video_tools._extract_frames",
            return_value=[SampledFrame(timestamp_seconds=0.5, path=str(frame_path))],
        ),
        patch("tools.video_tools._extract_audio_track", return_value=str(video_path)),
        patch(
            "tools.video_tools.vision_analyze_tool",
            new_callable=AsyncMock,
            return_value=json.dumps({"success": True, "analysis": "A person waves at the camera."}),
        ),
        patch(
            "tools.video_tools.transcribe_audio",
            return_value={"success": True, "transcript": "hello from the video"},
        ),
    ):
        result = await analyze_video_file(str(video_path))

    assert result["success"] is True
    assert result["sampled_frame_count"] == 1
    assert "A person waves at the camera." in result["analysis"]
    assert "hello from the video" in result["analysis"]
    assert str(video_path) in result["analysis"]
