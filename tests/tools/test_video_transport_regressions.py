"""Regression coverage for video-analysis review repairs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.video_analysis import core
from tools.video_analysis.frames import extract_video_frames


def test_core_reuses_single_frame_extraction_owner():
    assert core._extract_video_frames is extract_video_frames
    assert not hasattr(core, "_probe_video_duration")
    assert not hasattr(core, "_frame_timestamps")
    assert not hasattr(core, "_jpeg_data_url")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HTTP 429 rate limit exceeded", 429),
        ("429 rate limit exceeded", 429),
        ("stream ended after 429 bytes", None),
        ("model returned 404 tokens", None),
        ("attempt 429: retrying", None),
    ],
)
def test_status_code_string_fallback_is_leading_only(message, expected):
    assert core._status_code(RuntimeError(message)) == expected


def test_oversized_video_is_rejected_before_reading_file(tmp_path):
    video = tmp_path / "oversized.mp4"
    with video.open("wb") as handle:
        handle.truncate(core._max_raw_video_bytes("video/mp4") + 1)

    with patch.object(
        Path,
        "read_bytes",
        side_effect=AssertionError("oversized input must not be buffered"),
    ):
        with pytest.raises(ValueError, match="Video too large for API"):
            core._video_to_base64_data_url(video, "video/mp4")


def test_configured_video_timeout_is_not_silently_clamped():
    sentinel = object()

    def fake_cfg_get(_cfg, *keys, default=None):
        if keys == ("auxiliary", "video"):
            return {"timeout": 60, "temperature": 0.2}
        if keys == ("auxiliary", "vision"):
            return {}
        return default

    with (
        patch("hermes_cli.config.load_config", return_value=sentinel),
        patch("hermes_cli.config.cfg_get", side_effect=fake_cfg_get),
    ):
        timeout, temperature = core._resolve_video_call_settings()

    assert timeout == 60.0
    assert temperature == 0.2
