"""Tests for tools.youtube_tool."""

from __future__ import annotations

import json
import sys
import types

import pytest

from tools import youtube_tool


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ?feature=share", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(url, expected):
    assert youtube_tool.extract_video_id(url) == expected


def test_extract_video_id_rejects_invalid():
    with pytest.raises(ValueError):
        youtube_tool.extract_video_id("https://example.com/not-youtube")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (9.9, "0:09"), (65, "1:05"), (3661, "1:01:01")],
)
def test_format_timestamp(seconds, expected):
    assert youtube_tool.format_timestamp(seconds) == expected


def test_youtube_transcript_tool_fetches_and_formats_v1_api(monkeypatch):
    class Segment:
        def __init__(self, text, start, duration):
            self.text = text
            self.start = start
            self.duration = duration

    class FetchedTranscript(list):
        language_code = "en"

    fetched = FetchedTranscript([
        Segment("hello\nworld", 0.0, 2.5),
        Segment("second line", 65.0, 3.0),
    ])

    class MockApi:
        def fetch(self, video_id, languages=None):
            assert video_id == "dQw4w9WgXcQ"
            assert languages == ["en", "ko"]
            return fetched

    mock_mod = types.SimpleNamespace(YouTubeTranscriptApi=MockApi)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_mod)

    raw = youtube_tool.youtube_transcript_tool(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        languages="en,ko",
        include_timestamps=True,
        include_segments=True,
        include_metadata=False,
    )
    result = json.loads(raw)

    assert result["success"] is True
    assert result["video_id"] == "dQw4w9WgXcQ"
    assert result["language"] == "en"
    assert result["is_generated"] is None
    assert result["requested_languages"] == ["en", "ko"]
    assert result["segment_count"] == 2
    assert result["duration"] == "1:08"
    assert result["full_text"] == "hello world second line"
    assert result["timestamped_text"] == "0:00 hello world\n1:05 second line"
    assert result["segments"][0] == {"text": "hello world", "start": 0.0, "duration": 2.5}


def test_youtube_transcript_tool_supports_legacy_get_transcript(monkeypatch):
    class LegacyApi:
        @staticmethod
        def get_transcript(video_id, languages=None):
            assert video_id == "dQw4w9WgXcQ"
            assert languages is None
            return [{"text": "legacy text", "start": 1, "duration": 2}]

        def fetch(self, *args, **kwargs):
            raise TypeError("old api has no instance fetch")

    mock_mod = types.SimpleNamespace(YouTubeTranscriptApi=LegacyApi)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_mod)

    raw = youtube_tool.youtube_transcript_tool(
        "dQw4w9WgXcQ", include_timestamps=False, include_metadata=False)
    result = json.loads(raw)

    assert result["success"] is True
    assert result["full_text"] == "legacy text"
    assert "timestamped_text" not in result


def test_youtube_transcript_tool_returns_structured_error_for_bad_url():
    result = json.loads(youtube_tool.youtube_transcript_tool("not a youtube url"))

    assert result["success"] is False
    assert result["error_code"] == "INVALID_URL"
    assert result["error_message"] == result["error"]
    assert "Expected a YouTube URL" in result["error"]


def test_youtube_transcript_tool_truncates_text_fields(monkeypatch):
    class MockApi:
        def fetch(self, video_id, languages=None):
            return [
                {"text": "alpha beta gamma", "start": 0, "duration": 1},
                {"text": "delta epsilon zeta", "start": 2, "duration": 1},
            ]

    mock_mod = types.SimpleNamespace(YouTubeTranscriptApi=MockApi)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_mod)

    result = json.loads(youtube_tool.youtube_transcript_tool(
        "dQw4w9WgXcQ",
        include_metadata=False,
        include_timestamps=True,
        max_chars=20,
    ))

    assert result["success"] is True
    assert result["truncated"] is True
    assert result["text_stats"]["max_chars"] == 20
    assert result["text_stats"]["full_text_truncated"] is True
    assert result["text_stats"]["timestamped_text_truncated"] is True
    assert len(result["full_text"]) <= 20
    assert len(result["timestamped_text"]) <= 20


def test_youtube_transcript_tool_maps_errors_and_lists_languages(monkeypatch):
    class Transcript:
        language = "English"
        language_code = "en"
        is_generated = False
        is_translatable = True

    class MockApi:
        def fetch(self, video_id, languages=None):
            raise RuntimeError("No transcript found")

        def list(self, video_id):
            return [Transcript()]

    mock_mod = types.SimpleNamespace(YouTubeTranscriptApi=MockApi)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_mod)

    result = json.loads(youtube_tool.youtube_transcript_tool(
        "dQw4w9WgXcQ", include_metadata=False))

    assert result["success"] is False
    assert result["error_code"] == "NO_TRANSCRIPT_FOUND"
    assert result["video_id"] == "dQw4w9WgXcQ"
    assert result["available_languages"] == [
        {
            "language": "English",
            "language_code": "en",
            "is_generated": False,
            "is_translatable": True,
        }
    ]


def test_youtube_transcript_tool_includes_oembed_metadata(monkeypatch):
    class MockApi:
        def fetch(self, video_id, languages=None):
            return [{"text": "hello", "start": 0, "duration": 1}]

    mock_mod = types.SimpleNamespace(YouTubeTranscriptApi=MockApi)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_mod)
    monkeypatch.setattr(
        youtube_tool,
        "_fetch_oembed_metadata",
        lambda video_id: {"title": "Demo", "author_name": "Channel"},
    )

    result = json.loads(youtube_tool.youtube_transcript_tool("dQw4w9WgXcQ"))

    assert result["success"] is True
    assert result["metadata"] == {"title": "Demo", "author_name": "Channel"}
