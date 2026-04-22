"""Tests for Hermes DealSeek UGC analytics tools."""

from __future__ import annotations

import json

from tools.ugc_analytics_tool import (
    DEFAULT_SIDESHIFT_API_BASE_URL,
    _build_playback_url,
    _check_sideshift_available,
    _check_trackermane_available,
    _get_sideshift_base_url,
    _handle_sideshift_get_overview,
    _handle_tracker_get_video_details,
    _map_account,
    _map_video,
    _parse_json_each_row,
    get_sideshift_post,
    search_trackermane_accounts,
    search_trackermane_videos,
)


class TestAvailabilityChecks:
    def test_trackermane_check_accepts_explicit_env_vars(self, monkeypatch):
        monkeypatch.setenv("TRACKERMANE_POSTGRES_URL", "postgres://example")
        monkeypatch.setenv("TRACKERMANE_CLICKHOUSE_URL", "https://clickhouse")
        assert _check_trackermane_available() is True

    def test_trackermane_check_accepts_fallback_aliases(self, monkeypatch):
        monkeypatch.delenv("TRACKERMANE_POSTGRES_URL", raising=False)
        monkeypatch.delenv("TRACKERMANE_CLICKHOUSE_URL", raising=False)
        monkeypatch.setenv("POSTGRES_URL", "postgres://example")
        monkeypatch.setenv("CLICKHOUSE_URL", "https://clickhouse")
        assert _check_trackermane_available() is True

    def test_sideshift_check_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SIDESHIFT_API_KEY", raising=False)
        assert _check_sideshift_available() is False
        monkeypatch.setenv("SIDESHIFT_API_KEY", "secret")
        assert _check_sideshift_available() is True


class TestHelpers:
    def test_parse_json_each_row(self):
        rows = _parse_json_each_row('{"a":1}\n{"b":2}\n')
        assert rows == [{"a": 1}, {"b": 2}]

    def test_build_playback_url(self):
        assert _build_playback_url("uid-1", "customer.example.com") == "https://customer.example.com/uid-1/iframe"
        assert _build_playback_url("uid-1", None) == "https://iframe.videodelivery.net/uid-1"
        assert _build_playback_url(None, "customer.example.com") is None

    def test_get_sideshift_base_url_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SIDESHIFT_API_BASE_URL", raising=False)
        assert _get_sideshift_base_url() == DEFAULT_SIDESHIFT_API_BASE_URL
        monkeypatch.setenv("SIDESHIFT_API_BASE_URL", "https://custom.example/api/")
        assert _get_sideshift_base_url() == "https://custom.example/api"


class TestTrackerSearches:
    def test_search_videos_query_path_sorts_in_python(self, monkeypatch):
        monkeypatch.setattr(
            "tools.ugc_analytics_tool._search_content_metadata_candidates",
            lambda query, platform, published_within_days: [
                {"id": "video-a", "title": "A", "platform": "tiktok"},
                {"id": "video-b", "title": "B", "platform": "tiktok"},
            ],
        )
        monkeypatch.setattr(
            "tools.ugc_analytics_tool._fetch_latest_video_metrics_by_ids",
            lambda ids: [
                {"content_id": "video-a", "view_count": 10, "latest_snapshot_at": "2026-01-01T00:00:00+00:00"},
                {"content_id": "video-b", "view_count": 100, "latest_snapshot_at": "2026-01-02T00:00:00+00:00"},
            ],
        )

        result = search_trackermane_videos({"query": "test", "sortBy": "view_count", "limit": 1})

        assert result["results"][0]["contentId"] == "video-b"
        assert result["filters"]["query"] == "test"

    def test_search_accounts_query_path_sorts_in_python(self, monkeypatch):
        monkeypatch.setattr(
            "tools.ugc_analytics_tool._search_account_metadata_candidates",
            lambda query, platform, limit: [
                {"id": "acct-a", "handle": "a"},
                {"id": "acct-b", "handle": "b"},
            ],
        )
        monkeypatch.setattr(
            "tools.ugc_analytics_tool._fetch_latest_account_metrics_by_ids",
            lambda ids: [
                {"account_id": "acct-a", "total_views": 10, "latest_snapshot_at": "2026-01-01T00:00:00+00:00"},
                {"account_id": "acct-b", "total_views": 999, "latest_snapshot_at": "2026-01-02T00:00:00+00:00"},
            ],
        )

        result = search_trackermane_accounts({"query": "acct", "sortBy": "total_views", "limit": 1})

        assert result["results"][0]["accountId"] == "acct-b"
        assert result["filters"]["query"] == "acct"

    def test_tracker_video_details_requires_lookup_value(self):
        payload = json.loads(_handle_tracker_get_video_details({}))
        assert "error" in payload
        assert "contentId" in payload["error"]


class TestMapping:
    def test_map_video_prefers_metadata_and_derives_playback_url(self):
        mapped = _map_video(
            {"content_id": "video-1", "platform": "tiktok", "view_count": 77, "latest_snapshot_at": "2026-01-01T00:00:00+00:00"},
            {
                "title": "Hello",
                "platform": "instagram",
                "platform_content_id": "ig-1",
                "stream_uid": "stream-1",
                "stream_customer_subdomain": "stream.example.com",
            },
        )
        assert mapped["platform"] == "instagram"
        assert mapped["title"] == "Hello"
        assert mapped["playbackUrl"] == "https://stream.example.com/stream-1/iframe"

    def test_map_account_prefers_metadata(self):
        mapped = _map_account(
            {"account_id": "acct-1", "platform": "tiktok", "total_views": 123},
            {"handle": "creator", "platform": "instagram"},
        )
        assert mapped["platform"] == "instagram"
        assert mapped["handle"] == "creator"
        assert mapped["totalViews"] == 123


class TestSideShiftHandlers:
    def test_sideshift_handler_requires_program_id(self):
        payload = json.loads(_handle_sideshift_get_overview({}))
        assert payload == {"error": "programId is required."}

    def test_get_sideshift_post_unwraps_data(self, monkeypatch):
        monkeypatch.setattr(
            "tools.ugc_analytics_tool._call_sideshift",
            lambda path, params=None: {"data": {"id": "post-1", "views": 55}},
        )
        result = get_sideshift_post("post-1")
        assert result == {
            "source": "SideShift",
            "postId": "post-1",
            "data": {"id": "post-1", "views": 55},
        }
