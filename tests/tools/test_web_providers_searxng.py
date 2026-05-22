"""Tests for the SearXNG web search provider.

Covers:
- SearXNGSearchProvider.is_configured() env var gating
- SearXNGSearchProvider.search() — happy path, HTTP error, request error, bad JSON
- Result normalization (title, url, description, position)
- Score-based sorting and limit truncation
- _is_backend_available("searxng") integration
- _get_backend() recognizes "searxng" as a valid configured backend
- check_web_api_key() includes searxng in availability check
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SearXNGSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestSearXNGSearchProviderIsConfigured:
    def test_configured_when_url_set(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        assert SearXNGSearchProvider().is_configured() is True

    def test_not_configured_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        from tools.web_providers.searxng import SearXNGSearchProvider
        assert SearXNGSearchProvider().is_configured() is False

    def test_not_configured_when_url_empty_string(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "   ")
        from tools.web_providers.searxng import SearXNGSearchProvider
        assert SearXNGSearchProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.searxng import SearXNGSearchProvider
        assert SearXNGSearchProvider().provider_name() == "searxng"

    def test_implements_web_search_provider(self):
        from tools.web_providers.base import WebSearchProvider
        from tools.web_providers.searxng import SearXNGSearchProvider
        assert issubclass(SearXNGSearchProvider, WebSearchProvider)


class TestSearXNGSearchProviderSearch:
    """Happy path and error handling for SearXNGSearchProvider.search()."""

    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "Result A", "url": "https://a.example.com", "content": "Desc A", "score": 0.9},
            {"title": "Result B", "url": "https://b.example.com", "content": "Desc B", "score": 0.7},
            {"title": "Result C", "url": "https://c.example.com", "content": "Desc C", "score": 0.5},
        ]
    }

    def _make_mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_happy_path_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0]["title"] == "Result A"
        assert web[0]["url"] == "https://a.example.com"
        assert web[0]["description"] == "Desc A"
        assert web[0]["position"] == 1

    def test_results_sorted_by_score_descending(self, monkeypatch):
        """Results should be sorted by score before limit is applied."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        unordered = {
            "results": [
                {"title": "Low",  "url": "https://low.example.com",  "content": "", "score": 0.1},
                {"title": "High", "url": "https://high.example.com", "content": "", "score": 0.99},
                {"title": "Mid",  "url": "https://mid.example.com",  "content": "", "score": 0.5},
            ]
        }
        mock_resp = self._make_mock_response(unordered)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGSearchProvider().search("query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "High"
        assert result["data"]["web"][1]["title"] == "Mid"
        assert result["data"]["web"][2]["title"] == "Low"

    def test_limit_is_respected(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGSearchProvider().search("query", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2

    def test_position_is_one_indexed(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGSearchProvider().search("query", limit=5)

        positions = [r["position"] for r in result["data"]["web"]]
        assert positions == [1, 2, 3]

    def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        mock_resp = self._make_mock_response({"results": []})

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_score_falls_back_to_zero(self, monkeypatch):
        """Results without a score field should sort to the bottom."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider
        data = {
            "results": [
                {"title": "No score", "url": "https://noscore.example.com", "content": ""},
                {"title": "Has score", "url": "https://scored.example.com", "content": "", "score": 0.8},
            ]
        }
        mock_resp = self._make_mock_response(data)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGSearchProvider().search("query", limit=5)

        assert result["success"] is True
        # Has score should sort first (0.8 > 0)
        assert result["data"]["web"][0]["title"] == "Has score"

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        http_err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)

        with patch("httpx.get", side_effect=http_err):
            result = SearXNGSearchProvider().search("query", limit=5)

        assert result["success"] is False
        assert "500" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_providers.searxng import SearXNGSearchProvider

        with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
            result = SearXNGSearchProvider().search("query", limit=5)

        assert result["success"] is False
        assert "localhost:8080" in result["error"] or "connection" in result["error"].lower()

    def test_missing_url_returns_failure(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        from tools.web_providers.searxng import SearXNGSearchProvider

        result = SearXNGSearchProvider().search("query", limit=5)
        assert result["success"] is False
        assert "SEARXNG_URL" in result["error"]

    def test_trailing_slash_stripped_from_url(self, monkeypatch):
        """Base URL trailing slash should not produce double-slash in endpoint."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080/")
        from tools.web_providers.searxng import SearXNGSearchProvider
        mock_resp = self._make_mock_response({"results": []})

        calls = []
        def capture_get(url, **kwargs):
            calls.append(url)
            return mock_resp

        with patch("httpx.get", side_effect=capture_get):
            SearXNGSearchProvider().search("query", limit=5)

        assert calls[0] == "http://localhost:8080/search", f"Got: {calls[0]}"


# ---------------------------------------------------------------------------
# Integration: _is_backend_available recognizes "searxng"
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_searxng_available_when_url_set(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("searxng") is True

    def test_searxng_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("searxng") is False

    def test_unknown_backend_still_false(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("unknownbackend") is False


# ---------------------------------------------------------------------------
# Integration: _get_backend() accepts "searxng" as configured value
# ---------------------------------------------------------------------------


class TestGetBackendSearXNG:
    def test_configured_searxng_returns_searxng(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert web_tools._get_backend() == "searxng"

    def test_auto_detect_picks_searxng_when_only_url_set(self, monkeypatch):
        """When no backend is configured but SEARXNG_URL is set, auto-detect returns it."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        # Suppress tool gateway
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        assert web_tools._get_backend() == "searxng"

    def test_searxng_does_not_override_higher_priority_provider(self, monkeypatch):
        """Tavily (higher priority than searxng) should win in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        assert web_tools._get_backend() == "tavily"


# ---------------------------------------------------------------------------
# Integration: check_web_api_key includes searxng
# ---------------------------------------------------------------------------


class TestCheckWebApiKey:
    def test_searxng_satisfies_check_web_api_key(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert web_tools.check_web_api_key() is True

    def test_no_credentials_fails(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        assert web_tools.check_web_api_key() is False


# ---------------------------------------------------------------------------
# searxng-only: web_extract and web_crawl return clear errors
# ---------------------------------------------------------------------------


class TestSearXNGOnlyExtractCrawlErrors:
    """When searxng is the active backend, extract/crawl must return clear errors."""

    def test_web_crawl_searxng_returns_clear_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        import json
        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_crawl_tool("https://example.com")
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower() or "SearXNG" in result["error"]

    def test_web_extract_searxng_returns_clear_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        import json
        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower() or "SearXNG" in result["error"]
