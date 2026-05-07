"""Tests for the Brave Search web provider."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def brave_key(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave-key")


class TestBraveSearchProvider:
    def test_is_configured_true(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        from tools.web_providers.brave import BraveSearchProvider
        assert BraveSearchProvider().is_configured() is True

    def test_is_configured_false(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        from tools.web_providers.brave import BraveSearchProvider
        assert BraveSearchProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.brave import BraveSearchProvider
        assert BraveSearchProvider().provider_name() == "brave"

    def test_search_no_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        from tools.web_providers.brave import BraveSearchProvider
        result = BraveSearchProvider().search("test query")
        assert result["success"] is False
        assert "BRAVE_API_KEY" in result["error"]

    def test_search_success(self, brave_key):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"},
                    {"title": "Result 2", "url": "https://example.com/2", "description": "Desc 2"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            from tools.web_providers.brave import BraveSearchProvider
            result = BraveSearchProvider().search("test query", limit=2)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Result 1"
        assert web[0]["url"] == "https://example.com/1"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_search_respects_limit(self, brave_key):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": f"Result {i}", "url": f"https://example.com/{i}", "description": ""}
                    for i in range(10)
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            from tools.web_providers.brave import BraveSearchProvider
            result = BraveSearchProvider().search("test", limit=3)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3

    def test_search_401_error(self, brave_key):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("httpx.get", side_effect=httpx.HTTPStatusError("", request=MagicMock(), response=mock_resp)):
            from tools.web_providers.brave import BraveSearchProvider
            result = BraveSearchProvider().search("test")
        assert result["success"] is False
        assert "invalid or expired" in result["error"]

    def test_search_429_rate_limit(self, brave_key):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch("httpx.get", side_effect=httpx.HTTPStatusError("", request=MagicMock(), response=mock_resp)):
            from tools.web_providers.brave import BraveSearchProvider
            result = BraveSearchProvider().search("test")
        assert result["success"] is False
        assert "rate limit" in result["error"]

    def test_search_request_error(self, brave_key):
        import httpx
        with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
            from tools.web_providers.brave import BraveSearchProvider
            result = BraveSearchProvider().search("test")
        assert result["success"] is False
        assert "Could not reach" in result["error"]

    def test_search_empty_results(self, brave_key):
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_response):
            from tools.web_providers.brave import BraveSearchProvider
            result = BraveSearchProvider().search("obscure query no results")
        assert result["success"] is True
        assert result["data"]["web"] == []


class TestWebToolsBraveBackend:
    """Integration-style tests for Brave backend routing in web_tools."""

    def test_get_backend_brave_when_configured(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        with patch("tools.web_tools._load_web_config", return_value={"backend": "brave"}):
            from tools.web_tools import _get_backend
            assert _get_backend() == "brave"

    def test_brave_in_fallback_chain(self, monkeypatch):
        # Remove all other keys, set only BRAVE_API_KEY
        for key in ("FIRECRAWL_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY",
                    "EXA_API_KEY", "SEARXNG_URL"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            from tools.web_tools import _get_backend
            assert _get_backend() == "brave"

    def test_check_brave_api_key_true(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        from tools.web_tools import check_brave_api_key
        assert check_brave_api_key() is True

    def test_check_brave_api_key_false(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        from tools.web_tools import check_brave_api_key
        assert check_brave_api_key() is False
