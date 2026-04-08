"""Tests for SerpAPI web backend integration.

Coverage:
  _serpapi_search() — API key handling, endpoint construction, error propagation,
                      result normalization (organic_results → standard web format).
  web_search_tool  — SerpAPI dispatch path.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


# ─── _serpapi_search ─────────────────────────────────────────────────────────

class TestSerpApiSearch:
    """Test suite for the _serpapi_search helper."""

    def test_raises_without_api_key(self):
        """No SERPAPI_API_KEY → ValueError with guidance."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SERPAPI_API_KEY", None)
            from tools.web_tools import _serpapi_search
            with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
                _serpapi_search("test query")

    def test_sends_correct_params(self):
        """API key, engine, query, and num are sent as GET params."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic_results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-test-key"}), \
             patch("tools.web_tools.httpx.Client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _serpapi_search
            result = _serpapi_search("hello world", limit=5)

            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["api_key"] == "serp-test-key"
            assert params["engine"] == "google"
            assert params["q"] == "hello world"
            assert params["num"] == 5

    def test_limit_capped_at_20(self):
        """num param is capped at 20 even if limit is higher."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic_results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-test-key"}), \
             patch("tools.web_tools.httpx.Client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _serpapi_search
            _serpapi_search("query", limit=50)

            call_args = mock_client.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["num"] == 20

    def test_raises_on_http_error(self):
        """Non-2xx responses propagate as httpx.HTTPStatusError."""
        import httpx as _httpx
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-bad-key"}), \
             patch("tools.web_tools.httpx.Client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _serpapi_search
            with pytest.raises(_httpx.HTTPStatusError):
                _serpapi_search("test")

    def test_returns_early_when_interrupted(self):
        """If interrupted before HTTP call, returns error dict immediately."""
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_tools import _serpapi_search
            result = _serpapi_search("test")
            assert result["success"] is False
            assert "error" in result


# ─── Result normalization ────────────────────────────────────────────────────

class TestSerpApiNormalization:
    """Test that organic_results are normalized to the standard web format."""

    def _search_with_response(self, api_response: dict) -> dict:
        """Helper: run _serpapi_search with a mocked API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-test-key"}), \
             patch("tools.web_tools.httpx.Client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _serpapi_search
            return _serpapi_search("test")

    def test_basic_normalization(self):
        """organic_results fields map to standard title/url/description/position."""
        raw = {
            "organic_results": [
                {"title": "Python Docs", "link": "https://docs.python.org", "snippet": "Official docs"},
                {"title": "Tutorial", "link": "https://example.com", "snippet": "A tutorial"},
            ]
        }
        result = self._search_with_response(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Python Docs"
        assert web[0]["url"] == "https://docs.python.org"
        assert web[0]["description"] == "Official docs"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_empty_results(self):
        """No organic_results → empty web list, still success."""
        result = self._search_with_response({"organic_results": []})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_organic_results_key(self):
        """Response without organic_results key → empty web list."""
        result = self._search_with_response({"search_metadata": {}})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_fields_default_to_empty_string(self):
        """Result entries with missing link/title/snippet → empty strings."""
        result = self._search_with_response({"organic_results": [{}]})
        web = result["data"]["web"]
        assert web[0]["title"] == ""
        assert web[0]["url"] == ""
        assert web[0]["description"] == ""
        assert web[0]["position"] == 1

    def test_results_capped_by_limit(self):
        """Only `limit` results are returned even if API gives more."""
        raw = {
            "organic_results": [
                {"title": f"Result {i}", "link": f"https://r{i}.com", "snippet": f"Desc {i}"}
                for i in range(10)
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-test-key"}), \
             patch("tools.web_tools.httpx.Client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _serpapi_search
            result = _serpapi_search("test", limit=3)
            assert len(result["data"]["web"]) == 3
            assert result["data"]["web"][2]["position"] == 3


# ─── web_search_tool (SerpAPI dispatch) ──────────────────────────────────────

class TestWebSearchSerpApi:
    """Test web_search_tool dispatch to SerpAPI."""

    def test_search_dispatches_to_serpapi(self):
        """web_search_tool with serpapi backend → calls _serpapi_search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [
                {"title": "Result", "link": "https://r.com", "snippet": "desc"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("tools.web_tools._get_backend", return_value="serpapi"), \
             patch.dict(os.environ, {"SERPAPI_API_KEY": "serp-test"}), \
             patch("tools.web_tools.httpx.Client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("test query", limit=3))
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"
