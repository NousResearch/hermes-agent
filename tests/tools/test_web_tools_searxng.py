"""Tests for SearXNG web backend integration.

Coverage:
  _get_searxng_url() — URL handling, missing env var.
  _searxng_search() — search request construction, result normalization.
  web_search_tool — SearXNG dispatch path.
  web_extract_tool — SearXNG graceful fallback to Firecrawl.
  Backend selection — SearXNG in _get_backend and _is_backend_available.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


# ─── _get_searxng_url ───────────────────────────────────────────────────────

class TestGetSearxngUrl:
    """Test suite for the _get_searxng_url helper."""

    def test_raises_without_url(self):
        """No SEARXNG_URL → ValueError with guidance."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEARXNG_URL", None)
            from tools.web_tools import _get_searxng_url
            with pytest.raises(ValueError, match="SEARXNG_URL"):
                _get_searxng_url()

    def test_returns_trimmed_url(self):
        """Trailing slashes and whitespace are stripped."""
        with patch.dict(os.environ, {"SEARXNG_URL": "  https://searx.example.com/  "}):
            from tools.web_tools import _get_searxng_url
            assert _get_searxng_url() == "https://searx.example.com"

    def test_returns_url_without_trailing_slash(self):
        """Trailing slash is removed for clean URL joining."""
        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com/"}):
            from tools.web_tools import _get_searxng_url
            assert _get_searxng_url() == "https://searx.example.com"


# ─── _searxng_search ────────────────────────────────────────────────────────

class TestSearxngSearch:
    """Test suite for the _searxng_search helper."""

    def test_returns_normalized_results(self):
        """SearXNG JSON response is normalized to the standard format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Python Docs", "url": "https://docs.python.org", "content": "Official docs"},
                {"title": "Tutorial", "url": "https://example.com", "content": "A tutorial"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                with patch("tools.interrupt.is_interrupted", return_value=False):
                    from tools.web_tools import _searxng_search
                    result = _searxng_search("python docs", limit=5)

                    assert result["success"] is True
                    web = result["data"]["web"]
                    assert len(web) == 2
                    assert web[0]["title"] == "Python Docs"
                    assert web[0]["url"] == "https://docs.python.org"
                    assert web[0]["description"] == "Official docs"
                    assert web[0]["position"] == 1
                    assert web[1]["position"] == 2

                    # Verify correct URL and params
                    mock_get.assert_called_once()
                    call_args = mock_get.call_args
                    assert "searx.example.com/search" in call_args.args[0]
                    params = call_args.kwargs.get("params") or call_args[1].get("params")
                    assert params["q"] == "python docs"
                    assert params["format"] == "json"

    def test_respects_limit(self):
        """Results are truncated to the requested limit."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": f"https://r{i}.com", "content": f"desc {i}"}
                for i in range(10)
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response):
                with patch("tools.interrupt.is_interrupted", return_value=False):
                    from tools.web_tools import _searxng_search
                    result = _searxng_search("test", limit=3)
                    assert len(result["data"]["web"]) == 3

    def test_empty_results(self):
        """Empty SearXNG response returns empty web list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response):
                with patch("tools.interrupt.is_interrupted", return_value=False):
                    from tools.web_tools import _searxng_search
                    result = _searxng_search("nothing")
                    assert result["success"] is True
                    assert result["data"]["web"] == []

    def test_missing_fields(self):
        """Results with missing fields default to empty strings."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{}]}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response):
                with patch("tools.interrupt.is_interrupted", return_value=False):
                    from tools.web_tools import _searxng_search
                    result = _searxng_search("test")
                    web = result["data"]["web"]
                    assert web[0]["title"] == ""
                    assert web[0]["url"] == ""
                    assert web[0]["description"] == ""

    def test_interrupted_returns_error(self):
        """Interrupted search returns error dict."""
        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            with patch("tools.interrupt.is_interrupted", return_value=True):
                from tools.web_tools import _searxng_search
                result = _searxng_search("test")
                assert result["success"] is False
                assert "Interrupted" in result["error"]

    def test_raises_on_http_error(self):
        """Non-2xx responses propagate as httpx.HTTPStatusError."""
        import httpx as _httpx
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=mock_response
        )

        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response):
                with patch("tools.interrupt.is_interrupted", return_value=False):
                    from tools.web_tools import _searxng_search
                    with pytest.raises(_httpx.HTTPStatusError):
                        _searxng_search("test")


# ─── web_search_tool (SearXNG dispatch) ─────────────────────────────────────

class TestWebSearchSearxng:
    """Test web_search_tool dispatch to SearXNG."""

    def test_search_dispatches_to_searxng(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": "Result", "url": "https://r.com", "content": "desc"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="searxng"), \
             patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}), \
             patch("tools.web_tools.httpx.get", return_value=mock_response), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("test query", limit=3))
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"


# ─── web_extract_tool (SearXNG falls back to Firecrawl) ─────────────────────

class TestWebExtractSearxng:
    """Test web_extract_tool Firecrawl fallback when SearXNG is selected."""

    def test_extract_falls_back_to_firecrawl(self):
        mock_firecrawl = MagicMock()
        mock_firecrawl.scrape.return_value = {
            "markdown": "# Page Content",
            "metadata": {"title": "Example Page"},
        }

        with patch("tools.web_tools._get_backend", return_value="searxng"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools._get_firecrawl_client", return_value=mock_firecrawl), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_extract_tool
            import asyncio
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))
            assert "results" in result
            assert result["results"][0]["content"] == "# Page Content"
            mock_firecrawl.scrape.assert_called_once()

    def test_extract_fallback_handles_firecrawl_error(self):
        mock_firecrawl = MagicMock()
        mock_firecrawl.scrape.side_effect = ValueError("No API key")

        with patch("tools.web_tools._get_backend", return_value="searxng"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools._get_firecrawl_client", return_value=mock_firecrawl), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_extract_tool
            import asyncio
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))
            assert "results" in result
            assert "error" in result["results"][0]
            assert "failed" in result["results"][0]["error"].lower()


# ─── Backend selection ──────────────────────────────────────────────────────

class TestSearxngBackendSelection:
    """Test that SearXNG is correctly selected as a backend."""

    def test_searxng_selected_from_config(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng"}):
            from tools.web_tools import _get_backend
            assert _get_backend() == "searxng"

    def test_searxng_available_with_url(self):
        with patch.dict(os.environ, {"SEARXNG_URL": "https://searx.example.com"}):
            from tools.web_tools import _is_backend_available
            assert _is_backend_available("searxng") is True

    def test_searxng_unavailable_without_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEARXNG_URL", None)
            from tools.web_tools import _is_backend_available
            assert _is_backend_available("searxng") is False
