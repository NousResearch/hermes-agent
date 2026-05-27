"""Tests for Tavily web backend integration.

Coverage:
  _tavily_request() — API key handling, endpoint construction, error propagation.
  _normalize_tavily_search_results() — search response normalization.
  _normalize_tavily_documents() — extract/crawl response normalization, failed_results.
  web_search_tool / web_extract_tool / web_crawl_tool — Tavily dispatch paths.
"""

import json
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from tests.tools.conftest import register_all_web_providers


# ─── _tavily_request ─────────────────────────────────────────────────────────

class TestTavilyRequest:
    """Test suite for the _tavily_request helper."""

    def test_raises_without_api_key(self):
        """No TAVILY_API_KEY → ValueError with guidance."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TAVILY_API_KEY", None)
            from tools.web_tools import _tavily_request
            with pytest.raises(ValueError, match="TAVILY_API_KEY"):
                _tavily_request("search", {"query": "test"})

    def test_posts_with_api_key_in_body(self):
        """api_key is injected into the JSON payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test-key"}):
            with patch("tools.web_tools.httpx.post", return_value=mock_response) as mock_post:
                from tools.web_tools import _tavily_request
                result = _tavily_request("search", {"query": "hello"})

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args
                payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
                assert payload["api_key"] == "tvly-test-key"
                assert payload["query"] == "hello"
                assert "api.tavily.com/search" in call_kwargs.args[0]

    def test_raises_on_http_error(self):
        """Non-2xx responses propagate as httpx.HTTPStatusError."""
        import httpx as _httpx
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-bad-key"}):
            with patch("tools.web_tools.httpx.post", return_value=mock_response):
                from tools.web_tools import _tavily_request
                with pytest.raises(_httpx.HTTPStatusError):
                    _tavily_request("search", {"query": "test"})

    def test_search_http_error_is_redacted_and_actionable(self, monkeypatch):
        """Invalid Tavily credentials return bounded diagnostics without echoing the API key."""
        import httpx as _httpx
        from plugins.web.tavily.provider import TavilyWebSearchProvider

        secret = "tvly-super-secret-key"
        request = _httpx.Request("POST", "https://api.tavily.com/search")
        response = _httpx.Response(401, request=request, text=f"bad key {secret}")
        err = _httpx.HTTPStatusError(
            f"401 Unauthorized for key {secret}", request=request, response=response
        )

        def fake_request(endpoint, payload):
            raise err

        monkeypatch.setattr("plugins.web.tavily.provider._tavily_request", fake_request)
        monkeypatch.setenv("TAVILY_API_KEY", secret)

        result = TavilyWebSearchProvider().search("test")

        assert result["success"] is False
        assert "TAVILY_API_KEY" in result["error"]
        assert "401" in result["error"]
        assert secret not in result["error"]
        assert len(result["error"]) < 240

    def test_non_http_error_uses_allowlisted_diagnostic(self, monkeypatch):
        """Arbitrary exception strings may contain headers/bodies, so never echo them."""
        from plugins.web.tavily.provider import TavilyWebSearchProvider

        leaked = "Authorization: Bearer super-secret-token body={'api_key': 'tvly-super-secret-key'}"

        def fake_request(endpoint, payload):
            raise RuntimeError(leaked)

        monkeypatch.setattr("plugins.web.tavily.provider._tavily_request", fake_request)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-super-secret-key")

        result = TavilyWebSearchProvider().search("test")

        assert result["success"] is False
        assert result["error"] == "Tavily request failed (RuntimeError)."
        assert "super-secret-token" not in result["error"]
        assert "tvly-super-secret-key" not in result["error"]
        assert "Authorization" not in result["error"]


# ─── _normalize_tavily_search_results ─────────────────────────────────────────

class TestNormalizeTavilySearchResults:
    """Test search result normalization."""

    def test_basic_normalization(self):
        from tools.web_tools import _normalize_tavily_search_results
        raw = {
            "results": [
                {"title": "Python Docs", "url": "https://docs.python.org", "content": "Official docs", "score": 0.9},
                {"title": "Tutorial", "url": "https://example.com", "content": "A tutorial", "score": 0.8},
            ]
        }
        result = _normalize_tavily_search_results(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Python Docs"
        assert web[0]["url"] == "https://docs.python.org"
        assert web[0]["description"] == "Official docs"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_empty_results(self):
        from tools.web_tools import _normalize_tavily_search_results
        result = _normalize_tavily_search_results({"results": []})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_fields(self):
        from tools.web_tools import _normalize_tavily_search_results
        result = _normalize_tavily_search_results({"results": [{}]})
        web = result["data"]["web"]
        assert web[0]["title"] == ""
        assert web[0]["url"] == ""
        assert web[0]["description"] == ""


# ─── _normalize_tavily_documents ──────────────────────────────────────────────

class TestNormalizeTavilyDocuments:
    """Test extract/crawl document normalization."""

    def test_basic_document(self):
        from tools.web_tools import _normalize_tavily_documents
        raw = {
            "results": [{
                "url": "https://example.com",
                "title": "Example",
                "raw_content": "Full page content here",
            }]
        }
        docs = _normalize_tavily_documents(raw)
        assert len(docs) == 1
        assert docs[0]["url"] == "https://example.com"
        assert docs[0]["title"] == "Example"
        assert docs[0]["content"] == "Full page content here"
        assert docs[0]["raw_content"] == "Full page content here"
        assert docs[0]["metadata"]["sourceURL"] == "https://example.com"

    def test_falls_back_to_content_when_no_raw_content(self):
        from tools.web_tools import _normalize_tavily_documents
        raw = {"results": [{"url": "https://example.com", "content": "Snippet"}]}
        docs = _normalize_tavily_documents(raw)
        assert docs[0]["content"] == "Snippet"

    def test_failed_results_included(self):
        from tools.web_tools import _normalize_tavily_documents
        raw = {
            "results": [],
            "failed_results": [
                {"url": "https://fail.com", "error": "timeout"},
            ],
        }
        docs = _normalize_tavily_documents(raw)
        assert len(docs) == 1
        assert docs[0]["url"] == "https://fail.com"
        assert docs[0]["error"] == "timeout"
        assert docs[0]["content"] == ""

    def test_failed_urls_included(self):
        from tools.web_tools import _normalize_tavily_documents
        raw = {
            "results": [],
            "failed_urls": ["https://bad.com"],
        }
        docs = _normalize_tavily_documents(raw)
        assert len(docs) == 1
        assert docs[0]["url"] == "https://bad.com"
        assert docs[0]["error"] == "extraction failed"

    def test_fallback_url(self):
        from tools.web_tools import _normalize_tavily_documents
        raw = {"results": [{"content": "data"}]}
        docs = _normalize_tavily_documents(raw, fallback_url="https://fallback.com")
        assert docs[0]["url"] == "https://fallback.com"


# ─── web_search_tool (Tavily dispatch) ────────────────────────────────────────

class TestWebSearchTavily:
    """Test web_search_tool dispatch to Tavily."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_search_dispatches_to_tavily(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": "Result", "url": "https://r.com", "content": "desc", "score": 0.9}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="tavily"), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}), \
             patch("tools.web_tools.httpx.post", return_value=mock_response), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("test query", limit=3))
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"


# ─── web_extract_tool (Tavily dispatch) ───────────────────────────────────────

class TestWebExtractTavily:
    """Test web_extract_tool dispatch to Tavily."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_extract_dispatches_to_tavily(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"url": "https://example.com", "raw_content": "Extracted content", "title": "Page"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="tavily"), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}), \
             patch("tools.web_tools.httpx.post", return_value=mock_response), \
             patch("tools.web_tools.process_content_with_llm", return_value=None):
            from tools.web_tools import web_extract_tool
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["url"] == "https://example.com"


# ─── web_crawl_tool (Tavily dispatch) ─────────────────────────────────────────

class TestWebCrawlTavily:
    """Test web_crawl_tool dispatch to Tavily."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_crawl_dispatches_to_tavily(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com/page1", "raw_content": "Page 1 content", "title": "Page 1"},
                {"url": "https://example.com/page2", "raw_content": "Page 2 content", "title": "Page 2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="tavily"), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}), \
             patch("tools.web_tools.httpx.post", return_value=mock_response), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_crawl_tool
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_crawl_tool("https://example.com", use_llm_processing=False)
            ))
            assert "results" in result
            assert len(result["results"]) == 2
            assert result["results"][0]["title"] == "Page 1"

    def test_crawl_sends_instructions(self):
        """Instructions are included in the Tavily crawl payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="tavily"), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}), \
             patch("tools.web_tools.httpx.post", return_value=mock_response) as mock_post, \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_crawl_tool
            asyncio.get_event_loop().run_until_complete(
                web_crawl_tool("https://example.com", instructions="Find docs", use_llm_processing=False)
            )
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["instructions"] == "Find docs"
            assert payload["url"] == "https://example.com"
