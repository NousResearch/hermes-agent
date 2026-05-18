"""Tests for TinyFish web backend integration.

Coverage:
  _tinyfish_request() — API key handling, URL construction, error propagation.
  _normalize_tinyfish_search_results() — search response normalization.
  _normalize_tinyfish_documents() — extract response normalization, failed_results.
  TinyfishWebSearchProvider — availability, capabilities, search, extract.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


# ─── _tinyfish_request ─────────────────────────────────────────────────────────

class TestTinyfishRequest:
    """Test suite for the _tinyfish_request helper."""

    def test_raises_without_api_key(self):
        """No TINYFISH_API_KEY → ValueError with guidance."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TINYFISH_API_KEY", None)
            from plugins.web.tinyfish.provider import _tinyfish_request
            with pytest.raises(ValueError, match="TINYFISH_API_KEY"):
                _tinyfish_request("GET", "https://api.search.tinyfish.ai", params={"q": "test"})

    def test_get_includes_api_key_header(self):
        """X-API-Key header is sent on GET requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-test-key"}):
            with patch("plugins.web.tinyfish.provider.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                mock_client.get.return_value = mock_response

                from plugins.web.tinyfish.provider import _tinyfish_request
                _tinyfish_request("GET", "https://api.search.tinyfish.ai", params={"q": "hello"})

                mock_client.get.assert_called_once()
                call_kwargs = mock_client.get.call_args
                assert call_kwargs.kwargs["headers"]["X-API-Key"] == "tf-test-key"

    def test_post_includes_api_key_header(self):
        """X-API-Key header is sent on POST requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-test-key"}):
            with patch("plugins.web.tinyfish.provider.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_response

                from plugins.web.tinyfish.provider import _tinyfish_request
                _tinyfish_request("POST", "https://api.fetch.tinyfish.ai", json={"urls": ["https://example.com"]})

                mock_client.post.assert_called_once()
                call_kwargs = mock_client.post.call_args
                assert call_kwargs.kwargs["headers"]["X-API-Key"] == "tf-test-key"

    def test_raises_on_http_error(self):
        """Non-2xx responses propagate as httpx.HTTPStatusError."""
        import httpx as _httpx
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-bad-key"}):
            with patch("plugins.web.tinyfish.provider.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                mock_client.get.return_value = mock_response

                from plugins.web.tinyfish.provider import _tinyfish_request
                with pytest.raises(_httpx.HTTPStatusError):
                    _tinyfish_request("GET", "https://api.search.tinyfish.ai", params={"q": "test"})


# ─── _normalize_tinyfish_search_results ─────────────────────────────────────────

class TestNormalizeTinyfishSearchResults:
    """Test search result normalization."""

    def test_basic_normalization(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_search_results
        raw = {
            "results": [
                {"title": "Python Docs", "url": "https://docs.python.org", "snippet": "Official docs", "score": 0.9},
                {"title": "Tutorial", "url": "https://example.com", "snippet": "A tutorial", "score": 0.8},
            ]
        }
        result = _normalize_tinyfish_search_results(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Python Docs"
        assert web[0]["url"] == "https://docs.python.org"
        assert web[0]["description"] == "Official docs"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_empty_results(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_search_results
        result = _normalize_tinyfish_search_results({"results": []})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_fields(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_search_results
        result = _normalize_tinyfish_search_results({"results": [{}]})
        web = result["data"]["web"]
        assert web[0]["title"] == ""
        assert web[0]["url"] == ""
        assert web[0]["description"] == ""

    def test_description_falls_back_to_description_field(self):
        """When 'snippet' is absent, 'description' is used."""
        from plugins.web.tinyfish.provider import _normalize_tinyfish_search_results
        raw = {"results": [{"title": "T", "url": "https://x.com", "description": "from desc field"}]}
        result = _normalize_tinyfish_search_results(raw)
        assert result["data"]["web"][0]["description"] == "from desc field"


# ─── _normalize_tinyfish_documents ──────────────────────────────────────────────

class TestNormalizeTinyfishDocuments:
    """Test extract document normalization."""

    def test_basic_document(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_documents
        raw = {
            "results": [{
                "url": "https://example.com",
                "title": "Example",
                "raw_content": "Full page content here",
            }]
        }
        docs = _normalize_tinyfish_documents(raw)
        assert len(docs) == 1
        assert docs[0]["url"] == "https://example.com"
        assert docs[0]["title"] == "Example"
        assert docs[0]["content"] == "Full page content here"
        assert docs[0]["raw_content"] == "Full page content here"
        assert docs[0]["metadata"]["sourceURL"] == "https://example.com"

    def test_falls_back_to_content_when_no_raw_content(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_documents
        raw = {"results": [{"url": "https://example.com", "content": "Snippet"}]}
        docs = _normalize_tinyfish_documents(raw)
        assert docs[0]["content"] == "Snippet"

    def test_failed_results_included(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_documents
        raw = {
            "results": [],
            "failed_results": [
                {"url": "https://fail.com", "error": "timeout"},
            ],
        }
        docs = _normalize_tinyfish_documents(raw)
        assert len(docs) == 1
        assert docs[0]["url"] == "https://fail.com"
        assert docs[0]["error"] == "timeout"
        assert docs[0]["content"] == ""

    def test_fallback_url(self):
        from plugins.web.tinyfish.provider import _normalize_tinyfish_documents
        raw = {"results": [{"content": "data"}]}
        docs = _normalize_tinyfish_documents(raw, fallback_url="https://fallback.com")
        assert docs[0]["url"] == "https://fallback.com"


# ─── TinyfishWebSearchProvider ─────────────────────────────────────────────────

class TestTinyfishProvider:
    """Test the Tinyfish provider class."""

    def test_name(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        assert TinyfishWebSearchProvider().name == "tinyfish"

    def test_display_name(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        assert TinyfishWebSearchProvider().display_name == "TinyFish"

    def test_supports_search(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        assert TinyfishWebSearchProvider().supports_search() is True

    def test_supports_extract(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        assert TinyfishWebSearchProvider().supports_extract() is True

    def test_supports_crawl(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        assert TinyfishWebSearchProvider().supports_crawl() is False

    def test_is_available_requires_api_key(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TINYFISH_API_KEY", None)
            assert p.is_available() is False
        with patch.dict(os.environ, {"TINYFISH_API_KEY": "real"}):
            assert p.is_available() is True

    def test_search_returns_error_without_api_key(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TINYFISH_API_KEY", None)
            result = p.search("test")
            assert result["success"] is False
            assert "TINYFISH_API_KEY" in result["error"]

    def test_search_returns_error_on_network_failure(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-test"}):
            with patch("plugins.web.tinyfish.provider.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                mock_client.get.side_effect = Exception("network error")

                result = p.search("test")
                assert result["success"] is False
                assert "TinyFish search failed" in result["error"]

    def test_search_returns_error_when_interrupted(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch("tools.interrupt.is_interrupted", return_value=True):
            result = p.search("test")
            assert result["success"] is False
            assert result["error"] == "Interrupted"

    def test_extract_returns_error_without_api_key(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TINYFISH_API_KEY", None)
            result = p.extract(["https://example.com"])
            assert len(result) == 1
            assert "TINYFISH_API_KEY" in result[0]["error"]

    def test_extract_returns_per_url_errors_on_network_failure(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-test"}):
            with patch("plugins.web.tinyfish.provider.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                mock_client.post.side_effect = Exception("network error")

                result = p.extract(["https://example.com"])
                assert len(result) == 1
                assert "TinyFish extract failed" in result[0]["error"]

    def test_extract_returns_per_url_errors_when_interrupted(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        p = TinyfishWebSearchProvider()
        with patch("tools.interrupt.is_interrupted", return_value=True):
            result = p.extract(["https://a.com", "https://b.com"])
            assert len(result) == 2
            assert result[0]["error"] == "Interrupted"
            assert result[1]["error"] == "Interrupted"

    def test_setup_schema(self):
        from plugins.web.tinyfish.provider import TinyfishWebSearchProvider
        schema = TinyfishWebSearchProvider().get_setup_schema()
        assert schema["name"] == "TinyFish"
        assert schema["badge"] == "paid"
        env_keys = [e["key"] for e in schema["env_vars"]]
        assert "TINYFISH_API_KEY" in env_keys