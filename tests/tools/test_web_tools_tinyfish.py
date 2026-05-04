"""Tests for TinyFish web backend integration.

Coverage:
  _get_tinyfish_client() — API key handling, base URL override, singleton caching.
  _tinyfish_search() — search response normalization.
  _tinyfish_extract() — extract response normalization, error entries.
  web_search_tool / web_extract_tool — TinyFish dispatch paths.
"""

import json
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock


# ─── _get_tinyfish_client ───────────────────────────────────────────────────

class TestTinyFishClient:
    """Test suite for TinyFish client initialization."""

    def setup_method(self):
        import tools.web_tools
        tools.web_tools._tinyfish_client = None
        os.environ.pop("TINYFISH_API_KEY", None)
        os.environ.pop("TINYFISH_API_URL", None)

    def teardown_method(self):
        import tools.web_tools
        tools.web_tools._tinyfish_client = None
        os.environ.pop("TINYFISH_API_KEY", None)
        os.environ.pop("TINYFISH_API_URL", None)

    def test_raises_without_api_key(self):
        """No TINYFISH_API_KEY → ValueError with guidance."""
        with patch("tools.web_tools.TinyFish", create=True):
            from tools.web_tools import _get_tinyfish_client
            with pytest.raises(ValueError, match="TINYFISH_API_KEY"):
                _get_tinyfish_client()

    def test_creates_client_with_api_key(self):
        """TINYFISH_API_KEY set → client created."""
        mock_cls = MagicMock()
        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-test-key"}):
            with patch("tinyfish.TinyFish", mock_cls):
                from tools.web_tools import _get_tinyfish_client
                client = _get_tinyfish_client()
                mock_cls.assert_called_once_with(api_key="tf-test-key")
                assert client is mock_cls.return_value

    def test_base_url_override(self):
        """TINYFISH_API_URL overrides default base URL."""
        mock_cls = MagicMock()
        with patch.dict(os.environ, {
            "TINYFISH_API_KEY": "tf-test-key",
            "TINYFISH_API_URL": "https://staging.tinyfish.ai/",
        }):
            with patch("tinyfish.TinyFish", mock_cls):
                from tools.web_tools import _get_tinyfish_client
                _get_tinyfish_client()
                mock_cls.assert_called_once_with(
                    api_key="tf-test-key",
                    base_url="https://staging.tinyfish.ai",
                )

    def test_singleton_caching(self):
        """Repeated calls return the same client instance."""
        mock_cls = MagicMock()
        with patch.dict(os.environ, {"TINYFISH_API_KEY": "tf-test-key"}):
            with patch("tinyfish.TinyFish", mock_cls):
                from tools.web_tools import _get_tinyfish_client
                first = _get_tinyfish_client()
                second = _get_tinyfish_client()
                assert first is second
                assert mock_cls.call_count == 1


# ─── _tinyfish_search ───────────────────────────────────────────────────────

class TestTinyFishSearch:
    """Test search result normalization."""

    def setup_method(self):
        import tools.web_tools
        tools.web_tools._tinyfish_client = None

    def teardown_method(self):
        import tools.web_tools
        tools.web_tools._tinyfish_client = None

    def test_basic_normalization(self):
        mock_result_1 = MagicMock(title="Python Docs", url="https://docs.python.org", snippet="Official docs", position=1)
        mock_result_2 = MagicMock(title="Tutorial", url="https://example.com", snippet="A tutorial", position=2)
        mock_response = MagicMock(results=[mock_result_1, mock_result_2])
        mock_client = MagicMock()
        mock_client.search.query.return_value = mock_response

        with patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _tinyfish_search
            result = _tinyfish_search("python docs", limit=10)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Python Docs"
        assert web[0]["url"] == "https://docs.python.org"
        assert web[0]["description"] == "Official docs"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_empty_results(self):
        mock_response = MagicMock(results=[])
        mock_client = MagicMock()
        mock_client.search.query.return_value = mock_response

        with patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _tinyfish_search
            result = _tinyfish_search("nothing")

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_respects_limit(self):
        mock_results = [MagicMock(title=f"R{i}", url=f"https://r{i}.com", snippet=f"s{i}", position=i+1) for i in range(10)]
        mock_response = MagicMock(results=mock_results)
        mock_client = MagicMock()
        mock_client.search.query.return_value = mock_response

        with patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _tinyfish_search
            result = _tinyfish_search("test", limit=3)

        assert len(result["data"]["web"]) == 3

    def test_interrupted_returns_error(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_tools import _tinyfish_search
            result = _tinyfish_search("test")
            assert result["success"] is False


# ─── _tinyfish_extract ──────────────────────────────────────────────────────

class TestTinyFishExtract:
    """Test extract response normalization."""

    def setup_method(self):
        import tools.web_tools
        tools.web_tools._tinyfish_client = None

    def teardown_method(self):
        import tools.web_tools
        tools.web_tools._tinyfish_client = None

    def test_basic_extraction(self):
        mock_result = MagicMock(text="Page content here", final_url="https://example.com", url="https://example.com", title="Example")
        mock_response = MagicMock(results=[mock_result], errors=[])
        mock_client = MagicMock()
        mock_client.fetch.get_contents.return_value = mock_response

        with patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _tinyfish_extract
            results = _tinyfish_extract(["https://example.com"])

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert results[0]["title"] == "Example"
        assert results[0]["content"] == "Page content here"
        assert results[0]["raw_content"] == "Page content here"
        assert results[0]["metadata"]["sourceURL"] == "https://example.com"

    def test_error_entries_included(self):
        mock_error = MagicMock(url="https://fail.com", error="timeout")
        mock_response = MagicMock(results=[], errors=[mock_error])
        mock_client = MagicMock()
        mock_client.fetch.get_contents.return_value = mock_response

        with patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _tinyfish_extract
            results = _tinyfish_extract(["https://fail.com"])

        assert len(results) == 1
        assert results[0]["url"] == "https://fail.com"
        assert results[0]["error"] == "timeout"
        assert results[0]["content"] == ""

    def test_mixed_success_and_error(self):
        mock_ok = MagicMock(text="Good content", final_url="https://ok.com", url="https://ok.com", title="OK")
        mock_fail = MagicMock(url="https://bad.com", error="403 Forbidden")
        mock_response = MagicMock(results=[mock_ok], errors=[mock_fail])
        mock_client = MagicMock()
        mock_client.fetch.get_contents.return_value = mock_response

        with patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _tinyfish_extract
            results = _tinyfish_extract(["https://ok.com", "https://bad.com"])

        assert len(results) == 2
        assert results[0]["content"] == "Good content"
        assert results[1]["error"] == "403 Forbidden"

    def test_interrupted_returns_error_per_url(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_tools import _tinyfish_extract
            results = _tinyfish_extract(["https://a.com", "https://b.com"])
            assert len(results) == 2
            assert all(r["error"] == "Interrupted" for r in results)


# ─── web_search_tool (TinyFish dispatch) ────────────────────────────────────

class TestWebSearchTinyFish:
    """Test web_search_tool dispatch to TinyFish."""

    def test_search_dispatches_to_tinyfish(self):
        mock_result = MagicMock(title="Result", url="https://r.com", snippet="desc", position=1)
        mock_response = MagicMock(results=[mock_result])
        mock_client = MagicMock()
        mock_client.search.query.return_value = mock_response

        with patch("tools.web_tools._get_backend", return_value="tinyfish"), \
             patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("test query", limit=5))
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"


# ─── web_extract_tool (TinyFish dispatch) ───────────────────────────────────

class TestWebExtractTinyFish:
    """Test web_extract_tool dispatch to TinyFish."""

    def test_extract_dispatches_to_tinyfish(self):
        mock_result = MagicMock(text="Extracted content", final_url="https://example.com", url="https://example.com", title="Page")
        mock_response = MagicMock(results=[mock_result], errors=[])
        mock_client = MagicMock()
        mock_client.fetch.get_contents.return_value = mock_response

        with patch("tools.web_tools._get_backend", return_value="tinyfish"), \
             patch("tools.web_tools._get_tinyfish_client", return_value=mock_client), \
             patch("tools.web_tools.process_content_with_llm", return_value=None), \
             patch("tools.web_tools.is_safe_url", return_value=True):
            from tools.web_tools import web_extract_tool
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["url"] == "https://example.com"
