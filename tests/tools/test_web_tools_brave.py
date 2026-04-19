"""Tests for Brave Search web backend integration.

Coverage:
  _brave_search() — API key handling, header auth, query params, error propagation.
  _normalize_brave_search_results() — Brave → standard search response mapping.
  _is_backend_available() / check_web_api_key() — backend detection.
  web_search_tool — Brave dispatch.
  web_extract_tool — Firecrawl fallback when backend=brave, clear error otherwise.
"""

import json
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock


# ─── _brave_search ───────────────────────────────────────────────────────────

class TestBraveSearchRequest:
    """Test suite for the _brave_search helper."""

    def test_raises_without_api_key(self):
        """No BRAVE_API_KEY → ValueError with guidance."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_API_KEY", None)
            from tools.web_tools import _brave_search
            with pytest.raises(ValueError, match="BRAVE_API_KEY"):
                _brave_search("test")

    def test_sends_subscription_token_header(self):
        """API key is sent via X-Subscription-Token header (not JSON body)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "brave-test-key"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                from tools.web_tools import _brave_search
                _brave_search("hello world", limit=3)

                mock_get.assert_called_once()
                call = mock_get.call_args
                headers = call.kwargs.get("headers") or {}
                params = call.kwargs.get("params") or {}
                assert headers.get("X-Subscription-Token") == "brave-test-key"
                assert headers.get("Accept") == "application/json"
                assert params["q"] == "hello world"
                assert params["count"] == 3
                assert "api.search.brave.com/res/v1/web/search" in call.args[0]

    def test_clamps_limit_to_brave_max(self):
        """Brave caps count at 20 — our code must clamp before sending."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "k"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                from tools.web_tools import _brave_search
                _brave_search("q", limit=999)
                assert mock_get.call_args.kwargs["params"]["count"] == 20

    def test_clamps_limit_to_at_least_one(self):
        """Zero/negative limit should clamp to 1 (Brave rejects count=0)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "k"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                from tools.web_tools import _brave_search
                _brave_search("q", limit=0)
                assert mock_get.call_args.kwargs["params"]["count"] == 1

    def test_raises_on_http_error(self):
        """Non-2xx responses propagate as httpx.HTTPStatusError."""
        import httpx as _httpx
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "429 Too Many Requests", request=MagicMock(), response=mock_response
        )

        with patch.dict(os.environ, {"BRAVE_API_KEY": "k"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response):
                from tools.web_tools import _brave_search
                with pytest.raises(_httpx.HTTPStatusError):
                    _brave_search("q")


# ─── _normalize_brave_search_results ─────────────────────────────────────────

class TestNormalizeBraveSearchResults:
    """Test Brave response → standard web search format."""

    def test_basic_normalization(self):
        from tools.web_tools import _normalize_brave_search_results
        raw = {
            "web": {
                "results": [
                    {"title": "Python Docs", "url": "https://docs.python.org", "description": "Official docs"},
                    {"title": "Tutorial", "url": "https://example.com", "description": "A tutorial"},
                ]
            }
        }
        result = _normalize_brave_search_results(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Python Docs"
        assert web[0]["url"] == "https://docs.python.org"
        assert web[0]["description"] == "Official docs"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_empty_results(self):
        from tools.web_tools import _normalize_brave_search_results
        result = _normalize_brave_search_results({"web": {"results": []}})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_web_key(self):
        """Brave may omit the ``web`` key when no web results are returned."""
        from tools.web_tools import _normalize_brave_search_results
        result = _normalize_brave_search_results({})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_web_is_null(self):
        """Defensive: Brave returns ``web: null`` in some edge cases."""
        from tools.web_tools import _normalize_brave_search_results
        result = _normalize_brave_search_results({"web": None})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_fields(self):
        from tools.web_tools import _normalize_brave_search_results
        result = _normalize_brave_search_results({"web": {"results": [{}]}})
        web = result["data"]["web"]
        assert web[0]["title"] == ""
        assert web[0]["url"] == ""
        assert web[0]["description"] == ""
        assert web[0]["position"] == 1


# ─── Backend detection ───────────────────────────────────────────────────────

class TestBraveBackendDetection:
    """Brave recognised by _is_backend_available / _get_backend / check_web_api_key."""

    def test_is_backend_available_brave(self):
        from tools.web_tools import _is_backend_available
        with patch.dict(os.environ, {"BRAVE_API_KEY": "k"}):
            assert _is_backend_available("brave") is True
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_API_KEY", None)
            assert _is_backend_available("brave") is False

    def test_get_backend_honours_configured_brave(self):
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "brave"}):
            assert _get_backend() == "brave"


# ─── web_search_tool (Brave dispatch) ────────────────────────────────────────

class TestWebSearchBrave:
    """Test web_search_tool dispatch to Brave."""

    def test_search_dispatches_to_brave(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {"results": [{"title": "Result", "url": "https://r.com", "description": "desc"}]}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="brave"), \
             patch.dict(os.environ, {"BRAVE_API_KEY": "k"}), \
             patch("tools.web_tools.httpx.get", return_value=mock_response), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("test query", limit=3))
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"
            assert result["data"]["web"][0]["position"] == 1


# ─── web_extract_tool (Brave fallback behaviour) ─────────────────────────────

class TestWebExtractBraveFallback:
    """When backend is Brave, extract must fall back to Firecrawl or error."""

    def test_extract_errors_when_brave_and_no_firecrawl(self):
        """No Firecrawl key → clear tool_error, not a cryptic crash."""
        with patch("tools.web_tools._get_backend", return_value="brave"), \
             patch("tools.web_tools.check_firecrawl_api_key", return_value=False), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None):
            from tools.web_tools import web_extract_tool
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))
            assert result.get("success") is False
            assert "Brave backend supports web_search only" in result.get("error", "")
            assert "FIRECRAWL_API_KEY" in result.get("error", "")

    def test_extract_falls_back_to_firecrawl_when_available(self):
        """Brave + Firecrawl → extract routes through Firecrawl seamlessly."""
        fake_firecrawl_client = MagicMock()
        fake_scrape_result = MagicMock()
        fake_scrape_result.markdown = "Extracted markdown"
        fake_scrape_result.html = "<p>Extracted</p>"
        fake_scrape_result.metadata = {"title": "Example", "sourceURL": "https://example.com"}
        fake_firecrawl_client.scrape.return_value = fake_scrape_result

        with patch("tools.web_tools._get_backend", return_value="brave"), \
             patch("tools.web_tools.check_firecrawl_api_key", return_value=True), \
             patch("tools.web_tools._get_firecrawl_client", return_value=fake_firecrawl_client), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.web_tools.process_content_with_llm", return_value=None):
            from tools.web_tools import web_extract_tool
            raw = asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            )
            # Firecrawl path returns the Firecrawl-shape envelope; just
            # assert it didn't short-circuit to the Brave error and that
            # Firecrawl's scrape was actually invoked.
            assert "Brave backend supports web_search only" not in raw
            fake_firecrawl_client.scrape.assert_called()
