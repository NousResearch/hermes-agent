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

    def test_does_not_set_search_lang(self):
        """Hermes must NOT pin ``search_lang`` — Brave's auto-detection gives
        better results for non-English queries. Regression guard for a bug
        where an earlier approach hardcoded ``search_lang: \"en\"``."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "k"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                from tools.web_tools import _brave_search
                _brave_search("recette de pain au miel", limit=3)
                params = mock_get.call_args.kwargs.get("params") or {}
                assert "search_lang" not in params

    def test_brave_api_url_override(self):
        """``BRAVE_API_URL`` env var redirects the request to a custom host
        (useful for proxies / self-hosted gateways). Trailing slashes are
        stripped so both ``https://proxy/`` and ``https://proxy`` work."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "k", "BRAVE_API_URL": "https://brave.proxy.internal/v1/"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                from tools.web_tools import _brave_search
                _brave_search("q")
                called_url = mock_get.call_args.args[0]
                assert called_url == "https://brave.proxy.internal/v1/web/search"


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

    def test_extra_snippets_merged_into_description(self):
        """Brave's ``extra_snippets`` hold additional context from the page.
        We merge the first two into the description so the caller sees
        richer information without having to know about the Brave-specific
        field."""
        from tools.web_tools import _normalize_brave_search_results
        raw = {"web": {"results": [{
            "title": "T", "url": "https://x", "description": "Main description.",
            "extra_snippets": ["First extra.", "Second extra.", "Third dropped."],
        }]}}
        result = _normalize_brave_search_results(raw)
        desc = result["data"]["web"][0]["description"]
        assert "Main description." in desc
        assert "First extra." in desc
        assert "Second extra." in desc
        # Only first two are merged
        assert "Third dropped." not in desc

    def test_extra_snippets_used_when_description_empty(self):
        """When Brave returns no main description, fall back to snippets only."""
        from tools.web_tools import _normalize_brave_search_results
        raw = {"web": {"results": [{
            "title": "T", "url": "https://x", "description": "",
            "extra_snippets": ["Only snippet."],
        }]}}
        result = _normalize_brave_search_results(raw)
        assert result["data"]["web"][0]["description"] == "Only snippet."

    def test_no_extra_snippets(self):
        """Absent ``extra_snippets`` → description unchanged (no trailing space)."""
        from tools.web_tools import _normalize_brave_search_results
        raw = {"web": {"results": [{
            "title": "T", "url": "https://x", "description": "Just main.",
        }]}}
        result = _normalize_brave_search_results(raw)
        assert result["data"]["web"][0]["description"] == "Just main."


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
