"""Tests for the SearXNG web search provider.

Covers:
- SearXNGWebSearchProvider.is_available() env var gating
- SearXNGWebSearchProvider.search() — happy path, HTTP error, request error, bad JSON
- Result normalization (title, url, description, position)
- Score-based sorting and limit truncation
- _is_backend_available("searxng") integration
- _get_backend() recognizes "searxng" as a valid configured backend
- check_web_api_key() includes searxng in availability check
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


# ---------------------------------------------------------------------------
# SearXNGWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestSearXNGSearchProviderIsConfigured:
    def test_configured_when_url_set(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert SearXNGWebSearchProvider().is_available() is True


    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert issubclass(SearXNGWebSearchProvider, WebSearchProvider)


class TestSearXNGSearchProviderSearch:
    """Happy path and error handling for SearXNGWebSearchProvider.search()."""

    _SAMPLE_HTML = """
<article class="result result-general category-general">
<h3><a href="https://a.example.com" rel="noreferrer">Result A</a></h3>
<p class="content">Desc A</p>
</article>
<article class="result result-general category-general">
<h3><a href="https://b.example.com" rel="noreferrer">Result <span class="highlight">B</span></a></h3>
<p class="content">Desc B</p>
</article>
<article class="result result-general category-general">
<h3><a href="https://c.example.com" rel="noreferrer">Result C</a></h3>
<p class="content">Desc C</p>
</article>
"""

    def _make_mock_response(self, html_text, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = html_text
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_happy_path_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0]["title"] == "Result A"
        assert web[0]["url"] == "https://a.example.com"
        assert web[0]["description"] == "Desc A"
        assert web[0]["position"] == 1

    def test_results_in_document_order_and_limit_applied(self, monkeypatch):
        """Results follow HTML document order; limit truncates the tail."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=2)

        assert result["success"] is True
        assert [r["title"] for r in result["data"]["web"]] == ["Result A", "Result B"]
        assert result["data"]["web"][1]["position"] == 2

    def test_highlight_span_stripped_from_title(self, monkeypatch):
        """Nested <span class=\"highlight\"> must not leak into the title."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["data"]["web"][1]["title"] == "Result B"

    def test_no_results_page_returns_empty_web(self, monkeypatch):
        """A \"No results were found\" page is a valid empty search, not an error."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        no_results_html = (
            '<div class="dialog-error-block" role="alert">'
            "<p><strong>Sorry!</strong></p>"
            "<p>No results were found. You can try to:</p></div>"
        )
        mock_resp = self._make_mock_response(no_results_html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_http_error_returns_failure(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response("<html></html>", status_code=403)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is False
        assert "403" in result["error"]


    def test_trailing_slash_stripped_from_url(self, monkeypatch):
        """Base URL trailing slash should not produce double-slash in endpoint."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080/")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response("<html></html>")

        calls = []
        def capture_get(url, **kwargs):
            calls.append(url)
            return mock_resp

        with patch("httpx.get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("query", limit=5)

        assert calls[0] == "http://localhost:8080/search", f"Got: {calls[0]}"


# ---------------------------------------------------------------------------
# Integration: _is_backend_available recognizes "searxng"
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_searxng_available_when_url_set(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("searxng") is True


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

    def test_auto_detect_picks_searxng_when_url_only_in_hermes_config(self, monkeypatch):
        """#34290 follow-up: a config-only SEARXNG_URL (absent from process env)
        must still drive auto-detect via the now config-aware ``_has_env``."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(
            hermes_config,
            "get_env_value",
            lambda key: "http://config-only:8080" if key == "SEARXNG_URL" else None,
        )
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        assert web_tools._get_backend() == "searxng"


# ---------------------------------------------------------------------------
# Integration: check_web_api_key includes searxng
# ---------------------------------------------------------------------------


class TestCheckWebApiKey:
    def test_searxng_satisfies_check_web_api_key(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert web_tools.check_web_api_key() is True

    def test_searxng_config_only_satisfies_check_web_api_key(self, monkeypatch):
        """#34290 follow-up: config-only SEARXNG_URL satisfies the credential check."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(
            hermes_config,
            "get_env_value",
            lambda key: "http://config-only:8080" if key == "SEARXNG_URL" else None,
        )
        assert web_tools.check_web_api_key() is True

    def test_no_credentials_fails(self, monkeypatch):
        from tools import web_tools
        from agent import web_search_registry
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        # Disable the keyless free tier — with it on, zero credentials still
        # resolves (Parallel/Exa anonymous MCP; see test_web_keyless_fallback.py).
        monkeypatch.setattr(web_search_registry, "_keyless_tier_enabled", lambda: False)
        assert web_tools.check_web_api_key() is False


# ---------------------------------------------------------------------------
# searxng-only: web_extract returns a clear error
# ---------------------------------------------------------------------------


class TestSearXNGOnlyExtractCrawlErrors:
    """When searxng is the active backend, extract/crawl must return clear errors."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_searxng_returns_clear_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower() or "SearXNG" in result["error"]
