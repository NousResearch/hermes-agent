"""Tests for the Crawl4ai web extract provider.

Covers:
- Crawl4aiWebExtractProvider.is_available() env var gating (CRAWL4AI_URL + CRAWL4AI_API_TOKEN)
- Crawl4aiWebExtractProvider.extract() — happy path, HTTP error, request error, bad JSON
- Result normalization (title, url, content, metadata)
- _is_backend_available("crawl4ai") integration
- _get_backend() recognizes "crawl4ai" as a valid configured backend
- check_web_api_key() includes crawl4ai in availability check
- crawl4ai is extract-only: web_search returns a clear error
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


# ---------------------------------------------------------------------------
# Crawl4aiWebExtractProvider unit tests
# ---------------------------------------------------------------------------


class TestCrawl4aiProviderIsConfigured:
    def test_configured_when_both_vars_set(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().is_available() is True

    def test_not_configured_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().is_available() is False

    def test_not_configured_when_token_missing(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().is_available() is False

    def test_not_configured_when_both_missing(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().is_available() is False

    def test_not_configured_when_url_empty_string(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "   ")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().is_available() is False

    def test_not_configured_when_token_empty_string(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "   ")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().name == "crawl4ai"

    def test_display_name(self):
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().display_name == "Crawl4ai (self-hosted)"

    def test_supports_search_is_false(self):
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().supports_search() is False

    def test_supports_extract_is_true(self):
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert Crawl4aiWebExtractProvider().supports_extract() is True

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        assert issubclass(Crawl4aiWebExtractProvider, WebSearchProvider)

    def test_get_setup_schema(self):
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
        schema = Crawl4aiWebExtractProvider().get_setup_schema()
        assert schema["name"] == "Crawl4ai (self-hosted)"
        assert schema["badge"] == "self-hosted"
        assert "CRAWL4AI_URL" in str(schema["env_vars"])
        assert "CRAWL4AI_API_TOKEN" in str(schema["env_vars"])


class TestCrawl4aiProviderExtract:
    """Happy path and error handling for Crawl4aiWebExtractProvider.extract()."""

    _SAMPLE_MARKDOWN = "# Test Title\n\nThis is test content."

    def _make_mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_happy_path_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider

        mock_resp = self._make_mock_response(
            {"url": "https://example.com", "markdown": self._SAMPLE_MARKDOWN, "success": True}
        )

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            provider = Crawl4aiWebExtractProvider()
            result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com"
        assert result[0]["title"] == "Test Title"
        assert result[0]["content"] == self._SAMPLE_MARKDOWN
        assert result[0]["raw_content"] == self._SAMPLE_MARKDOWN
        assert result[0]["metadata"] == {}
        assert "error" not in result[0]

    @pytest.mark.asyncio
    async def test_handles_markdown_as_dict(self, monkeypatch):
        """Handle response where markdown is a dict with raw_markdown key."""
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider

        mock_resp = self._make_mock_response(
            {"url": "https://example.com", "markdown": {"raw_markdown": self._SAMPLE_MARKDOWN}, "success": True}
        )

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            provider = Crawl4aiWebExtractProvider()
            result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert result[0]["content"] == self._SAMPLE_MARKDOWN

    @pytest.mark.asyncio
    async def test_handles_markdown_dict_with_markdown_key(self, monkeypatch):
        """Handle response where markdown is a dict with markdown key."""
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider

        mock_resp = self._make_mock_response(
            {"url": "https://example.com", "markdown": {"markdown": self._SAMPLE_MARKDOWN}, "success": True}
        )

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            provider = Crawl4aiWebExtractProvider()
            result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert result[0]["content"] == self._SAMPLE_MARKDOWN

    @pytest.mark.asyncio
    async def test_multiple_urls(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai import provider as crawl4ai_provider

        mock_resp1 = self._make_mock_response(
            {"url": "https://a.example.com", "markdown": "# A\n\nContent A", "success": True}
        )
        mock_resp2 = self._make_mock_response(
            {"url": "https://b.example.com", "markdown": "# B\n\nContent B", "success": True}
        )

        with patch("httpx.AsyncClient.post", side_effect=[mock_resp1, mock_resp2]):
            with patch("plugins.web.crawl4ai.provider.is_safe_url", return_value=True):
                provider = crawl4ai_provider.Crawl4aiWebExtractProvider()
                result = await provider.extract(["https://a.example.com", "https://b.example.com"])

        assert len(result) == 2
        assert result[0]["url"] == "https://a.example.com"
        assert result[0]["title"] == "A"
        assert result[1]["url"] == "https://b.example.com"
        assert result[1]["title"] == "B"

    @pytest.mark.asyncio
    async def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai import provider as crawl4ai_provider

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        http_err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)

        with patch("httpx.AsyncClient.post", side_effect=http_err):
            with patch("plugins.web.crawl4ai.provider.is_safe_url", return_value=True):
                provider = crawl4ai_provider.Crawl4aiWebExtractProvider()
                result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert "error" in result[0]
        assert "500" in result[0]["error"]
        assert result[0]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai import provider as crawl4ai_provider

        with patch("httpx.AsyncClient.post", side_effect=httpx.RequestError("connection refused")):
            with patch("plugins.web.crawl4ai.provider.is_safe_url", return_value=True):
                provider = crawl4ai_provider.Crawl4aiWebExtractProvider()
                result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert "error" in result[0]
        assert "connection refused" in result[0]["error"].lower() or "failed" in result[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_config_returns_failure(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        from plugins.web.crawl4ai import provider as crawl4ai_provider

        with patch("plugins.web.crawl4ai.provider.is_safe_url", return_value=True):
            provider = crawl4ai_provider.Crawl4aiWebExtractProvider()
            result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert "error" in result[0]
        assert "not configured" in result[0]["error"].lower() or "url" in result[0]["error"].lower() or "token" in result[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_ssrf_blocked_url_returns_failure(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider

        # 10.0.0.1 is a private IP - should be blocked
        provider = Crawl4aiWebExtractProvider()
        result = await provider.extract(["http://10.0.0.1/secret"])

        assert len(result) == 1
        assert result[0]["url"] == "http://10.0.0.1/secret"
        assert "error" in result[0]
        assert "blocked" in result[0]["error"].lower() or "private" in result[0]["error"].lower()


# ---------------------------------------------------------------------------
# Integration: _is_backend_available recognizes "crawl4ai"
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_crawl4ai_available_when_both_vars_set(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("crawl4ai") is True

    def test_crawl4ai_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("crawl4ai") is False

    def test_crawl4ai_unavailable_when_token_missing(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("crawl4ai") is False

    def test_unknown_backend_still_false(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("unknownbackend") is False


# ---------------------------------------------------------------------------
# Integration: _get_backend() accepts "crawl4ai" as configured value
# ---------------------------------------------------------------------------


class TestGetBackendCrawl4ai:
    def test_configured_crawl4ai_returns_crawl4ai(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "crawl4ai"})
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        assert web_tools._get_backend() == "crawl4ai"

    def test_auto_detect_picks_crawl4ai_when_only_vars_set(self, monkeypatch):
        """When no backend is configured but CRAWL4AI vars are set, auto-detect returns it."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "crawl4ai"

    def test_crawl4ai_does_not_override_higher_priority_provider(self, monkeypatch):
        """Tavily (higher priority than crawl4ai) should win in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "tavily"

    def test_auto_detect_picks_crawl4ai_when_config_only(self, monkeypatch):
        """Config-only CRAWL4AI_URL/TOKEN (absent from process env) must still drive auto-detect."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY",
                    "CRAWL4AI_URL", "CRAWL4AI_API_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            hermes_config,
            "get_env_value",
            lambda key: "http://config-only:11235" if key == "CRAWL4AI_URL"
            else "config-token" if key == "CRAWL4AI_API_TOKEN"
            else None,
        )
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "crawl4ai"


# ---------------------------------------------------------------------------
# Integration: check_web_api_key includes crawl4ai
# ---------------------------------------------------------------------------


class TestCheckWebApiKey:
    def test_crawl4ai_satisfies_check_web_api_key(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "crawl4ai"})
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        assert web_tools.check_web_api_key() is True

    def test_crawl4ai_config_only_satisfies_check_web_api_key(self, monkeypatch):
        """Config-only CRAWL4AI_URL/TOKEN satisfies the credential check."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "crawl4ai"})
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        monkeypatch.setattr(
            hermes_config,
            "get_env_value",
            lambda key: "http://config-only:11235" if key == "CRAWL4AI_URL"
            else "config-token" if key == "CRAWL4AI_API_TOKEN"
            else None,
        )
        assert web_tools.check_web_api_key() is True

    def test_no_credentials_fails(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY",
                    "CRAWL4AI_URL", "CRAWL4AI_API_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools.check_web_api_key() is False


# ---------------------------------------------------------------------------
# crawl4ai is extract-only: web_search returns a clear error
# ---------------------------------------------------------------------------


class TestCrawl4aiSearchOnlyErrors:
    """When crawl4ai is the active backend, search must return clear errors."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_search_crawl4ai_returns_clear_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "crawl4ai"})
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = web_tools.web_search_tool("test query")
        result = json.loads(result_str)
        assert "error" in result
        assert "extract-only" in result["error"].lower() or "crawl4ai" in result["error"].lower()