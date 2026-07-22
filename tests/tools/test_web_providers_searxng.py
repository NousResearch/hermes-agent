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

    def test_not_configured_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert SearXNGWebSearchProvider().is_available() is False

    def test_not_configured_when_url_empty_string(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "   ")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert SearXNGWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert SearXNGWebSearchProvider().name == "searxng"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert issubclass(SearXNGWebSearchProvider, WebSearchProvider)


class TestSearXNGSearchProviderSearch:
    """Happy path and error handling for SearXNGWebSearchProvider.search()."""

    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "Result A", "url": "https://a.example.com", "content": "Desc A", "score": 0.9},
            {"title": "Result B", "url": "https://b.example.com", "content": "Desc B", "score": 0.7},
            {"title": "Result C", "url": "https://c.example.com", "content": "Desc C", "score": 0.5},
        ]
    }

    def _make_mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_happy_path_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0]["title"] == "Result A"
        assert web[0]["url"] == "https://a.example.com"
        assert web[0]["description"] == "Desc A"
        assert web[0]["position"] == 1

    def test_results_sorted_by_score_descending(self, monkeypatch):
        """Results should be sorted by score before limit is applied."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        unordered = {
            "results": [
                {"title": "Low",  "url": "https://low.example.com",  "content": "", "score": 0.1},
                {"title": "High", "url": "https://high.example.com", "content": "", "score": 0.99},
                {"title": "Mid",  "url": "https://mid.example.com",  "content": "", "score": 0.5},
            ]
        }
        mock_resp = self._make_mock_response(unordered)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "High"
        assert result["data"]["web"][1]["title"] == "Mid"
        assert result["data"]["web"][2]["title"] == "Low"

    def test_limit_is_respected(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2

    def test_position_is_one_indexed(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        positions = [r["position"] for r in result["data"]["web"]]
        assert positions == [1, 2, 3]

    def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response({"results": []})

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_score_falls_back_to_zero(self, monkeypatch):
        """Results without a score field should sort to the bottom."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        data = {
            "results": [
                {"title": "No score", "url": "https://noscore.example.com", "content": ""},
                {"title": "Has score", "url": "https://scored.example.com", "content": "", "score": 0.8},
            ]
        }
        mock_resp = self._make_mock_response(data)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is True
        # Has score should sort first (0.8 > 0)
        assert result["data"]["web"][0]["title"] == "Has score"

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        http_err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)

        with patch("httpx.get", side_effect=http_err):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is False
        assert "500" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is False
        assert "localhost:8080" in result["error"] or "connection" in result["error"].lower()

    def test_missing_url_returns_failure(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        result = SearXNGWebSearchProvider().search("query", limit=5)
        assert result["success"] is False
        assert "SEARXNG_URL" in result["error"]

    def test_trailing_slash_stripped_from_url(self, monkeypatch):
        """Base URL trailing slash should not produce double-slash in endpoint."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080/")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response({"results": []})

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

    def test_searxng_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("searxng") is False

    def test_unknown_backend_still_false(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("unknownbackend") is False


# ---------------------------------------------------------------------------
# Config contract: web.searxng makes searxng selectable without env vars
# ---------------------------------------------------------------------------


class TestIsBackendAvailableConfigContract:
    """_is_backend_available must detect searxng via web.searxng config too."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_config_only_url_makes_searxng_available(self, monkeypatch):
        """web.searxng.url in config.yaml alone (no SEARXNG_URL env) works."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        # Mock load_config so _searxng_config() finds the URL
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(
            searxng_provider,
            "_searxng_config",
            lambda: {"url": "http://config-only:8080"},
        )
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("searxng") is True

    def test_config_only_url_with_registered_provider_false(self, monkeypatch):
        """When web.searxng.url is empty, _is_backend_available returns False."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(
            searxng_provider,
            "_searxng_config",
            lambda: {"url": ""},
        )
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("searxng") is False


# ---------------------------------------------------------------------------
# Config contract: web.searxng.method/params/headers in search()
# ---------------------------------------------------------------------------


class TestSearXNGConfigSearchBehavior:
    """Verify that config knobs (method, params, headers) affect search()."""

    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "A", "url": "https://a.example.com", "content": "Desc", "score": 0.5},
        ]
    }

    def _make_mock_response(self, json_data, status_code=200):
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_post_method_sends_data_not_params(self, monkeypatch):
        """POST sends params as data=, not params=."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(
            searxng_provider,
            "_searxng_config",
            lambda: {"url": "http://localhost:8080", "method": "post"},
        )
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        calls = []

        def capture_post(url, **kwargs):
            calls.append((url, kwargs))
            return mock_resp

        import httpx

        with patch.object(httpx, "post", side_effect=capture_post):
            SearXNGWebSearchProvider().search("hello", limit=5)

        assert len(calls) == 1
        url, kwargs = calls[0]
        assert url == "http://localhost:8080/search"
        # POST sends q/format/pageno as data=
        assert kwargs.get("data") == {"q": "hello", "format": "json", "pageno": 1}
        # params= should NOT be set when method=post
        assert "params" not in kwargs

    def test_extra_params_merged(self, monkeypatch):
        """Extra params from web.searxng.params are merged with core params."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(
            searxng_provider,
            "_searxng_config",
            lambda: {"params": {"categories": "general", "language": "en-US", "safesearch": 0}},
        )
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        captured = {}

        def capture_get(url, **kwargs):
            captured.update(kwargs)
            return mock_resp

        import httpx

        with patch.object(httpx, "get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("query", limit=5)

        params = captured.get("params", {})
        assert params["q"] == "query"
        assert params["format"] == "json"
        assert params["categories"] == "general"
        assert params["language"] == "en-US"
        assert params["safesearch"] == 0

    def test_extra_params_cannot_override_core_params(self, monkeypatch):
        """User params cannot override q/format/pageno."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(
            searxng_provider,
            "_searxng_config",
            lambda: {"params": {"q": "evil-injection", "format": "xml"}},
        )
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        captured = {}

        def capture_get(url, **kwargs):
            captured.update(kwargs)
            return mock_resp

        import httpx

        with patch.object(httpx, "get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("real-query", limit=5)

        params = captured.get("params", {})
        assert params["q"] == "real-query"
        assert params["format"] == "json"

    def test_extra_headers_merged(self, monkeypatch):
        """Extra headers from web.searxng.headers are merged with defaults."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(
            searxng_provider,
            "_searxng_config",
            lambda: {"headers": {"Accept-Language": "en-US,en;q=0.9", "X-Custom": "test"}},
        )
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        captured = {}

        def capture_get(url, **kwargs):
            captured.update(kwargs)
            return mock_resp

        import httpx

        with patch.object(httpx, "get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("query", limit=5)

        headers = captured.get("headers", {})
        assert headers["Accept"] == "application/json"
        assert headers["Accept-Language"] == "en-US,en;q=0.9"
        assert headers["X-Custom"] == "test"

    def test_backwards_compat_no_config_same_get_behavior(self, monkeypatch):
        """Without web.searxng config, method defaults to GET with no extra params/headers."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        import plugins.web.searxng.provider as searxng_provider

        monkeypatch.setattr(searxng_provider, "_searxng_config", lambda: {})
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        captured = {}

        def capture_get(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return mock_resp

        import httpx

        with patch.object(httpx, "get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("query", limit=5)

        assert captured["url"] == "http://localhost:8080/search"
        params = captured.get("params", {})
        assert params == {"q": "query", "format": "json", "pageno": 1}
        headers = captured.get("headers", {})
        assert headers == {"Accept": "application/json"}


# ---------------------------------------------------------------------------
# Integration: _get_backend() accepts "searxng" as configured value
# ---------------------------------------------------------------------------


class TestGetBackendSearXNG:
    def test_configured_searxng_returns_searxng(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert web_tools._get_backend() == "searxng"

    def test_auto_detect_picks_searxng_when_only_url_set(self, monkeypatch):
        """When no backend is configured but SEARXNG_URL is set, auto-detect returns it."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        # Suppress tool gateway
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
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
