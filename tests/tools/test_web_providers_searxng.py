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

    def test_url_less_high_score_result_does_not_consume_limit(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        data = {
            "results": [
                {"title": "No URL", "content": "ignored", "score": 100},
                {"title": "Valid one", "url": "https://one.example.com", "content": "", "score": 0.9},
                {"title": "Valid two", "url": "https://two.example.com", "content": "", "score": 0.8},
            ]
        }
        mock_resp = self._make_mock_response(data)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=2)

        assert result["success"] is True
        titles = [item["title"] for item in result["data"]["web"]]
        assert titles == ["Valid one", "Valid two"]

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


# ---------------------------------------------------------------------------
# Integration: check_web_api_key includes searxng
# ---------------------------------------------------------------------------


class TestCheckWebApiKey:
    def test_searxng_satisfies_check_web_api_key(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
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


# ---------------------------------------------------------------------------
# SearXNG hardening: fallback attempts and HTML fallback
# ---------------------------------------------------------------------------


class TestSearXNGSearchProviderFallbacks:
    def _make_mock_response(self, json_data, status_code=200, text=""):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_general_search_uses_default_engine_pinning(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response({
            "results": [{"title": "Pinned", "url": "https://example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            return mock_resp

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("normal query", limit=5)

        assert result["success"] is True
        assert calls[0]["categories"] == "general"
        assert calls[0]["language"] == "en"
        assert calls[0]["engines"] == "bing,mojeek,presearch"

    def test_custom_general_engines_env_is_respected(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setenv("SEARXNG_GENERAL_ENGINES", "brave,duckduckgo")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        mock_resp = self._make_mock_response({
            "results": [{"title": "Custom", "url": "https://example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            return mock_resp

        with patch("httpx.get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("normal query", limit=5)

        assert calls[0]["engines"] == "brave,duckduckgo"

    def test_news_query_retries_general_when_news_category_http_errors(self, monkeypatch):
        import httpx
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        error_resp = MagicMock()
        error_resp.status_code = 400
        news_error = httpx.HTTPStatusError("bad category", request=MagicMock(), response=error_resp)
        general_hit = self._make_mock_response({
            "results": [{"title": "General after news error", "url": "https://news.example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            if len(calls) == 1:
                raise news_error
            return general_hit

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("latest ai news", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "General after news error"
        assert calls[0]["categories"] == "news"
        assert calls[1]["categories"] == "general"

    def test_news_query_retries_general_engines_when_news_empty(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        empty_news = self._make_mock_response({"results": []})
        general_hit = self._make_mock_response({
            "results": [{"title": "Fallback", "url": "https://news.example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            return empty_news if len(calls) == 1 else general_hit

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("latest ai news", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Fallback"
        assert calls[0]["categories"] == "news"
        assert calls[1]["categories"] == "general"
        assert calls[1]["engines"] == "bing,mojeek,presearch"

    def test_pinned_general_engine_http_error_retries_without_language(self, monkeypatch):
        import httpx
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        error_resp = MagicMock()
        error_resp.status_code = 400
        engine_error = httpx.HTTPStatusError("bad engines", request=MagicMock(), response=error_resp)
        no_language_hit = self._make_mock_response({
            "results": [{"title": "Recovered", "url": "https://example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            if len(calls) == 1:
                raise engine_error
            return no_language_hit

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("normal query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Recovered"
        assert calls[0]["engines"] == "bing,mojeek,presearch"
        assert "language" not in calls[1]

    def test_language_pinned_empty_retries_without_language(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        empty = self._make_mock_response({"results": []})
        no_language_hit = self._make_mock_response({
            "results": [{"title": "No lang", "url": "https://example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            return no_language_hit if len(calls) == 2 else empty

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("normal query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "No lang"
        assert "language" in calls[0]
        assert "language" not in calls[1]

    def test_pinned_engines_empty_retries_default_engines(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        empty = self._make_mock_response({"results": []})
        default_engine_hit = self._make_mock_response({
            "results": [{"title": "Default engines", "url": "https://example.com", "content": "ok", "score": 1}]
        })
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs["params"])
            return default_engine_hit if len(calls) == 3 else empty

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("normal query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Default engines"
        assert "engines" in calls[0]
        assert "engines" in calls[1]
        assert "engines" not in calls[2]

    def test_html_fallback_when_json_parse_fails(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        bad_json_resp = self._make_mock_response({})
        bad_json_resp.json.side_effect = ValueError("not json")
        html = """
        <html><body>
          <article class="result">
            <h3><a href="https://html-parse.example.com">HTML After Bad JSON</a></h3>
            <p class="content">Recovered from HTML</p>
          </article>
        </body></html>
        """
        html_resp = self._make_mock_response({}, text=html)
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs)
            return html_resp if kwargs.get("headers", {}).get("Accept") == "text/html" else bad_json_resp

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("parse fallback", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "HTML After Bad JSON"
        assert calls[-1]["headers"]["Accept"] == "text/html"

    def test_url_less_json_results_continue_to_html_fallback(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        url_less_json = self._make_mock_response({
            "results": [{"title": "No URL", "content": "not usable", "score": 10}]
        })
        html = """
        <html><body>
          <article class="result">
            <h3><a href="https://usable.example.com">Usable Fallback</a></h3>
            <p class="content">usable html snippet</p>
          </article>
        </body></html>
        """
        html_resp = self._make_mock_response({}, text=html)
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs)
            return html_resp if kwargs.get("headers", {}).get("Accept") == "text/html" else url_less_json

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("url less", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Usable Fallback"
        assert calls[-1]["headers"]["Accept"] == "text/html"

    def test_html_fallback_when_json_attempts_return_empty(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        empty = self._make_mock_response({"results": []})
        html = """
        <html><body>
          <article class="result">
            <h3><a href="https://html.example.com">HTML Result</a></h3>
            <p class="content">HTML snippet</p>
          </article>
        </body></html>
        """
        html_resp = self._make_mock_response({}, text=html)
        calls = []

        def capture_get(url, **kwargs):
            calls.append(kwargs)
            return html_resp if kwargs.get("headers", {}).get("Accept") == "text/html" else empty

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "HTML Result",
                "url": "https://html.example.com",
                "description": "HTML snippet",
                "position": 1,
            }
        ]
        assert calls[-1]["headers"]["Accept"] == "text/html"
