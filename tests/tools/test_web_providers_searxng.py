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




class TestSearXNGSearchProviderSearch:
    """Happy path and error handling for SearXNGWebSearchProvider.search().

    The provider requests ``format=html`` (a private instance answers 403 on
    ``format=json``) and parses the rendered result list. Each test mocks the
    HTTP reply with a representative SearXNG HTML page and asserts the
    normalized web-result contract.
    """

    # Three hits in source order — HTML output is already relevance-ordered.
    _SAMPLE_HTML = (
        '<html><body>'
        '<article class="result result-default">'
        '<h3><a href="https://a.example.com">Result <em>A</em></a></h3>'
        '<p class="content">Desc A</p>'
        '</article>'
        '<article class="result result-default">'
        '<h3><a href="https://b.example.com">Result B</a></h3>'
        '<p class="content">Desc B</p>'
        '</article>'
        '<article class="result result-default">'
        '<h3><a href="https://c.example.com">Result C</a></h3>'
        '<p class="content">Desc C</p>'
        '</article>'
        '</body></html>'
    )

    def _make_mock_response(self, text, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
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
        assert web[0]["title"] == "Result A"          # <em> stripped
        assert web[0]["url"] == "https://a.example.com"
        assert web[0]["description"] == "Desc A"
        assert web[0]["position"] == 1

    def test_html_tags_stripped_from_title_and_snippet(self, monkeypatch):
        """Inline markup must not leak into the normalized fields."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = (
            '<article class="result result-default">'
            '<h3><a href="https://x.example.com"><em>bold</em> and <strong>strong</strong></a></h3>'
            '<p class="content">plain <b>bold</b> text</p>'
            '</article>'
        )
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        web = result["data"]["web"]
        assert web[0]["title"] == "bold and strong"
        assert web[0]["description"] == "plain bold text"
        assert "  " not in web[0]["title"]
        assert "  " not in web[0]["description"]

    def test_html_entities_are_decoded(self, monkeypatch):
        """SearXNG renders title/snippet with `|safe`, so entity-encoded text
        must be decoded rather than passed through as &quot; / &#x27;."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = (
            '<article class="result result-default">'
            '<h3><a href="https://q.example.com">Quotes &amp; &#x27;apostrophes&#x27;</a></h3>'
            '<p class="content">He said &quot;hello&quot; &amp; left &mdash; done.</p>'
            '</article>'
        )
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        web = result["data"]["web"]
        assert web[0]["title"] == "Quotes & 'apostrophes'"
        assert web[0]["description"] == 'He said "hello" & left \u2014 done.'

    def test_engine_supplied_markup_escapes_survive_as_text(self, monkeypatch):
        """A literal &lt;b&gt; from the engine must not be stripped as markup."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = (
            '<article class="result result-default">'
            '<h3><a href="https://t.example.com">Escaped tag</a></h3>'
            '<p class="content">Use &lt;b&gt;bold&lt;/b&gt; in HTML.</p>'
            '</article>'
        )
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["data"]["web"][0]["description"] == "Use <b>bold</b> in HTML."

    def test_result_type_variants_all_parsed(self, monkeypatch):
        """``result-default``, ``result-images``, ``result-videos`` all match
        the tolerant ``result[^"]*`` class regex."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = (
            '<article class="result result-default"><h3><a href="https://d.example.com">D</a></h3><p class="content">d</p></article>'
            '<article class="result result-images"><h3><a href="https://i.example.com">I</a></h3><p class="content">i</p></article>'
            '<article class="result result-videos"><h3><a href="https://v.example.com">V</a></h3><p class="content">v</p></article>'
        )
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert [r["url"] for r in result["data"]["web"]] == [
            "https://d.example.com", "https://i.example.com", "https://v.example.com",
        ]

    def test_article_without_title_link_is_skipped(self, monkeypatch):
        """A container with no <h3><a> is skipped, not fatal."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = (
            '<article class="result result-default"><h3><a href="https://ok.example.com">OK</a></h3><p class="content">ok</p></article>'
            '<article class="result result-default"><p class="content">no title link</p></article>'
            '<article class="result result-default"><h3><a href="https://ok2.example.com">OK2</a></h3><p class="content">ok2</p></article>'
        )
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert [r["url"] for r in result["data"]["web"]] == ["https://ok.example.com", "https://ok2.example.com"]

    def test_request_asks_for_html_format(self, monkeypatch):
        """The wire request must use format=html — format=json is what a
        private instance rejects with 403."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)
        captured = {}

        def capture_get(url, **kwargs):
            captured.update(kwargs)
            return mock_resp

        with patch("httpx.get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("query", limit=5)

        assert captured["params"]["format"] == "html"
        assert captured["params"]["q"] == "query"
        assert "text/html" in captured["headers"]["Accept"]

    def test_limit_is_respected(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=2)

        assert result["success"] is True
        assert [r["url"] for r in result["data"]["web"]] == [
            "https://a.example.com", "https://b.example.com",
        ]

    def test_position_is_one_indexed(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert [r["position"] for r in result["data"]["web"]] == [1, 2, 3]

    def test_empty_results(self, monkeypatch):
        """A page with no result article returns an empty list, not an error."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response("<html><body>no results</body></html>")

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_article_without_snippet_is_kept(self, monkeypatch):
        """A hit with no <p class="content"> is returned with an empty description."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = '<article class="result result-default"><h3><a href="https://n.example.com">No Snippet</a></h3></article>'
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["description"] == ""

    def test_empty_element_placeholder_is_not_a_description(self, monkeypatch):
        """SearXNG fills a missing snippet with ``content empty_element`` and a
        UI placeholder; the placeholder must not surface as the description."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        html = (
            '<article class="result result-default">'
            '<h3><a href="https://e.example.com">No Desc</a></h3>'
            '<p class="content empty_element">This site did not provide any description.</p>'
            '</article>'
        )
        mock_resp = self._make_mock_response(html)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["description"] == ""

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
        monkeypatch.setattr("plugins.web.searxng.provider.provider_env", lambda name: "")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        result = SearXNGWebSearchProvider().search("query", limit=5)
        assert result["success"] is False
        assert "SEARXNG_URL" in result["error"]

    def test_trailing_slash_stripped_from_url(self, monkeypatch):
        """Base URL trailing slash should not produce a double-slash endpoint."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080/")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_HTML)

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
        """Exa (higher priority than searxng) should win in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.setenv("EXA_API_KEY", "exa_test_key")
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        assert web_tools._get_backend() == "exa"

    def test_auto_detect_picks_searxng_when_url_only_in_hermes_config(self, monkeypatch):
        """#34290 follow-up: a config-only SEARXNG_URL (absent from process env)
        must still drive auto-detect via the now config-aware ``_has_env``."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
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
        monkeypatch.delenv("EXA_API_KEY", raising=False)
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
