"""Tests for the TinyFish web search and fetch provider.

Covers:
- TinyFishSearchProvider / TinyFishExtractProvider is_configured() env var gating
- TinyFishSearchProvider.search() — happy path, HTTP error, rate limit, request error, bad JSON
- TinyFishExtractProvider.extract() — happy path, HTTP error, rate limit, request error, bad JSON
- Result normalization (title, url, description/content, position)
- Limit truncation
- _is_backend_available("tinyfish") integration
- _get_backend() recognizes "tinyfish" as a valid configured backend
- check_web_api_key() includes tinyfish in availability check
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# TinyFishSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestTinyFishSearchProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().is_configured() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().is_configured() is False

    def test_not_configured_when_key_whitespace(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "   ")
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().provider_name() == "tinyfish"

    def test_implements_web_search_provider(self):
        from tools.web_providers.base import WebSearchProvider
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert issubclass(TinyFishSearchProvider, WebSearchProvider)


class TestTinyFishSearchProviderSearch:
    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "A", "url": "https://a.example.com", "snippet": "desc A", "position": 1},
            {"title": "B", "url": "https://b.example.com", "snippet": "desc B", "position": 2},
            {"title": "C", "url": "https://c.example.com", "snippet": "desc C", "position": 3},
        ],
        "total_results": 3,
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = TinyFishSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0]["title"] == "A"
        assert web[0]["url"] == "https://a.example.com"
        assert web[0]["description"] == "desc A"
        assert web[0]["position"] == 1

    def test_sends_api_key_header(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["params"] = kwargs.get("params", {})
            return self._mock_resp({"results": [], "total_results": 0})

        with patch("httpx.get", side_effect=fake_get):
            TinyFishSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.search.tinyfish.ai"
        assert captured["headers"].get("X-API-Key") == "tf-key-123"
        assert captured["params"].get("query") == "q"

    def test_limit_is_respected_client_side(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = TinyFishSearchProvider().search("q", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2

    def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp({"results": [], "total_results": 0})):
            result = TinyFishSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_results_key_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp({})):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_rate_limit_429_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        m = MagicMock()
        m.status_code = 429

        with patch("httpx.get", return_value=m):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "rate limit" in result["error"].lower()

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        bad = MagicMock()
        bad.status_code = 500
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=bad)

        with patch("httpx.get", side_effect=err):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "500" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", side_effect=httpx.RequestError("boom")):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "boom" in result["error"] or "TinyFish" in result["error"]

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        result = TinyFishSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "TINYFISH_API_KEY" in result["error"]

    def test_bad_json_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = MagicMock()
        m.json.side_effect = ValueError("bad json")

        with patch("httpx.get", return_value=m):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "parse" in result["error"].lower()


# ---------------------------------------------------------------------------
# TinyFishExtractProvider unit tests
# ---------------------------------------------------------------------------


class TestTinyFishExtractProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert TinyFishExtractProvider().is_configured() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert TinyFishExtractProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert TinyFishExtractProvider().provider_name() == "tinyfish"

    def test_implements_web_extract_provider(self):
        from tools.web_providers.base import WebExtractProvider
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert issubclass(TinyFishExtractProvider, WebExtractProvider)


class TestTinyFishExtractProviderExtract:
    _SAMPLE_RESPONSE = {
        "results": [
            {
                "url": "https://example.com/1",
                "final_url": "https://example.com/1",
                "title": "Page 1",
                "text": "Content of page 1",
                "description": "A page",
                "language": "en",
                "latency_ms": 150,
            },
            {
                "url": "https://example.com/2",
                "final_url": "https://example.com/2",
                "title": "Page 2",
                "text": "Content of page 2",
                "description": "Another page",
                "language": "en",
                "latency_ms": 200,
            },
        ],
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = TinyFishExtractProvider().extract(
                ["https://example.com/1", "https://example.com/2"]
            )

        assert result["success"] is True
        data = result["data"]
        assert len(data) == 2
        assert data[0]["url"] == "https://example.com/1"
        assert data[0]["title"] == "Page 1"
        assert data[0]["content"] == "Content of page 1"
        assert data[0]["metadata"]["language"] == "en"

    def test_sends_api_key_header_and_json_body(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(["https://example.com"], format="markdown")

        assert captured["url"] == "https://api.fetch.tinyfish.ai"
        assert captured["headers"].get("X-API-Key") == "tf-key-123"
        assert captured["json"]["urls"] == ["https://example.com"]
        assert captured["json"]["format"] == "markdown"

    def test_empty_urls_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        result = TinyFishExtractProvider().extract([])
        assert result["success"] is True
        assert result["data"] == []

    def test_format_defaults_to_markdown(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(["https://example.com"])

        assert captured["json"]["format"] == "markdown"

    def test_invalid_format_falls_back_to_markdown(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(["https://example.com"], format="xml")

        assert captured["json"]["format"] == "markdown"

    def test_per_url_errors_reported(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        response = {
            "results": [
                {"url": "https://ok.example.com", "title": "OK", "text": "good"},
            ],
            "errors": [
                {"url": "https://fail.example.com", "error": "timeout"},
            ],
        }

        with patch("httpx.post", return_value=self._mock_resp(response)):
            result = TinyFishExtractProvider().extract(
                ["https://ok.example.com", "https://fail.example.com"]
            )

        assert result["success"] is True
        data = result["data"]
        assert len(data) == 2
        # Error entry added for failed URL
        error_entry = [d for d in data if d["url"] == "https://fail.example.com"][0]
        assert "error" in error_entry
        assert "timeout" in error_entry["error"]

    def test_rate_limit_429_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        m = MagicMock()
        m.status_code = 429

        with patch("httpx.post", return_value=m):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "rate limit" in result["error"].lower()

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        bad = MagicMock()
        bad.status_code = 500
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=bad)

        with patch("httpx.post", side_effect=err):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "500" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        with patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "boom" in result["error"] or "TinyFish" in result["error"]

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        result = TinyFishExtractProvider().extract(["https://example.com"])
        assert result["success"] is False
        assert "TINYFISH_API_KEY" in result["error"]

    def test_bad_json_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = MagicMock()
        m.json.side_effect = ValueError("bad json")

        with patch("httpx.post", return_value=m):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "parse" in result["error"].lower()

    def test_text_dict_converted_to_string(self, monkeypatch):
        """When text field is a dict (e.g. json format), it should be converted to string."""
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        response = {
            "results": [
                {"url": "https://example.com", "title": "JSON", "text": {"key": "value"}},
            ],
        }

        with patch("httpx.post", return_value=self._mock_resp(response)):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is True
        assert isinstance(result["data"][0]["content"], str)


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend / check_web_api_key
# ---------------------------------------------------------------------------


class TestTinyFishBackendWiring:
    def test_is_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("tinyfish") is True

    def test_is_backend_available_false_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("tinyfish") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "tinyfish"})
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        assert web_tools._get_backend() == "tinyfish"

    def test_auto_detect_picks_tinyfish_when_only_key_set(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "tinyfish"

    def test_tinyfish_does_not_override_paid_provider(self, monkeypatch):
        """Tavily (higher priority) should win in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly")
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "tavily"

    def test_check_web_api_key_true_when_tinyfish_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "tinyfish"})
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        assert web_tools.check_web_api_key() is True
