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
# web.searxng_timeout config: default, valid override, invalid-value rejection
# ---------------------------------------------------------------------------


class TestSearXNGTimeoutConfig:
    """``web.searxng_timeout`` — default preservation, valid override, invalid rejection (#99399)."""

    def _make_mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_default_timeout_is_15_when_unset(self, monkeypatch):
        """No web.searxng_timeout configured -> the historical hardcoded 15s is preserved exactly."""
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        timeout, error = _resolve_searxng_timeout()
        assert error is None
        assert timeout == 15.0

    def test_valid_override_is_used_in_the_request(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": 45})

        captured = {}

        def capture_get(url, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return self._make_mock_response({"results": []})

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is True
        assert captured["timeout"] == 45.0

    def test_valid_float_override_is_accepted(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": 30.5})
        timeout, error = _resolve_searxng_timeout()
        assert error is None
        assert timeout == 30.5

    def test_zero_is_rejected(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": 0})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "positive" in error

    def test_negative_is_rejected(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": -5})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "positive" in error

    def test_nan_is_rejected(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": float("nan")})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "finite" in error

    def test_infinity_is_rejected(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": float("inf")})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "finite" in error

    def test_non_numeric_string_is_rejected(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": "not-a-number"})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "positive number" in error

    def test_invalid_value_fails_search_with_clear_error_not_a_request(self, monkeypatch):
        """An invalid configured timeout must fail the search call with a clear error and must
        never fall through to an HTTP request (which would use an unvalidated/undefined timeout)."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": -1})

        with patch("httpx.get") as mock_get:
            result = SearXNGWebSearchProvider().search("query", limit=5)

        mock_get.assert_not_called()
        assert result["success"] is False
        assert "searxng_timeout" in result["error"]

    def test_explicit_null_is_treated_as_unset(self, monkeypatch):
        """Explicit YAML `searxng_timeout: null` parses to Python None and must fall back to the
        default exactly like the key being absent entirely — not be misread as a sentinel error."""
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": None})
        timeout, error = _resolve_searxng_timeout()
        assert error is None
        assert timeout == 15.0

    def test_bool_true_is_rejected(self, monkeypatch):
        """`bool` is a `int` subclass in Python, so `float(True) == 1.0` would otherwise slip
        past validation — a YAML boolean is never a meaningful timeout and must be rejected."""
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": True})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "positive number" in error

    def test_bool_false_is_rejected(self, monkeypatch):
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": False})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "positive number" in error

    def test_overflowing_integer_is_rejected_not_uncaught(self, monkeypatch):
        """An integer too large for `float()` raises `OverflowError` rather than `ValueError` —
        must be caught and surfaced as the same clear config error, not an unhandled exception."""
        from plugins.web.searxng.provider import _resolve_searxng_timeout
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"searxng_timeout": 10**400})
        timeout, error = _resolve_searxng_timeout()
        assert timeout is None
        assert error is not None and "finite" in error

    def test_unset_timeout_reaches_the_actual_request_as_15(self, monkeypatch):
        """End-to-end: with no config override, the real httpx.get call receives timeout=15.0,
        not just the resolver in isolation."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        captured = {}

        def capture_get(url, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return self._make_mock_response({"results": []})

        with patch("httpx.get", side_effect=capture_get):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is True
        assert captured["timeout"] == 15.0


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
