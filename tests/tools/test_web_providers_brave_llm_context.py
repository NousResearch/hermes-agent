"""Tests for the Brave Search LLM Context web search provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


class TestBraveLLMContextProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        assert BraveLLMContextWebSearchProvider().is_available() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        assert BraveLLMContextWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        assert BraveLLMContextWebSearchProvider().name == "brave-llm-context"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        assert issubclass(BraveLLMContextWebSearchProvider, WebSearchProvider)


class TestBraveLLMContextProviderSearch:
    _SAMPLE_RESPONSE = {
        "grounding": {
            "generic": [
                {
                    "title": "A",
                    "url": "https://a.example.com",
                    "snippets": [" first snippet ", "second snippet"],
                },
                {
                    "title": "B",
                    "url": "https://b.example.com",
                    "snippets": ["desc B"],
                },
            ],
            "poi": {
                "title": "Place",
                "url": "https://poi.example.com",
                "snippets": ["poi desc"],
            },
            "map": [
                {
                    "title": "Map",
                    "url": "https://map.example.com",
                    "snippets": ["map desc"],
                }
            ],
        },
        "sources": {
            "https://a.example.com": {
                "title": "Source A",
                "hostname": "a.example.com",
                "age": ["Wednesday, June 17, 2026", "2026-06-17", "2 days ago"],
            },
            "https://b.example.com": {
                "hostname": "b.example.com",
            },
        },
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_normalizes_grounding_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = BraveLLMContextWebSearchProvider().search("test query", limit=3)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "A",
            "url": "https://a.example.com",
            "description": "first snippet second snippet (a.example.com | 2 days ago)",
            "position": 1,
        }
        assert web[2]["url"] == "https://poi.example.com"
        assert web[2]["position"] == 3

    def test_sends_post_payload_and_configurable_defaults(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.setenv("BRAVE_LLM_CONTEXT_MAXIMUM_NUMBER_OF_TOKENS", "12000")
        monkeypatch.setenv("BRAVE_LLM_CONTEXT_MAXIMUM_NUMBER_OF_SNIPPETS", "7")
        monkeypatch.setenv("BRAVE_LLM_CONTEXT_MAXIMUM_NUMBER_OF_TOKENS_PER_URL", "3000")
        monkeypatch.setenv("BRAVE_LLM_CONTEXT_CONTEXT_THRESHOLD_MODE", "strict")
        monkeypatch.setenv("BRAVE_LLM_CONTEXT_TIMEOUT_SECONDS", "9")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            captured["timeout"] = kwargs.get("timeout")
            return self._mock_resp({"grounding": {"generic": []}, "sources": {}})

        with patch("httpx.post", side_effect=fake_post):
            BraveLLMContextWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.search.brave.com/res/v1/llm/context"
        assert captured["headers"].get("X-Subscription-Token") == "BSAkey123"
        assert captured["json"] == {
            "q": "q",
            "count": 5,
            "maximum_number_of_urls": 5,
            "maximum_number_of_tokens": 12000,
            "maximum_number_of_snippets": 7,
            "maximum_number_of_tokens_per_url": 3000,
            "context_threshold_mode": "strict",
        }
        assert captured["timeout"] == 9

    def test_limit_is_clamped_to_50(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"grounding": {"generic": []}, "sources": {}})

        with patch("httpx.post", side_effect=fake_post):
            BraveLLMContextWebSearchProvider().search("q", limit=100)

        assert captured["json"]["count"] == 50
        assert captured["json"]["maximum_number_of_urls"] == 50

    def test_empty_or_missing_grounding_returns_empty_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp({})):
            result = BraveLLMContextWebSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        bad = MagicMock()
        bad.status_code = 429
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=bad)

        with patch("httpx.post", side_effect=err):
            result = BraveLLMContextWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "429" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = BraveLLMContextWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "boom" in result["error"] or "Brave" in result["error"]

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider

        result = BraveLLMContextWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "BRAVE_SEARCH_API_KEY" in result["error"]


class TestBraveLLMContextBackendWiring:
    def test_is_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("brave-llm-context") is True

    def test_is_backend_available_false_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("brave-llm-context") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "brave-llm-context"})
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        assert web_tools._get_backend() == "brave-llm-context"

    def test_brave_free_still_wins_auto_detect_with_shared_key(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in (
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "PARALLEL_API_KEY",
            "TAVILY_API_KEY",
            "EXA_API_KEY",
            "SEARXNG_URL",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "brave-free"

    def test_check_web_api_key_true_when_configured(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "brave-llm-context"})
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        assert web_tools.check_web_api_key() is True


class TestBraveLLMContextSearchOnlyErrors:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests

        _reset_for_tests()

    def test_web_extract_returns_search_only_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "brave-llm-context"})
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        async def _safe_url(_url):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _safe_url)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower()
        assert "brave" in result["error"].lower()
