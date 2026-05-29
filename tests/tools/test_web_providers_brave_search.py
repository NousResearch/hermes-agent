"""Tests for the Brave Search API provider and client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx


def _mock_resp(json_data, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    return response


class TestBraveSearchAvailability:
    def test_unavailable_without_keys(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        assert BraveSearchWebProvider().is_available() is False

    def test_available_with_primary_key(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "primary")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        assert BraveSearchWebProvider().is_available() is True

    def test_available_with_legacy_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "legacy")

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        assert BraveSearchWebProvider().is_available() is True

    def test_provider_contract(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.brave_search.provider import BraveSearchWebProvider

        provider = BraveSearchWebProvider()
        assert isinstance(provider, WebSearchProvider)
        assert provider.name == "brave-search"
        assert provider.display_name == "Brave Search API"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    def test_registry_fallback_keeps_brave_free_before_brave_search(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        from agent import web_search_registry
        from plugins.web.brave_free.provider import BraveFreeWebSearchProvider
        from plugins.web.brave_search.provider import BraveSearchWebProvider

        original_providers = list(web_search_registry.list_providers())
        web_search_registry._reset_for_tests()
        web_search_registry.register_provider(BraveFreeWebSearchProvider())
        web_search_registry.register_provider(BraveSearchWebProvider())

        try:
            provider = web_search_registry._resolve(None, capability="search")
            assert provider is not None
            assert provider.name == "brave-free"
        finally:
            web_search_registry._reset_for_tests()
            for provider in original_providers:
                web_search_registry.register_provider(provider)


class TestBraveSearchProviderSearch:
    SAMPLE = {
        "web": {
            "results": [
                {"title": "A", "url": "https://a.example", "description": "A desc"},
                {"title": "B", "url": "https://b.example", "description": "B desc"},
            ]
        }
    }

    def test_search_returns_generic_web_shape(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp(self.SAMPLE)):
            result = BraveSearchWebProvider().search("query", limit=5)

        assert result == {
            "success": True,
            "data": {
                "web": [
                    {"title": "A", "url": "https://a.example", "description": "A desc", "position": 1},
                    {"title": "B", "url": "https://b.example", "description": "B desc", "position": 2},
                ]
            },
        }

    def test_search_sends_expected_headers_and_params(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["params"] = kwargs.get("params", {})
            return _mock_resp({"web": {"results": []}})

        with patch("plugins.web.brave_search.client.httpx.get", side_effect=fake_get):
            BraveSearchWebProvider().search("query", limit=100)

        assert captured["url"] == "https://api.search.brave.com/res/v1/web/search"
        assert captured["headers"]["X-Subscription-Token"] == "BSAkey123"
        assert captured["params"]["q"] == "query"
        assert captured["params"]["count"] == 20
        assert captured["params"]["extra_snippets"] == "true"

    def test_web_search_tool_uses_configured_brave_search_with_legacy_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "legacy-key")

        from agent import web_search_registry
        from plugins.web.brave_search.provider import BraveSearchWebProvider
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_backend": "brave-search"})
        original_providers = list(web_search_registry.list_providers())
        web_search_registry._reset_for_tests()
        web_search_registry.register_provider(BraveSearchWebProvider())

        try:
            with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp(self.SAMPLE)):
                result = json.loads(web_tools.web_search_tool("query", limit=1))
        finally:
            web_search_registry._reset_for_tests()
            for provider in original_providers:
                web_search_registry.register_provider(provider)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "A"

    def test_http_error_does_not_leak_key(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        response = MagicMock(status_code=403)
        err = httpx.HTTPStatusError("403 secret-value", request=MagicMock(), response=response)

        with patch("plugins.web.brave_search.client.httpx.get", side_effect=err):
            result = BraveSearchWebProvider().search("query", limit=5)

        assert result["success"] is False
        assert "403" in result["error"]
        assert "secret-value" not in result["error"]

    def test_request_error_does_not_echo_exception_text(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        err = httpx.RequestError("secret-value proxy detail", request=MagicMock())

        with patch("plugins.web.brave_search.client.httpx.get", side_effect=err):
            result = BraveSearchWebProvider().search("query", limit=5)

        assert result["success"] is False
        assert result["error"] == "Could not reach Brave Search API"
        assert "secret-value" not in result["error"]

    def test_bad_json_returns_clean_error(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

        from plugins.web.brave_search.provider import BraveSearchWebProvider

        response = _mock_resp({})
        response.json.side_effect = ValueError("secret-value")

        with patch("plugins.web.brave_search.client.httpx.get", return_value=response):
            result = BraveSearchWebProvider().search("query", limit=5)

        assert result["success"] is False
        assert result["error"] == "Could not parse Brave Search API response as JSON"
        assert "secret-value" not in result["error"]


class TestBraveSearchClientModes:
    def test_web_parser_extracts_extra_sections(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        payload = {
            "web": {"results": [{"title": "Web", "url": "https://web", "description": "desc", "extra_snippets": ["extra"]}]},
            "news": {"results": [{"title": "News", "url": "https://news"}]},
            "videos": {"results": [{"title": "Video", "url": "https://video"}]},
            "discussions": {"results": [{"title": "Discussion", "url": "https://discussion"}]},
            "faq": {"results": [{"title": "FAQ", "url": "https://faq"}]},
            "infobox": {"results": [{"title": "Info", "url": "https://info"}]},
            "locations": {"results": [{"title": "Place", "url": "https://place"}]},
        }

        from plugins.web.brave_search.client import BraveSearchApiClient

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp(payload)):
            result = BraveSearchApiClient().search_web("query", limit=5)

        assert result["success"] is True
        data = result["data"]
        assert data["web"][0]["extra_snippets"] == ["desc", "extra"]
        assert data["news"][0]["title"] == "News"
        assert data["videos"][0]["title"] == "Video"
        assert data["discussions"][0]["title"] == "Discussion"
        assert data["faq"][0]["title"] == "FAQ"
        assert data["infobox"][0]["title"] == "Info"
        assert data["locations"][0]["title"] == "Place"

    def test_llm_context_parses_grounding_generic(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")

        from plugins.web.brave_search.client import BraveSearchApiClient

        payload = {
            "grounding": {
                "generic": [
                    {"title": "Ground", "url": "https://ground", "snippets": ["one", "two"]}
                ]
            }
        }
        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp(payload)):
            result = BraveSearchApiClient().llm_context("query", limit=1)

        assert result["success"] is True
        assert result["data"]["llm_context"] == [
            {"title": "Ground", "url": "https://ground", "snippets": ["one", "two"], "position": 1}
        ]

    def test_images_news_videos_discussions_and_suggest(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")

        from plugins.web.brave_search.client import BraveSearchApiClient

        client = BraveSearchApiClient()
        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"results": [{"title": "Image", "url": "https://img"}]})):
            assert client.search_images("query", limit=1)["data"]["images"][0]["title"] == "Image"

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"results": [{"title": "News", "url": "https://news"}]})):
            assert client.search_news("query", limit=1)["data"]["news"][0]["title"] == "News"

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"news": {"results": [{"title": "Nested News", "url": "https://news"}]}})):
            assert client.search_news("query", limit=1)["data"]["news"][0]["title"] == "Nested News"

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"results": [{"title": "Video", "url": "https://video"}]})):
            assert client.search_videos("query", limit=1)["data"]["videos"][0]["title"] == "Video"

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"videos": {"results": [{"title": "Nested Video", "url": "https://video"}]}})):
            assert client.search_videos("query", limit=1)["data"]["videos"][0]["title"] == "Nested Video"

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"discussions": {"results": [{"title": "Discussion", "url": "https://discussion"}]}})):
            assert client.search_discussions("query", limit=1)["data"]["discussions"][0]["title"] == "Discussion"

        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"results": [{"query": "query suggestion"}]})):
            assert client.suggest("query")["data"]["suggestions"] == ["query suggestion"]
