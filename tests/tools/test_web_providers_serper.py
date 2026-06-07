"""Tests for the Serper web search provider.

Covers:
- SerperWebSearchProvider.is_available() env var gating
- SerperWebSearchProvider.search() normalization and error handling
- _is_backend_available("serper") / _get_backend() / check_web_api_key() wiring
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSerperProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sp-test")
        from plugins.web.serper.provider import SerperWebSearchProvider

        assert SerperWebSearchProvider().is_available() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        from plugins.web.serper.provider import SerperWebSearchProvider

        assert SerperWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.serper.provider import SerperWebSearchProvider

        assert SerperWebSearchProvider().name == "serper"


class TestSerperProviderSearch:
    _SAMPLE_RESPONSE = {
        "organic": [
            {"title": "A", "link": "https://a.example.com", "snippet": "desc A"},
            {"title": "B", "link": "https://b.example.com", "snippet": "desc B"},
            {"title": "C", "link": "https://c.example.com", "snippet": "desc C"},
        ]
    }

    @staticmethod
    def _mock_resp(json_data):
        m = MagicMock()
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sp-test")
        from plugins.web.serper.provider import SerperWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = SerperWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "A",
            "url": "https://a.example.com",
            "description": "desc A",
            "position": 1,
        }

    def test_sends_expected_endpoint_headers_and_payload(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sp-test")
        from plugins.web.serper.provider import SerperWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"organic": []})

        with patch("httpx.post", side_effect=fake_post):
            SerperWebSearchProvider().search("hello", limit=7)

        assert captured["url"] == "https://google.serper.dev/search"
        assert captured["headers"]["X-API-KEY"] == "sp-test"
        assert captured["json"]["q"] == "hello"
        assert captured["json"]["num"] == 7


class TestSerperBackendWiring:
    def test_is_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sp-test")
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("serper") is True

    def test_is_backend_available_false_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("serper") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "serper"})
        assert web_tools._get_backend() == "serper"

    def test_check_web_api_key_true_when_serper_configured(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "serper"})
        monkeypatch.setenv("SERPER_API_KEY", "sp-test")
        assert web_tools.check_web_api_key() is True
