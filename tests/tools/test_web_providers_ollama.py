"""Tests for the Ollama Cloud web search provider.

Covers:
- OllamaWebSearchProvider.is_available() env var gating
- OllamaWebSearchProvider.search() happy path + normalization
- OllamaWebSearchProvider.extract() happy path
- Missing API key handling
- _is_backend_available("ollama") integration
- _get_backend() recognizes "ollama" as a valid configured backend
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestOllamaProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().is_available() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().is_available() is False

    def test_not_configured_when_key_whitespace(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "   ")
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().name == "ollama"
        assert OllamaWebSearchProvider().display_name == "Ollama Cloud"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert issubclass(OllamaWebSearchProvider, WebSearchProvider)

    def test_supports_search_and_extract(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        p = OllamaWebSearchProvider()
        assert p.supports_search() is True
        assert p.supports_extract() is True


class TestOllamaProviderSearch:
    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "A", "url": "https://a.example.com", "content": "snippet A"},
            {"title": "B", "url": "https://b.example.com", "content": "snippet B"},
        ]
    }

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        def fake_urlopen(req, timeout=None):
            data = json.dumps(self._SAMPLE_RESPONSE).encode()
            m = MagicMock()
            m.read.return_value = data
            m.__enter__ = lambda s: s
            m.__exit__ = lambda *a: None
            return m

        with patch("urllib.request.urlopen", fake_urlopen):
            result = OllamaWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {"title": "A", "url": "https://a.example.com", "description": "snippet A", "position": 1}
        assert web[1]["position"] == 2

    def test_request_sends_json_with_max_results(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            captured["headers"] = dict(req.header_items())
            data = json.dumps({"results": []}).encode()
            m = MagicMock()
            m.read.return_value = data
            m.__enter__ = lambda s: s
            m.__exit__ = lambda *a: None
            return m

        with patch("urllib.request.urlopen", fake_urlopen):
            OllamaWebSearchProvider().search("q", limit=3)

        assert captured["body"]["query"] == "q"
        assert captured["body"]["max_results"] == 3
        assert captured["headers"].get("Authorization") == "Bearer osk_test"
        assert captured["headers"].get("Content-type") == "application/json"

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        result = OllamaWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "OLLAMA_API_KEY" in result["error"]

    def test_http_error_returns_failure(self, monkeypatch):
        import urllib.error
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {}, None
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            result = OllamaWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "429" in result["error"]


class TestOllamaProviderExtract:
    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        response = {"title": "T", "content": "body", "links": ["https://x.com"]}

        def fake_urlopen(req, timeout=None):
            data = json.dumps(response).encode()
            m = MagicMock()
            m.read.return_value = data
            m.__enter__ = lambda s: s
            m.__exit__ = lambda *a: None
            return m

        with patch("urllib.request.urlopen", fake_urlopen):
            results = OllamaWebSearchProvider().extract(["https://example.com"])

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert results[0]["title"] == "T"
        assert results[0]["content"] == "body"
        assert results[0]["raw_content"] == "body"
        assert results[0]["metadata"]["links"] == ["https://x.com"]

    def test_missing_key_returns_failure_per_url(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        results = OllamaWebSearchProvider().extract(["https://example.com"])
        assert len(results) == 1
        assert results[0]["error"]
        assert "OLLAMA_API_KEY" in results[0]["error"]


class TestOllamaBackendWiring:
    def test_is_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("ollama") is True

    def test_is_backend_available_false_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("ollama") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ollama"})
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        assert web_tools._get_backend() == "ollama"

    def test_auto_detect_picks_ollama_when_only_key_set(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OLLAMA_API_KEY", "osk_test")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "ollama"
