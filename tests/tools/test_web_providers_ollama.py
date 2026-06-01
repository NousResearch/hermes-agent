"""Tests for the Ollama Cloud web search + extract provider.

Covers:
- OllamaWebSearchProvider.is_available() — reflects OLLAMA_API_KEY presence
- OllamaWebSearchProvider.search() — happy path, missing key, HTTP errors
- OllamaWebSearchProvider.extract() — happy path, per-URL failure handling
- Result normalization (title, url, description/content, position)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx

# ---------------------------------------------------------------------------
# OllamaWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestOllamaProviderIsConfigured:
    def test_configured_when_api_key_set(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-test-key")
        assert OllamaWebSearchProvider().is_available() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        assert OllamaWebSearchProvider().is_available() is False

    def test_not_configured_when_key_empty(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        monkeypatch.setenv("OLLAMA_API_KEY", "")
        assert OllamaWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        assert OllamaWebSearchProvider().name == "ollama"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        assert issubclass(OllamaWebSearchProvider, WebSearchProvider)


class TestOllamaProviderSearch:
    def test_happy_path_normalizes_results(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "A", "url": "https://a.example.com", "content": "desc A"},
                {"title": "B", "url": "https://b.example.com", "content": "desc B"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        def fake_post(*args, **kwargs):
            return mock_response

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {"title": "A", "url": "https://a.example.com", "description": "desc A", "position": 1}
        assert web[1]["position"] == 2

    def test_limit_is_respected(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": f"R{i}", "url": f"https://r{i}.example.com", "content": f"desc {i}"}
                        for i in range(10)]
        }
        mock_response.raise_for_status = MagicMock()

        def fake_post(*args, **kwargs):
            return mock_response

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().search("q", limit=3)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3

    def test_missing_key_returns_failure(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        result = OllamaWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "OLLAMA_API_KEY" in result["error"]

    def test_http_error_returns_failure(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        mock_response = MagicMock()
        mock_response.status_code = 429
        exc = httpx.HTTPStatusError("too many requests", request=MagicMock(), response=mock_response)

        def fake_post(*args, **kwargs):
            raise exc

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "HTTP" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        def fake_post(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "Could not reach" in result["error"]

    def test_json_parse_error_returns_failure(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("invalid json")

        def fake_post(*args, **kwargs):
            return mock_response

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "parse" in result["error"].lower()

    def test_empty_results(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        def fake_post(*args, **kwargs):
            return mock_response

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().search("nothing", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []


class TestOllamaProviderExtract:
    def test_happy_path(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "title": "Example",
            "content": "page content here",
            "links": [],
        }
        mock_response.raise_for_status = MagicMock()

        def fake_post(*args, **kwargs):
            return mock_response

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().extract(["https://example.com"])

        assert len(result) == 1
        doc = result[0]
        assert doc["url"] == "https://example.com"
        assert doc["title"] == "Example"
        assert doc["content"] == "page content here"
        assert "error" not in doc

    def test_per_url_failure(self, monkeypatch):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        call_count = [0]

        def fake_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                mock = MagicMock()
                mock.json.return_value = {"title": "OK", "content": "ok", "links": []}
                mock.raise_for_status = MagicMock()
                return mock
            else:
                mock_resp = MagicMock()
                mock_resp.status_code = 500
                raise httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)

        monkeypatch.setenv("OLLAMA_API_KEY", "ol-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)
        monkeypatch.setattr("httpx.post", fake_post)

        result = OllamaWebSearchProvider().extract([
            "https://ok.example.com",
            "https://fail.example.com",
        ])

        assert len(result) == 2
        assert "error" not in result[0]
        assert "error" in result[1]


class TestOllamaProviderCapabilities:
    def test_supports_search(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().supports_search() is True

    def test_supports_extract(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().supports_extract() is True

    def test_supports_crawl(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        assert OllamaWebSearchProvider().supports_crawl() is False

    def test_setup_schema(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider

        schema = OllamaWebSearchProvider().get_setup_schema()
        assert "name" in schema
        assert "env_vars" in schema
        assert len(schema["env_vars"]) == 1
        assert schema["env_vars"][0]["key"] == "OLLAMA_API_KEY"
