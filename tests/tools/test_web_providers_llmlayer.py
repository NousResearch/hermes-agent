"""Tests for the LLMLayer web provider."""

from __future__ import annotations

import sys
import types

from plugins.web.llmlayer import provider as llmlayer_provider
from plugins.web.llmlayer.provider import LLMLayerWebSearchProvider


class _FakeLLMLayerClient:
    def __init__(self) -> None:
        self.search_calls = []
        self.scrape_calls = []

    def search_web(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return types.SimpleNamespace(
            results=[
                {
                    "title": "First",
                    "link": "https://example.com/1",
                    "snippet": "First result",
                },
                {
                    "title": "Second",
                    "link": "https://example.com/2",
                    "snippet": "Second result",
                },
            ]
        )

    def scrape(self, url, **kwargs):
        self.scrape_calls.append((url, kwargs))
        return {
            "url": url,
            "title": "Example",
            "markdown": "# Example\n\nBody",
            "html": "<h1>Example</h1><p>Body</p>",
            "metadata": {"title": "Example"},
        }


def test_get_llmlayer_client_uses_env_key_and_base_url(monkeypatch):
    llmlayer_provider._reset_client_for_tests()
    created = []
    fake_module = types.ModuleType("llmlayer")

    class LLMLayerClient:
        def __init__(self, *, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            created.append(self)

    fake_module.LLMLayerClient = LLMLayerClient
    monkeypatch.setitem(sys.modules, "llmlayer", fake_module)
    monkeypatch.setenv("LLMLAYER_API_KEY", "llmlayer-test")
    monkeypatch.setenv("LLMLAYER_BASE_URL", "https://llmlayer.local/")
    monkeypatch.setattr(llmlayer_provider, "_ensure_llmlayer_sdk_installed", lambda: None)

    client = llmlayer_provider._get_llmlayer_client()
    assert client.api_key == "llmlayer-test"
    assert client.base_url == "https://llmlayer.local"
    assert llmlayer_provider._get_llmlayer_client() is client
    assert len(created) == 1

    llmlayer_provider._reset_client_for_tests()


def test_llmlayer_search_normalizes_results(monkeypatch):
    fake_client = _FakeLLMLayerClient()
    provider = LLMLayerWebSearchProvider()

    monkeypatch.setattr(llmlayer_provider, "_get_llmlayer_client", lambda: fake_client)

    result = provider.search("hermes agent", limit=1)

    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "url": "https://example.com/1",
                    "title": "First",
                    "description": "First result",
                    "position": 1,
                }
            ]
        },
    }
    assert fake_client.search_calls == [
        (
            "hermes agent",
            {
                "search_type": "general",
                "timeout": 60,
            },
        )
    ]


def test_llmlayer_extract_uses_scrape_markdown(monkeypatch):
    fake_client = _FakeLLMLayerClient()
    provider = LLMLayerWebSearchProvider()

    monkeypatch.setattr(llmlayer_provider, "_get_llmlayer_client", lambda: fake_client)
    monkeypatch.setattr(llmlayer_provider, "check_website_access", lambda url: None)

    results = provider.extract(["https://example.com"], format="markdown")

    assert results == [
        {
            "url": "https://example.com",
            "title": "Example",
            "content": "# Example\n\nBody",
            "raw_content": "# Example\n\nBody",
            "metadata": {
                "title": "Example",
                "sourceURL": "https://example.com",
            },
        }
    ]
    assert fake_client.scrape_calls == [
        (
            "https://example.com",
            {
                "formats": ["markdown"],
                "include_images": False,
                "include_links": True,
                "timeout": 60,
            },
        )
    ]


def test_llmlayer_extract_uses_scrape_html_when_requested(monkeypatch):
    fake_client = _FakeLLMLayerClient()
    provider = LLMLayerWebSearchProvider()

    monkeypatch.setattr(llmlayer_provider, "_get_llmlayer_client", lambda: fake_client)
    monkeypatch.setattr(llmlayer_provider, "check_website_access", lambda url: None)

    results = provider.extract(["https://example.com"], format="html")

    assert results[0]["content"] == "<h1>Example</h1><p>Body</p>"
    assert fake_client.scrape_calls[0][1]["formats"] == ["html"]


def test_llmlayer_extract_returns_policy_block_without_scraping(monkeypatch):
    fake_client = _FakeLLMLayerClient()
    provider = LLMLayerWebSearchProvider()
    blocked = {
        "host": "example.com",
        "rule": "example.com",
        "source": "test",
        "message": "Blocked by policy",
    }

    monkeypatch.setattr(llmlayer_provider, "_get_llmlayer_client", lambda: fake_client)
    monkeypatch.setattr(llmlayer_provider, "check_website_access", lambda url: blocked)

    results = provider.extract(["https://example.com"], format="markdown")

    assert results == [
        {
            "url": "https://example.com",
            "title": "",
            "content": "",
            "raw_content": "",
            "error": "Blocked by policy",
            "blocked_by_policy": {
                "host": "example.com",
                "rule": "example.com",
                "source": "test",
            },
        }
    ]
    assert fake_client.scrape_calls == []
