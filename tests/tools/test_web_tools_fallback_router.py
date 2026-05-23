import asyncio
import json

import pytest

from agent.web_search_provider import WebSearchProvider


class FakeProvider(WebSearchProvider):
    def __init__(self, name, *, search_response=None, extract_response=None, raises=False, available=True):
        self._name = name
        self.search_response = search_response
        self.extract_response = extract_response
        self.raises = raises
        self.available = available
        self.search_calls = []
        self.extract_calls = []

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._name

    def is_available(self):
        return self.available

    def supports_search(self):
        return self.search_response is not None or self.raises

    def supports_extract(self):
        return self.extract_response is not None or self.raises

    def search(self, query, limit=5):
        self.search_calls.append((query, limit))
        if self.raises:
            raise RuntimeError(f"{self._name} quota exceeded")
        return self.search_response

    def extract(self, urls, **kwargs):
        self.extract_calls.append((list(urls), kwargs))
        if self.raises:
            raise RuntimeError(f"{self._name} quota exceeded")
        return self.extract_response


@pytest.fixture(autouse=True)
def reset_registry_and_config(monkeypatch):
    from agent import web_search_registry

    web_search_registry._reset_for_tests()
    monkeypatch.setattr("tools.web_tools.check_auxiliary_model", lambda: False)
    yield
    web_search_registry._reset_for_tests()


def test_web_search_fallback_tries_firecrawl_when_tavily_errors(monkeypatch):
    from agent.web_search_registry import register_provider
    from tools import web_tools

    tavily = FakeProvider("tavily", search_response={"success": False, "error": "quota exceeded"})
    firecrawl = FakeProvider(
        "firecrawl",
        search_response={
            "success": True,
            "data": {"web": [{"title": "ok", "url": "https://example.com", "description": "fallback"}]},
        },
    )
    register_provider(tavily)
    register_provider(firecrawl)

    monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: backend in {"tavily", "firecrawl"})
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        "search_backend": "tavily",
        "fallback_enabled": True,
        "fallback_backends": {"search": ["tavily", "firecrawl"]},
    })

    result = json.loads(web_tools.web_search_tool("credit limits", limit=3))

    assert result["success"] is True
    assert result["data"]["web"][0]["description"] == "fallback"
    assert len(tavily.search_calls) == 1
    assert len(firecrawl.search_calls) == 1


def test_web_search_does_not_fallback_when_disabled(monkeypatch):
    from agent.web_search_registry import register_provider
    from tools import web_tools

    tavily = FakeProvider("tavily", search_response={"success": False, "error": "quota exceeded"})
    firecrawl = FakeProvider("firecrawl", search_response={"success": True, "data": {"web": []}})
    register_provider(tavily)
    register_provider(firecrawl)

    monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: backend in {"tavily", "firecrawl"})
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        "search_backend": "tavily",
        "fallback_enabled": False,
        "fallback_backends": {"search": ["tavily", "firecrawl"]},
    })

    result = json.loads(web_tools.web_search_tool("credit limits", limit=3))

    assert result["success"] is False
    assert "quota" in result["error"]
    assert len(tavily.search_calls) == 1
    assert len(firecrawl.search_calls) == 0


def test_web_extract_fallback_tries_firecrawl_when_tavily_results_all_error(monkeypatch):
    from agent.web_search_registry import register_provider
    from tools import web_tools

    tavily = FakeProvider(
        "tavily",
        extract_response=[{"url": "https://example.com", "title": "", "content": "", "error": "quota exceeded"}],
    )
    firecrawl = FakeProvider(
        "firecrawl",
        extract_response=[{"url": "https://example.com", "title": "ok", "content": "fallback content"}],
    )
    register_provider(tavily)
    register_provider(firecrawl)

    monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: backend in {"tavily", "firecrawl"})
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        "extract_backend": "tavily",
        "fallback_enabled": True,
        "fallback_backends": {"extract": ["tavily", "firecrawl"]},
    })

    result = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)))

    assert result["results"][0]["content"] == "fallback content"
    assert len(tavily.extract_calls) == 1
    assert len(firecrawl.extract_calls) == 1
