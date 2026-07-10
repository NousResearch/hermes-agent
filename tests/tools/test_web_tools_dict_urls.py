"""Regression tests for model-forwarded web-search result objects."""

import json

import pytest

from agent import web_search_registry
from agent.web_search_provider import WebSearchProvider
from tools import web_tools


class _FakeExtractProvider(WebSearchProvider):
    def __init__(self) -> None:
        self.received_urls: list[str] = []

    @property
    def name(self) -> str:
        return "dict-url-test"

    @property
    def display_name(self) -> str:
        return "Dict URL Test"

    def is_available(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls, **kwargs):
        self.received_urls.extend(urls)
        return [
            {"url": url, "title": "", "content": "ok"}
            for url in urls
        ]


@pytest.fixture
def extract_provider(monkeypatch):
    with web_search_registry._lock:
        previous = dict(web_search_registry._providers)
        web_search_registry._providers.clear()

    provider = _FakeExtractProvider()
    web_search_registry.register_provider(provider)
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        web_tools,
        "_load_web_config",
        lambda: {"extract_backend": provider.name},
    )

    async def _safe(_url):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe)
    yield provider

    with web_search_registry._lock:
        web_search_registry._providers.clear()
        web_search_registry._providers.update(previous)


@pytest.mark.asyncio
async def test_web_extract_dispatches_urls_from_search_result_objects(extract_provider):
    result = json.loads(await web_tools.web_extract_tool([
        {"url": "https://example.com/a", "title": "A"},
        {"href": "https://example.org/b"},
    ]))

    assert extract_provider.received_urls == [
        "https://example.com/a",
        "https://example.org/b",
    ]
    assert [entry["url"] for entry in result["results"]] == extract_provider.received_urls


@pytest.mark.asyncio
async def test_web_extract_reports_invalid_items_without_dispatching_them(extract_provider):
    result = json.loads(await web_tools.web_extract_tool([
        {"url": "https://example.com/good"},
        {"title": "missing URL"},
        {"url": 123},
        None,
    ]))

    assert extract_provider.received_urls == ["https://example.com/good"]
    errors = [entry["error"] for entry in result["results"] if entry["error"]]
    assert errors == [
        "Invalid URL item at index 1: expected a URL string or an object "
        "with a string 'url' or 'href' field",
        "Invalid URL item at index 2: expected a URL string or an object "
        "with a string 'url' or 'href' field",
        "Invalid URL item at index 3: expected a URL string or an object "
        "with a string 'url' or 'href' field",
    ]


def test_web_extract_registry_dispatch_accepts_search_result_objects(
    extract_provider,
):
    """The model-facing registry path preserves object URLs through dispatch."""
    raw = web_tools.registry.dispatch("web_extract", {
        "urls": [{"url": "https://example.net/from-registry", "title": "R"}],
    })
    assert isinstance(raw, str)
    result = json.loads(raw)

    assert extract_provider.received_urls == ["https://example.net/from-registry"]
    assert result["results"][0]["url"] == "https://example.net/from-registry"
