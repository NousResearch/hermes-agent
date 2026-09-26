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


def test_each_fetched_page_stays_under_its_own_url_when_the_provider_omits_a_failure(
    extract_provider, monkeypatch,
):
    """An invalid item pins position 0, so the fetched pages are merged back by position. Exa omits
    the pages it could not fetch and backends reorder, so a page must land under the URL it names:
    the failed URL reports no result, and no page appears twice."""
    from tools.web_tools_extract import _NO_RESULT_ERROR

    async def _omit_and_reorder(urls, **kwargs):
        return [{"url": u, "title": "", "content": f"page {u}"} for u in reversed(urls) if not u.endswith("/down")]

    monkeypatch.setattr(extract_provider, "extract", _omit_and_reorder)
    urls = ["https://example.com/down", "https://example.com/a", "https://example.org/b"]
    result = json.loads(web_tools.registry.dispatch("web_extract", {"urls": [42, *urls]}))

    assert [(r["url"], r["content"] or r["error"]) for r in result["results"][1:]] == [
        (urls[0], _NO_RESULT_ERROR), (urls[1], f"page {urls[1]}"), (urls[2], f"page {urls[2]}"),
    ]
