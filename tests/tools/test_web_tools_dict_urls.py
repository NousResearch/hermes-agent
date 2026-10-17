"""Regression tests for model-supplied web_extract URL items (search-result objects, bare hosts)."""

import json
import socket

import pytest

from agent import web_search_registry
from agent.web_search_provider import WebSearchProvider
from tools import url_safety, web_tools


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


_DNS = {"localhost": "127.0.0.1", "example.com": "93.184.216.34"}  # anything else: a private address


@pytest.fixture
def gated_extract_provider(extract_provider, monkeypatch):
    """``extract_provider`` behind the real SSRF gate, with private-address blocking on and fake DNS."""
    monkeypatch.setattr(web_tools, "async_is_safe_url", url_safety.async_is_safe_url)
    monkeypatch.setattr(url_safety, "_global_allow_private_urls", lambda: False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *_a, **_k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (_DNS.get(host, "10.0.0.5"), 0))])
    return extract_provider


@pytest.mark.asyncio
async def test_web_extract_fetches_bare_host_url_over_https(gated_extract_provider):
    """A URL without a scheme names a public page; it must not be reported as a private address."""
    result = json.loads(await web_tools.web_extract_tool(["example.com/docs/page"]))

    assert gated_extract_provider.received_urls == ["https://example.com/docs/page"]
    assert result["results"][0]["error"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "intranet.example/admin",                        # resolves to a private address
    "metadata.google.internal/computeMetadata/v1/",  # cloud metadata hostname
    "localhost:8080/admin",
    "10.0.0.5/admin",
])
async def test_web_extract_still_blocks_bare_private_hosts(gated_extract_provider, url):
    result = json.loads(await web_tools.web_extract_tool([url]))

    assert gated_extract_provider.received_urls == []
    assert result["results"][0]["error"] == "Blocked: URL targets a private or internal network address"
