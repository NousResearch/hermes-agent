"""``web_extract`` forwards optional request headers to the extract backend.

Firecrawl's scrape API accepts a ``headers`` object; before #74177 Hermes never
populated it, so there was no way to set a descriptive ``User-Agent``, a
``Referer``, or an ``Authorization`` header on an extracted fetch. These tests
pin the plumbing: the tool sanitizes the caller-supplied mapping and threads it
to ``provider.extract(...)`` only when non-empty, leaving the no-header call
shape byte-for-byte unchanged.
"""

import asyncio
import json

import pytest

from agent.web_search_provider import WebSearchProvider
from tools import web_tools
from tools.web_tools import _sanitize_extract_headers


class TestSanitizeExtractHeaders:
    def test_none_and_non_mapping_become_none(self):
        assert _sanitize_extract_headers(None) is None
        assert _sanitize_extract_headers("User-Agent: x") is None
        assert _sanitize_extract_headers(["a", "b"]) is None

    def test_empty_mapping_becomes_none(self):
        assert _sanitize_extract_headers({}) is None

    def test_keeps_clean_string_pairs(self):
        assert _sanitize_extract_headers(
            {"User-Agent": "HermesBot/1.0", "Referer": "https://example.com"}
        ) == {"User-Agent": "HermesBot/1.0", "Referer": "https://example.com"}

    def test_drops_non_string_pairs(self):
        assert _sanitize_extract_headers(
            {"User-Agent": "ok", "X-Count": 3, 7: "nope"}
        ) == {"User-Agent": "ok"}

    def test_rejects_header_injection_via_control_chars(self):
        # A stray CR/LF/NUL in a name or value is a header-injection primitive.
        assert _sanitize_extract_headers(
            {
                "User-Agent": "ok",
                "X-Bad": "a\r\nSet-Cookie: evil=1",
                "X-Bad-Name\n": "v",
            }
        ) == {"User-Agent": "ok"}

    def test_all_bad_becomes_none(self):
        assert _sanitize_extract_headers({"a": "b\r\n"}) is None


class _CapturingProvider(WebSearchProvider):
    """Minimal extract provider that records the kwargs it was dispatched."""

    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "firecrawl"

    @property
    def display_name(self):
        return "Capturing Firecrawl"

    def is_available(self):
        return True

    def supports_extract(self):
        return True

    async def extract(self, urls, **kwargs):
        self.calls.append(kwargs)
        return [
            {"url": u, "title": "", "content": "ok", "raw_content": "ok", "metadata": {}}
            for u in urls
        ]


@pytest.fixture
def capturing_backend(monkeypatch):
    from agent import web_search_registry

    with web_search_registry._lock:
        original = dict(web_search_registry._providers)
        web_search_registry._providers.clear()

    provider = _CapturingProvider()
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"extract_backend": "firecrawl"}
    )
    web_search_registry.register_provider(provider)
    try:
        yield provider
    finally:
        with web_search_registry._lock:
            web_search_registry._providers.clear()
            web_search_registry._providers.update(original)


class TestHeadersReachBackend:
    def _run(self, **kwargs):
        return json.loads(
            asyncio.run(web_tools.web_extract_tool(["https://example.com"], **kwargs))
        )

    def test_headers_forwarded_when_supplied(self, capturing_backend):
        self._run(headers={"User-Agent": "HermesBot/1.0"})
        assert capturing_backend.calls, "extract() was never dispatched"
        assert capturing_backend.calls[0].get("headers") == {"User-Agent": "HermesBot/1.0"}

    def test_no_headers_kwarg_when_unset(self, capturing_backend):
        self._run()
        assert capturing_backend.calls
        assert "headers" not in capturing_backend.calls[0]

    def test_empty_headers_are_not_forwarded(self, capturing_backend):
        self._run(headers={})
        assert capturing_backend.calls
        assert "headers" not in capturing_backend.calls[0]

    def test_dirty_headers_sanitized_before_dispatch(self, capturing_backend):
        self._run(headers={"User-Agent": "ok", "X-Bad": "a\r\nInjected: 1"})
        assert capturing_backend.calls
        assert capturing_backend.calls[0].get("headers") == {"User-Agent": "ok"}


class _CapturingFirecrawlClient:
    """Fake Firecrawl SDK client that records the scrape() kwargs it receives."""

    def __init__(self):
        self.scrape_calls = []

    def scrape(self, **kwargs):
        self.scrape_calls.append(kwargs)
        return {"markdown": "ok", "metadata": {"sourceURL": kwargs["url"], "title": "Ex"}}


@pytest.fixture
def firecrawl_scrape_client(monkeypatch):
    """Patch the real Firecrawl provider's SDK seam + policy gates.

    Exercises ``FirecrawlWebSearchProvider.extract`` end-to-end down to the
    ``_get_firecrawl_client().scrape(**scrape_kwargs)`` boundary, so a
    regression in ``scrape_kwargs`` construction (e.g. dropping ``headers``)
    is caught — the generic-provider tests above only cover dispatcher
    plumbing, not this SDK call shape.
    """
    from plugins.web.firecrawl import provider as fc_provider

    client = _CapturingFirecrawlClient()
    monkeypatch.setattr(fc_provider, "_get_firecrawl_client", lambda: client)
    monkeypatch.setattr(fc_provider, "check_website_access", lambda url: None)
    monkeypatch.setattr(fc_provider, "is_safe_url", lambda url: True)
    # Force the keyed SDK path: without credentials the provider now short-
    # circuits into the keyless failover ring (which never touches the SDK
    # ``scrape`` seam these tests pin), so disable it here.
    monkeypatch.setattr(fc_provider, "_use_keyless_ring", lambda: False)
    return client


class TestScrapeSDKBoundary:
    """Verify the changed Firecrawl SDK boundary, not just the dispatcher."""

    def _extract(self, provider_client, **kwargs):
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        return asyncio.run(
            FirecrawlWebSearchProvider().extract(["https://example.com"], **kwargs)
        )

    def test_scrape_receives_headers_when_supplied(self, firecrawl_scrape_client):
        self._extract(
            firecrawl_scrape_client, headers={"User-Agent": "HermesBot/1.0"}
        )
        assert firecrawl_scrape_client.scrape_calls, "scrape() was never called"
        call = firecrawl_scrape_client.scrape_calls[0]
        assert call.get("headers") == {"User-Agent": "HermesBot/1.0"}
        assert call["url"] == "https://example.com"

    def test_scrape_omits_headers_when_unset(self, firecrawl_scrape_client):
        self._extract(firecrawl_scrape_client)
        assert firecrawl_scrape_client.scrape_calls
        assert "headers" not in firecrawl_scrape_client.scrape_calls[0]

    def test_scrape_omits_headers_when_empty(self, firecrawl_scrape_client):
        self._extract(firecrawl_scrape_client, headers={})
        assert firecrawl_scrape_client.scrape_calls
        assert "headers" not in firecrawl_scrape_client.scrape_calls[0]
