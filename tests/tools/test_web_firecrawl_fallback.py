"""Regression coverage for Firecrawl 400 recovery paths."""
from __future__ import annotations

import asyncio

from plugins.web.firecrawl import provider as firecrawl_provider


class _BadRequestClient:
    def scrape(self, **_kwargs):
        raise RuntimeError("Bad Request: HTTP 400")

    def search(self, **_kwargs):
        raise RuntimeError("Bad Request: HTTP 400")


def test_firecrawl_extract_400_uses_direct_http_fallback(monkeypatch):
    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: _BadRequestClient())
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda _url: None)
    monkeypatch.setattr(firecrawl_provider, "is_safe_url", lambda _url: True)

    fallback_calls = []

    def _fake_direct_fallback(url):
        fallback_calls.append(url)
        return {
            "url": url,
            "title": "Aviant - Enabling autonomous logistics",
            "content": "We do drone delivery.",
            "raw_content": "We do drone delivery.",
            "metadata": {"sourceURL": url, "fallback": "direct-http"},
        }

    monkeypatch.setattr(
        firecrawl_provider,
        "_direct_http_extract_fallback",
        _fake_direct_fallback,
    )

    result = asyncio.run(
        firecrawl_provider.FirecrawlWebSearchProvider().extract(["https://www.aviant.no/"])
    )

    assert fallback_calls == ["https://www.aviant.no/"]
    assert result == [
        {
            "url": "https://www.aviant.no/",
            "title": "Aviant - Enabling autonomous logistics",
            "content": "We do drone delivery.",
            "raw_content": "We do drone delivery.",
            "metadata": {"sourceURL": "https://www.aviant.no/", "fallback": "direct-http"},
        }
    ]


def test_firecrawl_search_400_returns_actionable_hint(monkeypatch):
    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: _BadRequestClient())
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    result = firecrawl_provider.FirecrawlWebSearchProvider().search(
        "Aviant drone delivery Norway",
        limit=5,
    )

    assert result["success"] is False
    assert "400 Bad Request" in result["error"]
    assert "web_extract" in result["error"]
    assert "direct HTTP fetch" in result["error"]
