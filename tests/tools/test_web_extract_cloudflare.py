"""Test Cloudflare interstitial detection and cache rejection in web_extract (#120502)."""

import asyncio
from unittest.mock import MagicMock
import pytest

from tools import web_tools_extract as wte
from tools import web_result_cache as wrc


class _FakeProvider:
    name = "fake-provider"

    def __init__(self, results):
        self._results = results

    async def extract(self, urls, format=None):
        return self._results


def test_cloudflare_interstitial_detected_as_error(monkeypatch):
    """A Cloudflare 'Just a moment...' page with '##' body should report an error and not cache."""
    cache_put_mock = MagicMock()
    monkeypatch.setattr(wrc, "extract_cache_put", cache_put_mock)
    monkeypatch.setattr(wte, "_rescue_eligible", lambda p: False)

    url = "https://example.com/protected"
    raw_results = [
        {
            "url": url,
            "title": "Just a moment...",
            "content": "##",
            "raw_content": "##",
        }
    ]
    provider = _FakeProvider(raw_results)
    results = asyncio.run(wte._dispatch_extract(provider, [url], None))

    assert len(results) == 1
    assert results[0]["error"] is not None
    assert "Cloudflare challenge" in results[0]["error"]
    assert results[0]["content"] == ""
    # Cache put must not be called
    cache_put_mock.assert_not_called()


def test_cloudflare_interstitial_triggers_keyless_rescue(monkeypatch):
    """When a provider returns a Cloudflare interstitial, keyless rescue is triggered if eligible."""
    rescue_mock = MagicMock(return_value=[{"url": "https://example.com/protected", "content": "rescued"}])
    monkeypatch.setattr(wte, "_rescue_eligible", lambda p: True)
    monkeypatch.setattr(wte, "_rescue_extract", rescue_mock)

    url = "https://example.com/protected"
    raw_results = [
        {
            "url": url,
            "title": "Just a moment...",
            "content": "##",
            "raw_content": "##",
        }
    ]
    provider = _FakeProvider(raw_results)
    results = asyncio.run(wte._dispatch_extract(provider, [url], None))

    assert results == [{"url": "https://example.com/protected", "content": "rescued"}]
    rescue_mock.assert_called_once()


def test_long_article_discussing_cloudflare_not_blocked(monkeypatch):
    """A long article (>2000 chars) discussing Cloudflare error strings must not be blocked as an interstitial."""
    cache_put_mock = MagicMock()
    monkeypatch.setattr(wrc, "extract_cache_put", cache_put_mock)
    monkeypatch.setattr(wte, "_rescue_eligible", lambda p: False)

    url = "https://example.com/article"
    # Over 2000 characters of prose that happens to mention Cloudflare error phrases
    long_content = (
        "Here is an in-depth article about troubleshooting CDN issues. "
        "Sometimes users see 'enable JavaScript and cookies to continue' or 'checking your browser before accessing'. "
        + ("Detailed explanation of CDN configuration and proxy settings. " * 50)
    )
    assert len(long_content) > 2000

    raw_results = [
        {
            "url": url,
            "title": "Troubleshooting CDN and Proxy Issues",
            "content": long_content,
            "raw_content": long_content,
        }
    ]
    provider = _FakeProvider(raw_results)
    results = asyncio.run(wte._dispatch_extract(provider, [url], None))

    assert len(results) == 1
    assert "error" not in results[0] or results[0]["error"] is None
    assert results[0]["content"] == long_content



def test_extract_cache_put_refuses_interstitial(tmp_path, monkeypatch):
    """extract_cache_put refuses to store '##' or 'Just a moment...' pages."""
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", lambda: {})

    url = "https://example.com/cf-page"
    wrc.extract_cache_put(url, "##", title="Just a moment...", provider="test")
    hit = wrc.extract_cache_get(url, provider="test")
    assert hit is None


def test_extract_cache_get_ignores_stale_interstitial(tmp_path, monkeypatch):
    """extract_cache_get treats an existing index entry with 'Just a moment...' or '##' as a cache miss."""
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", lambda: {})

    url = "https://example.com/stale-cf"
    digest = wrc._url_digest(url, None, "test")
    file_path = cache_dir / f"page-{digest}.cache.md"
    file_path.write_text("##", encoding="utf-8")

    index = {
        digest: {
            "url": url,
            "file": str(file_path),
            "title": "Just a moment...",
            "fetched_at": 9999999999.0,
        }
    }
    wrc._save_index(index)

    hit = wrc.extract_cache_get(url, provider="test")
    assert hit is None
