"""A web_extract cache hit re-runs the website policy / SSRF gates on the URL its content came from.

A redirector page cached while its target was allowed must stop serving that target once the target is
blocklisted (or resolves private): the cache may skip the vendor call, never a control.
"""

import asyncio

import pytest

from tools import web_result_cache as wrc
from tools import web_tools_extract as wte

REDIRECTOR = "https://redirector.example/to-target"
TARGET = "https://target.example/"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", dict)
    monkeypatch.setattr("tools.web_tools_extract._extract_timeout_seconds", lambda: 0)


class _RedirectingProvider:
    """Reports the redirect target in ``url`` and the requested URL in ``metadata.sourceURL``."""

    name = "firecrawl"

    def __init__(self):
        self.calls = 0

    async def extract(self, urls, format=None):
        self.calls += 1
        return [{"url": TARGET, "title": "Target", "content": "TARGET PAGE", "raw_content": "TARGET PAGE",
                 "metadata": {"sourceURL": urls[0]}}]


def _policy(blocked_host=None, unsafe_host=None):
    def check(url, *_a, **_k):
        if blocked_host and blocked_host in url:
            return {"host": blocked_host, "rule": blocked_host, "source": "config", "message": "Blocked by website policy"}
        return None

    async def safe(url):
        return not (unsafe_host and unsafe_host in url)

    return check, safe


def _extract(provider):
    return asyncio.run(wte._extract_safe_urls(provider, [REDIRECTOR], None))


@pytest.mark.parametrize("gate", ["blocklist", "ssrf"])
def test_cached_redirect_is_regated_on_its_final_url(monkeypatch, gate):
    check, safe = _policy()
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)
    provider = _RedirectingProvider()
    assert _extract(provider)[0]["content"] == "TARGET PAGE"
    assert _extract(provider)[0].get("cached") is True  # control: an allowed target is served from cache
    assert provider.calls == 1

    check, safe = _policy(**({"blocked_host": "target.example"} if gate == "blocklist" else {"unsafe_host": "target.example"}))
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)
    result = _extract(provider)[0]

    assert provider.calls == 2, "a hit whose final URL is now refused must be a miss (live fetch re-checks)"
    assert not result.get("cached")
