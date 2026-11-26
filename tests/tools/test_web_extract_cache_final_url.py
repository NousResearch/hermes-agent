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



def test_an_uncached_refetch_never_retargets_the_old_entry(monkeypatch):
    """A refused hit re-fetches; if that fetch lands somewhere allowed but is too big to cache, the old
    entry keeps its own final URL, so its blocked content is never served under the new target."""
    check, safe = _policy()
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)
    _extract(_RedirectingProvider())

    check, safe = _policy(blocked_host="target.example")
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)
    from tools.web_tools_truncate import MAX_STORED_TEXT_CHARS
    huge = "x" * (MAX_STORED_TEXT_CHARS + 1)

    class _NowElsewhere(_RedirectingProvider):
        async def extract(self, urls, format=None):
            self.calls += 1
            return [{"url": "https://allowed.example/", "content": huge, "raw_content": huge,
                     "metadata": {"sourceURL": urls[0]}}]

    _extract(_NowElsewhere())
    later = _RedirectingProvider()
    result = _extract(later)[0]
    assert later.calls == 1 and not result.get("cached"), "the old blocked page was served from cache"


def test_crlf_page_round_trips_through_the_cache(monkeypatch):
    """A page with CRLF line endings (Tavily passes text through untouched) must hit on the next
    call, not miss forever and re-buy the vendor fetch."""
    check, safe = _policy()
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)

    class _Crlf(_RedirectingProvider):
        async def extract(self, urls, format=None):
            self.calls += 1
            body = "line one\r\nline two\r\n"
            return [{"url": TARGET, "content": body, "raw_content": body, "metadata": {"sourceURL": urls[0]}}]

    provider = _Crlf()
    _extract(provider)
    assert _extract(provider)[0].get("cached") is True
    assert provider.calls == 1


def test_interleaved_writers_never_pair_one_body_with_anothers_final_url(monkeypatch):
    """Two fetches of one key: A writes its body, B writes its body and final URL, then A's late
    final-URL write lands. B's body must stay paired with B's destination, so blocking B's host
    makes the hit a miss instead of serving it under A's still-allowed destination."""
    check, safe = _policy()
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)

    def landing(host, body):
        class _Provider(_RedirectingProvider):
            async def extract(self, urls, format=None):
                self.calls += 1
                return [{"url": f"https://{host}/", "content": body, "raw_content": body,
                         "metadata": {"sourceURL": urls[0]}}]
        return _Provider()

    real_record, deferred = wrc.extract_cache_record_final_url, []
    monkeypatch.setattr(wrc, "extract_cache_record_final_url", lambda *a, **k: deferred.append((a, k)))
    def fetch(provider):  # straight to the fetch+cache-write path, as two concurrent misses would
        asyncio.run(wte._dispatch_extract(provider, [REDIRECTOR], None))

    fetch(landing("allowed-a.example", "BODY A"))  # A-put; A-final held back
    monkeypatch.setattr(wrc, "extract_cache_record_final_url", real_record)
    fetch(landing("origin-b.example", "BODY B"))  # B-put, B-final
    for args, kwargs in deferred:  # A-final lands last
        real_record(*args, **kwargs)

    check, safe = _policy(blocked_host="origin-b.example")
    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    later = _RedirectingProvider()
    result = _extract(later)[0]
    assert later.calls == 1 and not result.get("cached"), "B's blocked body was served under A's final URL"
