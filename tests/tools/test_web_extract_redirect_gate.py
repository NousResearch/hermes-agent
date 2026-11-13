"""web_extract re-gates every page a provider reports under a URL it was not asked for.

The website policy and SSRF checks run on the REQUESTED url; a vendor that follows a redirect reports
the target instead (Keenable: ``url`` = target, ``metadata.sourceURL`` = requested; Exa/Parallel/Tavily:
the vendor's URL in both). That target is checked once, in the shared dispatch, for every provider.
"""

import asyncio

import pytest

import plugins.web.keenable.provider as keenable
from tools import web_result_cache as wrc
from tools import web_tools_extract as wte

REDIRECTOR = "https://redirector.example/go"
BLOCKED = "https://blocked.example/secret"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", dict)
    monkeypatch.setattr(wte, "_extract_timeout_seconds", lambda: 0)
    _policy(monkeypatch)


def _policy(monkeypatch, blocked_host: str | None = "blocked.example", unsafe_host: str | None = None):
    def check(url, *_a, **_k):
        if blocked_host and blocked_host in url:
            return {"host": blocked_host, "rule": blocked_host, "source": "config",
                    "message": f"Blocked: {blocked_host} is blocked by website policy"}
        return None

    async def safe(url):
        return not (unsafe_host and unsafe_host in url)

    monkeypatch.setattr("tools.website_policy.check_website_access", check)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)


def _keenable(monkeypatch, target):
    """The real Keenable provider (keyed path) against a stubbed /v1/fetch that followed a redirect."""
    calls = []

    class _Response:
        status_code = 200

        def json(self):
            return {"url": target, "title": "T", "content": "TARGET PAGE"}

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (calls.append(k.get("params")), _Response())[1])
    monkeypatch.setattr(keenable, "provider_env", lambda _k: "test-key")
    monkeypatch.setattr(keenable, "use_keyless", lambda *_a: False)
    return keenable.KeenableWebSearchProvider(), calls


def _extract(provider, urls=(REDIRECTOR,)):
    return asyncio.run(wte._extract_safe_urls(provider, list(urls), None))


def test_keenable_redirect_into_a_blocked_host_is_refused_and_never_cached(monkeypatch):
    provider, calls = _keenable(monkeypatch, BLOCKED)
    for _ in range(3):
        result = _extract(provider)[0]
        assert result["content"] == "" and "TARGET PAGE" not in str(result)
        assert result["blocked_by_policy"] == {"host": "blocked.example", "rule": "blocked.example", "source": "config"}
        assert result["url"] == BLOCKED and result["error"].startswith("Blocked")
    # Nothing was cached, so no hit is ever refused into a paid re-fetch of a page we will not serve.
    assert wrc.extract_cache_get(REDIRECTOR, provider="keenable") is None
    assert len(calls) == 3


def test_keenable_allowed_redirect_is_served_from_cache_without_a_refetch_loop(monkeypatch):
    provider, calls = _keenable(monkeypatch, "https://allowed.example/page")
    assert _extract(provider)[0]["content"] == "TARGET PAGE"
    for _ in range(2):
        hit = _extract(provider)[0]
        assert hit.get("cached") is True and hit["content"] == "TARGET PAGE"
    assert len(calls) == 1


def test_keenable_redirect_to_a_private_address_is_refused(monkeypatch):
    _policy(monkeypatch, blocked_host=None, unsafe_host="internal.example")
    provider, _calls = _keenable(monkeypatch, "http://internal.example/admin")
    result = _extract(provider)[0]
    assert result["content"] == "" and "private or internal" in result["error"]
    assert "blocked_by_policy" not in result


class _VendorUrlProvider:
    """Exa/Parallel/Tavily shape: the vendor's final URL in ``url`` and ``sourceURL`` alike."""

    name = "exa"

    def extract(self, urls, format=None):
        return [{"url": BLOCKED, "title": "T", "content": "TARGET PAGE", "raw_content": "TARGET PAGE",
                 "metadata": {"sourceURL": BLOCKED, "title": "T"}}]


def test_a_vendor_reported_url_outside_the_request_is_regated():
    result = _extract(_VendorUrlProvider())[0]
    assert result["content"] == "" and result["blocked_by_policy"]["host"] == "blocked.example"


def test_a_rescued_batch_is_regated(monkeypatch):
    class _Down:
        name = "parallel"

        def extract(self, urls, format=None):
            raise RuntimeError("backend down")

    monkeypatch.setattr(wte, "_rescue_eligible", lambda _p: True)
    monkeypatch.setattr(wte, "_rescue_extract", lambda _n, urls, _f: _VendorUrlProvider().extract(urls))
    result = _extract(_Down())[0]
    assert result["content"] == "" and result["blocked_by_policy"]["host"] == "blocked.example"


def test_a_policy_check_that_raises_on_a_cached_final_url_is_a_miss(monkeypatch):
    provider, calls = _keenable(monkeypatch, "https://allowed.example/page")
    _extract(provider)

    def boom(url, *_a, **_k):
        if "allowed.example" in url:
            raise RuntimeError("policy unreadable")

    monkeypatch.setattr("tools.website_policy.check_website_access", boom)
    result = _extract(provider)[0]
    assert not result.get("cached") and len(calls) == 2
    assert result["content"] == "" and "policy check failed" in result["error"]
