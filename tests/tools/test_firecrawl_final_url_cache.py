"""Firecrawl's requested identity and final destination survive cache reuse."""

from types import SimpleNamespace

import pytest

from plugins.web.firecrawl import provider as firecrawl
from tools import web_result_cache as cache
from tools import web_tools_extract as extract


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["api", "typed_sdk"])
@pytest.mark.parametrize("destination", ["public", "policy", "private", "missing", "malformed"])
async def test_scrape_validates_reported_final_url_and_preserves_requested_identity(monkeypatch, shape, destination):
    requested = "https://request.invalid/start"
    final = {"public": "https://final.invalid/page", "policy": "https://denied.invalid/page",
             "private": "http://127.0.0.1/private", "missing": None, "malformed": 17}[destination]
    metadata = {"sourceURL" if shape == "api" else "source_url": requested, "url": final, "title": "page"}
    response_metadata = metadata if shape == "api" else SimpleNamespace(model_dump=lambda: dict(metadata))
    calls = []
    safety_checks = []
    policy_checks = []

    def scrape(**kwargs):
        calls.append(kwargs)
        return {"data": {"metadata": response_metadata, "markdown": "completed synthetic page"}}

    def safety(url):
        safety_checks.append(url)
        return isinstance(url, str) and url.startswith("https://")

    def policy(url):
        policy_checks.append(url)
        return ({"host": "denied.invalid", "rule": "denied.invalid", "source": "config",
                 "message": "Blocked by website policy"} if url == "https://denied.invalid/page" else None)

    monkeypatch.setattr(firecrawl, "_get_firecrawl_client", lambda: SimpleNamespace(scrape=scrape))
    monkeypatch.setattr(firecrawl, "is_safe_url", safety)
    monkeypatch.setattr(firecrawl, "check_website_access", policy)
    result = await firecrawl._scrape_one(requested, ["markdown"], "markdown")
    assert len(calls) == 1
    assert calls[0]["url"] == requested
    if destination == "public":
        assert result["url"] == final
        assert result["content"] == "completed synthetic page"
        assert result["metadata"]["sourceURL"] == requested
        assert result["metadata"]["url"] == final
    else:
        assert result.get("error")
        assert not result.get("content")
        assert not result.get("raw_content")
    if isinstance(final, str):
        assert final in safety_checks
        if destination == "policy":
            assert final in policy_checks
            assert result["blocked_by_policy"]["host"] == "denied.invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("source_key", ["sourceURL", "source_url"])
@pytest.mark.parametrize("change", ["none", "policy", "dns", "legacy", "request_collision"])
async def test_cached_redirect_keeps_final_identity_and_rechecks_destination(monkeypatch, tmp_path, source_key, change):
    requested = "https://request.invalid/start"
    final = "https://final.invalid/page"
    urls = [requested]
    if change == "request_collision":
        urls.append(final)
    index_dir = tmp_path / "cache"
    index_dir.mkdir()
    monkeypatch.setattr(cache, "_cache_dir", lambda: index_dir)
    monkeypatch.setattr(cache, "_web_config", lambda: {"cache_enabled": True})
    monkeypatch.setattr(extract, "_extract_timeout_seconds", lambda: 0)
    state = {"policy": False, "dns": False}
    fetched_batches = []

    def policy(url):
        return ({"host": "final.invalid", "rule": "final.invalid", "source": "config",
                 "message": "Blocked by website policy"} if state["policy"] and url == final else None)

    async def safe(url):
        return not (state["dns"] and url == final)

    async def fetch(batch, **kwargs):
        fetched_batches.append(list(batch))
        return [{"url": final, "title": "page", "content": "page for " + url,
                 "metadata": {source_key: url, "url": final}} for url in batch]

    monkeypatch.setattr("tools.website_policy.check_website_access", policy)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe)
    provider = SimpleNamespace(name="example-provider", extract=fetch)
    first = await extract._extract_safe_urls(provider, urls, "markdown")
    assert [r["content"] for r in first] == ["page for " + u for u in urls]
    assert len(fetched_batches) == 1
    index = cache._load_index()
    request_entry = index.get(cache._url_digest(requested, "markdown", provider.name))
    assert request_entry is not None
    assert request_entry.get("final_url") == final
    if change == "legacy":
        request_entry.pop("final_url")
        cache._save_index(index)
    state["policy"] = change == "policy"
    state["dns"] = change == "dns"
    second = await extract._extract_safe_urls(provider, urls, "markdown")
    if change in ("policy", "dns"):
        assert second[0].get("error")
        assert not second[0].get("content")
        assert second[0]["metadata"]["sourceURL"] == requested
        if change == "policy":
            assert second[0]["blocked_by_policy"]["host"] == "final.invalid"
        assert len(fetched_batches) == 1
    else:
        assert [r["content"] for r in second] == ["page for " + u for u in urls]
        assert len(fetched_batches) == (2 if change == "legacy" else 1)
        if change != "legacy":
            assert all(r["cached"] for r in second)
            assert second[0]["url"] == final
            assert second[0]["metadata"]["sourceURL"] == requested
