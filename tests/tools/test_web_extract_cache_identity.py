"""A redirect's source, not another requested target, owns its cached content."""

import json
from unittest.mock import AsyncMock

import pytest

from agent import web_search_registry
from plugins.web.firecrawl import provider as firecrawl
from tools import web_tools


@pytest.mark.parametrize("source_field", ["sourceURL", "source_url"])
def test_redirect_source_keeps_its_cache_when_target_is_in_batch(monkeypatch, source_field):
    source = "https://example.com/source"
    target = "https://example.com/target"
    calls = []

    class Scraper:
        def scrape(self, *, url, **kwargs):
            calls.append(url)
            return {
                "markdown": f"Content requested via {url}",
                "metadata": {"url": target, source_field: url},
            }

    provider = firecrawl.FirecrawlWebSearchProvider()
    monkeypatch.setattr(web_search_registry, "get_provider", lambda name: provider)
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_backend": "firecrawl"})
    monkeypatch.setattr(web_tools, "async_is_safe_url", AsyncMock(return_value=True))
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", AsyncMock(return_value=True))
    monkeypatch.setattr(firecrawl, "is_safe_url", lambda url: True)
    monkeypatch.setattr(firecrawl, "_use_keyless_ring", lambda capability: False)
    monkeypatch.setattr(firecrawl, "_get_firecrawl_client", lambda capability: Scraper())

    def extract(urls):
        raw = web_tools.registry.dispatch("web_extract", {"urls": urls})
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert "results" in result, result
        return result["results"]

    # Fetch target first: the later redirect must not overwrite its cache entry.
    extract([target, source])
    assert calls == [target, source]
    assert extract([target])[0]["content"] == f"Content requested via {target}"
    assert extract([source])[0]["content"] == f"Content requested via {source}"
    assert calls == [target, source], "Both requested identities should be cache hits"
