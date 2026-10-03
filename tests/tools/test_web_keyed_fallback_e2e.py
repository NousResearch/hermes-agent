"""End-to-end config and provider resolution with real plugin imports."""
import json

import pytest

from tools import web_tools
from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
from plugins.web.tavily.provider import TavilyWebSearchProvider


@pytest.mark.asyncio
async def test_configured_firecrawl_402_falls_back_to_keyed_tavily(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    cfg = home / "config.yaml"
    cfg.write_text("web:\n  extract_backend: firecrawl\n  extract_fallbacks: [tavily]\n  keyless_rescue: false\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_CONFIG", str(cfg))
    monkeypatch.setenv("TAVILY_API_KEY", "test-key-not-sent")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key-not-sent")
    calls = []

    async def firecrawl(self, urls, **kwargs):
        calls.append("firecrawl")
        return [{"url": u, "error": "HTTP 402 credits exhausted"} for u in urls]

    def tavily(self, urls, **kwargs):
        calls.append("tavily")
        return [{"url": u, "content": "real plugin dispatch", "error": None} for u in urls]

    async def safe(url):
        return True

    monkeypatch.setattr(FirecrawlWebSearchProvider, "extract", firecrawl)
    monkeypatch.setattr(TavilyWebSearchProvider, "extract", tavily)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe)
    out = json.loads(await web_tools.web_extract_tool(["https://example.com/article"]))
    assert calls == ["firecrawl", "tavily"]
    assert out["results"][0]["metadata"]["served_by"] == "tavily"
    assert out["results"][0]["metadata"]["fallback_from"] == "firecrawl"


def test_search_chain_reads_real_config_and_plugin_registry(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("web:\n  search_backend: firecrawl\n  search_fallbacks: [tavily]\n  keyless_rescue: false\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CONFIG", str(cfg))
    monkeypatch.setenv("TAVILY_API_KEY", "test-key-not-sent")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key-not-sent")
    calls = []

    def firecrawl(self, query, limit):
        calls.append(("firecrawl", query, limit))
        raise RuntimeError("HTTP 402 credits exhausted")

    def tavily(self, query, limit):
        calls.append(("tavily", query, limit))
        return {"success": True, "data": {"web": [{"url": "https://example.com"}]}}

    monkeypatch.setattr(FirecrawlWebSearchProvider, "search", firecrawl)
    monkeypatch.setattr(TavilyWebSearchProvider, "search", tavily)
    out = json.loads(web_tools.web_search_tool("real config", 3))
    assert [c[0] for c in calls] == ["firecrawl", "tavily"]
    assert calls[0][1:] == calls[1][1:]
    assert out["data"]["metadata"] == {"served_by": "tavily", "fallback_from": "firecrawl"}
