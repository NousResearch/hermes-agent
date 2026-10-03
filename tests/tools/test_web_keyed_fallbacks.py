"""Configured keyed web fallbacks precede the anonymous rescue ring."""
import json

import pytest

from tools import web_tools, web_tools_extract



class Provider:
    def __init__(self, name, calls, *, search=None, extract=None):
        self.name = name
        self.calls = calls
        self._search = search
        self._extract = extract

    def supports_search(self):
        return self._search is not None

    def supports_extract(self):
        return self._extract is not None

    def is_available(self):
        return True

    def search(self, query, limit):
        self.calls.append(self.name)
        return self._search

    def extract(self, urls, **kwargs):
        self.calls.append(self.name)
        return [dict(r) for r in self._extract]


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setattr("agent.web_search_provider.get_provider_env",
                        lambda key: "test-key" if key in ("FIRECRAWL_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY") else "")
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        "search_backend": "firecrawl", "extract_backend": "firecrawl",
        "search_fallbacks": ["tavily", "exa"],
        "extract_fallbacks": ["tavily", "exa"],
    })
    async def safe(url):
        return True
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe)
    calls = []
    providers = {}
    monkeypatch.setattr("agent.web_search_registry.get_provider", providers.get)
    return calls, providers


def test_search_order_and_identity(setup, monkeypatch):
    calls, providers = setup
    fail = {"success": False, "error": "HTTP 402"}
    ok = {"success": True, "data": {"web": [{"url": "https://ok.example"}]}}
    providers.update(firecrawl=Provider("firecrawl", calls, search=fail),
                     tavily=Provider("tavily", calls, search=fail),
                     exa=Provider("exa", calls, search=ok))
    monkeypatch.setattr(web_tools, "_rescue_search", lambda *args: pytest.fail("ring before keyed chain"))
    result = json.loads(web_tools.web_search_tool("unique keyed order"))
    assert calls == ["firecrawl", "tavily", "exa"]
    assert result["data"]["metadata"] == {"served_by": "exa", "fallback_from": "firecrawl"}


@pytest.mark.asyncio
async def test_extract_partial_failure_does_not_fallback(setup, monkeypatch):
    calls, providers = setup
    urls = ["https://a.example", "https://b.example"]
    providers["firecrawl"] = Provider("firecrawl", calls, extract=[
        {"url": urls[0], "content": "ok", "error": None},
        {"url": urls[1], "content": "", "error": "404"},
    ])
    providers["tavily"] = Provider("tavily", calls, extract=[{"url": u, "content": "wrong"} for u in urls])
    monkeypatch.setattr(web_tools_extract, "_rescue_extract", lambda *args: pytest.fail("ring called"))
    out = json.loads(await web_tools.web_extract_tool(urls))
    assert calls == ["firecrawl"]
    assert out["results"][1]["error"] == "404"
    assert out["results"][0]["metadata"] == {"served_by": "firecrawl"}


@pytest.mark.asyncio
async def test_extract_402_to_keyed_tavily_real_dispatch(setup, monkeypatch):
    calls, providers = setup
    urls = ["https://a.example"]
    providers.update(firecrawl=Provider("firecrawl", calls, extract=[
        {"url": urls[0], "content": "", "error": "HTTP 402 credits exhausted"}]),
        tavily=Provider("tavily", calls, extract=[
            {"url": urls[0], "content": "tavily content", "error": None}]))
    monkeypatch.setattr(web_tools_extract, "_rescue_extract", lambda *args: pytest.fail("ring before keyed"))
    out = json.loads(await web_tools.web_extract_tool(urls))
    assert calls == ["firecrawl", "tavily"]
    assert out["results"][0]["content"] == "tavily content"
    assert out["results"][0]["metadata"] == {"served_by": "tavily", "fallback_from": "firecrawl"}




def test_search_missing_key_skips_candidate(setup, monkeypatch):
    calls, providers = setup
    monkeypatch.setattr("agent.web_search_provider.get_provider_env", lambda key: "")
    providers.update(firecrawl=Provider("firecrawl", calls, search={"success": False, "error": "402"}),
                     tavily=Provider("tavily", calls, search={"success": True, "data": {"web": []}}))
    monkeypatch.setattr(web_tools, "_rescue_eligible", lambda provider: False)
    out = json.loads(web_tools.web_search_tool("missing-key-skip"))
    assert calls == ["firecrawl"]
    assert out["success"] is False


@pytest.mark.asyncio
async def test_policy_refusal_never_sent_to_keyed_fallback(setup, monkeypatch):
    calls, providers = setup
    url = "https://a.example"
    providers.update(firecrawl=Provider("firecrawl", calls, extract=[
        {"url": url, "error": "blocked by website policy", "blocked_by_policy": True}]),
        tavily=Provider("tavily", calls, extract=[{"url": url, "content": "bypass"}]))
    monkeypatch.setattr(web_tools, "_rescue_eligible", lambda provider: False)
    out = json.loads(await web_tools.web_extract_tool([url]))
    assert calls == ["firecrawl"]
    assert out["results"][0]["blocked_by_policy"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("known_vendor", [None, "parallel"])
async def test_exhausted_chain_reaches_ring(setup, monkeypatch, known_vendor):
    calls, providers = setup
    urls = ["https://a.example"]
    bad = [{"url": urls[0], "error": "HTTP 402"}]
    for name in ("firecrawl", "tavily", "exa"):
        providers[name] = Provider(name, calls, extract=bad)
    def rescue(name, urls, results):
        assert name == "firecrawl"
        meta = {"rescued_from": name}
        if known_vendor:
            meta["served_by"] = known_vendor
        return [{"url": urls[0], "content": "ring", "metadata": meta}]
    monkeypatch.setattr(web_tools_extract, "_rescue_extract", rescue)
    monkeypatch.setattr(web_tools_extract, "_rescue_eligible", lambda provider: True)
    out = json.loads(await web_tools.web_extract_tool(urls))
    assert calls == ["firecrawl", "tavily", "exa"]
    assert out["results"][0]["content"] == "ring"
    assert out["results"][0].get("metadata", {}).get("served_by") == known_vendor


@pytest.mark.parametrize("capability", ["search", "extract"])
def test_candidates_are_deduplicated_and_capability_checked(setup, monkeypatch, capability):
    from tools.web_tools_rescue import _keyed_fallbacks
    calls, providers = setup
    providers.update(tavily=Provider("tavily", calls, search={}, extract=[]),
                     exa=Provider("exa", calls, search={}),
                     broken=Provider("broken", calls, search={}, extract=[]))
    def broken():
        raise RuntimeError("unavailable")
    monkeypatch.setattr(providers["broken"], "is_available", broken)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        f"{capability}_fallbacks": [None, "unknown", "firecrawl", " TAVILY ", "tavily", "broken", "exa"]})
    names = [p.name for p in _keyed_fallbacks(capability, "firecrawl")]
    assert names == (["tavily", "exa"] if capability == "search" else ["tavily"])


@pytest.mark.parametrize("value", [None, "tavily", {}, []])
def test_invalid_or_empty_chain_does_not_change_primary_response(setup, monkeypatch, value):
    calls, providers = setup
    response = {"success": True, "data": {"web": []}}
    providers["firecrawl"] = Provider("firecrawl", calls, search=response)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        "search_backend": "firecrawl", "search_fallbacks": value})
    expected = json.loads(json.dumps(response))
    assert json.loads(web_tools.web_search_tool("unchanged primary")) == expected
    assert calls == ["firecrawl"]


@pytest.mark.parametrize("raised", [False, True])
def test_search_stops_on_success_and_retries_primary_next_call(setup, monkeypatch, raised):
    calls, providers = setup
    providers.update(firecrawl=Provider("firecrawl", calls, search={"success": False, "error": "402"}),
                     tavily=Provider("tavily", calls, search={"success": True, "data": {"web": []}}),
                     exa=Provider("exa", calls, search={"success": True, "data": {"web": []}}))
    if raised:
        def boom(*args):
            calls.append("firecrawl")
            raise RuntimeError("402")
        monkeypatch.setattr(providers["firecrawl"], "search", boom)
    for _ in range(2):
        result = json.loads(web_tools.web_search_tool("same query"))
        assert result["data"]["metadata"] == {"served_by": "tavily", "fallback_from": "firecrawl"}
    assert calls == ["firecrawl", "tavily"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["result", "exception", "timeout"])
async def test_extract_stops_on_success_without_poisoning_primary_cache(setup, monkeypatch, failure):
    calls, providers = setup
    url = "https://a.example"
    providers.update(firecrawl=Provider("firecrawl", calls, extract=[{"url": url, "error": "402"}]),
                     tavily=Provider("tavily", calls, extract=[{"url": url, "content": "ok"}]),
                     exa=Provider("exa", calls, extract=[{"url": url, "content": "wrong"}]))
    if failure != "result":
        async def boom(*args, **kwargs):
            calls.append("firecrawl")
            if failure == "timeout":
                import asyncio
                raise asyncio.TimeoutError
            raise RuntimeError("402")
        monkeypatch.setattr(providers["firecrawl"], "extract", boom)
    for _ in range(2):
        result = json.loads(await web_tools.web_extract_tool([url]))
        assert result["results"][0]["metadata"]["served_by"] == "tavily"
    assert calls == ["firecrawl", "tavily"] * 2


def test_search_exhaustion_preserves_original_rescue_and_error(setup, monkeypatch):
    calls, providers = setup
    for name in ("firecrawl", "tavily", "exa"):
        providers[name] = Provider(name, calls, search={"success": False, "error": name + " 402"})
    def rescue(name, error, query, limit):
        assert calls == ["firecrawl", "tavily", "exa"]
        assert (name, error) == ("firecrawl", "firecrawl 402")
        return {"success": True, "data": {"web": [], "served_by": "parallel", "rescued_from": name}}
    monkeypatch.setattr(web_tools, "_rescue_eligible", lambda p: True)
    monkeypatch.setattr(web_tools, "_rescue_search", rescue)
    result = json.loads(web_tools.web_search_tool("exhaustion"))
    assert result["data"]["served_by"] == "parallel"
    assert "metadata" not in result["data"]


@pytest.mark.asyncio
async def test_extract_async_fallback_failure_advances_to_next_candidate(setup, monkeypatch):
    calls, providers = setup
    url = "https://a.example"
    providers.update(firecrawl=Provider("firecrawl", calls, extract=[{"url": url, "error": "402"}]),
                     tavily=Provider("tavily", calls, extract=[]),
                     exa=Provider("exa", calls, extract=[{"url": url, "content": "ok"}]))
    async def boom(*args, **kwargs):
        calls.append("tavily")
        raise RuntimeError("429")
    monkeypatch.setattr(providers["tavily"], "extract", boom)
    result = json.loads(await web_tools.web_extract_tool([url]))
    assert calls == ["firecrawl", "tavily", "exa"]
    assert result["results"][0]["metadata"] == {"served_by": "exa", "fallback_from": "firecrawl"}


def test_explicit_free_tier_is_not_a_keyed_candidate(setup, monkeypatch):
    from tools.web_tools_rescue import _keyed_fallbacks
    calls, providers = setup
    providers["exa"] = Provider("exa", calls, search={})
    monkeypatch.setattr("plugins.web.keyless_mcp.provider_tier", lambda name: "free")
    assert list(_keyed_fallbacks("search", "firecrawl")) == []


def test_web_picker_displays_both_chains_without_writing_config(monkeypatch, capsys):
    import copy
    from hermes_cli import tools_config_providers as picker
    config = {"web": {"search_fallbacks": ["tavily", "exa"], "extract_fallbacks": ["exa"]}}
    before = copy.deepcopy(config)
    monkeypatch.setattr(picker, "_visible_providers", lambda *a, **k: [{"name": "example"}])
    monkeypatch.setattr(picker, "_configure_provider", lambda *a, **k: None)
    picker._configure_tool_category("web", {"name": "Web"}, config)
    output = capsys.readouterr().out
    assert "search keyed fallbacks: tavily → exa" in output
    assert "extract keyed fallbacks: exa" in output
    assert config == before
