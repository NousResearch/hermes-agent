"""Extract delivery facts distinguish cached content from this call's fetch."""

import json

import pytest


def install_provider(monkeypatch, tmp_path, *, failing=False):
    from agent.web_search_provider import WebSearchProvider
    from agent import web_search_registry as registry
    from tools import web_tools

    class Fixture(WebSearchProvider):
        name = "fixture-extract"
        display_name = "Fixture extract"
        calls = 0

        def is_available(self):
            return True

        def supports_extract(self):
            return True

        async def extract(self, urls, **kwargs):
            self.calls += 1
            return [{"url": url, "content": "" if failing else "synthetic page", "error": "outage" if failing else None} for url in urls]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("web:\n  extract_backend: fixture-extract\n  keyless_rescue: false\n  cache_enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(registry._registry, "_providers", {})
    monkeypatch.setattr(registry._registry, "_scoped_providers", {})
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    async def safe_url(url):
        return True
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    provider = Fixture()
    registry.register_provider(provider)
    return provider, web_tools


@pytest.mark.asyncio
async def test_cache_hit_preserves_original_retrieval_and_records_no_provider_call(monkeypatch, tmp_path):
    provider, tools = install_provider(monkeypatch, tmp_path)
    first = json.loads(await tools.web_extract_tool(["https://example.com/a"]))
    second = json.loads(await tools.web_extract_tool(["https://example.com/a"]))
    a, b = first["provenance"], second["provenance"]
    assert provider.calls == 1
    assert a["cache_status"] == "miss" and b["cache_status"] == "hit"
    assert a["provider_call_attempted"] is True and b["provider_call_attempted"] is False
    assert a["retrieved_at"] == b["retrieved_at"]
    assert a["served_at"] <= b["served_at"]
    assert b["requested_backend"] == b["served_by"] == "fixture-extract"
    assert b["success_count"] == b["returned_count"] == b["requested_count"] == 1
    assert b["failure_count"] == 0
    assert second["results"][0]["content"] == "synthetic page"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["mixed-cache", "failure", "invalid", "rescue"])
async def test_mixed_and_failure_provenance_do_not_claim_fresh_content(monkeypatch, tmp_path, case):
    provider, tools = install_provider(monkeypatch, tmp_path, failing=case in {"failure", "rescue"})
    if case == "mixed-cache":
        old = json.loads(await tools.web_extract_tool(["https://example.com/a"]))
        result = json.loads(await tools.web_extract_tool(["https://example.com/a", "https://example.com/b"]))
        assert result["provenance"]["cache_status"] == "mixed"
        assert result["provenance"]["retrieved_at"] == old["provenance"]["retrieved_at"]
        assert result["provenance"]["success_count"] == 2
    elif case == "rescue":
        from tools import web_tools_extract
        monkeypatch.setattr(web_tools_extract, "_rescue_eligible", lambda _: True)
        monkeypatch.setattr(web_tools_extract, "_rescue_extract", lambda name, urls, failures: [
            {"url": url, "content": "rescued page", "metadata": {"_hermes_served_by": "fixture-backup"}, "error": None}
            for url in urls
        ])
        for _ in range(2):
            result = json.loads(await tools.web_extract_tool(["https://example.com/a"]))
            provenance = result["provenance"]
            assert provenance["fallback_attempted"] is True and provenance["fallback_used"] is True
            assert provenance["served_by"] == "fixture-backup"
            assert provenance["retrieved_at"] is not None
            assert provenance["cache_status"] == "miss"
        assert provider.calls == 2
    else:
        urls = [{"invalid": True}] if case == "invalid" else ["https://example.com/a"]
        result = json.loads(await tools.web_extract_tool(urls))
        provenance = result["provenance"]
        assert provenance["retrieved_at"] is None
        assert provenance["served_by"] is None
        assert provenance["success_count"] == 0 and provenance["failure_count"] == 1
        assert provenance["provider_call_attempted"] is (case == "failure")
