"""Search metadata must not masquerade as a fresh page retrieval."""

import json

import pytest


def _install_provider(monkeypatch, tmp_path, responses):
    from agent.web_search_provider import WebSearchProvider
    from agent import web_search_registry
    from tools import web_tools, web_result_cache

    class FixtureProvider(WebSearchProvider):
        name = "fixture-search"
        display_name = "Fixture search"

        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def search(self, query, limit=5):
            self.calls += 1
            return responses[self.calls - 1]

    provider = FixtureProvider()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "web:\n  search_backend: fixture-search\n  cache_enabled: true\n  keyless_rescue: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_search_registry._registry, "_providers", {})
    monkeypatch.setattr(web_search_registry._registry, "_scoped_providers", {})
    web_search_registry.register_provider(provider)
    monkeypatch.setattr(web_result_cache, "search_memo", web_result_cache.SearchMemo())
    return provider, web_tools


def test_cached_search_preserves_retrieval_time_and_reports_metadata_scope(monkeypatch, tmp_path):
    response = {"success": True, "data": {"web": [
        {"url": f"https://example.invalid/{i}", "title": str(i)} for i in range(10)
    ], "provenance": {"page_fetched": True, "fresh": True, "engine": "fixture"}}}
    provider, tools = _install_provider(monkeypatch, tmp_path, [response])
    first = json.loads(tools.web_search_tool("example", 3))
    second = json.loads(tools.web_search_tool("example", 5))
    a, b = first["data"]["provenance"], second["data"]["provenance"]
    assert provider.calls == 1
    assert a["cache"]["status"] == "miss" and b["cache"]["status"] == "hit"
    assert a["retrieved_at"] == b["retrieved_at"]
    assert a["served_at"] <= b["served_at"]
    assert b["page_fetched"] is False
    assert b["evidence_scope"] == "search_result_metadata_only"
    assert b["returned_count"] == 5 and b["fetched_result_count"] == 10
    assert b["engine"] == "fixture" and "fresh" not in b
    assert response["data"]["provenance"]["page_fetched"] is True


@pytest.mark.parametrize("response", [
    {"success": "true", "data": {"web": []}},
    {"success": True, "data": {"web": ["not a result object"]}},
    {"success": True, "data": {"web": [{"score": float("nan")}]}}
])
def test_invalid_search_responses_are_rejected_without_cache_or_truth_claims(monkeypatch, tmp_path, response):
    provider, tools = _install_provider(monkeypatch, tmp_path, [response, response])
    first = json.loads(tools.web_search_tool("example"))
    second = json.loads(tools.web_search_tool("example"))
    assert first["success"] is False and second["success"] is False
    assert "Invalid web search provider response" in first["error"]
    assert "provenance" not in first.get("data", {})
    assert provider.calls == 2
