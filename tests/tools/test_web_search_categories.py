"""Category filters reach the provider and cannot reuse unfiltered cached results."""
import json
from unittest.mock import patch

from plugins.web.searxng.provider import SearXNGWebSearchProvider
from tools import web_tools as wt
from tools.web_result_cache import SearchMemo


def test_dispatch_categories_and_cache_are_bound_to_filter(monkeypatch):
    provider = SearXNGWebSearchProvider()
    calls = []

    def get_json(*args, **kwargs):
        calls.append(kwargs["params"])
        label = kwargs["params"].get("categories", "general")
        return {"results": [{"url": "https://example.org/" + label, "title": label, "content": label}]}, None

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "searxng")
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"cache_enabled": True})
    with patch("agent.web_search_registry.get_provider", return_value=provider), patch("plugins.web.searxng.provider.provider_env", return_value="https://search.example"), patch("plugins.web.searxng.provider.http_get_json", side_effect=get_json), patch("tools.web_result_cache.search_memo", SearchMemo()):
        from tools.registry import registry
        handler = registry._tools["web_search"].handler
        for categories in [None, ["science"], [" SCIENCE "], ["news"]]:
            result = json.loads(handler({"query": "physics", "categories": categories}))
            assert result["success"]
        assert [c.get("categories") for c in calls] == [None, "science", "news"]


def test_legacy_provider_is_compatible_and_filtered_failure_is_not_rescued(monkeypatch):
    class Legacy:
        name = "legacy"
        def search(self, query, limit=5):
            return {"success": True, "data": {"web": []}}
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"cache_enabled": False})
    assert wt._memoized_search(Legacy(), "query", 5)["success"]
    assert not wt._memoized_search(Legacy(), "query", 5, categories=["science"])["success"]
    assert not json.loads(wt.web_search_tool("query", categories="science"))["success"]


def test_generated_code_execution_stub_forwards_categories():
    from tools.code_execution_tool import generate_hermes_tools_module
    namespace = {}
    exec(generate_hermes_tools_module(["web_search"]), namespace)
    calls = []
    namespace["_call"] = lambda name, args: calls.append((name, args)) or {"success": True}
    assert namespace["web_search"]("physics", categories=["science"])["success"]
    assert calls == [("web_search", {"query": "physics", "limit": 5, "categories": ["science"]})]
