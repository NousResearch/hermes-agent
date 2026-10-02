"""An explicit backend is a routing choice, including a misspelled name."""

import json

import pytest


def install_providers(monkeypatch, tmp_path, config):
    from agent.web_search_provider import WebSearchProvider
    from agent import web_search_registry as registry
    from tools import web_tools

    class Fixture(WebSearchProvider):
        name = "fixture"
        display_name = "Fixture"

        def __init__(self, name, search=True, extract=True):
            self.name, self.search_capable, self.extract_capable = name, search, extract
            self.calls = 0

        def is_available(self):
            return True

        def supports_search(self):
            return self.search_capable

        def supports_extract(self):
            return self.extract_capable

        def search(self, query, limit=5):
            self.calls += 1
            return {"success": True, "data": {"web": []}}

        async def extract(self, urls, format=None):
            self.calls += 1
            return [{"url": url, "content": "fixture", "error": None} for url in urls]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setattr(registry._registry, "_providers", {})
    monkeypatch.setattr(registry._registry, "_scoped_providers", {})
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    fallback = Fixture("fixture-fallback")
    wrong = Fixture("fixture-wrong", search=False, extract=False)
    registry.register_provider(fallback)
    registry.register_provider(wrong)
    return registry, web_tools, fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["search", "extract"])
@pytest.mark.parametrize("backend", ["missing-provider", "fixture-wrong"])
async def test_explicit_backend_never_walks_to_another_provider(monkeypatch, tmp_path, capability, backend):
    registry, tools, fallback = install_providers(
        monkeypatch, tmp_path, f"web:\n  {capability}_backend: {backend}\n  keyless_rescue: false\n",
    )
    resolver = registry.get_active_search_provider if capability == "search" else registry.get_active_extract_provider
    if capability == "search":
        result = json.loads(tools.web_search_tool("example"))
    else:
        async def safe_url(url):
            return True
        monkeypatch.setattr(tools, "async_is_safe_url", safe_url)
        result = json.loads(await tools.web_extract_tool(["https://example.invalid/page"]))
    assert result["success"] is False
    assert resolver() is None
    assert fallback.calls == 0


@pytest.mark.parametrize("capability", ["search", "extract"])
def test_unconfigured_install_keeps_capability_autodetection(monkeypatch, tmp_path, capability):
    registry, _, fallback = install_providers(monkeypatch, tmp_path, "web:\n  keyless_fallback: false\n")
    resolver = registry.get_active_search_provider if capability == "search" else registry.get_active_extract_provider
    assert resolver() is fallback
