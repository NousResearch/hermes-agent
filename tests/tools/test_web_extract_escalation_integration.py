"""Extraction escalation must preserve the registered tool's upstream controls."""

import asyncio
import json
import threading

import pytest

from agent.web_search_provider import WebSearchProvider


class ExtractProvider(WebSearchProvider):
    name = "integration-extract"

    def is_available(self):
        return True

    def supports_extract(self):
        return True


@pytest.fixture
def extract_context(tmp_path, monkeypatch):
    # Import discovery before registering the test providers, then exercise the
    # real tool registry, backend selection, config, dispatch, and disk cache.
    import model_tools  # noqa: F401
    from agent import web_search_registry
    from tools import web_tools
    from tools.registry import registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)

    async def public_url(_url):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", public_url)
    web_search_registry._reset_for_tests()

    def configure(provider, selected=None, timeout=120):
        web_search_registry.register_provider(provider)
        tmp_path.joinpath("config.yaml").write_text(
            json.dumps({"web": {
                "extract_backend": selected or provider.name,
                "extract_timeout": timeout,
                "cache_enabled": True,
            }}),
            encoding="utf-8",
        )

    yield configure, registry
    web_search_registry._reset_for_tests()


@pytest.mark.parametrize("provider_mode", ["sync", "async"])
@pytest.mark.parametrize("browser_outcome", ["recovered", "blocked"])
def test_registered_extract_escalates_and_caches_only_usable_content(
    extract_context, monkeypatch, provider_mode, browser_outcome
):
    from tools import browser_tool_cloud, web_routing

    configure, registry = extract_context
    primary_threads, browser_threads = [], []

    def blocked(urls, **kwargs):
        primary_threads.append(threading.get_ident())
        return [{"url": urls[0], "content": "captcha", "metadata": {"status_code": 403}}]

    class SyncProvider(ExtractProvider):
        def extract(self, urls, **kwargs):
            return blocked(urls, **kwargs)

    class AsyncProvider(ExtractProvider):
        async def extract(self, urls, **kwargs):
            return blocked(urls, **kwargs)

    class BrowserUseProvider:
        name = "browser-use"

        def __init__(self):
            self.created, self.closed = [], []

        def is_available(self):
            return True

        def create_session(self, task_id):
            self.created.append(task_id)
            return {"bb_session_id": "integration-session", "cdp_url": "wss://example.test/cdp"}

        def close_session(self, session_id):
            self.closed.append(session_id)
            return True

    cloud = BrowserUseProvider()

    async def allow_navigation(url, lane, **kwargs):
        return None

    async def browser_extract(url, cdp_url, lane):
        browser_threads.append(threading.get_ident())
        assert lane == "browser_use_cloud"
        if browser_outcome == "blocked":
            return {"url": url, "error": "browser challenge remains"}
        return {"url": url, "title": "Recovered", "content": "Recovered public page"}

    monkeypatch.setattr(browser_tool_cloud, "_get_cloud_provider", lambda: cloud)
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_navigation)
    monkeypatch.setattr(web_routing, "_extract_via_cdp", browser_extract)
    configure(SyncProvider() if provider_mode == "sync" else AsyncProvider())

    args = {"urls": ["https://public.example/recover"]}
    first = json.loads(registry.dispatch("web_extract", args))
    second = json.loads(registry.dispatch("web_extract", args))

    expected = "Recovered public page" if browser_outcome == "recovered" else "captcha"
    assert first["results"][0]["content"] == expected
    assert second["results"][0]["content"] == first["results"][0]["content"]
    expected_calls = 1 if browser_outcome == "recovered" else 2
    assert len(primary_threads) == len(browser_threads) == len(cloud.created) == expected_calls
    assert cloud.closed == ["integration-session"] * expected_calls
    assert all(
        (primary == browser) is (provider_mode == "async")
        for primary, browser in zip(primary_threads, browser_threads)
    )


@pytest.mark.parametrize("guard", ["unknown_selection", "search_only", "timeout"])
def test_registered_extract_preserves_selection_and_timeout_guards(
    extract_context, monkeypatch, guard
):
    from tools import web_routing

    configure, registry = extract_context
    called, cancelled, browser_calls = [], [], []

    class GuardProvider(ExtractProvider):
        def supports_extract(self):
            return guard != "search_only"

        async def extract(self, urls, **kwargs):
            called.append(urls)
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

    async def unexpected_browser(url, lanes):
        browser_calls.append(url)
        raise AssertionError("selection errors and transport timeouts must not create browser sessions")

    monkeypatch.setattr(web_routing, "_extract_via_browser_use_cloud", unexpected_browser)
    configure(
        GuardProvider(),
        selected="unregistered-selection" if guard == "unknown_selection" else None,
        timeout=0.02,
    )

    result = json.loads(registry.dispatch("web_extract", {"urls": ["https://public.example/guard"]}))

    if guard == "timeout":
        assert "timed out" in result["results"][0]["error"].lower()
        assert called == [["https://public.example/guard"]]
        assert cancelled == [True]
    else:
        assert result["success"] is False
        assert ("unregistered-selection" if guard == "unknown_selection" else "search-only") in result["error"]
        assert called == cancelled == []
    assert browser_calls == []
