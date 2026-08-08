"""Wall-clock cap on the web_extract provider dispatch.

``web_extract_tool`` awaited ``provider.extract()`` unbounded: the HTTP
timeouts inside providers cap a single request, but a hung stream or a
provider that retries internally can stall far past them, hanging the tool
(and any idle-limited caller above it) indefinitely. The dispatch is now
capped by ``web.extract_timeout`` (default
``tools.web_tools.DEFAULT_EXTRACT_TIMEOUT_S``) and degrades to per-URL
structured timeout errors instead of hanging.
"""

import asyncio
import json
import time

import pytest

from agent import web_search_registry
from agent.web_search_provider import WebSearchProvider
from tools import web_tools

_URLS = ["https://example.com/a", "https://example.org/b"]


class _BaseFakeProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "extract-timeout-test"

    @property
    def display_name(self) -> str:
        return "Extract Timeout Test"

    def is_available(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True


class _HangingAsyncProvider(_BaseFakeProvider):
    """Async extract() that hangs far past the cap (cancelled by wait_for)."""

    async def extract(self, urls, **kwargs):
        await asyncio.sleep(30)
        return [{"url": u, "title": "", "content": "SHOULD-NOT-RETURN"} for u in urls]


class _HangingSyncProvider(_BaseFakeProvider):
    """Sync extract() dispatched via asyncio.to_thread.

    The sleep is deliberately short: wait_for abandons the worker thread but
    cannot kill it, so a long sleep here would stall suite teardown.  2s is
    still 40x the cap the tests configure.
    """

    def extract(self, urls, **kwargs):
        time.sleep(2)
        return [{"url": u, "title": "", "content": "SHOULD-NOT-RETURN"} for u in urls]


class _FastProvider(_BaseFakeProvider):
    async def extract(self, urls, **kwargs):
        return [{"url": u, "title": "T", "content": "real content"} for u in urls]


def _install(monkeypatch, provider, web_config):
    with web_search_registry._lock:
        previous = dict(web_search_registry._providers)
        web_search_registry._providers.clear()
    web_search_registry.register_provider(provider)
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: web_config)

    async def _safe(_url):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe)
    return previous


def _restore(previous):
    with web_search_registry._lock:
        web_search_registry._providers.clear()
        web_search_registry._providers.update(previous)


@pytest.fixture
def hanging_async_provider(monkeypatch):
    provider = _HangingAsyncProvider()
    previous = _install(
        monkeypatch,
        provider,
        {"extract_backend": provider.name, "extract_timeout": 0.05},
    )
    yield provider
    _restore(previous)


@pytest.fixture
def hanging_sync_provider(monkeypatch):
    provider = _HangingSyncProvider()
    previous = _install(
        monkeypatch,
        provider,
        {"extract_backend": provider.name, "extract_timeout": 0.05},
    )
    yield provider
    _restore(previous)


@pytest.fixture
def fast_provider(monkeypatch):
    provider = _FastProvider()
    previous = _install(
        monkeypatch, provider, {"extract_backend": provider.name}
    )
    yield provider
    _restore(previous)


def _assert_structured_timeouts(raw: str):
    result = json.loads(raw)
    assert [e["url"] for e in result["results"]] == _URLS
    for entry in result["results"]:
        assert entry["content"] == ""
        assert "timed out" in entry["error"]
    assert "SHOULD-NOT-RETURN" not in raw


@pytest.mark.asyncio
async def test_hanging_async_provider_returns_structured_timeouts(
    hanging_async_provider,
):
    # Outer guard: if the cap is not honored this raises instead of hanging CI.
    raw = await asyncio.wait_for(web_tools.web_extract_tool(_URLS), timeout=5.0)
    _assert_structured_timeouts(raw)


@pytest.mark.asyncio
async def test_hanging_sync_provider_returns_structured_timeouts(
    hanging_sync_provider,
):
    raw = await asyncio.wait_for(web_tools.web_extract_tool(_URLS), timeout=5.0)
    _assert_structured_timeouts(raw)


@pytest.mark.asyncio
async def test_fast_provider_unaffected_by_default_cap(fast_provider):
    result = json.loads(await web_tools.web_extract_tool(_URLS))
    assert [e["url"] for e in result["results"]] == _URLS
    for entry in result["results"]:
        assert entry["content"] == "real content"
        assert entry["error"] is None


def test_extract_timeout_config(monkeypatch):
    # Default when the key is absent.
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
    assert web_tools._get_extract_timeout_s() == web_tools.DEFAULT_EXTRACT_TIMEOUT_S

    # Explicit override wins.
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"extract_timeout": 7}
    )
    assert web_tools._get_extract_timeout_s() == 7.0

    # Garbage and non-positive values fall back to the default.
    for bad in ("nope", None, 0, -3):
        monkeypatch.setattr(
            web_tools, "_load_web_config", lambda bad=bad: {"extract_timeout": bad}
        )
        assert (
            web_tools._get_extract_timeout_s()
            == web_tools.DEFAULT_EXTRACT_TIMEOUT_S
        )
