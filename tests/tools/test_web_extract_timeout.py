"""web_extract provider dispatch must be wall-clock bounded (#57155, salvage #57180).

A backend that keeps the response open without finishing (hanging HTTP server,
stuck SDK) must produce per-URL timeout errors instead of stalling the tool
call — and the event loop — indefinitely.
"""
from __future__ import annotations

import asyncio

import pytest

from tools import web_tools_extract as wte


class _HangingAsyncProvider:
    name = "hanging-async"

    async def extract(self, urls, format=None):
        await asyncio.sleep(9999)


class _HangingSyncProvider:
    name = "hanging-sync"

    def extract(self, urls, format=None):
        import time

        # Longer than the patched 0.2s cap, short enough that asyncio.run's
        # executor-join at loop shutdown doesn't hang the test.
        time.sleep(2)


@pytest.mark.parametrize("provider", [_HangingAsyncProvider(), _HangingSyncProvider()],
                         ids=["async", "sync-to-thread"])
def test_hanging_provider_returns_per_url_timeout_errors(monkeypatch, provider):
    monkeypatch.setattr(wte, "_extract_timeout_seconds", lambda: 0.2)
    monkeypatch.setattr(wte, "_rescue_eligible", lambda p: False)
    urls = ["https://example.com/a", "https://example.com/b"]
    results = asyncio.run(wte._dispatch_extract(provider, urls, None))
    assert [r["url"] for r in results] == urls
    for r in results:
        assert "timed out" in r["error"].lower()
        assert provider.name in r["error"]


def test_timeout_zero_disables_the_cap(monkeypatch):
    class _FastProvider:
        name = "fast"

        async def extract(self, urls, format=None):
            return [{"url": u, "content": "ok"} for u in urls]

    monkeypatch.setattr(wte, "_extract_timeout_seconds", lambda: 0.0)
    results = asyncio.run(wte._dispatch_extract(_FastProvider(), ["https://example.com/x"], None))
    assert results[0]["content"] == "ok"
