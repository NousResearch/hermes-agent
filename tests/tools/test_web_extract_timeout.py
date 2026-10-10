"""Provider deadlines use native dispatch without real time in the test cases."""
from __future__ import annotations

import asyncio

import pytest

from tools import web_tools_extract as wte


class _AsyncProvider:
    name = "fixture-async"

    async def extract(self, urls, format=None):
        return [{"url": url, "content": "ok"} for url in urls]


class _SyncProvider:
    name = "fixture-sync"

    def extract(self, urls, format=None):
        return [{"url": url, "content": "ok"} for url in urls]


@pytest.fixture(autouse=True)
def controls(monkeypatch):
    async def allowed(url):
        return None

    monkeypatch.setattr(wte, "_page_controls", allowed)
    monkeypatch.setattr(wte, "_rescue_eligible", lambda provider, capability=None: False)


@pytest.mark.parametrize("provider", [_AsyncProvider(), _SyncProvider()], ids=["async", "sync-to-thread"])
def test_dispatch_deadline_returns_typed_per_url_timeout(monkeypatch, provider):
    calls = []

    class ExpiredDeadline:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise TimeoutError()

        def expired(self):
            return True

    def deadline(timeout):
        calls.append(timeout)
        return ExpiredDeadline()

    monkeypatch.setattr(wte, "_extract_timeout_seconds", lambda: 12.0)
    monkeypatch.setattr(wte.asyncio, "timeout", deadline)
    urls = ["https://example.com/a", "https://example.com/b"]
    results = asyncio.run(wte._dispatch_extract(provider, urls, None))
    assert calls == [12.0]
    assert results == [{"url": url, "title": "", "content": "", "error": f"Extract timed out after 12s via {provider.name}",
                        "failure": {"kind": "timeout", "retry": "transient", "scope": "provider", "timeout_seconds": 12.0}}
                       for url in urls]


def test_timeout_zero_disables_the_cap(monkeypatch):
    def unexpected(timeout):
        pytest.fail("A disabled deadline must not create a timeout scope")

    monkeypatch.setattr(wte, "_extract_timeout_seconds", lambda: 0.0)
    monkeypatch.setattr(wte.asyncio, "timeout", unexpected)
    results = asyncio.run(wte._dispatch_extract(_AsyncProvider(), ["https://example.com/x"], None))
    assert results == [{"url": "https://example.com/x", "requested_url": "https://example.com/x", "title": "", "content": "ok",
                        "raw_content": "ok", "coverage": "unknown", "error": None}]
