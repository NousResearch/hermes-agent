"""Regression tests for web_extract result ordering and URL identity matching (#127697).

Ensures that:
1. Out-of-order provider results are mapped to the correct requested URLs by identity.
2. Omitted results receive the typed _NO_RESULT_ERROR without shifting other entries.
3. Unexpected URLs returned by a provider are not misattributed to requested positions.
4. Redirected URLs with metadata.sourceURL match the original requested URL.
5. _extract_safe_urls preserves input order regardless of cache hits/misses.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import tools.web_tools as wt
from tools import web_tools_extract as wte
from tools.web_result_cache import extract_cache_put


def test_merge_in_order_reordered_results():
    """Providers returning successes out of order must be reordered by URL identity."""
    fetch_urls = ["https://a.example.com", "https://b.example.com"]
    results = [
        {"url": "https://b.example.com", "content": "Content B", "title": "B"},
        {"url": "https://a.example.com", "content": "Content A", "title": "A"},
    ]
    merged = wte._merge_in_order(2, {}, [0, 1], fetch_urls, results)
    assert len(merged) == 2
    assert merged[0]["url"] == "https://a.example.com"
    assert merged[0]["content"] == "Content A"
    assert merged[1]["url"] == "https://b.example.com"
    assert merged[1]["content"] == "Content B"


def test_merge_in_order_omitted_result_preserves_identity():
    """When a provider omits an entry, subsequent entries are not shifted."""
    fetch_urls = ["https://a.example.com", "https://b.example.com"]
    # Provider omitted 'a', returned only 'b'
    results = [
        {"url": "https://b.example.com", "content": "Content B", "title": "B"},
    ]
    merged = wte._merge_in_order(2, {}, [0, 1], fetch_urls, results)
    assert len(merged) == 2
    assert merged[0]["url"] == "https://a.example.com"
    assert merged[0]["error"] == wte._NO_RESULT_ERROR
    assert merged[1]["url"] == "https://b.example.com"
    assert merged[1]["content"] == "Content B"


def test_merge_in_order_unexpected_url_not_misattributed():
    """An unrequested URL returned by a provider must not be assigned to another requested URL."""
    fetch_urls = ["https://a.example.com", "https://b.example.com"]
    results = [
        {"url": "https://unrequested.example.com", "content": "Sneaky Content"},
        {"url": "https://b.example.com", "content": "Content B"},
    ]
    merged = wte._merge_in_order(2, {}, [0, 1], fetch_urls, results)
    assert len(merged) == 2
    assert merged[0]["url"] == "https://a.example.com"
    assert merged[0]["error"] == wte._NO_RESULT_ERROR
    assert merged[1]["url"] == "https://b.example.com"
    assert merged[1]["content"] == "Content B"


def test_merge_in_order_redirect_source_url():
    """A redirect URL with metadata.sourceURL matches the original requested URL."""
    fetch_urls = ["https://short.example.com"]
    results = [
        {
            "url": "https://destination.example.com/actual/path",
            "metadata": {"sourceURL": "https://short.example.com"},
            "content": "Redirected Page",
        }
    ]
    merged = wte._merge_in_order(1, {}, [0], fetch_urls, results)
    assert len(merged) == 1
    assert merged[0]["url"] == "https://destination.example.com/actual/path"
    assert merged[0]["content"] == "Redirected Page"


def test_merge_in_order_trailing_slash_normalization():
    """Trailing slash differences between request and response still match."""
    fetch_urls = ["https://example.com/dir/"]
    results = [
        {"url": "https://example.com/dir", "content": "Dir Content"}
    ]
    merged = wte._merge_in_order(1, {}, [0], fetch_urls, results)
    assert len(merged) == 1
    assert merged[0]["content"] == "Dir Content"


def test_merge_in_order_anonymous_results_fallback():
    """Results without url or metadata.sourceURL fall back to positional order if counts match."""
    fetch_urls = ["https://a.example.com", "https://b.example.com"]
    results = [
        {"content": "Content 0"},
        {"content": "Content 1"},
    ]
    merged = wte._merge_in_order(2, {}, [0, 1], fetch_urls, results)
    assert len(merged) == 2
    assert merged[0]["content"] == "Content 0"
    assert merged[1]["content"] == "Content 1"


def test_merge_in_order_duplicate_urls_consume_distinct_results():
    """Each duplicate requested position consumes one distinct provider row."""
    url = "https://dup.example.com"
    fetch_urls = [url, url]
    results = [
        {"url": url, "content": "First", "title": "First"},
        {"url": url, "content": "Second", "title": "Second"},
    ]
    merged = wte._merge_in_order(2, {}, [0, 1], fetch_urls, results)
    assert len(merged) == 2
    assert merged[0]["content"] == "First"
    assert merged[0].get("error") in (None, "")
    assert merged[1]["content"] == "Second"
    assert merged[1].get("error") in (None, "")


def test_merge_in_order_duplicate_urls_fewer_rows_than_positions():
    """3 duplicate positions with 2 provider rows: the third gets _NO_RESULT_ERROR, not a reuse."""
    url = "https://dup.example.com"
    fetch_urls = [url, url, url]
    results = [
        {"url": url, "content": "First"},
        {"url": url, "content": "Second"},
    ]
    merged = wte._merge_in_order(3, {}, [0, 1, 2], fetch_urls, results)
    assert len(merged) == 3
    assert merged[0]["content"] == "First"
    assert merged[1]["content"] == "Second"
    # Intended semantics: one provider row per requested position; shortage is an
    # explicit miss, never a reuse of the last row.
    assert merged[2]["url"] == url
    assert merged[2]["content"] == ""
    assert merged[2]["error"] == wte._NO_RESULT_ERROR
    assert merged[2] != merged[1]


def test_merge_in_order_duplicate_urls_single_row():
    """2 duplicate positions with 1 provider row: the second gets _NO_RESULT_ERROR."""
    url = "https://dup.example.com"
    fetch_urls = [url, url]
    results = [
        {"url": url, "content": "Only"},
    ]
    merged = wte._merge_in_order(2, {}, [0, 1], fetch_urls, results)
    assert len(merged) == 2
    assert merged[0]["content"] == "Only"
    assert merged[1]["url"] == url
    assert merged[1]["content"] == ""
    assert merged[1]["error"] == wte._NO_RESULT_ERROR


def test_merge_in_order_duplicate_urls_interleaved_with_distinct():
    """Duplicate positions interleaved with another URL still consume per-URL rows independently."""
    dup = "https://dup.example.com"
    other = "https://other.example.com"
    fetch_urls = [dup, other, dup]
    results = [
        {"url": dup, "content": "Dup Only"},
        {"url": other, "content": "Other"},
    ]
    merged = wte._merge_in_order(3, {}, [0, 1, 2], fetch_urls, results)
    assert len(merged) == 3
    assert merged[0]["content"] == "Dup Only"
    assert merged[1]["content"] == "Other"
    assert merged[2]["url"] == dup
    assert merged[2]["error"] == wte._NO_RESULT_ERROR


@pytest.mark.asyncio
async def test_extract_safe_urls_reordered_without_cache(tmp_path, monkeypatch):
    """Even with zero cache hits, _extract_safe_urls must return results in safe_urls order."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _ReorderingProvider:
        name = "mock-reordering"

        async def extract(self, urls, format=None):
            # Reverse order of requested urls
            return [
                {"url": u, "content": f"Content for {u}", "title": f"Title {u}"}
                for u in reversed(urls)
            ]

    provider = _ReorderingProvider()
    safe_urls = [
        "https://first.example.com",
        "https://second.example.com",
        "https://third.example.com",
    ]
    results = await wte._extract_safe_urls(provider, safe_urls, format="markdown")
    assert len(results) == 3
    assert [r["url"] for r in results] == safe_urls
    assert [r["content"] for r in results] == [f"Content for {u}" for u in safe_urls]


@pytest.mark.asyncio
async def test_extract_safe_urls_with_partial_cache_and_reorder(tmp_path, monkeypatch):
    """Partial cache hits combined with out-of-order backend results merge in correct sequence."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Pre-populate cache for the middle URL
    extract_cache_put(
        "https://second.example.com",
        "Cached Content 2",
        "Cached Title 2",
        format="markdown",
        provider="mock-provider",
    )

    class _PartialReorderProvider:
        name = "mock-provider"

        async def extract(self, urls, format=None):
            # Backend only receives first and third, returns them reversed
            assert urls == ["https://first.example.com", "https://third.example.com"]
            return [
                {"url": "https://third.example.com", "content": "Fresh Content 3"},
                {"url": "https://first.example.com", "content": "Fresh Content 1"},
            ]

    provider = _PartialReorderProvider()
    safe_urls = [
        "https://first.example.com",
        "https://second.example.com",
        "https://third.example.com",
    ]
    results = await wte._extract_safe_urls(provider, safe_urls, format="markdown")
    assert len(results) == 3
    assert results[0]["url"] == "https://first.example.com"
    assert results[0]["content"] == "Fresh Content 1"
    assert results[1]["url"] == "https://second.example.com"
    assert results[1]["content"] == "Cached Content 2"
    assert results[2]["url"] == "https://third.example.com"
    assert results[2]["content"] == "Fresh Content 3"


@pytest.mark.asyncio
async def test_web_extract_tool_with_invalid_and_reordered(tmp_path, monkeypatch):
    """End-to-end web_extract_tool with invalid input URL and reordering provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _StubProvider:
        name = "stub"

        def supports_extract(self):
            return True

        async def extract(self, urls, format=None):
            return [
                {"url": urls[1], "content": "Content B"},
                {"url": urls[0], "content": "Content A"},
            ]

    monkeypatch.setattr(wt, "_get_extract_backend", lambda: "stub")
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "async_is_safe_url", AsyncMock(return_value=True))
    monkeypatch.setattr(wt, "_resolve_extract_provider", lambda backend: (_StubProvider(), None))

    urls = [
        {"not_a_valid": "item"},
        "https://a.example.com",
        "https://b.example.com",
    ]
    raw = await wt.web_extract_tool(urls=urls)
    parsed = json.loads(raw)
    res = parsed["results"]
    assert len(res) == 3
    assert "Invalid URL item at index 0" in res[0]["error"]
    assert res[1]["url"] == "https://a.example.com"
    assert res[1]["content"] == "Content A"
    assert res[2]["url"] == "https://b.example.com"
    assert res[2]["content"] == "Content B"
