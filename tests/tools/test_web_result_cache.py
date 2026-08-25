"""Tests for tools/web_result_cache.py — TTL memo for web_search and the
disk-backed extract cache, plus their wiring into web_tools.

The cache sits AFTER safety gates and around the paid vendor call only, so
these tests focus on: hit/miss semantics, TTL expiry, limit bucketing +
slicing, single-flight coalescing, error non-caching, the disable flag, and
extract index integrity (tamper = miss, oversized = not indexed).
"""

import json
import threading
import time

import pytest

import tools.web_result_cache as wrc
from tools.web_result_cache import (
    SearchMemo,
    bucket_limit,
    extract_cache_get,
    extract_cache_put,
    normalize_query,
    slice_search_response,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the extract cache at a temp dir and force cache-on defaults."""
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", lambda: {})
    yield cache_dir


def _ok_response(n=10):
    return {
        "success": True,
        "data": {"web": [
            {"title": f"t{i}", "url": f"https://e.com/{i}", "description": "d"}
            for i in range(n)
        ]},
    }


# ── bucketing / normalization ────────────────────────────────────────────

def test_bucket_limit_rounds_up():
    assert bucket_limit(1) == 10
    assert bucket_limit(10) == 10
    assert bucket_limit(11) == 20
    assert bucket_limit(50) == 50
    assert bucket_limit(99) == 100
    assert bucket_limit(500) == 100


def test_normalize_query_folds_case_and_whitespace():
    assert normalize_query("  Weather  in\tVegas ") == "weather in vegas"


def test_slice_search_response_trims_to_requested_limit():
    sliced = slice_search_response(_ok_response(10), 3)
    assert len(sliced["data"]["web"]) == 3
    # original untouched (defensive copy)
    assert len(_ok_response(10)["data"]["web"]) == 10


# ── search memo ──────────────────────────────────────────────────────────

def test_search_memo_hit_within_ttl():
    memo = SearchMemo()
    memo.store("firecrawl", "weather in vegas", 5, _ok_response())
    hit = memo.lookup("firecrawl", "Weather In Vegas", 8)  # same bucket (10)
    assert hit is not None and hit["success"]


def test_search_memo_miss_across_providers_and_buckets():
    memo = SearchMemo()
    memo.store("firecrawl", "q", 5, _ok_response())
    assert memo.lookup("tavily", "q", 5) is None          # different provider
    assert memo.lookup("firecrawl", "q", 15) is None      # different bucket
    assert memo.lookup("firecrawl", "other", 5) is None   # different query


def test_search_memo_expires_after_ttl(monkeypatch):
    memo = SearchMemo()
    memo.store("firecrawl", "q", 5, _ok_response())
    monkeypatch.setattr(wrc, "ttl_seconds", lambda: 0.0)
    # store used the old TTL; force expiry by faking monotonic forward
    real = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real() + 100 * 3600)
    assert memo.lookup("firecrawl", "q", 5) is None


def test_search_memo_never_caches_failures():
    memo = SearchMemo()
    memo.store("firecrawl", "q", 5, {"success": False, "error": "boom"})
    assert memo.lookup("firecrawl", "q", 5) is None


def test_search_memo_disabled_by_config(monkeypatch):
    monkeypatch.setattr(wrc, "_web_config", lambda: {"cache_enabled": False})
    memo = SearchMemo()
    memo.store("firecrawl", "q", 5, _ok_response())
    assert memo.lookup("firecrawl", "q", 5) is None


def test_search_memo_hit_returns_copy():
    memo = SearchMemo()
    memo.store("firecrawl", "q", 5, _ok_response())
    first = memo.lookup("firecrawl", "q", 5)
    first["data"]["web"].clear()
    second = memo.lookup("firecrawl", "q", 5)
    assert len(second["data"]["web"]) == 10


def test_single_flight_coalesces_concurrent_identical_queries():
    """Two threads race the same query: exactly one paid call happens."""
    memo = SearchMemo()
    calls = []
    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait()
        resp = memo.lookup("p", "q", 5)
        if resp is None:
            with memo.flight_lock("p", "q", 5):
                resp = memo.lookup("p", "q", 5)
                if resp is None:
                    calls.append(1)          # the "paid" request
                    time.sleep(0.05)          # widen the race window
                    resp = _ok_response()
                    memo.store("p", "q", 5, resp)
        results.append(resp)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, "concurrent identical queries must share one request"
    assert len(results) == 2 and all(r["success"] for r in results)


# ── extract cache ────────────────────────────────────────────────────────

def _put_and_get(url="https://example.com/a", content="hello world",
                 fmt=None, **kw):
    extract_cache_put(url, content, title="T", format=fmt, **kw)
    return extract_cache_get(url, format=fmt)


def test_extract_cache_roundtrip(monkeypatch, _isolated_cache):
    # _store_full_text writes to the real hermes dir; redirect it here.
    stored = {}

    def fake_store(url, content):
        p = _isolated_cache / "page.md"
        p.write_text(content, encoding="utf-8")
        stored["path"] = str(p)
        return str(p)

    import tools.web_tools as wt
    monkeypatch.setattr(wt, "_store_full_text", fake_store)

    hit = _put_and_get()
    assert hit is not None
    assert hit["content"] == "hello world"
    assert hit["title"] == "T"
    assert hit["cached"] is True


def test_extract_cache_expired_entry_is_miss(monkeypatch, _isolated_cache):
    import tools.web_tools as wt
    p = _isolated_cache / "page.md"
    p.write_text("x", encoding="utf-8")
    monkeypatch.setattr(wt, "_store_full_text", lambda u, c: str(p))
    extract_cache_put("https://e.com", "x")
    monkeypatch.setattr(wrc, "ttl_seconds", lambda: 0.0)
    assert extract_cache_get("https://e.com") is None


def test_extract_cache_format_participates_in_key(monkeypatch, _isolated_cache):
    import tools.web_tools as wt
    p = _isolated_cache / "page.md"
    p.write_text("md", encoding="utf-8")
    monkeypatch.setattr(wt, "_store_full_text", lambda u, c: str(p))
    extract_cache_put("https://e.com", "md", format="markdown")
    assert extract_cache_get("https://e.com", format="html") is None
    assert extract_cache_get("https://e.com", format="markdown") is not None


def test_extract_cache_oversized_page_not_indexed(monkeypatch, _isolated_cache):
    import tools.web_tools as wt
    monkeypatch.setattr(
        wt, "_store_full_text",
        lambda u, c: str(_isolated_cache / "should-not-happen.md"),
    )
    big = "x" * (wt.MAX_STORED_TEXT_CHARS + 1)
    extract_cache_put("https://big.com", big)
    assert extract_cache_get("https://big.com") is None


def test_extract_cache_tampered_index_path_is_miss(_isolated_cache, tmp_path):
    """An index entry pointing outside cache/web must never be read."""
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    index = {
        wrc._url_digest("https://evil.com", None): {
            "url": "https://evil.com",
            "file": str(outside),
            "title": "",
            "fetched_at": time.time(),
        }
    }
    (_isolated_cache / wrc._INDEX_FILENAME).write_text(json.dumps(index))
    assert extract_cache_get("https://evil.com") is None


def test_extract_cache_missing_file_is_miss(_isolated_cache):
    index = {
        wrc._url_digest("https://gone.com", None): {
            "url": "https://gone.com",
            "file": str(_isolated_cache / "pruned.md"),
            "title": "",
            "fetched_at": time.time(),
        }
    }
    (_isolated_cache / wrc._INDEX_FILENAME).write_text(json.dumps(index))
    assert extract_cache_get("https://gone.com") is None


def test_extract_cache_corrupt_index_is_empty(_isolated_cache):
    (_isolated_cache / wrc._INDEX_FILENAME).write_text("{not json")
    assert extract_cache_get("https://any.com") is None


def test_extract_cache_disabled_by_config(monkeypatch, _isolated_cache):
    import tools.web_tools as wt
    p = _isolated_cache / "page.md"
    p.write_text("x", encoding="utf-8")
    monkeypatch.setattr(wt, "_store_full_text", lambda u, c: str(p))
    extract_cache_put("https://e.com", "x")
    monkeypatch.setattr(wrc, "_web_config", lambda: {"cache_enabled": False})
    assert extract_cache_get("https://e.com") is None


def test_index_eviction_keeps_newest(monkeypatch, _isolated_cache):
    monkeypatch.setattr(wrc, "_INDEX_MAX_ENTRIES", 3)
    now = time.time()
    index = {
        f"digest{i}": {"url": f"u{i}", "file": "f", "fetched_at": now + i}
        for i in range(6)
    }
    wrc._save_index(index)
    saved = json.loads((_isolated_cache / wrc._INDEX_FILENAME).read_text())
    assert len(saved) == 3
    assert set(saved) == {"digest3", "digest4", "digest5"}


def test_ttl_clamping(monkeypatch):
    monkeypatch.setattr(wrc, "_web_config", lambda: {"cache_ttl_minutes": 0})
    assert wrc.ttl_seconds() == 60.0          # floor 1 minute
    monkeypatch.setattr(wrc, "_web_config", lambda: {"cache_ttl_minutes": 99999})
    assert wrc.ttl_seconds() == 1440 * 60.0   # ceiling 24h
    monkeypatch.setattr(wrc, "_web_config", lambda: {"cache_ttl_minutes": "bogus"})
    assert wrc.ttl_seconds() == 20 * 60.0     # default on garbage
