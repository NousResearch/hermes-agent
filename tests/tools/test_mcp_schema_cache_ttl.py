"""SEP-2549 schema-cache TTL expiry (tools/mcp_schema_cache.py)."""

import time

import pytest

from tools import mcp_schema_cache as sc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_cache_path", lambda: tmp_path / "cache.json")
    yield


def test_entry_without_ttl_never_expires():
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}])
    assert sc.get_cached_entry("srv", "fp") is not None


def test_entry_with_zero_ttl_never_expires(monkeypatch):
    """A ``ttl_ms`` of 0 is SEP-2549's "no hint" and must not expire the
    entry on every read — every current server reports 0, so expiring would
    defeat ``lazy`` mode entirely (#101007)."""
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=0)
    entry = sc.get_cached_entry("srv", "fp")
    assert entry is not None
    assert "written_at" not in entry, "0-TTL hints should not be persisted"
    real_time = time.time
    monkeypatch.setattr(sc.time, "time", lambda: real_time() + 3600.0)
    assert sc.get_cached_entry("srv", "fp") is not None


def test_legacy_zero_ttl_cache_row_still_served(tmp_path, monkeypatch):
    """Caches written before the fix persist ``ttl_ms: 0`` with a
    ``written_at``; those rows must read back as hits, not expire."""
    monkeypatch.setattr(sc, "_cache_path", lambda: tmp_path / "cache.json")
    sc._save_all(
        {
            "srv": {
                "fingerprint": "fp",
                "tools": [{"name": "t"}],
                "ttl_ms": 0,
                "written_at": 1.0,
            }
        }
    )
    assert sc.get_cached_entry("srv", "fp") is not None


def test_entry_within_ttl_served():
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    entry = sc.get_cached_entry("srv", "fp")
    assert entry is not None
    assert entry["ttl_ms"] == 60_000
    assert "written_at" in entry


def test_entry_past_ttl_is_a_miss(monkeypatch):
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=1_000)
    real_time = time.time
    monkeypatch.setattr(sc.time, "time", lambda: real_time() + 2.0)
    assert sc.get_cached_entry("srv", "fp") is None


def test_ttl_rewrite_advances_written_at():
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    first = sc.get_cached_entry("srv", "fp")["written_at"]
    time.sleep(0.01)
    # Identical payload would previously short-circuit; TTL'd entries must
    # rewrite so written_at advances on every live reconfirmation.
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    second = sc.get_cached_entry("srv", "fp")["written_at"]
    assert second > first


def test_cache_scope_round_trips():
    sc.write_cache_entry(
        "srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000, cache_scope="private"
    )
    assert sc.get_cached_entry("srv", "fp")["cache_scope"] == "private"
