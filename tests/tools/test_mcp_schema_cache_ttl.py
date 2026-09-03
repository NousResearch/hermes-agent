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


def test_boolean_ttl_is_not_treated_as_a_number():
    """``bool`` subclasses ``int``, so a bare isinstance would accept it.

    ``ttl_ms=False`` would become a 0 ms TTL that expires the entry the
    instant it is written, and ``True`` a 1 ms one — the exact never-cache
    failure this module's TTL handling exists to avoid.
    """
    for bogus in (True, False):
        sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=bogus)
        entry = sc.get_cached_entry("srv", "fp")
        assert entry is not None
        assert "ttl_ms" not in entry
        assert "written_at" not in entry
        sc.clear_cache_entry("srv")


def test_boolean_written_at_on_disk_does_not_expire_the_entry(monkeypatch):
    """A cache file carrying a bogus ``written_at`` must not silently expire.

    The entry is re-read from disk, which this module does not own — an older
    version or a hand edit can put anything there. ``written_at=True`` reads
    as epoch+1s under a bare isinstance, making every entry look ancient.
    """
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    data = sc._load_all()
    data["srv"]["written_at"] = True
    sc._save_all(data)
    assert sc.get_cached_entry("srv", "fp") is not None
