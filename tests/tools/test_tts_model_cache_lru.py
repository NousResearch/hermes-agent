"""LRU bound on the Piper/KittenTTS model caches.

Each cached entry is a whole loaded TTS model (tens of MB). Without a cap the
caches pinned one model per distinct voice/model for the process lifetime, so a
surface that sweeps voices grew memory with no ceiling. `_tts_cache_get_or_load`
loads on a miss and evicts the least-recently-used entry beyond the cap.
"""
from __future__ import annotations

import tools.tts_tool as tts


def test_loads_on_miss_and_serves_from_cache_on_hit():
    cache: dict = {}
    calls = []

    def load():
        calls.append(1)
        return "model"

    assert tts._tts_cache_get_or_load(cache, "a", load) == "model"
    assert tts._tts_cache_get_or_load(cache, "a", load) == "model"
    assert len(calls) == 1  # loaded once, second call served from cache


def test_evicts_least_recently_used_beyond_cap(monkeypatch):
    monkeypatch.setattr(tts, "_TTS_MODEL_CACHE_MAX", 2)
    cache: dict = {}
    for k in ("a", "b", "c"):
        tts._tts_cache_get_or_load(cache, k, lambda k=k: k)
    assert set(cache) == {"b", "c"}  # "a", the oldest, was evicted
    assert len(cache) == 2


def test_hit_refreshes_recency_so_eviction_is_lru_not_fifo(monkeypatch):
    monkeypatch.setattr(tts, "_TTS_MODEL_CACHE_MAX", 2)
    cache: dict = {}
    tts._tts_cache_get_or_load(cache, "a", lambda: "a")
    tts._tts_cache_get_or_load(cache, "b", lambda: "b")
    # Touch "a": "b" is now the least recently used.
    tts._tts_cache_get_or_load(cache, "a", lambda: "a")
    tts._tts_cache_get_or_load(cache, "c", lambda: "c")
    assert "b" not in cache
    assert set(cache) == {"a", "c"}
