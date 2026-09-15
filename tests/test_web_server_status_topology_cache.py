"""Regression tests for the /api/status profile-topology cache.

The desktop app polls /api/status ~1/s while waiting for the backend to become
ready. Before the cache, every poll ran a full _collect_profile_gateway_topology
scan (per-profile yaml.safe_load with the pure-Python loader + psutil
process-table probes + realpath walks) in the default executor; on multi-profile
installs the concurrent scans held the GIL for 14-16s and starved the event
loop, so the desktop WS never received gateway.ready and boot escalated to the
"Hermes couldn't start" overlay (#60800).
"""

import threading
import time

from hermes_cli import web_server


def _reset_cache():
    web_server._TOPOLOGY_CACHE["ts"] = 0.0
    web_server._TOPOLOGY_CACHE["data"] = None
    web_server._TOPOLOGY_CACHE["fn"] = None


def _fake_topology(calls, delay=0.0):
    def _collect():
        if delay:
            time.sleep(delay)
        calls.append(1)
        return {"profiles": ["default"], "gateway_mode": "single", "gateways": []}

    return _collect


def test_topology_cache_returns_cached_result_within_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(
        web_server, "_collect_profile_gateway_topology", _fake_topology(calls)
    )
    _reset_cache()
    try:
        first = web_server._collect_profile_gateway_topology_cached()
        second = web_server._collect_profile_gateway_topology_cached()
    finally:
        _reset_cache()

    assert len(calls) == 1
    assert first is second


def test_topology_cache_rescans_after_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(
        web_server, "_collect_profile_gateway_topology", _fake_topology(calls)
    )
    _reset_cache()
    try:
        web_server._collect_profile_gateway_topology_cached()
        # Age the cache entry past the TTL instead of sleeping through it.
        web_server._TOPOLOGY_CACHE["ts"] -= web_server._TOPOLOGY_CACHE_TTL + 1.0
        web_server._collect_profile_gateway_topology_cached()
    finally:
        _reset_cache()

    assert len(calls) == 2


def test_topology_cache_collapses_concurrent_scans(monkeypatch):
    """Concurrent status polls must not each run their own scan — that pile-up
    is exactly the GIL storm the cache exists to prevent."""
    calls = []
    monkeypatch.setattr(
        web_server,
        "_collect_profile_gateway_topology",
        _fake_topology(calls, delay=0.05),
    )
    _reset_cache()
    results = []
    try:
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    web_server._collect_profile_gateway_topology_cached()
                )
            )
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        _reset_cache()

    assert len(calls) == 1
    assert len(results) == 8
    assert all(r == results[0] for r in results)
def test_topology_cache_misses_when_collector_is_swapped(monkeypatch):
    """Tests (and hot-reload scenarios) monkeypatch the collector; a swapped
    function identity must be a cache miss so stale data from the previous
    collector never leaks across the swap."""
    calls_a, calls_b = [], []
    monkeypatch.setattr(
        web_server, "_collect_profile_gateway_topology", _fake_topology(calls_a)
    )
    _reset_cache()
    try:
        first = web_server._collect_profile_gateway_topology_cached()
        monkeypatch.setattr(
            web_server, "_collect_profile_gateway_topology", _fake_topology(calls_b)
        )
        second = web_server._collect_profile_gateway_topology_cached()
    finally:
        _reset_cache()

    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert first is not second
