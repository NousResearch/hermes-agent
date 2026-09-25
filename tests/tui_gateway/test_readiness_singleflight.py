"""Overlapping readiness polls must not saturate the shared RPC pool (#65151).

``setup.runtime_check`` / ``setup.status`` are Desktop-polled and execute on the
shared RPC executor (``_LONG_HANDLERS``). Before the single-flight, every
overlapping poll ran its own provider resolution while a slow one (blocked
keyring, OAuth refresh, GIL pressure) was still in flight — one shared worker
per poll, all resolving the same state, until unrelated RPCs starved.

These tests drive the REAL dispatch path (``server.dispatch`` → the shared
pool) with a recording transport and a blocking resolver. On the old behavior
the pool saturates and an unrelated long-handler RPC is never answered; with
the single-flight there is exactly one probe and the pool stays responsive.
"""

from __future__ import annotations

import threading
import time

from tui_gateway import server


class _RecordingTransport:
    """Collect worker-written responses; the tests wait on them by request id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self.responses = {}

    def write(self, response):
        with self._lock:
            self.responses[response.get("id")] = response
            self._changed.set()
        return True

    def wait_for(self, request_id, timeout=2.0):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                response = self.responses.get(request_id)
                if response is not None:
                    return response
                self._changed.clear()
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"timed out waiting for response {request_id!r}"
            self._changed.wait(timeout=remaining)


def _dispatch(transport, request_id, method, params=None):
    assert server.dispatch(
        {"id": request_id, "method": method, "params": params or {}}, transport) is None


def _patch_fast_probe_env(monkeypatch, resolve):
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("custom/m", None))


def test_overlapping_runtime_checks_share_one_probe_and_keep_pool_responsive(monkeypatch):
    """Same-key polls join the in-flight probe; the shared pool answers an unrelated RPC."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_resolve(requested=None, **kwargs):
        calls.append(requested)
        started.set()
        release.wait(timeout=10)
        return {"provider": "custom", "api_key": "no-key-required", "source": "config"}

    _patch_fast_probe_env(monkeypatch, slow_resolve)
    transport = _RecordingTransport()

    _dispatch(transport, "owner", "setup.runtime_check")
    assert started.wait(timeout=2)

    try:
        # Saturate every remaining shared RPC worker with overlapping polls.
        for index in range(1, server._rpc_pool_workers):
            _dispatch(transport, f"poll-{index}", "setup.runtime_check")

        # An unrelated long-handler RPC must still be answered promptly: with the
        # single-flight the overlapping polls joined instead of each running a
        # resolver, so they did not occupy the pool.
        monkeypatch.setitem(
            server._methods, "process.list",
            lambda rid, _params: server._ok(rid, {"processes": []}))
        before = time.monotonic()
        _dispatch(transport, "unrelated", "process.list", {"session_id": "x"})
        unrelated = transport.wait_for("unrelated", timeout=2)
        elapsed = time.monotonic() - before

        assert unrelated["result"] == {"processes": []}
        assert elapsed < 1.0, f"unrelated RPC waited {elapsed:.2f}s — the shared pool saturated"
        assert len(calls) == 1, "overlapping polls must not each run provider resolution"

        # An overlapping poll answered with the retryable unknown — a JSON-RPC
        # error, never a fabricated ok=False result.
        joiner = transport.wait_for("poll-1", timeout=2)
        assert "result" not in joiner
        assert joiner["error"]["code"] == server._READINESS_IN_PROGRESS_ERR
    finally:
        # Never leave pool workers blocked in the fake resolver, even on failure.
        release.set()

    owner = transport.wait_for("owner", timeout=3)
    assert owner["result"]["ok"] is True
    assert owner["result"]["provider"] == "custom"

    # The in-flight entry is cleared once the probe settles: a later poll
    # starts a fresh probe instead of reading a stale answer.
    deadline = time.monotonic() + 2
    while server._readiness_inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not server._readiness_inflight


def test_blocked_provider_does_not_serialize_a_healthy_provider_check(monkeypatch):
    """The single-flight key includes the requested provider; distinct providers probe in parallel."""
    provider_a_started = threading.Event()
    release_a = threading.Event()
    calls = {"provider-a": 0, "provider-b": 0}

    def resolve_by_provider(requested=None, **kwargs):
        calls[requested] = calls.get(requested, 0) + 1
        if requested == "provider-a":
            provider_a_started.set()
            release_a.wait(timeout=10)
        return {"provider": requested, "api_key": "no-key-required", "source": "config"}

    _patch_fast_probe_env(monkeypatch, resolve_by_provider)
    transport = _RecordingTransport()

    _dispatch(transport, "a", "setup.runtime_check", {"provider": "provider-a"})
    assert provider_a_started.wait(timeout=2)

    try:
        before = time.monotonic()
        _dispatch(transport, "b", "setup.runtime_check", {"provider": "provider-b"})
        provider_b = transport.wait_for("b", timeout=2)
        elapsed = time.monotonic() - before

        assert provider_b["result"]["ok"] is True
        assert provider_b["result"]["provider"] == "provider-b"
        assert elapsed < 1.0
        assert calls == {"provider-a": 1, "provider-b": 1}
    finally:
        release_a.set()
    provider_a = transport.wait_for("a", timeout=3)
    assert provider_a["result"]["ok"] is True
    assert provider_a["result"]["provider"] == "provider-a"


def test_probe_outliving_its_budget_answers_retryable_unknown_then_reprobes(monkeypatch):
    """A probe slower than the join budget answers the retryable error (unknown, not ok=False)
    while it keeps running; the next poll starts a fresh probe with the real answer."""
    monkeypatch.setattr(server, "_READINESS_SHARE_WAIT_SECONDS", 0.2)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_resolve(requested=None, **kwargs):
        calls.append(requested)
        started.set()
        release.wait(timeout=10)
        return {"provider": "custom", "api_key": "no-key-required", "source": "config"}

    _patch_fast_probe_env(monkeypatch, slow_resolve)
    transport = _RecordingTransport()

    _dispatch(transport, "first", "setup.runtime_check")
    assert started.wait(timeout=2)
    try:
        first = transport.wait_for("first", timeout=3)

        # Unknown readiness, expressed as an unanswered RPC — the desktop keeps the
        # last authoritative result and setup.status stays the credential source.
        assert "result" not in first
        assert first["error"]["code"] == server._READINESS_IN_PROGRESS_ERR
    finally:
        release.set()

    deadline = time.monotonic() + 2
    while server._readiness_inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not server._readiness_inflight

    _dispatch(transport, "second", "setup.runtime_check")
    second = transport.wait_for("second", timeout=3)
    assert second["result"]["ok"] is True
    assert second["result"]["provider"] == "custom"
    assert len(calls) == 2
