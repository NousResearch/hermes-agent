"""Tests for MCP discovery synchronization in TUI agent build.

Validates that MCP tools are present in the agent's tool snapshot after
``_start_agent_build`` + ``_make_agent`` complete, eliminating the race
condition where the agent builds before background discovery finishes.

Fixes: #47121, #61891, #41625
"""

import threading
import time
import types

import pytest

from hermes_cli import mcp_startup


def test_mcp_discovery_timeout_default_is_10s():
    """Verify mcp_discovery_timeout default is >= 10s to cover cold-start HTTP MCPs."""
    import hermes_cli.config_defaults as cfg_defaults

    actual = cfg_defaults.DEFAULT_CONFIG.get("mcp_discovery_timeout", 1.5)
    assert actual >= 10.0, \
        f"mcp_discovery_timeout default is {actual}, expected >= 10.0"


def test_wait_for_mcp_discovery_no_thread_returns_instantly(monkeypatch):
    """When no discovery thread exists, wait_for_mcp_discovery returns ~immediately."""
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery()
    elapsed = time.time() - t0

    assert elapsed < 0.2, f"Blocked for {elapsed:.1f}s with no thread"


def test_wait_for_mcp_discovery_respects_timeout_bound(monkeypatch):
    """wait_for_mcp_discovery must not hang longer than the configured bound."""
    class SlowThread:
        def __init__(self):
            self._event = threading.Event()

        def is_alive(self):
            return not self._event.is_set()

        def join(self, timeout=None):
            self._event.wait(timeout=timeout)

    slow_thread = SlowThread()
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", slow_thread)

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery(timeout=0.5)
    elapsed = time.time() - t0

    assert elapsed < 2.0, f"wait_for_mcp_discovery blocked too long: {elapsed:.1f}s"


def test_wait_for_mcp_discovery_waits_for_completion(monkeypatch):
    """When discovery completes within timeout, wait_for_mcp_discovery returns promptly."""
    class FastThread:
        def __init__(self):
            self._done = threading.Event()

        def is_alive(self):
            return not self._done.is_set()

        def join(self, timeout=None):
            self._done.set()

    fast_thread = FastThread()
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", fast_thread)

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery(timeout=5.0)
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"Did not return promptly after completion: {elapsed:.1f}s"


def test_ensure_mcp_discovery_calls_start_then_wait(monkeypatch):
    """ensure_mcp_discovery_before_agent_build starts discovery and waits for it."""
    calls = []

    def fake_start(*, logger, thread_name):
        calls.append("start")

    def fake_wait(*, timeout=None, single_query=False):
        calls.append("wait")

    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", fake_start)
    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", fake_wait)

    mcp_startup.ensure_mcp_discovery_before_agent_build(
        logger=types.SimpleNamespace(debug=lambda *a, **k: None,
                                     warning=lambda *a, **k: None),
        timeout=10.0,
    )

    assert "start" in calls, "Discovery was not started"
    assert "wait" in calls, "Did not wait for discovery completion"
