"""Regression test: MCP discovery that completes within the default timeout.

Validates that a realistic HTTP MCP cold-start (~2-3s) lands inside the
mcp_discovery_timeout window, so the agent's first-turn tool snapshot
includes connected MCP tools.

Fixes: #47121, #61891, #41625
"""

import time
import types

import pytest

from hermes_cli import mcp_startup


def _simulate_discovery(duration: float):
    """Return a discovery callable that sleeps *duration* seconds."""
    def _discover():
        time.sleep(duration)
    return _discover


def test_default_timeout_covers_realistic_cold_start():
    """A 2s HTTP MCP cold-start must complete within the default window.

    Before the fix, the default was 1.5s — any server taking longer would
    miss the agent's one-time tool snapshot and be invisible for the
    session lifetime.
    """
    import hermes_cli.config_defaults as cfg

    timeout = cfg.DEFAULT_CONFIG.get("mcp_discovery_timeout", 1.5)
    assert timeout >= 5.0, (
        f"mcp_discovery_timeout={timeout}s is too short; "
        "realistic HTTP MCP cold-start needs ~2-4s"
    )


def test_discovery_completes_within_timeout(monkeypatch):
    """Discovery finishing before the timeout must not be dropped."""
    monkeypatch.setattr(
        mcp_startup,
        "_discover_mcp_tools_without_interactive_oauth",
        _simulate_discovery(2.0),
    )

    logger = types.SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )

    mcp_startup.start_background_mcp_discovery(
        logger=logger, thread_name="test-discovery"
    )
    mcp_startup.wait_for_mcp_discovery()

    thread = mcp_startup._mcp_discovery_thread
    assert thread is None or not thread.is_alive(), (
        "Discovery thread still running after wait_for_mcp_discovery — "
        "timeout is too short"
    )


def test_wait_returns_immediately_when_no_discovery(monkeypatch):
    """No MCP servers configured → wait_for_mcp_discovery returns ~instantly."""
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery()
    elapsed = time.time() - t0

    assert elapsed < 0.2, f"Blocked for {elapsed:.1f}s with no discovery thread"
