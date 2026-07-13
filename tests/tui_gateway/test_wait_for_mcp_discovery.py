"""Tests for tui_gateway.entry.wait_for_mcp_discovery (PR #35245).

MCP tool discovery runs in a background daemon thread so a slow/dead server
can't freeze ``gateway.ready``.  The agent snapshots its tool list once at
build time and never re-reads it, so ``_make_agent`` briefly joins the
discovery thread before building — bounded, so a dead server can't re-introduce
the startup hang, and a no-op once discovery has finished.
"""

import threading
import time

import tui_gateway.entry as entry


def _restore_thread_slot(saved):
    entry._mcp_discovery_thread = saved


def test_no_thread_is_noop():
    """When no discovery thread was started (the common no-MCP case), the
    helper returns immediately and never blocks."""
    saved = entry._mcp_discovery_thread
    try:
        entry._mcp_discovery_thread = None
        start = time.monotonic()
        entry.wait_for_mcp_discovery(timeout=5.0)
        assert time.monotonic() - start < 0.1
    finally:
        _restore_thread_slot(saved)


def test_already_finished_thread_is_noop():
    """A thread that has already finished is not joined-on (dead thread)."""
    saved = entry._mcp_discovery_thread
    try:
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        t.join()  # ensure it's finished
        entry._mcp_discovery_thread = t
        start = time.monotonic()
        entry.wait_for_mcp_discovery(timeout=5.0)
        assert time.monotonic() - start < 0.1
    finally:
        _restore_thread_slot(saved)


def test_fast_thread_is_joined():
    """A reachable-but-still-connecting (fast) server lands before the agent
    snapshots tools — the helper waits for it to finish."""
    saved = entry._mcp_discovery_thread
    try:
        t = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
        t.start()
        entry._mcp_discovery_thread = t
        entry.wait_for_mcp_discovery(timeout=1.0)
        assert not t.is_alive()  # joined to completion
    finally:
        _restore_thread_slot(saved)


def test_hung_thread_is_bounded_by_timeout():
    """A slow/dead server must NOT re-introduce the startup hang — the join is
    bounded by the timeout and returns even though the thread is still alive."""
    saved = entry._mcp_discovery_thread
    stop = threading.Event()
    try:
        t = threading.Thread(target=stop.wait, daemon=True)  # blocks until set
        t.start()
        entry._mcp_discovery_thread = t
        start = time.monotonic()
        entry.wait_for_mcp_discovery(timeout=0.3)
        elapsed = time.monotonic() - start
        assert 0.25 <= elapsed < 1.0  # bounded near the timeout, not forever
        assert t.is_alive()  # thread still running; we did not block on it
    finally:
        stop.set()
        _restore_thread_slot(saved)
