"""Regression tests: the stdio TUI consults the shared MCP discovery owner.

The stdio ``hermes --tui`` path used to spawn its own discovery thread and
``wait_for_mcp_discovery`` only ever joined that local handle. Now the spawn
goes through ``hermes_cli.mcp_startup.start_background_mcp_discovery`` (single
owner, restart-after-zero-connected semantics), so the entry-side wait must
fall through to the shared owner when no local thread exists.
"""

import threading
import time

from hermes_cli import mcp_startup
from tui_gateway import entry


def test_wait_falls_through_to_shared_owner(monkeypatch):
    monkeypatch.setattr(entry, "_mcp_discovery_thread", None)
    # The fall-through to the shared owner only exists for the stdio TUI,
    # which arms this flag in main(); other surfaces call the startup wait
    # directly from _make_agent and must NOT be waited twice.
    monkeypatch.setattr(entry, "_mcp_discovery_enabled", True)
    monkeypatch.setattr(
        mcp_startup, "start_background_mcp_discovery", lambda **kw: None
    )
    thread = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
    thread.start()
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", thread)

    start = time.monotonic()
    entry.wait_for_mcp_discovery(timeout=2.0)
    elapsed = time.monotonic() - start

    assert not thread.is_alive()
    assert elapsed >= 0.04


def test_wait_noop_when_no_owner_has_a_thread(monkeypatch):
    monkeypatch.setattr(entry, "_mcp_discovery_thread", None)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

    start = time.monotonic()
    entry.wait_for_mcp_discovery(timeout=2.0)

    assert time.monotonic() - start < 0.5


def test_wait_still_joins_entry_local_thread(monkeypatch):
    thread = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
    thread.start()
    monkeypatch.setattr(entry, "_mcp_discovery_thread", thread)

    entry.wait_for_mcp_discovery(timeout=2.0)

    assert not thread.is_alive()


def test_wait_reinvokes_shared_spawn_when_discovery_enabled(monkeypatch):
    """The TUI wait path must give the shared owner a retry opportunity.

    start_background_mcp_discovery() allows a retry after a run that
    connected zero servers — but only when it is CALLED again. main() calls
    it exactly once, so the per-agent-build wait must re-invoke the
    idempotent spawn when this process is MCP-enabled.
    """
    monkeypatch.setattr(entry, "_mcp_discovery_thread", None)
    monkeypatch.setattr(entry, "_mcp_discovery_enabled", True)

    calls = []

    def _fake_start(*, logger, thread_name):
        calls.append(thread_name)

    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", _fake_start)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

    entry.wait_for_mcp_discovery(timeout=0.1)

    assert calls == ["tui-mcp-discovery"]


def test_wait_skips_spawn_when_discovery_not_enabled(monkeypatch):
    """Non-MCP sessions must not import/spawn discovery on the wait path."""
    monkeypatch.setattr(entry, "_mcp_discovery_thread", None)
    monkeypatch.setattr(entry, "_mcp_discovery_enabled", False)

    calls = []

    def _fake_start(*, logger, thread_name):
        calls.append(thread_name)

    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", _fake_start)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

    entry.wait_for_mcp_discovery(timeout=0.1)

    assert calls == []
