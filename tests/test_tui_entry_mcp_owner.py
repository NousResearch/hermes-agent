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
