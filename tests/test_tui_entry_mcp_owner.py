"""Regression tests: the stdio TUI consults the shared MCP discovery owner.

The stdio ``hermes --tui`` path used to spawn its own discovery thread and
``wait_for_mcp_discovery`` only ever joined that local handle. Now the spawn
goes through ``hermes_cli.mcp_startup.start_background_mcp_discovery`` (single
owner, restart-after-zero-connected semantics), so the entry-side wait must
fall through to the shared owner when no local thread exists.
"""

import threading
import time

import pytest

from hermes_cli import mcp_startup
from hermes_constants import hermes_home_key
from tui_gateway import entry


@pytest.fixture
def shared_owner_thread(monkeypatch):
    """Install/clear the shared owner's discovery thread for THIS profile.

    The owner keys its slot per resolved ``HERMES_HOME`` (#67605), so these
    white-box tests set the active profile's entry rather than a single
    process-global handle.
    """
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_threads", {})

    def _set(thread):
        if thread is None:
            mcp_startup._mcp_discovery_threads.pop(hermes_home_key(), None)
        else:
            mcp_startup._mcp_discovery_threads[hermes_home_key()] = thread

    return _set


def test_tui_uses_shared_portable_mcp_gate(monkeypatch):
    monkeypatch.setattr(mcp_startup, "_has_configured_mcp_servers", lambda: True)

    assert entry._has_configured_mcp_servers() is True


def test_wait_falls_through_to_shared_owner(monkeypatch, shared_owner_thread):
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
    shared_owner_thread(thread)

    start = time.monotonic()
    entry.wait_for_mcp_discovery(timeout=2.0)
    elapsed = time.monotonic() - start

    assert not thread.is_alive()
    assert elapsed >= 0.04


def test_wait_noop_when_no_owner_has_a_thread(monkeypatch, shared_owner_thread):
    monkeypatch.setattr(entry, "_mcp_discovery_thread", None)
    shared_owner_thread(None)

    start = time.monotonic()
    entry.wait_for_mcp_discovery(timeout=2.0)

    assert time.monotonic() - start < 0.5


def test_wait_still_joins_entry_local_thread(monkeypatch):
    thread = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
    thread.start()
    monkeypatch.setattr(entry, "_mcp_discovery_thread", thread)

    entry.wait_for_mcp_discovery(timeout=2.0)

    assert not thread.is_alive()


def test_wait_reinvokes_shared_spawn_when_discovery_enabled(
    monkeypatch, shared_owner_thread
):
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
    shared_owner_thread(None)

    entry.wait_for_mcp_discovery(timeout=0.1)

    assert calls == ["tui-mcp-discovery"]


def test_wait_skips_spawn_when_discovery_not_enabled(
    monkeypatch, shared_owner_thread
):
    """Non-MCP sessions must not import/spawn discovery on the wait path."""
    monkeypatch.setattr(entry, "_mcp_discovery_thread", None)
    monkeypatch.setattr(entry, "_mcp_discovery_enabled", False)

    calls = []

    def _fake_start(*, logger, thread_name):
        calls.append(thread_name)

    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", _fake_start)
    shared_owner_thread(None)

    entry.wait_for_mcp_discovery(timeout=0.1)

    assert calls == []
