"""Regression test for issue #51587.

MCP tools connected/enabled but never surfaced into the agent's session
toolset on the desktop app + dashboard WebUI.

Root cause: there are two independent background MCP discovery thread owners
by surface:

  * ``tui_gateway.entry`` — the stdio ``hermes --tui`` path.
  * ``hermes_cli.mcp_startup`` — the desktop app + dashboard WebSocket sidecar
    (``tui_gateway/ws.py``) and ``hermes dashboard``.

The late-refresh scheduler (``tui_gateway.server._schedule_mcp_late_refresh``)
gates on ``tui_gateway.entry.mcp_discovery_in_flight()``. Before the fix that
function read ONLY ``tui_gateway.entry._mcp_discovery_thread``. On the
desktop/dashboard surfaces that thread is ``None`` (the thread lives on
``hermes_cli.mcp_startup``), so the scheduler bailed immediately and a slow MCP
server's tools never surfaced for the whole session — even after a container
restart. The fix makes ``mcp_discovery_in_flight`` / ``join_mcp_discovery``
consult BOTH thread owners.
"""

import threading

import pytest

import hermes_cli.mcp_startup as startup
import tui_gateway.entry as entry


@pytest.fixture
def clean_discovery_globals():
    """Snapshot and restore both modules' discovery-thread globals."""
    saved_entry = entry._mcp_discovery_thread
    saved_startup = startup._mcp_discovery_thread
    entry._mcp_discovery_thread = None
    startup._mcp_discovery_thread = None
    try:
        yield
    finally:
        entry._mcp_discovery_thread = saved_entry
        startup._mcp_discovery_thread = saved_startup


def _alive_thread(stop: threading.Event) -> threading.Thread:
    t = threading.Thread(target=lambda: stop.wait(5.0), daemon=True)
    t.start()
    return t


def test_entry_in_flight_sees_startup_thread(clean_discovery_globals):
    """Desktop/dashboard surface: discovery thread lives on hermes_cli.mcp_startup.

    The entry-level in-flight check must report True so the late-refresh
    scheduler does not bail (the #51587 bug).
    """
    stop = threading.Event()
    startup._mcp_discovery_thread = _alive_thread(stop)
    try:
        # Entry's own thread is None, but the startup thread is alive.
        assert entry._mcp_discovery_thread is None
        assert entry.mcp_discovery_in_flight() is True
    finally:
        stop.set()
        startup._mcp_discovery_thread.join(timeout=2.0)

    # After the thread exits, neither owner is in flight.
    assert entry.mcp_discovery_in_flight() is False


def test_entry_join_waits_on_startup_thread(clean_discovery_globals):
    """join_mcp_discovery must report not-done while the startup thread runs."""
    stop = threading.Event()
    t = _alive_thread(stop)
    startup._mcp_discovery_thread = t

    assert entry.join_mcp_discovery(timeout=0.1) is False

    stop.set()
    t.join(timeout=2.0)
    assert entry.join_mcp_discovery(timeout=2.0) is True


def test_entry_in_flight_still_sees_own_thread(clean_discovery_globals):
    """stdio TUI surface: discovery thread lives on tui_gateway.entry (unchanged)."""
    stop = threading.Event()
    entry._mcp_discovery_thread = _alive_thread(stop)
    try:
        assert startup._mcp_discovery_thread is None
        assert entry.mcp_discovery_in_flight() is True
    finally:
        stop.set()
        entry._mcp_discovery_thread.join(timeout=2.0)

    assert entry.mcp_discovery_in_flight() is False


def test_no_mcp_threads_not_in_flight(clean_discovery_globals):
    """No discovery anywhere → not in flight, join reports done immediately."""
    assert entry.mcp_discovery_in_flight() is False
    assert entry.join_mcp_discovery(timeout=0.1) is True


def test_startup_module_exposes_in_flight_helpers(clean_discovery_globals):
    """hermes_cli.mcp_startup gains the in-flight/join helpers entry delegates to."""
    assert startup.mcp_discovery_in_flight() is False
    assert startup.join_mcp_discovery(timeout=0.1) is True

    stop = threading.Event()
    t = _alive_thread(stop)
    startup._mcp_discovery_thread = t
    try:
        assert startup.mcp_discovery_in_flight() is True
        assert startup.join_mcp_discovery(timeout=0.1) is False
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert startup.mcp_discovery_in_flight() is False
    assert startup.join_mcp_discovery(timeout=2.0) is True
