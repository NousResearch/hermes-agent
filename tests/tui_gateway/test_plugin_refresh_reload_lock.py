"""Activating a plugin must not rebuild session tools while a reload has the registry torn down.

``reload.mcp`` runs shutdown -> rediscover -> refresh under ``_mcp_reload_lock``, and its own
refresh says why: "so a concurrent reload can't tear the registry down mid-refresh". The plugin
activation path (``refresh_plugin_sessions``, reached from ``hermes_cli.plugins_activation``)
rebuilds the same per-agent snapshots from the same process-global registry, so it needs the same
lock — installing a plugin is itself a common way to trigger a reload, which makes the two overlap.
A snapshot taken mid-teardown leaves those chats without their MCP tools until the next reload, and
the per-session failure is only logged.
"""

from __future__ import annotations

import threading
import time

import pytest

import tui_gateway.server as srv


@pytest.fixture()
def registry(monkeypatch):
    """Record the registry state each session rebuild observes."""
    state = {"registry": "ready", "observed": []}

    def _fake_refresh_live(home=None, **_kwargs):
        state["observed"].append(state["registry"])

    monkeypatch.setattr(srv, "_refresh_live_sessions", _fake_refresh_live)
    return state


def test_plugin_refresh_holds_the_reload_lock(registry, monkeypatch, tmp_path):
    held = []
    monkeypatch.setattr(srv, "_refresh_live_sessions",
                        lambda home=None, **_kw: held.append(srv._mcp_reload_lock.locked()))

    srv.refresh_plugin_sessions(tmp_path, "plugin is live")

    assert held == [True]
    assert srv._mcp_reload_lock.locked() is False  # and it is released afterwards


def test_plugin_refresh_waits_for_an_in_flight_reload(registry, tmp_path):
    """The rebuild must observe the rebuilt registry, never the torn-down one."""
    torn_down = threading.Event()

    def _reload_like():
        with srv._mcp_reload_lock:
            registry["registry"] = "torn down"
            torn_down.set()
            time.sleep(0.2)  # the shutdown -> rediscover window
            registry["registry"] = "rebuilt"

    reload_thread = threading.Thread(target=_reload_like)
    reload_thread.start()
    assert torn_down.wait(5), "reload thread never started"
    try:
        srv.refresh_plugin_sessions(tmp_path, "plugin is live")
    finally:
        reload_thread.join(10)

    assert registry["observed"] == ["rebuilt"]


def test_the_refresh_still_appends_and_carries_its_note(monkeypatch, tmp_path):
    """Locking must not change what the refresh asks for: append-only, scoped to the profile home."""
    calls = []
    monkeypatch.setattr(srv, "_refresh_live_sessions",
                        lambda home=None, **kwargs: calls.append((home, kwargs)))

    srv.refresh_plugin_sessions(tmp_path, "plugin is live")

    assert calls == [(tmp_path, {"preserve_prefix": True, "note": "plugin is live"})]
