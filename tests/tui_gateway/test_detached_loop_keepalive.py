"""A detached Desktop/TUI session that owns an active /loop must survive the orphan reapers.

The per-session notification poller is the only driver of a route-less (Desktop/TUI) loop; reaping the
detached session stops that poller and freezes the loop until a client reattaches. Gateway-routed loops are
fired by the gateway's own scanner, so they must not pin the session.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending_ws_reaps.clear()


class _Timer:
    """Captures the reap callback instead of sleeping on it."""

    created: list = []

    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback
        _Timer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture()
def detached(server, monkeypatch):
    _Timer.created = []
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    popped = []
    monkeypatch.setattr(server, "_teardown_popped_session", lambda s, **kw: popped.append(s))
    sid = "sid-detached-loop"
    session = {
        "session_key": "desktop-loop-session",
        "transport": server._detached_ws_transport,
        "running": False,
        "history_lock": threading.Lock(),
        "last_active": 0.0,
        "created_at": 0.0,
    }
    server._sessions[sid] = session
    return sid, session, popped


def _set_loop(session_key: str, route: dict | None = None):
    from hermes_cli.loops import LoopManager

    LoopManager(session_key).set("reconcile the spec", interval_seconds=1800, route=route)


def test_orphan_reap_keeps_detached_session_with_active_desktop_loop(server, detached):
    sid, session, popped = detached
    _set_loop(session["session_key"])

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert server._sessions.get(sid) is session and not popped
    assert server._pending_ws_reaps[sid] is _Timer.created[-1]  # re-armed, re-checked each grace


def test_orphan_reap_still_collects_session_once_loop_stops(server, detached):
    sid, session, popped = detached
    _set_loop(session["session_key"])
    from hermes_cli.loops import LoopManager

    LoopManager(session["session_key"]).clear()

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert sid not in server._sessions and popped == [session]


def test_gateway_routed_loop_does_not_pin_detached_session(server, detached):
    sid, session, popped = detached
    _set_loop(session["session_key"], route={"platform": "discord", "chat_id": "123"})

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert sid not in server._sessions and popped == [session]


def test_ttl_and_cap_reapers_spare_session_with_active_desktop_loop(server, detached, monkeypatch):
    sid, session, _ = detached
    monkeypatch.setattr(server, "_SESSION_TTL_S", 1)
    now = time.time()
    assert server._session_is_evictable(sid, session, now)  # idle, detached, past TTL: reapable today

    _set_loop(session["session_key"])

    assert not server._session_is_evictable(sid, session, now)
    assert not server._session_is_reapable(sid, session)
    # An idle looping session must not block a memory trim the way live work does.
    assert server._session_is_lru_evictable(sid, session)
