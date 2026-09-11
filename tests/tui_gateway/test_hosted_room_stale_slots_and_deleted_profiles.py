"""Focused regressions for hosted-room stale bot_room slots and .deleted profiles.

See NousResearch/hermes-agent#106847.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from tui_gateway import server
from tui_gateway.hosted_room_driver import ROOM_SESSION_SOURCE
from tui_gateway.hosted_room_service import HostedRoomService


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def _quiescent_session(**overrides):
    ready = threading.Event()
    ready.set()
    session = {
        "running": False,
        "agent_ready": ready,
        "transport": None,
        "source": "tui",
    }
    session.update(overrides)
    return session


def test_local_profiles_skips_dot_directories(tmp_path: Path):
    # HostedRoomService.local_profiles() reads <db_path.parent>/profiles/
    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "alice").mkdir(parents=True)
    (profiles_dir / ".deleted").mkdir()
    (profiles_dir / ".deleted" / "bob").write_text("deleted")
    svc = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    names = svc.local_profiles()
    assert ".deleted" not in names
    assert "alice" in names


def test_idle_room_session_without_transport_is_lru_evictable(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    session = _quiescent_session(source=ROOM_SESSION_SOURCE, room_plumbing=True, transport=None)
    assert server._session_is_lru_evictable("room-sid", session) is True


def test_idle_room_session_stdio_fallback_is_lru_evictable(monkeypatch):
    """session.create falls back to _stdio_transport when there is no WS client."""
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    session = _quiescent_session(
        source=ROOM_SESSION_SOURCE,
        room_plumbing=True,
        transport=server._stdio_transport,
    )
    assert server._session_is_lru_evictable("room-sid", session) is True


def test_ordinary_session_with_live_transport_is_not_lru_evictable(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    live_transport = type("T", (), {"_closed": False})()
    session = _quiescent_session(source="tui", transport=live_transport)
    assert server._session_is_lru_evictable("tui-sid", session) is False


def test_ordinary_stdio_session_is_not_unconditionally_lru_evictable(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    session = _quiescent_session(source="tui", transport=server._stdio_transport)
    assert server._session_is_lru_evictable("tui-sid", session) is False


def test_room_session_with_live_transport_is_not_lru_evictable(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    live_transport = type("T", (), {"_closed": False})()
    session = _quiescent_session(
        source=ROOM_SESSION_SOURCE, room_plumbing=True, transport=live_transport)
    assert server._session_is_lru_evictable("room-sid", session) is False


def test_running_room_session_is_not_lru_evictable(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    session = _quiescent_session(
        source=ROOM_SESSION_SOURCE, room_plumbing=True, transport=None, running=True)
    assert server._session_is_lru_evictable("room-sid", session) is False
