"""An idle resident Desktop record stops fencing its chat from every other surface (#146735).

Desktop keeps a chat's runtime alive after its tab closes (no RPC is sent) and the app's live socket
exempts that record from the TTL/LRU reapers, so the active-session lease its first turn claimed used to
refuse Telegram, the dashboard's ``session.resume`` and ``message_agent`` until the app quit. The reaper
tick now releases the lease of an idle record; a record mid-turn keeps it.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session
from tui_gateway import server


class _LiveSocket:
    _closed = False  # the Desktop app is still connected


def _desktop_record(key: str, **extra) -> dict:
    ready = threading.Event()
    ready.set()
    long_ago = time.time() - 3600
    return {
        "agent": None, "session_key": key, "source": "desktop", "history": [],
        "history_lock": threading.Lock(), "history_version": 0, "running": False, "agent_ready": ready,
        "transport": _LiveSocket(), "created_at": long_ago, "last_active": long_ago, **extra}


@pytest.fixture
def resident(monkeypatch):
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    added: list[str] = []

    def add(sid: str, record: dict) -> dict:
        server._sessions[sid] = record
        added.append(sid)
        assert server._ensure_active_session_slot(sid, record) is None
        return record

    yield add
    for sid in added:
        server._release_active_session_slot(server._sessions.pop(sid, None))


def _held() -> list[str]:
    return sorted(entry["session_id"] for entry in active_session_registry_snapshot())


def test_idle_desktop_record_releases_its_lease_on_the_reaper_tick(resident):
    idle = resident("closed-tab", _desktop_record("stored-idle"))
    resident("mid-turn", _desktop_record("stored-busy", running=True))
    assert _held() == ["stored-busy", "stored-idle"]

    server._reap_idle_sessions()

    other_surface, refusal = try_acquire_active_session(
        session_id="stored-idle", surface="telegram", config={})
    try:
        assert {
            "held": _held(), "other_surface_refusal": refusal,
            "record_resident": server._sessions.get("closed-tab") is idle,
        } == {"held": ["stored-busy", "stored-idle"], "other_surface_refusal": None, "record_resident": True}
    finally:
        if other_surface is not None:
            other_surface.release()
    assert _held() == ["stored-busy"]
