"""A hosted room member turn releases the profile's ``bot_room`` active-session slot when it ends.

Room turns are serialized by the room driver's lease, not by the per-profile active-session
registry. When two room workers share one home (messaging gateway + Desktop ``serve``), a slot
held past the turn by the worker that drove turn 1 made every later worker's turn for that
member fail with ``Refused active session … already held by pid=…`` until the first process
exited (#106847). Ordinary sessions keep their slot for the life of the live session.
"""

from __future__ import annotations

import threading
import types

from tui_gateway import server
from tui_gateway.hosted_room_driver import ROOM_SESSION_SOURCE


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _Lease:
    track_liveness = False
    enabled = True

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _session(source: str, lease: _Lease):
    agent = types.SimpleNamespace(
        session_id="agent-sid-1", run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None)
    return {
        "agent": agent, "agent_error": None, "session_key": "gw-session-key", "history": [],
        "history_lock": threading.RLock(), "history_version": 0, "running": True, "attached_images": [],
        "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all", "inflight_turn": None, "source": source, "active_session_lease": lease}


def _turn_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda event_type, sid, payload=None: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def test_hosted_room_turn_releases_bot_room_slot_when_it_ends(monkeypatch, tmp_path):
    _turn_env(monkeypatch, tmp_path)
    lease = _Lease()
    session = _session(ROOM_SESSION_SOURCE, lease)

    assert server._run_prompt_submit("rid", "ui-sid", session, "@sentinel ping") is True

    assert session["running"] is False
    assert lease.released is True
    assert "active_session_lease" not in session


def test_ordinary_session_keeps_its_slot_after_a_turn(monkeypatch, tmp_path):
    """Control: a Desktop-driven session's exclusivity is the slot itself — it must survive the turn."""
    _turn_env(monkeypatch, tmp_path)
    lease = _Lease()
    session = _session("desktop", lease)

    assert server._run_prompt_submit("rid", "ui-sid", session, "hello") is True

    assert lease.released is False
    assert session["active_session_lease"] is lease


def test_mobile_turn_releases_slot_when_it_ends(monkeypatch, tmp_path):
    """Mobile turns on :9119 release their active-session slot at turn end so Desktop/CLI can resume cleanly."""
    _turn_env(monkeypatch, tmp_path)
    lease = _Lease()
    session = _session("mobile", lease)

    assert server._run_prompt_submit("rid", "ui-sid", session, "hello from phone") is True

    assert session["running"] is False
    assert lease.released is True
    assert "active_session_lease" not in session


def test_okhttp_transport_on_desktop_sourced_session_releases_slot_when_turn_ends(monkeypatch, tmp_path):
    """A Desktop-created session (source='desktop') resumed from hermes-mobile (User-Agent: okhttp/...)
    is recognized as a mobile turn and releases its active-session slot when the turn ends."""
    _turn_env(monkeypatch, tmp_path)
    lease = _Lease()
    session = _session("desktop", lease)
    session["transport"] = types.SimpleNamespace(_ws=types.SimpleNamespace(headers={"user-agent": "okhttp/4.12.0"}))

    assert server._run_prompt_submit("rid", "ui-sid", session, "hello from phone on desktop session") is True

    assert session["running"] is False
    assert lease.released is True
    assert "active_session_lease" not in session


def test_detached_ws_turn_releases_slot_when_it_ends(monkeypatch, tmp_path):
    """If a client disconnected mid-turn, the finished turn releases its active-session slot immediately."""
    _turn_env(monkeypatch, tmp_path)
    lease = _Lease()
    session = _session("desktop", lease)
    session["transport"] = server._detached_ws_transport

    assert server._run_prompt_submit("rid", "ui-sid", session, "hello before disconnect") is True

    assert session["running"] is False
    assert lease.released is True
    assert "active_session_lease" not in session


def test_desktop_turn_marks_lease_idle_and_allows_cross_surface_mobile_handoff(monkeypatch, tmp_path):
    """An active turn refuses cross-surface takeover; once the turn ends, an idle lease hands over cleanly
    between Desktop and Mobile and re-claims when returning to Desktop."""
    from hermes_cli.active_sessions import SESSION_NOT_OWNED, active_session_registry_snapshot

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_load_cfg", dict)
    _turn_env(monkeypatch, tmp_path)

    desktop_session = _session("desktop", None)
    desktop_session.pop("active_session_lease", None)
    desktop_session["session_key"] = "shared-chat-1"
    desktop_session["profile_home"] = tmp_path
    desktop_session["running"] = True

    assert server._ensure_active_session_slot("desktop-sid", desktop_session) is None
    desktop_lease = desktop_session["active_session_lease"]
    (entry_running,) = active_session_registry_snapshot(registry_home=tmp_path)
    assert entry_running["metadata"]["turn_active"] is True

    # Polarity 1 (mid-turn): while Desktop's turn is running, Mobile is refused with SESSION_NOT_OWNED.
    mobile_session = _session("mobile", None)
    mobile_session.pop("active_session_lease", None)
    mobile_session["session_key"] = "shared-chat-1"
    mobile_session["profile_home"] = tmp_path
    mobile_session["running"] = True
    refusal = server._ensure_active_session_slot("mobile-sid", mobile_session)
    assert getattr(refusal, "reason", None) == SESSION_NOT_OWNED
    assert "active_session_lease" not in mobile_session

    # Finish Desktop's turn: lease stays attached (not released) but is marked turn_active=False on disk.
    assert server._run_prompt_submit("rid-1", "desktop-sid", desktop_session, "hello from desktop") is True
    assert desktop_lease.released is False
    assert getattr(desktop_lease, "turn_idle", False) is True
    (entry_idle,) = active_session_registry_snapshot(registry_home=tmp_path)
    assert entry_idle["metadata"]["turn_active"] is False

    # Polarity 2 (post-turn idle): Mobile now takes over the idle Desktop lease cleanly and releases at turn end.
    assert server._ensure_active_session_slot("mobile-sid", mobile_session) is None
    assert server._run_prompt_submit("rid-2", "mobile-sid", mobile_session, "hello from mobile") is True
    assert "active_session_lease" not in mobile_session
    assert active_session_registry_snapshot(registry_home=tmp_path) == []

    # Returning to Desktop re-acquires a fresh lease because desktop_lease.turn_idle is True.
    desktop_session["running"] = True
    assert server._ensure_active_session_slot("desktop-sid", desktop_session) is None
    assert desktop_session["active_session_lease"] is not desktop_lease
    (entry_returned,) = active_session_registry_snapshot(registry_home=tmp_path)
    assert entry_returned["surface"] == "desktop"
    assert entry_returned["metadata"]["turn_active"] is True


def test_park_forever_ws_disconnect_releases_idle_slot(monkeypatch, tmp_path):
    """When ws_orphan_reap_grace_s <= 0, disconnecting an idle WS client releases its active-session slot."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.0)
    lease = _Lease()
    transport = object()
    session = _session("desktop", lease)
    session["running"] = False
    session["transport"] = transport
    with server._sessions_lock:
        server._sessions.clear()
        server._sessions["parked-sid"] = session
    try:
        reaped, detached = server._close_sessions_for_transport(transport, end_reason="ws_disconnect")
        assert reaped == 0
        assert detached == 1
        assert lease.released is True
        assert "active_session_lease" not in session
    finally:
        with server._sessions_lock:
            server._sessions.clear()
