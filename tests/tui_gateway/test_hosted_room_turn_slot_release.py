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
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
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
