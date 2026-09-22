"""A turn must never run against a session record whose agent is ``None``.

The deferred agent build can finish WITHOUT attaching an agent: when the record is replaced or
closed while the build runs, ``_build`` leaves early while its ``finally`` still sets
``agent_ready``.  Such a prompt used to reach the turn body, which dereferenced
``session["agent"]`` twice (``_invoke_agent`` and the turn's ``finally``) — the turn thread died
with ``running`` still True, the prompt vanished and the session stayed "busy" (#111531).
"""

from __future__ import annotations

import threading
import types

from tui_gateway import server
from tui_gateway.user_messages import AGENT_BUILD_ABANDONED


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent, **extra):
    return {
        "agent": agent, "agent_error": None, "session_key": "gw-session-key", "history": [],
        "history_lock": threading.RLock(), "history_version": 0, "running": True, "attached_images": [],
        "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all", "inflight_turn": None, **extra}


def _turn_env(monkeypatch, tmp_path) -> list:
    emitted: list[tuple] = []
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda event_type, sid, payload=None: emitted.append((event_type, sid, payload)))
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    return emitted


def test_turn_without_agent_is_refused_with_retryable_frame(monkeypatch, tmp_path):
    """The recorded build reason reaches the client as a retryable runtime frame; ``running`` is released."""
    emitted = _turn_env(monkeypatch, tmp_path)
    session = _session(None, agent_error=AGENT_BUILD_ABANDONED)

    assert server._run_prompt_submit("rid", "ui-sid", session, "继续") is False

    frames = [p for (t, _sid, p) in emitted if t == "message.complete"]
    assert len(frames) == 1
    assert frames[0]["status"] == "error" and frames[0]["recoverable"] is True
    assert frames[0]["error"] == AGENT_BUILD_ABANDONED
    assert frames[0]["error_surface"] == {"layer": "runtime", "code": "agent_init_failed", "retryable": True}
    assert session["running"] is False
    assert session["inflight_turn"]["status"] == "error"  # retained for session.resume

    # Control: a built agent still runs the turn to completion and clears the interim closure.
    agent = types.SimpleNamespace(
        session_id="agent-sid-1", run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None)
    emitted.clear()
    server._run_prompt_submit("rid", "ui-sid", _session(agent), "go")
    assert [p["status"] for (t, _sid, p) in emitted if t == "message.complete"] == ["complete"]
    assert agent.interim_assistant_callback is None


def test_turn_without_agent_with_missing_credentials_is_actionable(monkeypatch, tmp_path):
    """An init failure caused by an empty pool offers model setup, not a generic retry (#119105)."""
    emitted = _turn_env(monkeypatch, tmp_path)
    session = _session(None, agent_error="No LLM provider configured. Run `hermes model` to select a provider.")

    assert server._run_prompt_submit("rid", "ui-sid", session, "go") is False

    frame = next(p for (t, _sid, p) in emitted if t == "message.complete")
    assert frame["error_surface"] == {"layer": "auth", "code": "credentials_missing", "retryable": False}


def test_replaced_record_build_records_reason_and_leaves_agent_unset(monkeypatch, tmp_path):
    """A build whose record was swapped mid-flight sets ``agent_ready`` AND records why nothing attached."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_await_resume_history", lambda sid, current: False)

    sid = "replaced-record"
    session = _session(None, agent_ready=threading.Event(), cwd=str(tmp_path), profile_home=None)
    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
    finally:
        server._sessions.pop(sid, None)

    assert session["agent"] is None
    assert session["agent_ready"].is_set()
    assert session["agent_error"] == AGENT_BUILD_ABANDONED
