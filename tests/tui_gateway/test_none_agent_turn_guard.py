"""A turn must never be run against a session record whose agent is ``None``.

The deferred agent build can finish WITHOUT attaching an agent: when the record
is replaced while the build runs, ``_await_resume_history`` returns False and
``_build`` leaves through its ``try`` while the ``finally`` still sets
``agent_ready``.  ``_wait_agent_for_prompt`` decides readiness from
``agent_ready``/``agent_error`` alone, so such a prompt was handed straight to
the turn body, which dereferenced ``session["agent"]``:

    AttributeError: 'NoneType' object has no attribute 'interim_assistant_callback'

raised once in ``_invoke_agent`` and again in the turn's ``finally`` — the turn
thread died, the prompt was never run, and the client got no frame it could
show (the only trace was the dispatcher's crash log).

These tests pin the guard at both seams:

* the prompt gate reports the failure instead of green-lighting the turn, and
* the turn body fails with a retryable runtime error frame (and its ``finally``
  no longer dereferences a missing agent).
"""

from __future__ import annotations

import logging
import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


def _waitable_session(ready_set: bool = True, agent_error=None):
    """A minimal record in the state ``_wait_agent_for_prompt`` inspects."""
    ready = threading.Event()
    if ready_set:
        ready.set()
    return {
        "agent": None,
        "agent_error": agent_error,
        "agent_ready": ready,
        "history": [],
        "history_lock": threading.RLock(),
        "running": True,
        "session_key": "gw-session-key",
    }


# ── prompt gate (server._wait_agent_for_prompt) ──────────────────────────────


def test_prompt_gate_rejects_ready_event_without_agent():
    """``agent_ready`` set + ``agent`` None must fail the prompt, not run it."""
    err = server._wait_agent_for_prompt(_waitable_session(), "rid", "ui-sid")

    assert err is not None
    assert err["error"]["code"] == 5032
    assert "agent initialization failed" in err["error"]["message"]


def test_prompt_gate_surfaces_recorded_build_reason():
    """A reason recorded by ``_build`` reaches the client verbatim."""
    err = server._wait_agent_for_prompt(
        _waitable_session(agent_error=server._RECORD_REPLACED_BUILD_ERROR), "rid", "ui-sid")

    assert err is not None
    assert err["error"]["message"] == server._RECORD_REPLACED_BUILD_ERROR


def test_prompt_gate_still_passes_a_built_agent():
    """The new check must not touch the healthy path."""
    session = _waitable_session()
    session["agent"] = types.SimpleNamespace(session_id="agent-sid-1")

    assert server._wait_agent_for_prompt(session, "rid", "ui-sid") is None


# ── deferred build records why it attached nothing ───────────────────────────


def test_replaced_record_build_records_reason_and_leaves_agent_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    # What _await_resume_history returns when _sessions[sid] was swapped mid-build.
    monkeypatch.setattr(server, "_await_resume_history", lambda sid, current: False)

    sid = "replaced-record"
    session = _waitable_session(ready_set=False)
    session.update({"cwd": str(tmp_path), "profile_home": None})
    server._sessions[sid] = session

    try:
        server._start_agent_build(sid, session)

        assert session["agent"] is None          # nothing was attached
        assert session["agent_ready"].is_set()   # yet the finally still reports ready
        assert session["agent_error"] == server._RECORD_REPLACED_BUILD_ERROR
    finally:
        server._sessions.pop(sid, None)


# ── turn body (prompt_turn.run) ──────────────────────────────────────────────


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    emitted: list[tuple] = []
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload=None: emitted.append((event_type, sid, payload)))
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    return emitted


def _complete_frames(emitted):
    return [p for (t, _sid, p) in emitted if t == "message.complete"]


def test_turn_without_agent_fails_with_retryable_frame(turn_env, caplog):
    session = _session()
    session["agent"] = None
    session["running"] = True

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, "继续")

    frames = _complete_frames(turn_env)
    assert len(frames) == 1
    payload = frames[0]
    assert payload["status"] == "error"
    assert payload["recoverable"] is True
    assert payload["error"] == server._NO_AGENT_TURN_ERROR
    assert payload["error_surface"] == {
        "layer": "runtime", "code": "agent_init_failed", "retryable": True}

    # No partial assistant output was invented for a turn that never ran.
    assert not any(t == "message.delta" for (t, _sid, _p) in turn_env)

    # The finally still runs and reports the retained failure (it used to die on
    # ``None.interim_assistant_callback`` before reaching these lines).
    finished = [r.getMessage() for r in caplog.records if "tui turn finished" in r.getMessage()]
    assert len(finished) == 1
    assert "status=error" in finished[0]
    assert "error_retained=True" in finished[0]

    assert session["running"] is False


def test_turn_with_agent_never_sets_the_interim_callback_on_none(turn_env):
    """Guard path must not leak into the healthy path's callback wiring."""
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    server._run_prompt_submit("rid", "ui-sid", session, "go")

    assert agent.interim_assistant_callback is None
    payload = _complete_frames(turn_env)[0]
    assert payload["status"] == "complete"
