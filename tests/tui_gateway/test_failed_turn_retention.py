"""Failed turns must retain a replayable ``inflight`` snapshot.

A turn that ended in error used to clear ``inflight_turn`` and emit its
terminal frame in the same breath. If the client was disconnected during that
window, the frame went to the detached drop-transport and the in-memory state
was already gone — the desktop reconnected to a session with no trace of the
failure (stuck spinner or a silently missing turn).

Contract pinned here:

* ``_fail_inflight_turn`` keeps the user prompt, partial assistant text, and
  error semantics; ``_inflight_snapshot`` exposes status/error/recoverable.
* The returned-error path (``run_conversation()`` returning ``error``) retains
  the snapshot — not just the exception path.
* The exception path closes the turn with a terminal ``message.complete``
  (``status: "error"``, same shape as the returned-error path) instead of a
  bare ``error`` event.
* ``session.resume``'s live payload carries the retained snapshot.
* A retained failure never leaks into the next turn's inflight state.
"""

from __future__ import annotations

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
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
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


@pytest.fixture()
def emits(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _events(captured, name):
    return [payload for event, _sid, payload in captured if event == name]


# ── Unit: retention helpers ───────────────────────────────────────────


def test_fail_inflight_turn_retains_partial_and_error():
    session = _session()
    server._start_inflight_turn(session, "do the thing")
    server._append_inflight_delta(session, "partial answer")

    server._fail_inflight_turn(session, RuntimeError("provider exploded"))

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["user"] == "do the thing"
    assert snapshot["assistant"] == "partial answer"
    assert snapshot["streaming"] is False
    assert snapshot["error"] == "provider exploded"
    assert snapshot["status"] == "error"
    assert snapshot["recoverable"] is True


def test_snapshot_returned_for_error_only_turn():
    """An init failure has no user/assistant text yet — the error alone must
    survive the emptiness check, or resume shows nothing."""
    session = _session()
    server._fail_inflight_turn(session, "agent initialization failed")

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["error"] == "agent initialization failed"


def test_healthy_snapshot_carries_no_error_keys():
    session = _session()
    server._start_inflight_turn(session, "hi")
    server._append_inflight_delta(session, "hello")

    snapshot = server._inflight_snapshot(session)
    assert snapshot == {"assistant": "hello", "streaming": True, "user": "hi"}


# ── Returned-error path (run_conversation returns an error result) ────


def test_returned_error_result_retains_snapshot_and_emits_terminal_frame(
    emits, turn_env
):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "provider 402: billing wall",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    payload = completes[0]
    assert payload["status"] == "error"
    assert payload["error"] == "provider 402: billing wall"
    assert payload["recoverable"] is True

    # The retained snapshot survives the finally block for resume replay.
    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["status"] == "error"
    assert snapshot["error"] == "provider 402: billing wall"
    assert snapshot["user"] == "do the thing"
    assert session["running"] is False


def test_returned_error_result_carries_error_surface(emits, turn_env):
    """A classified failure_reason rides the terminal frame AND the retained
    snapshot as a structured {layer, code, retryable} descriptor, so the
    desktop names the failing layer instead of sniffing the message."""
    agent = types.SimpleNamespace(
        session_id="session-key",
        provider="openrouter",
        model="test/model",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "Rate limit exceeded",
            "failed": True,
            "failure_reason": "rate_limit",
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    payload = _events(emits, "message.complete")[0]
    assert payload["error_surface"] == {
        "layer": "provider",
        "code": "rate_limit",
        "retryable": True,
        # The failing session's identity rides the descriptor so clients
        # report the model that actually failed, not the composer's current.
        "provider": "openrouter",
        "model": "test/model",
    }

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["error_surface"]["layer"] == "provider"


def test_returned_error_without_reason_omits_no_frame(emits, turn_env):
    """Legacy result dicts (no failure_reason) still get a best-effort
    descriptor — never a crash, never a missing terminal frame."""
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "something odd",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "go")

    server._run_prompt_submit("rid", "sid", session, "go")

    payload = _events(emits, "message.complete")[0]
    assert payload["status"] == "error"
    assert payload["error_surface"]["layer"] == "provider"
    assert payload["error_surface"]["code"] == "unknown"


def test_completed_turn_still_clears_inflight(emits, turn_env):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {"final_response": "all done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "complete"
    assert "error" not in completes[0]
    assert server._inflight_snapshot(session) is None


# ── Exception path ─────────────────────────────────────────────────────


def test_exception_closes_turn_with_terminal_complete_and_partial(emits, turn_env):
    def _boom(message, stream_callback=None, **kwargs):
        if stream_callback is not None:
            stream_callback("half an ans")
        raise RuntimeError("connection reset mid-stream")

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=_boom,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    # Terminal frame, not a bare error event.
    assert not _events(emits, "error")
    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    payload = completes[0]
    assert payload["status"] == "error"
    assert payload["error"] == "connection reset mid-stream"
    assert payload["recoverable"] is True
    assert payload["partial"] is True
    assert payload["text"] == "half an ans"

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["assistant"] == "half an ans"
    assert snapshot["error"] == "connection reset mid-stream"
    assert session["running"] is False

    # Dispatcher-side exceptions (not API errors) classify as gateway-layer.
    assert payload["error_surface"]["layer"] == "gateway"
    assert snapshot["error_surface"]["layer"] == "gateway"


# ── Resume replay (the reason retention exists) ───────────────────────


def test_live_session_payload_exposes_retained_failure(emits, turn_env, monkeypatch):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "budget exhausted",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "long job")
    server._run_prompt_submit("rid", "sid", session, "long job")

    # What session.resume's live fast path hands a reconnecting client.
    monkeypatch.setattr(server, "_get_db", lambda: None)
    payload = server._live_session_payload("sid", session)

    assert payload["running"] is False
    inflight = payload.get("inflight")
    assert inflight is not None
    assert inflight["status"] == "error"
    assert inflight["error"] == "budget exhausted"
    assert inflight["user"] == "long job"


# ── Retained failure must not leak into the next turn ─────────────────


def test_next_turn_replaces_retained_error_snapshot(emits, turn_env):
    seen_inflight_user: list = []

    def _run_ok(message, **kwargs):
        # Capture what the inflight turn looks like while the new turn runs.
        turn = server._inflight_snapshot(_run_ok.session)
        seen_inflight_user.append(turn and turn["user"])
        return {"final_response": "fresh answer"}

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=_run_ok,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    _run_ok.session = session

    # Leftover retained failure from a previous turn.
    server._start_inflight_turn(session, "old failed prompt")
    server._fail_inflight_turn(session, "previous turn failed")

    server._run_prompt_submit("rid", "sid", session, "new prompt")

    # The new turn must have started a fresh inflight turn, not inherited the
    # failed one (the retained dict used to satisfy the isinstance guard).
    assert seen_inflight_user == ["new prompt"]
    snapshot = server._inflight_snapshot(session)
    assert snapshot is None
    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "complete"
