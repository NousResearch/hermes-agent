"""Crash-interrupted turns auto-continue on the next session.resume.

A turn's durable marker (tui_gateway/turn_marker.py) is written when the turn
starts running and cleared when it concludes — success, handled error, or
interrupt. Only a process death leaves it behind, so a marker found at resume
time is positive proof the turn never finished. Contract pinned here:

* the marker module round-trips, prunes stale entries, and tolerates a
  corrupt sidecar;
* ``_run_prompt_submit`` writes the marker before the turn and clears it in
  the ``finally`` on both the success and exception paths (a handled failure
  is a concluded turn — its terminal frame + retained snapshot own recovery);
* ``_maybe_schedule_auto_continue`` re-submits a fresh interrupted prompt as
  a continuation note (display_kind ``auto_continue``), refuses stale /
  disabled / crash-looping / already-running cases, and bounds attempts via
  the marker's attempt counter.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

from tui_gateway import server
from tui_gateway.turn_marker import (
    clear_turn_marker,
    read_turn_marker,
    record_turn_start,
)


class _InlineThread:
    """Run threads synchronously so tests observe final state."""

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
def marker_home(monkeypatch, tmp_path):
    """Point the server's marker storage at a temp HERMES_HOME."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture()
def turn_env(monkeypatch, tmp_path, marker_home):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


# ── Marker module ──────────────────────────────────────────────────────


def test_marker_roundtrip(tmp_path):
    record_turn_start(tmp_path, "abc", "fix the bug", attempts=1)

    marker = read_turn_marker(tmp_path, "abc")
    assert marker is not None
    assert marker["prompt"] == "fix the bug"
    assert marker["attempts"] == 1
    assert marker["started_at"] == pytest.approx(time.time(), abs=5)

    clear_turn_marker(tmp_path, "abc")
    assert read_turn_marker(tmp_path, "abc") is None


def test_marker_clear_is_noop_without_entry(tmp_path):
    clear_turn_marker(tmp_path, "missing")
    assert read_turn_marker(tmp_path, "missing") is None


def test_marker_write_prunes_expired_entries(tmp_path):
    import json

    path = tmp_path / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"old": {"attempts": 0, "prompt": "ancient prompt", "started_at": 1.0}})
    )

    record_turn_start(tmp_path, "new", "current prompt")

    assert read_turn_marker(tmp_path, "old") is None
    assert read_turn_marker(tmp_path, "new") is not None


def test_marker_survives_corrupt_sidecar(tmp_path):
    path = tmp_path / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    assert read_turn_marker(tmp_path, "abc") is None
    record_turn_start(tmp_path, "abc", "prompt")
    assert read_turn_marker(tmp_path, "abc")["prompt"] == "prompt"


# ── Turn lifecycle owns the marker ─────────────────────────────────────


def test_concluded_turn_clears_marker(emits, turn_env, marker_home):
    seen_mid_turn: list = []

    def _run(message, **kwargs):
        seen_mid_turn.append(read_turn_marker(marker_home, "session-key"))
        return {"final_response": "done"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True)

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    # Written before the turn ran (this is what survives a process death) …
    assert seen_mid_turn and seen_mid_turn[0] is not None
    assert seen_mid_turn[0]["prompt"] == "do the thing"
    assert seen_mid_turn[0]["attempts"] == 0
    # … and cleared once the turn concluded.
    assert read_turn_marker(marker_home, "session-key") is None


@pytest.mark.parametrize(
    "result", [{"final_response": "done"}, {"error": "provider said no"}]
)
def test_marker_is_gone_by_the_terminal_frame(monkeypatch, turn_env, marker_home, result):
    """The client treats message.complete as "turn over" and may quit right
    then; post-turn work (titles, memory, goal hooks) keeps the thread alive
    for a second or more afterwards. If the marker outlived the frame, that
    quit looked like a crash and re-ran a finished turn on the next launch."""
    at_frame: list = []

    def _emit(event, sid, payload=None):
        if event == "message.complete":
            at_frame.append(read_turn_marker(marker_home, "session-key"))

    monkeypatch.setattr(server, "_emit", _emit)
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda message, **kwargs: result,
        clear_interrupt=lambda: None,
    )

    server._run_prompt_submit("rid", "sid", _session(agent=agent, running=True), "do the thing")

    assert at_frame == [None]


def test_handled_failure_still_clears_marker(emits, turn_env, marker_home):
    """An exception is a CONCLUDED turn (terminal frame + retained snapshot own
    recovery) — only a process death may leave the marker behind."""

    def _boom(message, **kwargs):
        raise RuntimeError("provider exploded")

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_boom, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True)

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    assert read_turn_marker(marker_home, "session-key") is None


def test_continuation_turn_records_attempt_and_original_prompt(
    emits, turn_env, marker_home
):
    """A continuation's marker must carry the attempt count (crash-loop
    breaker) and the ORIGINAL prompt — recording its own recovery note would
    nest note inside note on a second crash."""
    seen: list = []

    def _run(message, **kwargs):
        seen.append(read_turn_marker(marker_home, "session-key"))
        return {"final_response": "done"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(
        agent=agent,
        running=True,
        _auto_continue_attempt=2,
        _auto_continue_prompt="the original prompt",
    )

    server._run_prompt_submit("rid", "sid", session, server._auto_continue_note("the original prompt"))

    assert [(m["attempts"], m["prompt"]) for m in seen] == [(2, "the original prompt")]
    # Consumed, so the NEXT user turn starts from a clean slate.
    assert "_auto_continue_attempt" not in session
    assert "_auto_continue_prompt" not in session


# ── Scheduling decision ────────────────────────────────────────────────


@pytest.fixture()
def schedule_env(monkeypatch, marker_home):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda session, rid, timeout=30.0: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    submitted: list = []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, session, text, **kw: submitted.append((text, kw)),
    )
    return submitted


def test_fresh_marker_schedules_continuation(emits, schedule_env, marker_home):
    record_turn_start(marker_home, "session-key", "fix the flaky test")
    session = _session()

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert result is not None
    assert result["attempt"] == 1
    assert session["running"] is True
    assert session["_auto_continue_attempt"] == 1
    (text, kwargs), = schedule_env
    assert text.startswith("[System note: Your previous turn was interrupted")
    assert "fix the flaky test" in text
    assert kwargs["display_kind"] == "auto_continue"
    assert ("message.start", "sid", None) in [(e, s, p) for e, s, p in emits]


def test_stale_marker_is_cleared_not_continued(schedule_env, marker_home, monkeypatch):
    record_turn_start(marker_home, "session-key", "old prompt")
    monkeypatch.setattr(
        server, "time", types.SimpleNamespace(time=lambda: time.time() + 3600)
    )

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is None
    assert not schedule_env
    assert read_turn_marker(marker_home, "session-key") is None


def test_config_widens_freshness_window(emits, schedule_env, marker_home, monkeypatch):
    record_turn_start(marker_home, "session-key", "old prompt")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"desktop": {"auto_continue": {"freshness_minutes": 120}}},
    )
    monkeypatch.setattr(
        server, "time", types.SimpleNamespace(time=lambda: time.time() + 3600)
    )

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is not None
    assert len(schedule_env) == 1


def test_exhausted_attempts_break_the_loop(schedule_env, marker_home):
    record_turn_start(marker_home, "session-key", "crashy prompt", attempts=2)

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is None
    assert not schedule_env
    assert read_turn_marker(marker_home, "session-key") is None


def test_disabled_by_config(schedule_env, marker_home, monkeypatch):
    record_turn_start(marker_home, "session-key", "prompt")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"desktop": {"auto_continue": {"enabled": False}}},
    )

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is None
    assert not schedule_env


def test_no_marker_means_no_continuation(schedule_env, marker_home):
    assert server._maybe_schedule_auto_continue("sid", _session(), "session-key") is None
    assert not schedule_env


def test_running_session_wins_over_continuation(emits, schedule_env, marker_home):
    """A real user prompt that raced the kickoff keeps its turn; the marker is
    left for that turn's own conclusion to clear."""
    record_turn_start(marker_home, "session-key", "prompt")
    session = _session(running=True)

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    # Scheduled (the descriptor is returned), but the kickoff bailed.
    assert result is not None
    assert not schedule_env
    assert session["_auto_continue_scheduled"] is False
    assert read_turn_marker(marker_home, "session-key") is not None
    # Nothing left behind for the racing user turn to inherit.
    assert "_auto_continue_attempt" not in session
    assert "_auto_continue_prompt" not in session


def test_double_schedule_is_guarded(emits, schedule_env, marker_home):
    record_turn_start(marker_home, "session-key", "prompt")
    session = _session()

    first = server._maybe_schedule_auto_continue("sid", session, "session-key")
    second = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert first is not None
    assert second is None
    assert len(schedule_env) == 1


def test_failed_agent_build_leaves_marker_for_retry(
    emits, schedule_env, marker_home, monkeypatch
):
    record_turn_start(marker_home, "session-key", "prompt")
    monkeypatch.setattr(
        server,
        "_wait_agent",
        lambda session, rid, timeout=30.0: {"error": {"message": "boom"}},
    )
    session = _session()

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert result is not None
    assert not schedule_env
    assert session["_auto_continue_scheduled"] is False
    assert read_turn_marker(marker_home, "session-key") is not None


# ── End to end: continuation runs a real turn and clears the marker ────


def test_continuation_runs_through_turn_pipeline(emits, turn_env, marker_home, monkeypatch):
    record_turn_start(marker_home, "session-key", "finish the migration")
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda session, rid, timeout=30.0: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    prompts: list = []

    def _run(message, **kwargs):
        prompts.append(message)
        return {"final_response": "continued and finished"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(agent=agent)

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert result == {
        "attempt": 1,
        "interrupted_at": pytest.approx(time.time(), abs=5),
    }
    assert len(prompts) == 1
    assert "finish the migration" in prompts[0]
    # The concluded continuation cleared both the marker and the turn state.
    assert read_turn_marker(marker_home, "session-key") is None
    assert session["running"] is False
    completes = [p for e, _s, p in emits if e == "message.complete"]
    assert completes and completes[0]["status"] == "complete"
