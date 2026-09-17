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

import json
import multiprocessing
import os
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

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
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


def test_marker_clear_is_bound_to_turn_identity(tmp_path):
    record_turn_start(tmp_path, "abc", "old", turn_id="old-turn")
    record_turn_start(tmp_path, "abc", "new", turn_id="new-turn")

    clear_turn_marker(tmp_path, "abc", expected_turn_id="old-turn")

    marker = read_turn_marker(tmp_path, "abc")
    assert marker is not None
    assert marker["prompt"] == "new"
    assert marker["turn_id"] == "new-turn"


def test_stale_marker_cleanup_cannot_delete_new_turn_marker(monkeypatch, tmp_path):
    record_turn_start(tmp_path, "abc", "old", turn_id="old-turn")
    real_clear = clear_turn_marker

    def replace_then_clear(home, key, **kwargs):
        record_turn_start(home, key, "new", turn_id="new-turn")
        real_clear(home, key, **kwargs)

    monkeypatch.setattr(server, "clear_turn_marker", replace_then_clear)
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"desktop": {"auto_continue": {"enabled": False}}},
    )

    session = _session(profile_home=str(tmp_path))
    assert server._maybe_schedule_auto_continue("sid", session, "abc") is None
    marker = read_turn_marker(tmp_path, "abc")
    assert marker is not None
    assert marker["turn_id"] == "new-turn"
    assert marker["prompt"] == "new"


@pytest.mark.skipif(os.name == "nt", reason="fork-only interprocess race harness")
def test_marker_compare_and_swap_is_serialized_between_processes(tmp_path):
    record_turn_start(tmp_path, "same-key", "old", turn_id="turn-old")
    ctx = multiprocessing.get_context("fork")
    loaded = ctx.Event()
    resume = ctx.Event()

    def stale_clear():
        from tui_gateway import turn_marker

        real_load = turn_marker._load

        def paused_load(path):
            entries = real_load(path)
            loaded.set()
            assert resume.wait(5)
            return entries

        turn_marker._load = paused_load
        turn_marker.clear_turn_marker(
            tmp_path, "same-key", expected_turn_id="turn-old"
        )

    def write_new():
        record_turn_start(tmp_path, "same-key", "new", turn_id="turn-new")

    clearer = ctx.Process(target=stale_clear)
    writer = ctx.Process(target=write_new)
    clearer.start()
    assert loaded.wait(5)
    writer.start()
    time.sleep(0.1)
    assert writer.is_alive()
    resume.set()
    clearer.join(5)
    writer.join(5)
    assert clearer.exitcode == 0
    assert writer.exitcode == 0
    marker = read_turn_marker(tmp_path, "same-key")
    assert marker is not None
    assert marker["turn_id"] == "turn-new"


def test_personal_crash_marker_persists_only_non_secret_blocked_flag(tmp_path):
    secret = "never-write-this-bearer"
    record_turn_start(
        tmp_path,
        "personal",
        "continue safely",
        personal_authorization_blocked=True,
    )

    raw = (tmp_path / "desktop" / "interrupted_turns.json").read_text()
    marker = read_turn_marker(tmp_path, "personal")

    assert secret not in raw
    assert json.loads(raw)["personal"]["personal_authorization_blocked"] is True
    assert marker["personal_authorization_blocked"] is True


def test_marker_survives_corrupt_sidecar(tmp_path):
    path = tmp_path / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    assert read_turn_marker(tmp_path, "abc") is None
    record_turn_start(tmp_path, "abc", "prompt")
    assert read_turn_marker(tmp_path, "abc")["prompt"] == "prompt"


def _patch_local_interrupt(monkeypatch, session):
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda current: False)
    monkeypatch.setattr(server, "_clear_pending", lambda sid=None: None)


def test_interrupt_ack_retires_marker_before_run_thread_exits(monkeypatch, marker_home):
    """A confirmed Stop must not auto-continue if the backend dies afterward."""

    class _AliveThread:
        def is_alive(self):
            return True

    interrupted = []
    agent = types.SimpleNamespace(interrupt=lambda: interrupted.append(True))
    session = _session(
        agent=agent,
        running=True,
        _active_turn_id="turn-1",
        _run_thread=_AliveThread(),
        _active_turn_marker_key="original-key",
    )
    session["session_key"] = "rotated-key"
    session_home = marker_home / "remote-profile"
    session["profile_home"] = str(session_home)
    record_turn_start(
        session_home, "original-key", "do not resume me", turn_id="turn-1"
    )

    _patch_local_interrupt(monkeypatch, session)

    response = server._methods["session.interrupt"]("request-1", {"session_id": "runtime-1"})

    assert response["result"]["status"] == "interrupted"
    assert interrupted == [True]
    assert read_turn_marker(session_home, "original-key") is None
    assert read_turn_marker(session_home, "rotated-key") is None
    assert "_active_turn_marker_key" not in session


def test_interrupt_racing_marker_write_cannot_leave_recovery_state(
    monkeypatch, emits, turn_env, marker_home
):
    """Stop before the disk write must still prevent later auto-continue."""

    interrupted = []
    agent = types.SimpleNamespace(
        session_id="session-key",
        clear_interrupt=lambda: None,
        interrupt=lambda: interrupted.append(True),
        run_conversation=lambda message, **kwargs: {"final_response": "stopped"},
    )
    session = _session(agent=agent, running=True)
    _patch_local_interrupt(monkeypatch, session)

    def write_after_stop(
        home, key, prompt, *, attempts=0, auto_continue=True,
        personal_authorization_blocked=False, turn_id=None,
        should_record=None,
    ):
        response = server._methods["session.interrupt"](
            "stop-during-write", {"session_id": "runtime-race"}
        )
        assert response["result"]["status"] == "interrupted"
        record_turn_start(
            home, key, prompt, attempts=attempts, auto_continue=auto_continue,
            personal_authorization_blocked=personal_authorization_blocked,
            turn_id=turn_id,
            should_record=should_record,
        )

    monkeypatch.setattr(server, "record_turn_start", write_after_stop)

    server._run_prompt_submit("request-race", "runtime-race", session, "race me")

    assert interrupted == [True]
    assert read_turn_marker(marker_home, "session-key") is None
    assert "_active_turn_marker_key" not in session


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


def test_hosted_terminal_receipt_commits_before_marker_retire(
    emits, turn_env, marker_home
):
    observed = []

    def _run(message, **kwargs):
        return {"final_response": "done"}

    def _terminal(receipt):
        observed.append((receipt, read_turn_marker(marker_home, "session-key")))

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True, source="bot_room")

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        "do the thing",
        terminal_callback=_terminal,
    )

    assert observed[0][0]["status"] == "settled"
    assert observed[0][1] is not None
    assert read_turn_marker(marker_home, "session-key") is None


def test_hosted_terminal_receipt_failure_keeps_crash_marker(
    emits, turn_env, marker_home
):
    def _run(message, **kwargs):
        return {"final_response": "done"}

    def _terminal(_receipt):
        raise RuntimeError("state store unavailable")

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True, source="bot_room")

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        "do the thing",
        terminal_callback=_terminal,
    )

    assert read_turn_marker(marker_home, "session-key") is not None


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


def test_older_agent_still_gets_the_post_turn_stamp(emits, turn_env, marker_home):
    """An agent whose run_conversation predates turn-start typing keeps the
    original behavior — the row is typed once the turn concludes."""
    stamped: list = []

    class _LegacyDB:
        def set_latest_matching_message_display_kind(self, session_id, **kwargs):
            stamped.append((session_id, kwargs["display_kind"]))
            return True

    def _run(message, conversation_history=None, stream_callback=None, **_kwargs):
        return {"final_response": "done"}

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=_run,
        clear_interrupt=lambda: None,
        _session_db=_LegacyDB(),
    )
    note = server._auto_continue_note("the original prompt")

    server._run_prompt_submit(
        "rid", "sid", _session(agent=agent, running=True), note,
        display_kind="auto_continue",
    )

    assert stamped == [("session-key", "auto_continue")]


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
    assert kwargs["turn_authorization"].is_personal is False
    assert kwargs["expected_turn_id"] == session["_active_turn_id"]
    assert ("message.start", "sid", None) in [(e, s, p) for e, s, p in emits]


def test_auto_continue_diagnostic_failure_still_releases_claim(
    monkeypatch, marker_home
):
    record_turn_start(marker_home, "session-key", "retry safely")
    session = _session()
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("dispatch failed")
        ),
    )
    monkeypatch.setattr(
        server,
        "_notif_log_failure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("diagnostic failed")
        ),
    )

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert result is not None
    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_route" not in session


def test_personal_marker_auto_continue_uses_blocked_authorization(
    emits, schedule_env, marker_home
):
    record_turn_start(
        marker_home,
        "session-key",
        "personal work",
        personal_authorization_blocked=True,
    )

    assert server._maybe_schedule_auto_continue("sid", _session(), "session-key") is not None

    (_text, kwargs), = schedule_env
    authorization = kwargs["turn_authorization"]
    assert authorization.is_personal is True
    assert authorization.has_token is False


def test_hosted_room_marker_is_left_to_the_driver(schedule_env, marker_home):
    record_turn_start(marker_home, "session-key", "hosted prompt")

    result = server._maybe_schedule_auto_continue(
        "sid",
        _session(source="bot_room"),
        "session-key",
    )

    assert result is None
    assert not schedule_env
    assert read_turn_marker(marker_home, "session-key") is not None


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

