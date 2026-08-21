"""Crash-interrupted turns auto-continue on the next session.resume.

A turn's durable record (``interrupted_turns`` in state.db) is written when the
turn starts running and cleared when it concludes — success, handled error, or
interrupt. Only a process death leaves it behind, so a record found at resume
time is positive proof the turn never finished. Contract pinned here:

* ``_run_prompt_submit`` writes the record before the turn and clears it in
  the ``finally`` on both the success and exception paths (a handled failure
  is a concluded turn — its terminal frame + retained snapshot own recovery);
* the retire is owner-checked: a process that never ran the turn cannot delete
  the record of one still running elsewhere;
* ``_maybe_schedule_auto_continue`` re-submits a fresh interrupted prompt as
  a continuation note (display_kind ``auto_continue``), refuses stale /
  disabled / crash-looping / already-running cases, and bounds attempts via
  the record's attempt counter;
* the legacy JSON sidecar is imported once per HERMES_HOME and renamed aside,
  with its keys resolved to the compression-lineage root the table uses.
"""

from __future__ import annotations

import json
import threading
import time
import types

import pytest

from hermes_state import SessionDB
from tui_gateway import server, turn_marker

_FOREIGN_OWNER = "pid=999999:platform=other"


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


def _record(db, session_key, prompt, *, attempts=0, owner=_FOREIGN_OWNER):
    """A record left by some other process's interrupted turn."""
    assert db.record_interrupted_turn(
        session_key, prompt, attempts=attempts, owner=owner
    )


def _read(db, session_key):
    return db.read_interrupted_turn(session_key)


def _write_sidecar(home, entries):
    path = home / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


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
    """Point the server's session home at a temp HERMES_HOME."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture()
def turn_db(monkeypatch, marker_home):
    """Point the server's state.db handle at a temp database."""
    db = SessionDB(marker_home / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    return db


@pytest.fixture()
def turn_env(monkeypatch, tmp_path, turn_db):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


# ── Record storage ─────────────────────────────────────────────────────


def test_record_roundtrip(turn_db):
    _record(turn_db, "abc", "fix the bug", attempts=1)

    marker = _read(turn_db, "abc")
    assert marker is not None
    assert marker["prompt"] == "fix the bug"
    assert marker["attempts"] == 1
    assert marker["started_at"] == pytest.approx(time.time(), abs=5)

    assert turn_db.clear_interrupted_turn("abc", owner=_FOREIGN_OWNER)
    assert _read(turn_db, "abc") is None


def test_a_foreign_process_cannot_retire_a_live_record(turn_db, marker_home):
    """The record of a turn running elsewhere survives this process's retire.

    A second process that submits on a busy conversation waits for the turn
    lease, times out, and emits a terminal error frame — and the frame path
    retires the record as it goes. Before the owner check that deleted the
    scheduler's only durable pointer to the running turn: the prompt row
    itself survives in the session DB, so what it cost was the automatic
    resume, not the data.
    """
    _record(turn_db, "session-key", "the turn that is actually running")

    server._retire_turn_marker(_session())

    survivor = _read(turn_db, "session-key")
    assert survivor is not None
    assert survivor["prompt"] == "the turn that is actually running"


def test_retire_tolerates_the_keyless_call(turn_env, turn_db):
    """``_emit_terminal_turn_error`` retires with no explicit keys."""
    session = _session()
    server._record_interrupted_turn(session, "session-key", "own turn")
    assert _read(turn_db, "session-key") is not None

    server._retire_turn_marker(session)

    assert _read(turn_db, "session-key") is None


def test_retire_covers_a_key_compression_rotated_mid_turn(turn_env, turn_db):
    session = _session(session_key="rotated-key")
    server._record_interrupted_turn(session, "pre-rotation-key", "own turn")

    server._retire_turn_marker(session, "pre-rotation-key")

    assert _read(turn_db, "pre-rotation-key") is None


# ── Turn lifecycle owns the record ─────────────────────────────────────


def test_concluded_turn_clears_marker(emits, turn_env, turn_db):
    seen_mid_turn: list = []

    def _run(message, **kwargs):
        seen_mid_turn.append(_read(turn_db, "session-key"))
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
    assert _read(turn_db, "session-key") is None


def test_handled_failure_still_clears_marker(emits, turn_env, turn_db):
    """An exception is a CONCLUDED turn (terminal frame + retained snapshot own
    recovery) — only a process death may leave the record behind."""

    def _boom(message, **kwargs):
        raise RuntimeError("provider exploded")

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_boom, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True)

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    assert _read(turn_db, "session-key") is None


def test_continuation_turn_records_attempt_and_original_prompt(
    emits, turn_env, turn_db
):
    """A continuation's record must carry the attempt count (crash-loop
    breaker) and the ORIGINAL prompt — recording its own recovery note would
    nest note inside note on a second crash."""
    seen: list = []

    def _run(message, **kwargs):
        seen.append(_read(turn_db, "session-key"))
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


def test_continuation_takes_over_a_dead_process_record(emits, turn_env, turn_db):
    """The attempts counter has to advance across a restart.

    The record being continued was written by the process that died. Refusing
    to overwrite another owner's row would freeze the counter and defeat the
    crash-loop breaker, so a turn start takes the row over and stamps itself.
    """
    _record(turn_db, "session-key", "the original prompt", attempts=1)
    seen: list = []

    def _run(message, **kwargs):
        seen.append(_read(turn_db, "session-key"))
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

    server._run_prompt_submit("rid", "sid", session, "note")

    assert seen[0]["attempts"] == 2
    assert seen[0]["owner"] == server._turn_record_owner()
    # This process owns it now, so its own retire lands.
    assert _read(turn_db, "session-key") is None


def test_older_agent_still_gets_the_post_turn_stamp(emits, turn_env, turn_db):
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
def schedule_env(monkeypatch, turn_db):
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


def test_fresh_marker_schedules_continuation(emits, schedule_env, turn_db):
    _record(turn_db, "session-key", "fix the flaky test")
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


def test_stale_marker_is_cleared_not_continued(schedule_env, turn_db, monkeypatch):
    _record(turn_db, "session-key", "old prompt")
    monkeypatch.setattr(
        server, "time", types.SimpleNamespace(time=lambda: time.time() + 3600)
    )

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is None
    assert not schedule_env
    assert _read(turn_db, "session-key") is None


def test_config_widens_freshness_window(emits, schedule_env, turn_db, monkeypatch):
    _record(turn_db, "session-key", "old prompt")
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


def test_exhausted_attempts_break_the_loop(schedule_env, turn_db):
    _record(turn_db, "session-key", "crashy prompt", attempts=2)

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is None
    assert not schedule_env
    assert _read(turn_db, "session-key") is None


def test_disabled_by_config(schedule_env, turn_db, monkeypatch):
    """Disabled here says nothing about a record another process owns.

    ``enabled`` is this process's config, and the process that wrote the
    record may be running with auto-continue on — so declining to continue is
    not licence to retire someone else's live record.
    """
    _record(turn_db, "session-key", "the turn that is actually running")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"desktop": {"auto_continue": {"enabled": False}}},
    )

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is None
    assert not schedule_env
    survivor = _read(turn_db, "session-key")
    assert survivor is not None
    assert survivor["prompt"] == "the turn that is actually running"


def test_disabled_by_config_still_clears_own_record(schedule_env, turn_db, monkeypatch):
    """Its own record is retired: this process will never continue that turn."""
    session = _session()
    server._record_interrupted_turn(session, "session-key", "own turn")
    assert _read(turn_db, "session-key") is not None
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"desktop": {"auto_continue": {"enabled": False}}},
    )

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert result is None
    assert not schedule_env
    assert _read(turn_db, "session-key") is None


def test_no_marker_means_no_continuation(schedule_env, turn_db):
    assert server._maybe_schedule_auto_continue("sid", _session(), "session-key") is None
    assert not schedule_env


def test_running_session_wins_over_continuation(emits, schedule_env, turn_db):
    """A real user prompt that raced the kickoff keeps its turn; the record is
    left for that turn's own conclusion to clear."""
    _record(turn_db, "session-key", "prompt")
    session = _session(running=True)

    result = server._maybe_schedule_auto_continue("sid", session, "session-key")

    # Scheduled (the descriptor is returned), but the kickoff bailed.
    assert result is not None
    assert not schedule_env
    assert session["_auto_continue_scheduled"] is False
    assert _read(turn_db, "session-key") is not None
    # Nothing left behind for the racing user turn to inherit.
    assert "_auto_continue_attempt" not in session
    assert "_auto_continue_prompt" not in session


def test_double_schedule_is_guarded(emits, schedule_env, turn_db):
    _record(turn_db, "session-key", "prompt")
    session = _session()

    first = server._maybe_schedule_auto_continue("sid", session, "session-key")
    second = server._maybe_schedule_auto_continue("sid", session, "session-key")

    assert first is not None
    assert second is None
    assert len(schedule_env) == 1


def test_failed_agent_build_leaves_marker_for_retry(
    emits, schedule_env, turn_db, monkeypatch
):
    _record(turn_db, "session-key", "prompt")
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
    assert _read(turn_db, "session-key") is not None


# ── Legacy sidecar import ──────────────────────────────────────────────


def test_legacy_sidecar_imports_and_is_renamed_aside(schedule_env, turn_db, marker_home):
    path = _write_sidecar(
        marker_home,
        {
            "session-key": {
                "attempts": 1,
                "prompt": "the prompt the old build recorded",
                "started_at": time.time(),
            }
        },
    )

    result = server._maybe_schedule_auto_continue("sid", _session(), "session-key")

    assert result is not None
    assert result["attempt"] == 2
    (text, _kwargs), = schedule_env
    assert "the prompt the old build recorded" in text
    assert not path.exists()
    assert (marker_home / "desktop" / "interrupted_turns.json.migrated").is_file()


def test_legacy_sidecar_key_is_resolved_to_the_conversation_root(
    schedule_env, turn_db, marker_home
):
    """The sidecar filed records under the compression SEGMENT.

    A record written before a rotation has to land on the row the resume after
    the rotation reads, which is the lineage root.
    """
    turn_db.create_session("root", source="test")
    turn_db.end_session("root", "compression")
    turn_db.create_session("child", source="test", parent_session_id="root")
    _write_sidecar(
        marker_home,
        {
            "root": {
                "attempts": 0,
                "prompt": "recorded before the rotation",
                "started_at": time.time(),
            }
        },
    )

    result = server._maybe_schedule_auto_continue(
        "sid", _session(session_key="child"), "child"
    )

    assert result is not None
    (text, _kwargs), = schedule_env
    assert "recorded before the rotation" in text


def test_imported_record_carries_no_owner_and_stays_retirable(turn_db, marker_home):
    _write_sidecar(
        marker_home,
        {
            "session-key": {
                "attempts": 0,
                "prompt": "legacy prompt",
                "started_at": time.time(),
            }
        },
    )
    session = _session()

    server._migrate_turn_markers(session)

    record = _read(turn_db, "session-key")
    assert record["owner"] is None
    assert record["cause"] == "migrated"
    server._retire_turn_marker(session)
    assert _read(turn_db, "session-key") is None


def test_import_does_not_overwrite_a_live_record(turn_db, marker_home):
    _record(turn_db, "session-key", "the live record", attempts=1)
    _write_sidecar(
        marker_home,
        {
            "session-key": {
                "attempts": 0,
                "prompt": "the legacy record",
                "started_at": time.time() - 60,
            }
        },
    )

    server._migrate_turn_markers(_session())

    record = _read(turn_db, "session-key")
    assert record["prompt"] == "the live record"
    assert record["attempts"] == 1


def test_corrupt_legacy_sidecar_is_retired_without_breaking_resume(
    schedule_env, turn_db, marker_home
):
    """The unreadable-file case the sidecar had to tolerate, now at import."""
    path = marker_home / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    session = _session()

    assert server._maybe_schedule_auto_continue("sid", session, "session-key") is None
    assert not path.exists()

    # Recording still works afterwards.
    server._record_interrupted_turn(session, "session-key", "prompt")
    assert _read(turn_db, "session-key")["prompt"] == "prompt"


def test_failed_import_leaves_the_sidecar_for_the_next_resume(
    turn_db, marker_home, monkeypatch
):
    path = _write_sidecar(
        marker_home,
        {
            "session-key": {
                "attempts": 0,
                "prompt": "legacy prompt",
                "started_at": time.time(),
            }
        },
    )

    def _boom(home):
        raise OSError("rename denied")

    monkeypatch.setattr(server, "retire_sidecar", _boom)
    server._migrate_turn_markers(_session())
    assert path.is_file()

    monkeypatch.setattr(server, "retire_sidecar", turn_marker.retire_sidecar)
    server._migrate_turn_markers(_session())
    assert not path.exists()
    assert _read(turn_db, "session-key")["prompt"] == "legacy prompt"


# ── End to end: continuation runs a real turn and clears the record ────
