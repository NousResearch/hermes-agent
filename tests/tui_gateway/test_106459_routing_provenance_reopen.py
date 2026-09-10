"""#106459 routing provenance: the TUI clears a stale explicit-close stamp only for a session it
still has registered, under ``_sessions_lock`` as it starts a turn, and never once the session has
been claimed for teardown.

Fourth review probe on #106543: the TUI accepts a prompt and starts a worker before that worker
reaches ``run_conversation()`` and takes its turn lease, so ``session.close`` can pop the session,
wait out its grace and stamp ``tui_close`` in between. Clearing the stamp on lease acquisition
turned that deliberate close back into a live row. The clear now happens in ``_run_prompt_submit``
while ``_sessions_lock`` is held and the session is still registered -- the lock
``_pop_session_by_id`` claims teardown under -- and a late lease clears nothing.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway import server

TURN = f"pid={os.getpid()}:turn=tui:platform=tui"
# Captured at import: several tests replace ``threading.Thread`` (module-global) with a synchronous stand-in,
# and a lock probe must run on a genuinely different thread or an RLock simply re-enters.
_RealThread = threading.Thread


class _ImmediateThread:
    """Run the turn inside ``start()`` so tests observe its final state synchronously."""

    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _Agent:
    model = "test-model"
    provider = "test-provider"

    def __init__(self, session_id: str, db: SessionDB, *, before_lease=None):
        self.session_id = session_id
        self._db = db
        self._before_lease = before_lease
        self.turns: list = []
        self.stamp_seen_by_turn: list = []

    def clear_interrupt(self):
        return None

    def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
        if self._before_lease is not None:
            self._before_lease()
        # What admit_durable_turn_lease does at the top of the real run_conversation.
        self._db.try_acquire_session_turn_lease(self.session_id, TURN, ttl_seconds=300.0)
        self.stamp_seen_by_turn.append(self._db.get_session(self.session_id)["end_reason"])
        self.turns.append(prompt)
        return {"final_response": "", "messages": []}


def _session(agent: _Agent, **extra) -> dict:
    return {
        "agent": agent,
        "session_key": agent.session_id,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture
def db(tmp_path: Path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


@pytest.fixture
def turn_env(monkeypatch, tmp_path, db):
    """The immediate-prompt harness of tests/test_tui_gateway_server.py, with a real SessionDB."""
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_session_info", lambda *_a: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    return monkeypatch


@pytest.fixture
def registry():
    """Sessions registered by a test, removed afterwards."""
    added: list[str] = []

    def _add(sid: str, session: dict) -> dict:
        server._sessions[sid] = session
        added.append(sid)
        return session

    yield _add
    for sid in added:
        server._sessions.pop(sid, None)


def _stamp(db: SessionDB, row_id: str, reason: str = "tui_close", *, age: float = 60.0) -> None:
    db.end_session(row_id, reason)
    if age:
        db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time() - age, row_id))


def test_a_registered_session_is_reopened_before_its_turn_starts(turn_env, db, registry):
    """The #106459 field shape: the TUI keeps accepting prompts on a session whose row carries a
    stale ``tui_close``. The row must already be clear when the worker runs."""
    turn_env.setattr(server.threading, "Thread", _ImmediateThread)
    db.create_session("row-1", source="tui")
    _stamp(db, "row-1")
    agent = _Agent("row-1", db)
    session = registry("ui-1", _session(agent))

    assert server._run_prompt_submit("rid", "ui-1", session, "go") is True

    assert agent.turns == ["go"]
    assert agent.stamp_seen_by_turn == [None], "the stamp must be cleared before the worker starts"
    row = db.get_session("row-1")
    assert row["ended_at"] is None and row["end_reason"] is None


@pytest.mark.parametrize("reason", ["session_reset", "new_session", "compression", "ws_disconnect"])
def test_only_explicit_closes_are_reopened(turn_env, db, registry, reason):
    turn_env.setattr(server.threading, "Thread", _ImmediateThread)
    db.create_session("row-2", source="tui")
    _stamp(db, "row-2", reason)
    session = registry("ui-2", _session(_Agent("row-2", db)))

    server._run_prompt_submit("rid", "ui-2", session, "go")

    assert db.get_session("row-2")["end_reason"] == reason


def test_an_unregistered_session_runs_its_turn_but_keeps_its_stamp(turn_env, db):
    """``can_start`` also admits a session that is not in the registry; nothing proves it is routed
    here, so its turn runs as before but its stamp is not cleared."""
    turn_env.setattr(server.threading, "Thread", _ImmediateThread)
    db.create_session("row-3", source="tui")
    _stamp(db, "row-3")
    agent = _Agent("row-3", db)

    assert server._run_prompt_submit("rid", "ui-3", _session(agent), "go") is True

    assert agent.turns == ["go"]
    assert agent.stamp_seen_by_turn == ["tui_close"]
    assert db.get_session("row-3")["end_reason"] == "tui_close"


def test_a_session_already_claimed_for_teardown_is_not_reopened(turn_env, db, registry):
    turn_env.setattr(server.threading, "Thread", _ImmediateThread)
    db.create_session("row-4", source="tui")
    _stamp(db, "row-4")
    agent = _Agent("row-4", db)
    session = registry("ui-4", _session(agent))
    assert server._pop_session_by_id("ui-4") is session  # session.close claimed it

    assert server._run_prompt_submit("rid", "ui-4", session, "go") is False

    assert agent.turns == []
    assert db.get_session("row-4")["end_reason"] == "tui_close"


def test_a_close_that_wins_the_race_after_admission_is_not_reopened(turn_env, db, registry):
    """``_admit_prompt_turn`` checks ``_closing`` under ``history_lock`` only, which does not exclude
    ``_pop_session_by_id``. A close claimed between admission and the start gate must keep its row's
    stamp -- the heal belongs under ``_sessions_lock`` with the gate, not in admission."""
    db.create_session("row-5", source="tui")
    _stamp(db, "row-5")
    agent = _Agent("row-5", db)
    session = registry("ui-5", _session(agent))
    emit_entered, release_emit = threading.Event(), threading.Event()

    def _blocking_emit(event, *_a, **_k):
        if event == "message.start":
            emit_entered.set()
            assert release_emit.wait(timeout=2.0)

    turn_env.setattr(server, "_emit", _blocking_emit)
    results: list = []
    dispatch = threading.Thread(target=lambda: results.append(
        server._run_prompt_submit("rid", "ui-5", session, "go")))
    try:
        dispatch.start()
        assert emit_entered.wait(timeout=1.0)
        assert server._pop_session_by_id("ui-5") is session
    finally:
        release_emit.set()
        dispatch.join(timeout=2.0)

    assert results == [False]
    assert agent.turns == []
    assert db.get_session("row-5")["end_reason"] == "tui_close"


def test_a_late_worker_lease_after_session_close_leaves_the_close(turn_env, db, registry):
    """Fourth review probe: the prompt is accepted and the worker started, the user closes the session
    (pop, bounded grace, ``tui_close``) while the worker is still in pre-turn work, and only then does
    the worker take its turn lease. Nothing re-applies the close afterwards, so it must survive."""
    db.create_session("row-6", source="tui")
    worker_waiting, release_worker = threading.Event(), threading.Event()

    def _hold_before_lease():
        worker_waiting.set()
        assert release_worker.wait(timeout=5.0)

    agent = _Agent("row-6", db, before_lease=_hold_before_lease)
    session = registry("ui-6", _session(agent))
    stamped: list = []

    def _teardown(popped, *, end_reason="tui_close"):
        db.end_session(popped["agent"].session_id, end_reason)  # what _finalize_session writes
        stamped.append(end_reason)

    turn_env.setattr(server, "_teardown_session", _teardown)
    turn_env.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.2, raising=False)
    try:
        assert server._run_prompt_submit("rid", "ui-6", session, "go") is True
        assert worker_waiting.wait(timeout=2.0)
        assert server._close_session_by_id("ui-6") is True
        assert stamped == ["tui_close"]
    finally:
        release_worker.set()
        worker = session.get("_run_thread")
        if worker is not None:
            worker.join(timeout=5.0)

    assert agent.turns == ["go"]  # the late worker still runs its turn, as at the merge base
    assert agent.stamp_seen_by_turn == ["tui_close"]
    assert db.get_session("row-6")["end_reason"] == "tui_close"


def test_the_session_db_is_resolved_before_the_sessions_lock_is_taken(turn_env, db, registry):
    """Resolving a profile session's handle goes through the state registry; it must not run under the
    lock that gates every create/close/prompt on this backend."""
    turn_env.setattr(server.threading, "Thread", _ImmediateThread)
    db.create_session("row-7", source="tui")
    _stamp(db, "row-7")
    session = registry("ui-7", _session(_Agent("row-7", db)))
    real_session_db = server._session_db
    lock_free_at_resolution: list[bool] = []

    @contextlib.contextmanager
    def _probing_session_db(sess):
        callers = {sys._getframe(depth).f_code.co_name for depth in range(1, 5)}
        if "_run_prompt_submit" in callers:
            probe: list[bool] = []

            def _try_lock_from_another_thread():
                # Acquire and release on the SAME thread: an RLock left owned by a finished thread stays held.
                got = server._sessions_lock.acquire(timeout=0.5)
                probe.append(got)
                if got:
                    server._sessions_lock.release()

            prober = _RealThread(target=_try_lock_from_another_thread)
            prober.start()
            prober.join()
            lock_free_at_resolution.append(bool(probe and probe[0]))
        with real_session_db(sess) as handle:
            yield handle

    turn_env.setattr(server, "_session_db", _probing_session_db)

    assert server._run_prompt_submit("rid", "ui-7", session, "go") is True

    assert lock_free_at_resolution == [True]
    assert db.get_session("row-7")["end_reason"] is None


def test_a_failed_routing_reopen_never_blocks_the_turn(turn_env, db, registry):
    turn_env.setattr(server.threading, "Thread", _ImmediateThread)
    db.create_session("row-8", source="tui")
    _stamp(db, "row-8")
    agent = _Agent("row-8", db)
    session = registry("ui-8", _session(agent))

    def _unavailable(*_a, **_k):
        raise RuntimeError("database is locked")

    turn_env.setattr(db, "reopen_if_explicitly_closed", _unavailable)

    assert server._run_prompt_submit("rid", "ui-8", session, "go") is True

    assert agent.turns == ["go"]
    assert db.get_session("row-8")["end_reason"] == "tui_close"
