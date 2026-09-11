"""#106459: a stale explicit-close stamp (``tui_close`` on a session the TUI still routes) wedged
compression forever, because publish fails closed on it and ``end_session()`` is first-stamp-wins.
The TUI clears it under ``_sessions_lock`` as it starts a turn for a still-registered session --
and only then: the store never decides on its own that an explicit close is stale.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway import server

TURN = f"pid={os.getpid()}:turn=tui:platform=tui"


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

    def __init__(self, session_id: str, db: SessionDB):
        self.session_id = session_id
        self._db = db
        self.turns: list = []
        self.stamp_seen_by_turn: list = []

    def clear_interrupt(self):
        return None

    def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
        # What admit_durable_turn_lease does at the top of the real run_conversation.
        self._db.try_acquire_session_turn_lease(self.session_id, TURN, ttl_seconds=300.0)
        self.stamp_seen_by_turn.append(self._db.get_session(self.session_id)["end_reason"])
        self.turns.append(prompt)
        return {"final_response": "", "messages": []}


def _session(agent: _Agent) -> dict:
    return {
        "agent": agent, "session_key": agent.session_id, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
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
    for name, stub in {
        "_emit": lambda *_a, **_k: None, "make_stream_renderer": lambda _cols: None,
        "render_message": lambda _raw, _cols: None, "_wire_callbacks": lambda _sid: None,
        "_sync_agent_model_with_config": lambda *_a: None, "_session_cwd": lambda _session: str(tmp_path),
        "_register_session_cwd": lambda _session: None, "_set_session_context": lambda *_a, **_k: [],
        "_clear_session_context": lambda _tokens: None, "_session_info": lambda *_a: {},
        "_get_usage": lambda _agent: {}, "_sync_session_key_after_compress": lambda *_a, **_k: None,
        "_drain_queued_prompt": lambda *_a: False, "_voice_tts_enabled": lambda: False, "_get_db": lambda: db,
    }.items():
        monkeypatch.setattr(server, name, stub)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    yield monkeypatch
    for sid in [s for s in server._sessions if s.startswith("ui-106459-")]:
        server._sessions.pop(sid, None)


def _stamp(db: SessionDB, row_id: str, reason: str) -> None:
    db.end_session(row_id, reason)
    db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time() - 60.0, row_id))


def test_a_registered_stale_close_is_cleared_before_the_turn_and_rotation_publishes(turn_env, db):
    """The field shape: the TUI keeps accepting prompts on a row stamped ``tui_close`` an hour ago.
    The row is clear when the worker runs, and the rotation that follows publishes instead of wedging."""
    db.create_session("row-1", source="tui")
    _stamp(db, "row-1", "tui_close")
    agent = _Agent("row-1", db)
    session = server._sessions["ui-106459-1"] = _session(agent)

    assert server._run_prompt_submit("rid", "ui-106459-1", session, "go") is True

    assert agent.turns == ["go"]
    assert agent.stamp_seen_by_turn == [None], "the stamp must be cleared before the worker starts"
    assert db.try_acquire_compression_lock("row-1", "holder")
    db.publish_compression_child(
        parent_session_id="row-1", child_session_id="row-1-child", source="tui",
        messages=[{"role": "user", "content": "go"}], model="m", compression_lock_holder="holder")
    assert db.get_session("row-1")["end_reason"] == "compression"


@pytest.mark.parametrize("reason, claimed_for_teardown", [
    ("session_reset", False), ("new_session", False), ("compression", False), ("ws_disconnect", False),
    ("tui_close", True),
])
def test_only_an_explicit_close_on_a_still_registered_session_is_cleared(turn_env, db, reason, claimed_for_teardown):
    """Boundaries, compression and automatic stamps own lineage elsewhere; a session ``session.close``
    already claimed under ``_sessions_lock`` is refused a turn and its deliberate close survives."""
    db.create_session("row-2", source="tui")
    _stamp(db, "row-2", reason)
    agent = _Agent("row-2", db)
    session = server._sessions["ui-106459-2"] = _session(agent)
    if claimed_for_teardown:
        assert server._pop_session_by_id("ui-106459-2") is session

    started = server._run_prompt_submit("rid", "ui-106459-2", session, "go")

    assert started is not claimed_for_teardown
    assert db.get_session("row-2")["end_reason"] == reason
