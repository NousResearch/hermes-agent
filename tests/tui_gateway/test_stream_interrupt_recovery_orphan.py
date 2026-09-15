"""A stream that dies mid-answer must not leave an anonymous session behind (#111999).

The two recovery injections — the gateway's crash auto-continue note and the loop's "Continue exactly
where you left off" stub — both write into the session that was interrupted. When that session's
durable row has not landed yet, three separate facts turned the recovery into a phantom:

* ``update_token_counts`` is the only writer that mints ``source='unknown'`` (its "ensure the row
  exists" guard, reached from the first API call's queued token delta);
* the row upsert keeps whatever the FIRST writer set, so the session's own creator could never repair
  that placeholder — the record stayed anonymous for life;
* the startup orphan sweep skipped ``unknown``, so such a row stayed ``ended_at IS NULL`` forever.

Contracts pinned here:

* the recovery dispatch persists the session's OWN row (original session key, real source) before the
  turn writes anything — so the recovery is written into the original session record;
* the session's real creator repairs the accounting guard's placeholder source on the same id;
* the startup sweep collects a phantom an older build already left on disk.
"""

from __future__ import annotations

import threading
import time
import types
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_registry import acquire, release_or_close
from tui_gateway import server
from tui_gateway.session_reaper import _ORPHAN_SWEEP_SOURCES

# One of the four orphan ids from the report.
ORPHAN_SID = "20260913_210721_c89ac8"
IDLE_S = 6 * 3600  # mirror the TUI gateway's default session TTL


class _InlineThread:
    """Run threads synchronously so tests observe final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
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
        "agent": agent, "session_key": ORPHAN_SID, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
        "source": "desktop", **extra,
    }


@pytest.fixture()
def recovery_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths (same set the auto-continue suite uses)."""
    monkeypatch.setattr(server, "threading", types.SimpleNamespace(Thread=_InlineThread))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _recovery_dispatch(server_module, session, text):
    """Dispatch a turn the way the crash auto-continue / queued-prompt drain do (straight into the turn
    pipeline, bypassing prompt.submit's row persistence)."""
    return server_module._run_prompt_submit("rid", "sid", session, text, display_kind="auto_continue")


def test_recovery_dispatch_binds_the_original_session_row(recovery_env, tmp_path):
    """The interrupted turn's row never landed; the recovery must create it before writing, on the
    SAME id and with the session's real source — not leave the store to materialize it later as an
    anonymous ``unknown`` session holding only the half-finished assistant text."""
    agent = types.SimpleNamespace(
        session_id=ORPHAN_SID, clear_interrupt=lambda: None,
        run_conversation=lambda message, **kwargs: {"final_response": "resumed"})
    session = _session(agent, profile_home=str(tmp_path))

    db = acquire(Path(tmp_path) / "state.db")
    try:
        assert db.get_session(ORPHAN_SID) is None  # the state the recovery resumes from

        _recovery_dispatch(server, session, server._auto_continue_note("the original prompt"))

        row = db.get_session(ORPHAN_SID)
        assert row is not None, "the recovery turn must own a durable row for its own session"
        assert row["source"] == "desktop"
        assert db.list_sessions_rich(source="unknown", limit=50) == []
    finally:
        release_or_close(db)


def test_creator_repairs_the_accounting_placeholder_source(tmp_path):
    """The accounting guard's mint is a placeholder, not an identity: the session's own creator
    stamps the real surface on the same id (the upsert used to keep the first writer's value)."""
    db = SessionDB(tmp_path / "state.db")
    # First writer wins the INSERT because the row creation lost the race with the SQLite lock.
    db.update_token_counts(ORPHAN_SID, input_tokens=1200, output_tokens=90, model="grok-4.6")
    assert db.get_session(ORPHAN_SID)["source"] == "unknown"

    # The interrupted session's creator (prompt.submit / the recovery dispatch) claims it.
    db.create_session(ORPHAN_SID, source="desktop")

    row = db.get_session(ORPHAN_SID)
    assert row["source"] == "desktop"
    assert row["model"] == "grok-4.6"  # nothing else the first writer set is clobbered
    assert db.list_sessions_rich(source="unknown", limit=50) == []


def test_startup_sweep_collects_a_legacy_unknown_phantom(tmp_path):
    """A phantom an older build already left on disk is collected, not left open forever."""
    db = SessionDB(tmp_path / "state.db")
    db.update_token_counts(ORPHAN_SID, input_tokens=10, output_tokens=5, model="claude-sonnet-5")
    db.append_message(ORPHAN_SID, role="assistant", content="review table, cut mid-stream")
    stale = time.time() - 8 * 3600
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (stale, ORPHAN_SID))
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (stale, ORPHAN_SID))
    db._conn.commit()

    assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S, sources=_ORPHAN_SWEEP_SOURCES) == [ORPHAN_SID]
