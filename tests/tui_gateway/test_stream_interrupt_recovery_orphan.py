"""A stream that dies mid-answer must not leave an anonymous session behind (#111999).

``update_token_counts`` is the only writer that mints ``source='unknown'`` (its "ensure the row
exists" guard). Two contracts keep such a placeholder from becoming a permanent phantom:

* the session's real creator repairs the placeholder source on the same id (the upsert used to
  keep whatever the first writer set);
* the startup orphan sweep collects an ``unknown`` row an older build already left on disk.
"""

from __future__ import annotations

import time

from hermes_state import SessionDB
from tui_gateway.session_reaper import _ORPHAN_SWEEP_SOURCES

# One of the four orphan ids from the report.
ORPHAN_SID = "20260913_210721_c89ac8"
IDLE_S = 6 * 3600  # mirror the TUI gateway's default session TTL


def test_creator_repairs_the_accounting_placeholder_source(tmp_path):
    """The accounting guard's mint is a placeholder, not an identity: the session's own creator
    stamps the real surface on the same id, and a real source is never downgraded."""
    db = SessionDB(tmp_path / "state.db")
    # First writer wins the INSERT because the row creation lost the race with the SQLite lock.
    db.update_token_counts(ORPHAN_SID, input_tokens=1200, output_tokens=90, model="grok-4.6")
    assert db.get_session(ORPHAN_SID)["source"] == "unknown"

    db.create_session(ORPHAN_SID, source="desktop")

    row = db.get_session(ORPHAN_SID)
    assert row["source"] == "desktop"
    assert row["model"] == "grok-4.6"  # nothing else the first writer set is clobbered
    assert db.list_sessions_rich(source="unknown", limit=50) == []
    # Control: a later writer with a different real surface never overwrites the creator's.
    db.create_session(ORPHAN_SID, source="tui")
    assert db.get_session(ORPHAN_SID)["source"] == "desktop"


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
