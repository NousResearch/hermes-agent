"""Regression: a session whose live context was compacted away must still be able to
find its own archived history via the OR-relaxed retry.

Shape observed on a long-lived production session (2026-10-01): the FTS AND of a multi-term query
only matches rows inside the searcher's OWN live lineage (today's query echo), so the
DB layer returns non-zero and never OR-relaxes; the tool layer then drops every hit in
the own-lineage guard, and the caller gets ZERO despite archived rows matching
individual terms. The rewind/carried originals (active=0, compacted=0) stay excluded —
only archived rows (compacted=1) become discoverable via the retry.
"""
import json
import time

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


def _seed_own_lineage_with_archived_history(db):
    """One lineage: old archived rows (compacted=1) + a live tail with the query echo."""
    db.create_session("s_live", source="cli")
    now = int(time.time())
    # Archived history rows — what a completed compaction leaves behind.
    for i, text in enumerate([
        "the treat dispenser servo drops kibble when Mack bats the lever",
        "Mack loves the shoot-the-treat play module on the robot buddy",
    ]):
        db.append_message("s_live", role="assistant", content=text, timestamp=now - 86400 * (3 - i))
    # Compress: archives the two rows above and writes a summary row.
    db.archive_and_compact("s_live", compacted_messages=[{"role": "assistant", "content": "old robot talk summarized"}])
    # The live tail (post-compression): an echo of a failed search — matches the full AND
    # but is pure tool noise.
    db.append_message("s_live", role="assistant",
                      content=f"I searched '{QUERY}' and found nothing about the robot buddy.")


QUERY = "treat dispenser servo drop shoot cat play robot buddy"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_own_lineage_zero_still_finds_archived_history_via_or(db):
    _seed_own_lineage_with_archived_history(db)

    result = json.loads(session_search(query=QUERY, limit=8, sort="oldest",
                                       current_session_id="s_live", db=db))

    assert result["success"] is True
    assert result["count"] >= 1, result.get("message")
    hit_ids = {r["session_id"] for r in result["results"]}
    assert hit_ids == {"s_live"}, hit_ids


def test_or_retry_never_surfaces_rewound_rows(db):
    """The retry must not un-exclude rewound/carried duplicates (active=0, compacted=0)."""
    _seed_own_lineage_with_archived_history(db)
    # A rewound row: superseded duplicate that search must never return.
    db.append_message("s_live", role="assistant", content="rewound duplicate treat dispenser text",
                      timestamp=int(time.time()) - 3600)
    conn = db._conn if hasattr(db, "_conn") else db._open_writer_conn()
    row = conn.execute("SELECT id FROM messages WHERE content LIKE 'rewound duplicate%'").fetchone()
    conn.execute("UPDATE messages SET active=0, compacted=0 WHERE id=?", (row[0],))
    conn.commit()

    result = json.loads(session_search(query=QUERY, limit=8,
                                       current_session_id="s_live", db=db))

    snippets = " ".join(str(r.get("snippet", "")) for r in result["results"])
    assert "rewound duplicate" not in snippets
