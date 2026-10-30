"""session_search discovery results carry the outcomes recorded on a session (#138653).

Recall should tell work that shipped from work that was reverted: each hit gets ``outcomes`` (newest
first, at most three, ``{signal, verdict, source, recorder}`` only, never note/ref) when its compression
lineage has any; a /reset successor of the hit's conversation never lends it outcomes.
"""

import json
import sqlite3

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _session(db, sid, text, *, parent=None, end_reason=None, model_config=None):
    db.create_session(sid, source="cli", parent_session_id=parent, model_config=model_config)
    db.append_message(sid, role="user", content=text)
    db.append_message(sid, role="assistant", content=f"Working on it: {text}")
    if end_reason:
        db.end_session(sid, end_reason)


def _rec(db, sid, signal, **kw):
    db.record_session_outcome(sid, signal=signal, source="github", recorder="cli", **kw)


def _hits(db, query):
    result = json.loads(session_search(query=query, limit=5, db=db))
    assert result["success"] is True, result
    return {hit["session_id"]: hit for hit in result["results"]}


def test_hits_carry_newest_three_outcomes_without_note_or_ref(db):
    _session(db, "shipped", "rewrite the zebrafish uploader retry loop")
    _session(db, "quiet", "zebrafish uploader docs pass")
    for n, (signal, ref) in enumerate((("ci_passed", "run/1"), ("review_accepted", "pr/5"), ("merged", "pr/5"))):
        _rec(db, "shipped", signal, ref=ref, note="never shown to the model", observed_at=1_700_000_000.0 + n)
    _rec(db, "shipped", "reverted", ref="pr/6")

    hits = _hits(db, "zebrafish uploader")

    assert hits["shipped"]["outcomes"] == [
        {"signal": "reverted", "verdict": "bad", "source": "github", "recorder": "cli"},
        {"signal": "merged", "verdict": "good", "source": "github", "recorder": "cli"},
        {"signal": "review_accepted", "verdict": "good", "source": "github", "recorder": "cli"},
    ]
    assert "outcomes" not in hits["quiet"]
    assert "never shown" not in json.dumps(hits) and "pr/5" not in json.dumps(hits)


def test_outcome_on_the_compression_tip_reaches_a_hit_in_the_root(db):
    _session(db, "root", "migrate the okapi billing cron", end_reason="compression")
    _session(db, "tip", "continue", parent="root")
    _rec(db, "tip", "merged")

    hits = list(_hits(db, "okapi billing").values())

    assert [hit.get("outcomes") for hit in hits] == [
        [{"signal": "merged", "verdict": "good", "source": "github", "recorder": "cli"}]]


def test_outcome_on_a_reset_successor_does_not_reach_the_earlier_conversation(db):
    _session(db, "before", "plan the pangolin dashboard", end_reason="session_reset")
    _session(db, "after", "unrelated work", parent="before", model_config={"_reset_from": "before"})
    _rec(db, "after", "reverted")

    assert "outcomes" not in _hits(db, "pangolin dashboard")["before"]


def test_search_of_a_store_without_the_table_still_works(tmp_path):
    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    _session(writer, "old", "the quokka exporter incident")
    writer.close()
    with sqlite3.connect(path) as legacy:
        legacy.execute("DROP TABLE session_outcomes")
    reader = SessionDB(db_path=path, read_only=True)
    try:
        hits = _hits(reader, "quokka exporter")
    finally:
        reader.close()
    assert list(hits) == ["old"] and "outcomes" not in hits["old"]
