"""session_search results carry the outcomes recorded on a session (#138653).

Recall should tell work that shipped from work that was reverted: discover hits, browse rows and the read /
scroll ``session_meta`` get ``outcomes`` (effective rows only, newest first, at most three, ``{signal,
verdict, source}``; never note, ref, or a row the agent recorded itself) when the compression lineage has
any; a /reset successor of the conversation never lends it outcomes.
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
    db.record_session_outcome(sid, signal=signal, source="github", recorder=kw.pop("recorder", "cli"), **kw)


def _call(db, **kw):
    result = json.loads(session_search(db=db, **kw))
    assert result["success"] is True, result
    return result


SHIPPED = [  # pr/5's merge is superseded by its revert; the agent's own (newest) grade is never shown
    {"signal": "reverted", "verdict": "bad", "source": "github"},
    {"signal": "review_accepted", "verdict": "good", "source": "github"},
    {"signal": "ci_passed", "verdict": "good", "source": "github"},
]


def _leaks(payload) -> list[str]:
    """The note, the ref, the agent's own row, and the recorder field must never reach the model."""
    text = json.dumps(payload)
    return [w for w in ("never shown", "pr/5", "task.done", "recorder") if w in text]


def _shipped(db):
    _session(db, "shipped", "rewrite the zebrafish uploader retry loop")
    _session(db, "quiet", "zebrafish uploader docs pass")
    for n, (signal, ref) in enumerate((("ci_passed", "run/1"), ("review_accepted", "pr/5"), ("merged", "pr/5"),
                                       ("reverted", "pr/5"))):
        _rec(db, "shipped", signal, ref=ref, note="never shown to the model", observed_at=1_700_000_000.0 + n)
    _rec(db, "shipped", "task.done", verdict="good", recorder="agent", observed_at=1_700_000_100.0)


def _hits(db, query):
    result = json.loads(session_search(query=query, limit=5, db=db))
    assert result["success"] is True, result
    return {hit["session_id"]: hit for hit in result["results"]}


def test_hits_carry_the_newest_three_effective_outcomes_without_note_ref_or_agent_rows(db):
    _shipped(db)

    hits = _hits(db, "zebrafish uploader")

    assert hits["shipped"]["outcomes"] == SHIPPED
    assert "outcomes" not in hits["quiet"]
    assert not _leaks(hits)


def test_reading_scrolling_and_browsing_a_session_show_the_same_outcomes(db):
    _shipped(db)
    anchor = db.get_messages("shipped")[0]["id"]

    read = _call(db, session_id="shipped")["session_meta"]
    scroll = _call(db, session_id="shipped", around_message_id=anchor)["session_meta"]
    browse = {r["session_id"]: r for r in _call(db)["results"]}

    assert read["outcomes"] == scroll["outcomes"] == browse["shipped"]["outcomes"] == SHIPPED
    assert "outcomes" not in _call(db, session_id="quiet")["session_meta"] and "outcomes" not in browse["quiet"]
    assert not _leaks([read, scroll, browse])


def test_outcome_on_the_compression_tip_reaches_a_hit_in_the_root(db):
    _session(db, "root", "migrate the okapi billing cron", end_reason="compression")
    _session(db, "tip", "continue", parent="root")
    _rec(db, "tip", "merged")

    hits = list(_hits(db, "okapi billing").values())

    assert [hit.get("outcomes") for hit in hits] == [[{"signal": "merged", "verdict": "good", "source": "github"}]]


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
