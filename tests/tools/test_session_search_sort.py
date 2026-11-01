"""``sort=newest|oldest`` orders the relevant sessions by time; it must not replace relevance.

The FTS scan used ``ORDER BY m.timestamp DESC, rank`` inside its 300-row window, so the pool was the
300 newest matching rows, not the 300 best: on a real 40-question recall set hit@5 fell from 36/40
(no sort) to 2/40 (newest) and 3/40 (oldest), and 12 of 26 real discovery calls passed
``sort=newest``. Discovery now ranks by relevance and then orders the deduped sessions by start time.
"""

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    d.create_session("s_relevant_old", source="cli")
    d.append_message("s_relevant_old", role="assistant",
                     content="kestrel deploy codename TEAL-HERON-4417 kestrel rollback window kestrel")
    # Many newer sessions that mention the term only incidentally fill any time-ordered window.
    for i in range(320):
        d.create_session(f"s_noise_{i:03d}", source="cli")
        d.append_message(f"s_noise_{i:03d}", role="user", content=f"note {i}: kestrel mentioned in passing")
    yield d
    d.close()


def _ids(db, **kw):
    return [r["session_id"] for r in json.loads(session_search(db=db, **kw))["results"]]


def test_newest_still_returns_the_relevant_session(db):
    # An OR query (what recall guidance and the OR-relaxed retry produce) lets every incidental
    # mention match, so a time-ordered scan window fills with newer weak rows.
    q = "kestrel OR codename OR rollback"
    assert "s_relevant_old" in _ids(db, query=q, limit=5)
    assert "s_relevant_old" in _ids(db, query=q, limit=5, sort="newest")


def test_newest_orders_returned_sessions_by_start_time(db):
    got = _ids(db, query="kestrel OR codename", limit=5, sort="newest")
    started = [db.get_session(s)["started_at"] for s in got]
    assert started == sorted(started, reverse=True)
