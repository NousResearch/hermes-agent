"""The OR fill ranks by relevance even when the caller passes ``sort``.

An OR query matches every incidental mention, so a fill scan ordered by time holds only the newest
weak matches. With ``sort="newest"`` and a thin AND result, the relevant older session was dropped.
"""

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    d.create_session("s_incidental", source="cli")  # holds every query word, incidentally
    d.append_message("s_incidental", role="user", content="kestrel codename rollback note")
    d.create_session("s_answer", source="cli")  # older answer; lacks "rollback", reachable only via the fill
    d.append_message("s_answer", role="assistant", content="Kestrel codename TEAL-HERON-4417 kestrel codename kestrel")
    for i in range(320):  # newer sessions mentioning one query word in passing
        d.create_session(f"s_noise_{i:03d}", source="cli")
        d.append_message(f"s_noise_{i:03d}", role="user", content=f"note {i}: kestrel in passing")
    yield d
    d.close()


def test_or_fill_ranks_by_relevance_under_sort_newest(db):
    got = [r["session_id"] for r in json.loads(session_search(query="kestrel codename rollback", limit=5, sort="newest", db=db))["results"]]
    assert "s_answer" in got, got
