"""Discovery fills a thin implicit-AND result with the OR-relaxed ranking.

The storage layer's OR-relaxed retry fires only when the AND query returns zero rows. An AND query
that matches ONE incidental session therefore hides the session that holds the answer under a
paraphrase (it lacks a single query word). Measured on a real 40-question recall set: AND finds the
source session in the top 5 for 24/40 questions, any-term OR for 36/40. Discovery now keeps the
AND sessions first and fills the remaining slots from the OR ranking.
"""

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    # Holds every query word, but only incidentally (an unrelated chat).
    d.create_session("s_incidental", source="cli")
    d.append_message("s_incidental", role="user", content="rollback the kestrel deploy codename window later")
    d.append_message("s_incidental", role="assistant", content="Noted.")
    # Holds the answer, worded differently: no "rollback" anywhere.
    d.create_session("s_answer", source="cli")
    d.append_message("s_answer", role="user", content="What does my note say about project Kestrel deploys?")
    d.append_message("s_answer", role="assistant", content="Kestrel deploy codename TEAL-HERON-4417; revert window 30 minutes.")
    # Shares no term: must never appear.
    d.create_session("s_unrelated", source="cli")
    d.append_message("s_unrelated", role="user", content="graphiti daemon looks healthy today")
    yield d
    d.close()


def _sessions(db, query, limit=5):
    payload = json.loads(session_search(query=query, limit=limit, db=db))
    return [r["session_id"] for r in payload["results"]]


def test_thin_and_result_is_filled_from_the_or_ranking(db):
    got = _sessions(db, "kestrel deploy codename rollback window")
    assert got[0] == "s_incidental"          # exact AND matches keep the lead
    assert "s_answer" in got                  # the paraphrased answer session is no longer hidden
    assert "s_unrelated" not in got           # OR fill only adds sessions sharing a query term


def test_explicit_boolean_queries_keep_exact_semantics(db):
    # Explicit OR/NOT means the caller chose the semantics: no extra fill.
    assert _sessions(db, "kestrel NOT rollback") == ["s_answer"]


def test_fill_counts_lineages_not_raw_sessions(tmp_path):
    """Compression children share their root's lineage, and discovery dedupes by lineage. A gate that
    counted raw session ids saw a "full" AND result made of one conversation and skipped the fill."""
    d = SessionDB(tmp_path / "state.db")
    d.create_session("root", source="cli")
    d.append_message("root", role="user", content="kestrel deploy codename rollback window")
    d.end_session("root", "compression") if hasattr(d, "end_session") else None
    for i in range(3):  # three continuations of the same conversation, each holding every term
        d.create_session(f"cont{i}", source="cli", parent_session_id="root" if i == 0 else f"cont{i-1}")
        d.append_message(f"cont{i}", role="assistant", content="kestrel deploy codename rollback window noted")
    d.create_session("s_answer", source="cli")
    d.append_message("s_answer", role="assistant", content="Kestrel deploy codename TEAL-HERON-4417; revert window 30 minutes.")
    try:
        got = _sessions(d, "kestrel deploy codename rollback window", limit=3)
        assert "s_answer" in got
    finally:
        d.close()
