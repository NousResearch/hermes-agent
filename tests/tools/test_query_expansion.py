"""Behaviour contracts for session_search's bounded keyword expansion (tools/query_expansion.py).

Expansion fires only when the strict FTS5 query covers fewer sessions than ``limit``; its rows
rank below every strict row, and it never fires for explicit FTS5 syntax.
"""
import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _session(db, sid, text, source="cli"):
    db.create_session(sid, source=source)
    db.append_message(sid, role="assistant", content=text)


def _ids(result):
    return [r["session_id"] for r in result["results"]]


QUERY = "what did we decide about the zebra migration"


def _seed_thin_strict(db, decoy_source="cli"):
    # The decoy carries every query word, so strict AND returns exactly one session; the
    # target lacks the filler words and is reachable only through the keyword probe.
    _session(db, "decoy", f"{QUERY}? no idea, that was a quiz question", source=decoy_source)
    _session(db, "target", "zebra tables were renamed during the quarterly migration")
    _session(db, "noise", "unrelated notes about the garden")


def test_thin_strict_pool_is_supplemented_below_strict_hits(db):
    _seed_thin_strict(db)
    result = json.loads(session_search(query=QUERY, db=db, limit=3))
    assert _ids(result) == ["decoy", "target"]
    assert result["expanded_terms"] == ["decide", "zebra", "migration"]


def test_strict_cron_hit_still_outranks_expanded_interactive_hit(db):
    _seed_thin_strict(db, decoy_source="cron")
    result = json.loads(session_search(query=QUERY, db=db, limit=3))
    assert _ids(result) == ["decoy", "target"]


def test_punctuated_keyword_expansion_recovers_target(db):
    # "$HOME/.config/hermes" sanitized bare leaves ".config", which FTS5 rejects: the
    # probe must quote it as a phrase or the supplement silently finds nothing.
    query = "what about $HOME/.config/hermes"
    _session(db, "decoy", f"{query} was a riddle, never mind")
    _session(db, "target", "your settings live in HOME/.config/hermes on that box")
    result = json.loads(session_search(query=query, db=db, limit=3))
    assert "target" in _ids(result)


@pytest.mark.parametrize("query", ['"zebra migration" decide', "zebra OR migration", "zebra*"])
def test_explicit_fts_syntax_is_never_expanded(db, query):
    _seed_thin_strict(db)
    result = json.loads(session_search(query=query, db=db, limit=3))
    assert "expanded_terms" not in result
