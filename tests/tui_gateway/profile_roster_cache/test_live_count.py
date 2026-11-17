"""``profiles.list`` does not recount a long chat's display rows on every poll.

``live_message_count`` (``SessionDB.display_message_count``) reads every display row of a session.
The store-wide roster memo cannot absorb it, because a store that background workers write all the
time never holds still, so the canonical chat of every profile was recounted on every 5 s poll. The
count is now reused while a cheap, index-only fingerprint of the session's display rows holds, and
recounted the moment any transition that changes the count moves it.
"""
from __future__ import annotations

import sqlite3

import pytest

import tui_gateway.server as srv
from hermes_state import SessionDB
from tui_gateway import profile_roster_cache as cache


@pytest.fixture(autouse=True)
def _clean_memo():
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    handle.create_session(session_id="s1", source="cli")
    yield handle
    handle.close()


@pytest.fixture
def counts(monkeypatch):
    """Session ids of every full display count that ran."""
    calls: list[str] = []
    real = SessionDB.display_message_count

    def counting(self, session_id):
        calls.append(session_id)
        return real(self, session_id)

    monkeypatch.setattr(SessionDB, "display_message_count", counting)
    return calls


def _live(db, session_id="s1"):
    return srv._live_count_field(db, session_id)["live_message_count"]


def _write(db, sql, params=()):
    with db._lock:
        db._conn.execute(sql, params)
        db._conn.commit()


def test_an_unchanged_session_is_counted_once(db, counts):
    for text in ("a", "b", "c"):
        db.append_message(session_id="s1", role="user", content=text)

    assert [_live(db) for _ in range(5)] == [3] * 5
    assert counts == ["s1"]


def test_the_memo_serves_the_same_int_the_count_returns(db):
    for text in ("a", "b"):
        db.append_message(session_id="s1", role="assistant", content=text)

    fresh = db.display_message_count("s1")
    assert _live(db) == fresh
    served = _live(db)  # from the memo
    assert served == fresh and isinstance(served, int)


def test_an_appended_message_is_counted_on_the_next_call(db, counts):
    db.append_message(session_id="s1", role="user", content="a")
    assert _live(db) == 1

    db.append_message(session_id="s1", role="assistant", content="b")

    assert _live(db) == 2
    assert counts == ["s1", "s1"]


def test_a_rewind_that_leaves_the_display_set_is_counted(db, counts):
    db.append_message(session_id="s1", role="user", content="a")
    second = db.append_message(session_id="s1", role="assistant", content="b")
    assert _live(db) == 2

    _write(db, "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (second,))

    assert _live(db) == 1
    assert counts == ["s1", "s1"]


def test_a_compaction_archive_with_the_same_row_count_is_counted(db, counts):
    """Compaction flips active -> archived inside the display set: the set's size and highest id
    are unchanged, so the fingerprint must carry how many rows are still active."""
    db.append_message(session_id="s1", role="user", content="a")
    db.append_message(session_id="s1", role="assistant", content="b")
    before = cache.live_count_signature(db, "s1")
    assert _live(db) == 2

    _write(db, "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = 's1' AND role = 'user'")

    after = cache.live_count_signature(db, "s1")
    assert after != before and after[0] == before[0] and after[2] == before[2]
    _live(db)
    assert counts == ["s1", "s1"]


def test_rows_folded_into_one_display_group_are_counted(db, counts):
    """The count is one per ``display_order`` group, and folding rows into one group (the
    re-fold after a rotation, a display-slot backfill) bumps no revision: the fingerprint must
    carry the groups."""
    first = db.append_message(session_id="s1", role="user", content="a")
    second = db.append_message(session_id="s1", role="assistant", content="b")
    assert _live(db) == 2

    with db._lock:
        order = db._conn.execute("SELECT display_order FROM messages WHERE id = ?", (first,)).fetchone()[0]
    _write(db, "UPDATE messages SET display_order = ? WHERE id = ?", (order, second))

    assert db.display_message_count("s1") == 1  # the premise: one group now
    counts.clear()
    assert _live(db) == 1
    assert counts == ["s1"]


def test_unindexed_rows_count_as_one_group_until_backfilled(db, counts):
    db.append_message(session_id="s1", role="user", content="a")
    db.append_message(session_id="s1", role="assistant", content="b")
    _write(db, "UPDATE messages SET display_order = NULL WHERE session_id = 's1'")
    assert _live(db) == db.display_message_count("s1") == 1

    _write(db, "UPDATE messages SET display_order = id WHERE session_id = 's1'")

    counts.clear()
    assert _live(db) == 2
    assert counts == ["s1"]


def test_an_in_place_rewrite_is_recounted(db, counts):
    row = db.append_message(session_id="s1", role="user", content="a")
    assert _live(db) == 1

    _write(db, "UPDATE messages SET content = 'rewritten' WHERE id = ?", (row,))

    assert _live(db) == 1
    assert counts == ["s1", "s1"]


def test_each_session_and_each_store_has_its_own_entry(db, tmp_path, counts):
    db.create_session(session_id="s2", source="cli")
    db.append_message(session_id="s1", role="user", content="a")
    db.append_message(session_id="s2", role="user", content="a")
    db.append_message(session_id="s2", role="user", content="b")
    other = SessionDB(db_path=tmp_path / "other.db")
    try:
        other.create_session(session_id="s1", source="cli")
        assert (_live(db, "s1"), _live(db, "s2"), _live(other, "s1")) == (1, 2, 0)
        assert (_live(db, "s1"), _live(db, "s2"), _live(other, "s1")) == (1, 2, 0)
    finally:
        other.close()
    assert len(counts) == 3


def test_the_fingerprint_reads_the_display_index_not_the_rows(db, monkeypatch):
    """The point of the memo: the fingerprint must not touch the message rows the count reads.
    SQLite's plan text calls this index "not covering" because the query names ``compacted``, but
    that term is the partial index's own WHERE clause and is coded away, so the program never reads
    a column of the ``messages`` table itself; assert on the program, not on the plan text."""
    db.append_message(session_id="s1", role="user", content="a")
    statements: list[tuple[str, tuple]] = []
    real_one, real_all = db._read_one, db._read_all

    def capture(sql, params=()):
        statements.append((sql, params))
        return real_one(sql, params)

    monkeypatch.setattr(db, "_read_one", capture)
    cache.live_count_signature(db, "s1")
    [(sql, params)] = statements

    plan = " ".join(str(row[3]) for row in real_all("EXPLAIN QUERY PLAN " + sql, params))
    assert "idx_messages_display_page" in plan and "SCAN messages" not in plan, plan
    rootpage = real_one("SELECT rootpage FROM sqlite_master WHERE type = 'table' AND name = 'messages'")[0]
    program = [tuple(row)[:4] for row in real_all("EXPLAIN " + sql, params)]
    table_cursors = {p1 for _addr, op, p1, p2 in program if op == "OpenRead" and p2 == rootpage}
    assert table_cursors, program
    assert not [step for step in program if step[1] == "Column" and step[2] in table_cursors], program


def test_a_store_that_cannot_give_a_fingerprint_still_counts(db, counts, monkeypatch):
    db.append_message(session_id="s1", role="user", content="a")

    def refuse(_db, _session_id):
        raise sqlite3.OperationalError("no such column: transcript_revision")

    monkeypatch.setattr(cache, "live_count_signature", refuse)

    assert _live(db) == 1 and _live(db) == 1
    assert counts == ["s1", "s1"]  # no memo without a fingerprint: the old behaviour


def test_invalidate_drops_one_profiles_counts(db, tmp_path, counts):
    db.append_message(session_id="s1", role="user", content="a")
    _live(db)

    cache.invalidate(tmp_path / "elsewhere")
    _live(db)
    cache.invalidate(tmp_path)
    _live(db)

    assert counts == ["s1", "s1"]


def test_profiles_list_does_not_recount_an_unchanged_chat_on_a_busy_store(tmp_path, monkeypatch, counts):
    """End to end through the handler, with the store-wide memo defeated the way a busy store defeats it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "cached_session_fields", lambda _path, compute: compute())
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="canon", source="cli")
        db.set_session_title("canon", "Bot Chat")
        for text in ("a", "b", "c"):
            db.append_message(session_id="canon", role="user", content=text)
    finally:
        db.close()

    def poll():
        rows = srv._methods["profiles.list"](1, {})["result"]["profiles"]
        default = next(p for p in rows if p["is_default"])
        return default["canonical_session"], default["last_session"]

    first = poll()
    walks_after_first = len(counts)
    for _ in range(4):
        assert poll() == first

    assert first[0]["live_message_count"] == 3 and first[1]["live_message_count"] == 3
    assert walks_after_first == 1  # canonical and newest chat are the same session: one count
    assert len(counts) == walks_after_first, "the roster recounted an unchanged chat on every poll"
