"""Contract tests for the shared /undo -> /redo core (hermes_undo).

These exercise a REAL SessionDB against a temp path rather than mocks: the
whole feature is about row id lifecycle, which mocks cannot model.
"""

import sqlite3

import pytest

import hermes_undo
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    hermes_undo._session_db = database
    hermes_undo.clear_state()
    yield database
    hermes_undo.clear_state()
    hermes_undo._session_db = None
    database.close()


def _seed(db, sid="s1", turns=3):
    db.create_session(sid, source="test")
    for i in range(1, turns + 1):
        db.append_message(sid, "user", f"q{i}")
        db.append_message(sid, "assistant", f"a{i}")
    return sid


def _active(db, sid):
    return [m["content"] for m in db.get_messages(sid, include_inactive=False)]


def _undo_once(db, sid, n=1):
    """Drive one rewind through the real engine and bank it, as callers do."""
    msgs = db.get_messages(sid, include_inactive=False)
    users = [m for m in msgs if m["role"] == "user"]
    target = users[-n]["id"]
    result = db.rewind_to_message(sid, target)
    hermes_undo.record_undo(sid, n, result.get("rewound_ids") or [])
    return result


class TestRewoundIdsContract:
    def test_rewind_reports_exactly_the_rows_it_deactivated(self, db):
        sid = _seed(db, turns=2)
        before = {m["id"] for m in db.get_messages(sid, include_inactive=False)}
        result = _undo_once(db, sid)

        ids = result["rewound_ids"]
        assert ids, "rewind must name the rows it archived"
        assert len(ids) == result["rewound_count"]
        # Every reported id was active before and is inactive now.
        assert set(ids) <= before
        still_active = {m["id"] for m in db.get_messages(sid, include_inactive=False)}
        assert set(ids).isdisjoint(still_active)

    def test_rewound_ids_exclude_already_inactive_rows(self, db):
        """A second rewind past the same target must not re-claim old rows."""
        sid = _seed(db, turns=3)
        first = _undo_once(db, sid)
        second = _undo_once(db, sid)
        assert set(first["rewound_ids"]).isdisjoint(second["rewound_ids"])


class TestRedoStackBanking:
    def test_redo_restores_exactly_the_undone_rows(self, db):
        sid = _seed(db, turns=3)
        before = _active(db, sid)
        _undo_once(db, sid)
        assert _active(db, sid) == before[:-2]

        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 2
        assert _active(db, sid) == before

    def test_redo_without_a_banked_undo_reports_nothing(self, db):
        sid = _seed(db, turns=2)
        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 0
        assert "nothing to redo" in result["message"]

    def test_unbanked_rewind_is_not_redoable(self, db):
        """The redo stack is what makes a rewind replayable.

        A rewind that was never recorded (for example /retry, which replaces
        the turn) must not be resurrectable by /redo.
        """
        sid = _seed(db, turns=2)
        msgs = db.get_messages(sid, include_inactive=False)
        target = [m for m in msgs if m["role"] == "user"][-1]["id"]
        db.rewind_to_message(sid, target)  # deliberately NOT recorded

        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 0
        assert len(_active(db, sid)) == 2

    def test_empty_rewind_is_never_banked(self, db):
        """Banking a no-op would let /redo claim success for nothing."""
        sid = _seed(db, turns=1)
        hermes_undo.record_undo(sid, 1, [])
        assert hermes_undo.has_redoable(sid) is False
        assert hermes_undo.redo(sid, 1)["reactivated_count"] == 0

    def test_multi_step_undo_redo_round_trips(self, db):
        sid = _seed(db, turns=3)
        before = _active(db, sid)
        _undo_once(db, sid)
        _undo_once(db, sid)
        assert _active(db, sid) == before[:-4]

        result = hermes_undo.redo(sid, 2)
        assert result["reactivated_count"] == 4
        assert _active(db, sid) == before

    def test_redo_replays_newest_operation_first(self, db):
        sid = _seed(db, turns=3)
        before = _active(db, sid)
        _undo_once(db, sid)
        _undo_once(db, sid)

        hermes_undo.redo(sid, 1)
        # Only the most recent undo came back, so we are one turn short.
        assert _active(db, sid) == before[:-2]


class TestHalfTurnBoundary:
    def test_undo_one_turn_removes_the_user_and_its_reply(self, db):
        sid = _seed(db, turns=3)
        result = _undo_once(db, sid, n=1)
        assert result["rewound_count"] == 2
        assert _active(db, sid) == ["q1", "a1", "q2", "a2"]

    def test_undo_two_turns_removes_both_exchanges(self, db):
        sid = _seed(db, turns=3)
        result = _undo_once(db, sid, n=2)
        assert result["rewound_count"] == 4
        assert _active(db, sid) == ["q1", "a1"]

    def test_rewound_rows_are_soft_deleted_not_destroyed(self, db):
        """Redo is only possible because /undo archives rather than deletes."""
        sid = _seed(db, turns=2)
        _undo_once(db, sid)
        every = db.get_messages(sid, include_inactive=True)
        assert [m["content"] for m in every] == ["q1", "a1", "q2", "a2"]


class TestRedoInvalidation:
    def test_new_user_message_clears_the_redo_branch(self, db):
        sid = _seed(db, turns=2)
        _undo_once(db, sid)
        assert hermes_undo.has_redoable(sid) is True

        hermes_undo.on_user_message_appended(sid)

        assert hermes_undo.has_redoable(sid) is False
        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 0
        assert _active(db, sid) == ["q1", "a1"]

    def test_a_fresh_undo_clears_stale_redo_history(self, db):
        sid = _seed(db, turns=3)
        _undo_once(db, sid)
        hermes_undo.redo(sid, 1)
        assert hermes_undo.get_state(sid).redo_stack

        _undo_once(db, sid)
        assert hermes_undo.get_state(sid).redo_stack == []


class TestRestartAndEviction:
    def test_lost_stack_after_rewind_explains_why(self, db):
        """A rewound session with no stack means the process restarted."""
        sid = _seed(db, turns=2)
        _undo_once(db, sid)
        hermes_undo.clear_state(sid)  # simulates a restart

        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 0
        assert "restart" in result["message"]

    def test_never_rewound_session_gets_the_plain_message(self, db):
        sid = _seed(db, turns=2)
        result = hermes_undo.redo(sid, 1)
        assert result["message"] == "nothing to redo"

    def test_state_holder_is_bounded(self, db):
        hermes_undo.clear_state()
        for i in range(hermes_undo._STATE_CAP + 25):
            hermes_undo.get_state(f"sess-{i}")
        assert len(hermes_undo._states) == hermes_undo._STATE_CAP

    def test_eviction_drops_the_oldest_session(self, db):
        hermes_undo.clear_state()
        hermes_undo.get_state("oldest")
        for i in range(hermes_undo._STATE_CAP):
            hermes_undo.get_state(f"filler-{i}")
        assert "oldest" not in hermes_undo._states


class TestTranscriptRewrite:
    def test_redo_across_a_rewrite_degrades_instead_of_raising(self, db):
        """/compress and /retry hard-delete and renumber rows.

        A redo whose banked ids no longer exist must report a dead branch
        rather than raising at the user.
        """
        sid = _seed(db, turns=2)
        _undo_once(db, sid)
        # Every banked id vanishes, as a transcript rewrite would do.
        state = hermes_undo.get_state(sid)
        state.undo_stack[-1] = hermes_undo.UndoOp(n=1, rewound_ids=[999001, 999002])

        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 0
        assert "transcript changed" in result["message"]
        assert hermes_undo.get_state(sid).undo_stack == []

    def test_partial_progress_is_reported_not_discarded(self, db):
        """Earlier committed operations must survive a later dead one."""
        sid = _seed(db, turns=3)
        _undo_once(db, sid)  # older op, still replayable
        _undo_once(db, sid)  # newer op, we will kill this one's rows
        state = hermes_undo.get_state(sid)
        # Kill the OLDER op (replayed second) so the first replay commits.
        state.undo_stack[0] = hermes_undo.UndoOp(n=1, rewound_ids=[999101])

        result = hermes_undo.redo(sid, 2)
        assert result["reactivated_count"] == 2
        assert result["partial"] is True
        assert result["partial_retryable"] is False

    def test_single_op_partial_restore_fails_loud(self, db):
        """Some-but-not-all rows back is corruption, not a clean rewrite."""
        sid = _seed(db, turns=2)
        _undo_once(db, sid)
        state = hermes_undo.get_state(sid)
        real = list(state.undo_stack[-1].rewound_ids)
        state.undo_stack[-1] = hermes_undo.UndoOp(n=1, rewound_ids=real + [999201])

        with pytest.raises(RuntimeError, match="redo invariant violated"):
            hermes_undo.redo(sid, 1)


class TestErrorHonesty:
    def test_transient_error_is_classified_retryable(self):
        exc = sqlite3.OperationalError("database is locked")
        assert hermes_undo._is_transient_db_error(exc) is True

    def test_real_bug_is_not_classified_retryable(self):
        assert hermes_undo._is_transient_db_error(ValueError("boom")) is False
        assert (
            hermes_undo._is_transient_db_error(
                sqlite3.OperationalError("no such column: nope")
            )
            is False
        )

    def test_first_op_failure_propagates(self, db, monkeypatch):
        """Nothing committed yet, so the caller must see the real error."""
        sid = _seed(db, turns=2)
        _undo_once(db, sid)

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(db, "restore_ids", boom)
        with pytest.raises(sqlite3.OperationalError):
            hermes_undo.redo(sid, 1)
        # The failed op stays on the stack — it never committed.
        assert hermes_undo.has_redoable(sid) is True

    def test_transient_failure_after_progress_is_a_retryable_partial(
        self, db, monkeypatch
    ):
        sid = _seed(db, turns=3)
        _undo_once(db, sid)
        _undo_once(db, sid)

        real = db.restore_ids
        calls = {"n": 0}

        def flaky(session_id, ids, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise sqlite3.OperationalError("database is locked")
            return real(session_id, ids, **kwargs)

        monkeypatch.setattr(db, "restore_ids", flaky)
        result = hermes_undo.redo(sid, 2)
        assert result["reactivated_count"] == 2
        assert result["partial"] is True
        assert result["partial_retryable"] is True
        # The un-replayed op is still banked, so the retry can finish it.
        assert hermes_undo.has_redoable(sid) is True

    def test_post_commit_counter_failure_does_not_lose_the_redo(
        self, db, monkeypatch
    ):
        """The rows are durable; a cosmetic counter must not report failure."""
        sid = _seed(db, turns=2)
        _undo_once(db, sid)

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(db, "bump_redo_count", boom)
        result = hermes_undo.redo(sid, 1)
        assert result["reactivated_count"] == 2
        assert _active(db, sid) == ["q1", "a1", "q2", "a2"]


class TestRestoreIdsPrimitive:
    def test_restores_only_the_named_rows(self, db):
        sid = _seed(db, turns=3)
        first = _undo_once(db, sid)
        second = _undo_once(db, sid)

        restored = db.restore_ids(sid, second["rewound_ids"])
        assert restored == len(second["rewound_ids"])
        # The other operation's rows stay archived.
        active = {m["id"] for m in db.get_messages(sid, include_inactive=False)}
        assert set(first["rewound_ids"]).isdisjoint(active)

    def test_is_idempotent_and_scoped(self, db):
        sid = _seed(db, turns=2)
        result = _undo_once(db, sid)
        assert db.restore_ids(sid, result["rewound_ids"]) == 2
        # Already active -> zero, not an error.
        assert db.restore_ids(sid, result["rewound_ids"]) == 0
        assert db.restore_ids(sid, []) == 0
        assert db.restore_ids(sid, [987654]) == 0

    def test_does_not_cross_session_boundaries(self, db):
        a = _seed(db, sid="sa", turns=2)
        _seed(db, sid="sb", turns=2)
        result = _undo_once(db, a)
        # Asking session B to restore session A's rows must do nothing.
        assert db.restore_ids("sb", result["rewound_ids"]) == 0

    def test_redo_count_bumps_once_per_command(self, db):
        sid = _seed(db, turns=3)
        _undo_once(db, sid)
        _undo_once(db, sid)
        assert (db.get_session(sid) or {}).get("redo_count") in (0, None)

        hermes_undo.redo(sid, 2)  # one command, two operations
        assert (db.get_session(sid) or {}).get("redo_count") == 1


class TestSchemaMigration:
    def test_redo_count_is_added_to_a_pre_existing_database(self, tmp_path):
        """An existing state.db must gain the column on open, not error."""
        path = tmp_path / "old.db"
        first = SessionDB(db_path=path)
        first.create_session("s", source="test")
        first.close()

        # Drop redo_count, emulating an older database. DROP COLUMN (not a table rebuild): views
        # over ``sessions`` (the FTS source views) must keep resolving.
        con = sqlite3.connect(path)
        con.execute("ALTER TABLE sessions DROP COLUMN redo_count")
        con.commit()
        after_drop = [r[1] for r in con.execute("PRAGMA table_info(sessions)")]
        con.close()
        assert "redo_count" not in after_drop

        reopened = SessionDB(db_path=path)
        try:
            con = sqlite3.connect(path)
            restored = [r[1] for r in con.execute("PRAGMA table_info(sessions)")]
            con.close()
            assert "redo_count" in restored
            reopened.bump_redo_count("s")
            assert (reopened.get_session("s") or {}).get("redo_count") == 1
        finally:
            reopened.close()


class TestCompositeCarrierRedo:
    """Rewinding a compaction carrier INSERTS a hidden handoff row as the new head. Redo must retire it
    in the same write, or the restored carrier sits next to its own scaffold (summary shown twice)."""

    def test_redo_of_a_carrier_rewind_restores_the_exact_active_set(self, tmp_path):
        from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER

        db = SessionDB(db_path=tmp_path / "carrier.db")
        try:
            sid = "carrier-redo"
            db.create_session(sid, source="tui")
            db.append_message(sid, "user", "older ask")
            db.append_message(sid, "assistant", "older reply")
            db.append_message(
                sid, "user",
                f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nREAL ASK")
            db.append_message(sid, "assistant", "failed")
            before = [m["_row_id"] for m in db.get_messages_as_conversation(sid, include_row_ids=True)]

            outcome = db.rewind_user_turn(sid, -1)
            assert isinstance(outcome.replacement_id, int)  # precondition: the carrier path ran
            assert outcome.replacement_id not in before
            hermes_undo.clear_state()
            hermes_undo.record_rewind(sid, db, outcome.turns_undone, outcome)

            result = hermes_undo.redo(sid, 1)

            after = [m["_row_id"] for m in db.get_messages_as_conversation(sid, include_row_ids=True)]
            assert result["reactivated_count"] == len(outcome.rewound_ids)
            assert after == before
        finally:
            db.close()
            hermes_undo.clear_state()
