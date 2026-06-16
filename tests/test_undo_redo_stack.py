import ast
import inspect

import pytest

import hermes_undo
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_undo, "_session_db", session_db)
    hermes_undo.clear_state()
    yield session_db
    session_db.close()
    hermes_undo.clear_state()


def _make_session(db, sid="s1"):
    db.create_session(sid, source="cli")
    return sid


def _active_ids(db, sid):
    return [m["id"] for m in db.get_messages(sid)]


def _seed_three_half_turns(db, sid):
    u1 = db.append_message(sid, "user", "u1")
    a1 = db.append_message(sid, "assistant", "a1")
    u2 = db.append_message(sid, "user", "u2")
    a2 = db.append_message(sid, "assistant", "a2")
    return u1, a1, u2, a2


def test_undo_redo_transition_table_identity_and_redo_count(db):
    sid = _make_session(db)
    ids = _seed_three_half_turns(db, sid)
    before = _active_ids(db, sid)

    undone = hermes_undo.undo(sid, 1)
    assert undone["rewound_ids"] == [ids[-1]]
    assert undone["prefill_text"] == "u2"
    state = hermes_undo.get_state(sid)
    assert [op.rewound_ids for op in state.undo_stack] == [[ids[-1]]]
    assert state.redo_stack == []
    assert _active_ids(db, sid) == before[:-1]

    redone = hermes_undo.redo(sid, 1)
    assert redone == {"reactivated_count": 1, "new_tail_id": ids[-1], "prefill_text": None}
    assert _active_ids(db, sid) == before
    assert state.undo_stack == []
    assert [op.rewound_ids for op in state.redo_stack] == [[ids[-1]]]
    assert db.get_session(sid)["redo_count"] == 1


def test_stacked_ops_have_disjoint_ids_and_redo_lifo_pop_order(db):
    sid = _make_session(db)
    ids = _seed_three_half_turns(db, sid)

    op1 = hermes_undo.undo(sid, 1)
    op2 = hermes_undo.undo(sid, 1)
    op3 = hermes_undo.undo(sid, 1)
    rewound_sets = [set(op["rewound_ids"]) for op in (op1, op2, op3)]
    assert rewound_sets == [{ids[3]}, {ids[2]}, {ids[1]}]
    assert rewound_sets[0].isdisjoint(rewound_sets[1])
    assert rewound_sets[0].isdisjoint(rewound_sets[2])
    assert rewound_sets[1].isdisjoint(rewound_sets[2])

    redone = hermes_undo.redo(sid, 2)
    assert redone["reactivated_count"] == 2
    assert _active_ids(db, sid) == [ids[0], ids[1], ids[2]]
    state = hermes_undo.get_state(sid)
    assert [op.rewound_ids for op in state.redo_stack] == [[ids[1]], [ids[2]]]
    assert [op.rewound_ids for op in state.undo_stack] == [[ids[3]]]


def test_redo_m_non_positive_and_empty_stack_do_not_bump(db):
    sid = _make_session(db)
    db.append_message(sid, "user", "u")

    assert hermes_undo.redo(sid, 0)["message"] == "nothing to redo"
    assert hermes_undo.redo(sid, -1)["message"] == "nothing to redo"
    assert db.get_session(sid)["redo_count"] is None

    assert hermes_undo.redo(sid, 10)["message"] == "nothing to redo"
    assert db.get_session(sid)["redo_count"] is None

    hermes_undo.undo(sid, 1)
    hermes_undo.clear_state(sid)
    cold = hermes_undo.redo(sid, 1)
    assert "doesn't survive a restart" in cold["message"]
    assert db.get_session(sid)["redo_count"] is None


def test_redo_degrades_gracefully_when_no_rows_restorable(monkeypatch, db):
    """Zero rows restorable == transcript was rewritten → graceful no-op, no raise.

    A redo op whose rows ALL fail to restore means the transcript was rewritten
    out from under the stack (/compress, /retry); redo across that is impossible,
    same as after a restart. This must NOT raise (it's reachable by normal user
    actions: /undo then /compress then /redo). A PARTIAL restore is different —
    that still fails loud (see test_redo_fail_loud_requires_each_op_restore_all_rewound_ids).
    """
    sid = _make_session(db)
    _seed_three_half_turns(db, sid)
    hermes_undo.undo(sid, 1)
    monkeypatch.setattr(db, "restore_ids", lambda _sid, _ids: 0)

    r = hermes_undo.redo(sid, 1)
    assert r["reactivated_count"] == 0
    assert "transcript changed" in r["message"]
    state = hermes_undo.get_state(sid)
    assert state.undo_stack == []
    assert state.redo_stack == []


def test_user_message_append_invalidates_redo_branch(db):
    """Typing a new message after an undo must discard the redo branch.

    ``undo_stack`` semantically holds the redo branch (ops that were undone and
    can be redone — ``redo()`` pops it). A text editor discards that branch the
    moment you type after undoing, so a later /redo must NOT resurrect the
    undone content out of order. Regression: previously on_user_message_appended
    cleared only the vestigial ``redo_stack``, leaving ``undo_stack`` live so a
    /redo wedged stale rows back in.
    """
    sid = _make_session(db)
    _seed_three_half_turns(db, sid)
    hermes_undo.undo(sid, 1)
    hermes_undo.redo(sid, 1)
    state = hermes_undo.get_state(sid)
    assert len(state.redo_stack) == 1
    state.undo_stack.append(hermes_undo.UndoOp(n=99, rewound_ids=[12345]))

    db.append_message(sid, "user", "new branch")
    hermes_undo.on_user_message_appended(sid)

    assert state.redo_stack == []
    assert state.undo_stack == []


def test_undo_then_type_then_redo_does_not_resurrect(db):
    """End-to-end repro: undo, type a new message, redo → nothing resurrected."""
    sid = _make_session(db)
    u1, a1, u2, a2 = _seed_three_half_turns(db, sid)

    r = hermes_undo.undo(sid, 1)
    assert r["rewound_ids"] == [a2]
    assert _active_ids(db, sid) == [u1, a1, u2]

    new = db.append_message(sid, "user", "different question")
    hermes_undo.on_user_message_appended(sid)
    assert _active_ids(db, sid) == [u1, a1, u2, new]

    r = hermes_undo.redo(sid, 1)
    assert r["reactivated_count"] == 0
    # a2 must NOT be wedged back between u2 and the new message
    assert _active_ids(db, sid) == [u1, a1, u2, new]


def test_redo_after_transcript_rewrite_does_not_raise(db):
    """A /compress (replace_messages) after /undo must not crash a later /redo.

    replace_messages hard-deletes and renumbers rows, so the in-memory
    undo_stack references dead ids. redo() must degrade gracefully (clear the
    branch, report "transcript changed") instead of raising the restore-count
    invariant. Regression: previously this raised an unhandled RuntimeError.
    """
    sid = _make_session(db)
    _seed_three_half_turns(db, sid)
    hermes_undo.undo(sid, 1)
    state = hermes_undo.get_state(sid)
    assert state.undo_stack  # a redo op is pending

    # Simulate /compress: replace the transcript with the active-only rows.
    active = db.get_messages(sid, include_inactive=False)
    db.replace_messages(
        sid, [{"role": m["role"], "content": m.get("content", "")} for m in active]
    )

    r = hermes_undo.redo(sid, 1)
    assert r["reactivated_count"] == 0
    assert "transcript changed" in r["message"]
    # Stacks invalidated so a subsequent redo is also a clean no-op.
    assert state.undo_stack == []
    assert state.redo_stack == []


def test_undo_with_no_rewound_rows_touches_neither_stack(monkeypatch, db):
    """A rewind that deactivates nothing must not push an empty op or wipe redo.

    Matches the None-path contract: a no-op undo leaves both stacks untouched.
    Guards the degenerate/raced rewind_to_message return.
    """
    sid = _make_session(db)
    _seed_three_half_turns(db, sid)
    hermes_undo.undo(sid, 1)
    hermes_undo.redo(sid, 1)
    state = hermes_undo.get_state(sid)
    assert state.redo_stack  # a redo op is pending
    redo_before = list(state.redo_stack)
    undo_before = list(state.undo_stack)

    # Force rewind_to_message to report it deactivated nothing.
    monkeypatch.setattr(
        db, "rewind_to_message", lambda *a, **k: {"rewound_ids": []}
    )
    r = hermes_undo.undo(sid, 1)
    assert r["rewound_ids"] == []
    assert r["message"] == "nothing to undo"
    assert state.undo_stack == undo_before
    assert state.redo_stack == redo_before


def test_new_undo_clears_redo_and_discarded_ops_are_unreachable(db):
    sid = _make_session(db)
    _seed_three_half_turns(db, sid)
    hermes_undo.undo(sid, 1)
    hermes_undo.redo(sid, 1)
    state = hermes_undo.get_state(sid)
    assert state.redo_stack

    hermes_undo.undo(sid, 1)
    assert state.redo_stack == []
    source = inspect.getsource(hermes_undo)
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "UndoOp"
    ]
    assert len(calls) == 1


def test_d13_fires_on_lone_multimodal_user_tail_and_redo_round_trips(db):
    sid = _make_session(db)
    u_prev = db.append_message(sid, "user", "plain")
    a1 = db.append_message(sid, "assistant", "answer one")
    mm = db.append_message(
        sid,
        "user",
        [{"type": "text", "text": "see this"}, {"type": "image_url", "image_url": "x"}],
    )
    a2 = db.append_message(sid, "assistant", "answer two")
    before = _active_ids(db, sid)
    assert hermes_undo.compute_half_turn_target(db.get_messages(sid), 1) == a2

    result = hermes_undo.undo(sid, 1)

    assert set(result["rewound_ids"]) == {mm, a2}
    assert result["prefill_text"] is None
    assert _active_ids(db, sid) == [u_prev, a1]
    assert db.get_messages(sid)[-1]["id"] == a1

    redo = hermes_undo.redo(sid, 1)
    assert redo["new_tail_id"] == a2
    assert _active_ids(db, sid) == before


def test_d13_firing_fixture_is_reachable_by_two_turn_operation_sequence(db):
    sid = _make_session(db)
    db.append_message(sid, "user", "first text")
    a1 = db.append_message(sid, "assistant", "first answer")
    mm = db.append_message(
        sid,
        "user",
        [{"type": "text", "text": "image prompt"}, {"type": "image_url", "image_url": "x"}],
    )
    a2 = db.append_message(sid, "assistant", "image answer")

    result = hermes_undo.undo(sid, 1)

    assert set(result["rewound_ids"]) == {mm, a2}
    assert _active_ids(db, sid)[-1] == a1
    assert result["prefill_text"] is None


def test_d13_plain_string_control_does_not_lower_target(db):
    sid = _make_session(db)
    db.append_message(sid, "user", "plain")
    db.append_message(sid, "assistant", "answer one")
    user = db.append_message(sid, "user", "editable")
    a2 = db.append_message(sid, "assistant", "answer two")

    result = hermes_undo.undo(sid, 1)

    assert result["rewound_ids"] == [a2]
    assert result["prefill_text"] == "editable"
    assert _active_ids(db, sid)[-1] == user


def test_zero_non_other_none_return_noop_and_redo_stack_survives(db):
    sid = _make_session(db)
    db.append_message(sid, "system", "rules only")
    state = hermes_undo.get_state(sid)
    state.undo_stack.append(hermes_undo.UndoOp(n=7, rewound_ids=[700]))
    state.redo_stack.append(hermes_undo.UndoOp(n=8, rewound_ids=[800]))

    assert hermes_undo.compute_half_turn_target(db.get_messages(sid), 1) is None
    result = hermes_undo.undo(sid, 1)

    assert result["rewound_ids"] == []
    assert result["prefill_text"] is None
    assert result["message"] == "nothing to undo"
    assert state.undo_stack == [hermes_undo.UndoOp(n=7, rewound_ids=[700])]
    assert state.redo_stack == [hermes_undo.UndoOp(n=8, rewound_ids=[800])]
    assert [m["active"] for m in db.get_messages(sid, include_inactive=True)] == [1]


def test_tool_monotonicity_violation_raises_before_orphaning(db):
    sid = _make_session(db)
    db.append_message(sid, "user", "run")
    assistant = db.append_message(sid, "assistant", None, tool_calls=[{"id": "ok"}])
    tool = db.append_message(sid, "tool", "result", tool_call_id="ok")
    assert tool > assistant

    bad = _make_session(db, "bad-tool-order")
    db.append_message(bad, "user", "run")
    early_tool = db.append_message(bad, "tool", "too early", tool_call_id="call-1")
    owner = db.append_message(
        bad, "assistant", None, tool_calls=[{"id": "call-1"}]
    )
    assert early_tool < owner

    with pytest.raises(ValueError, match="orphan"):
        db.rewind_to_message(bad, owner, require_user_role=False)

    rows = db.get_messages(bad)
    assert [row["id"] for row in rows] == [m["id"] for m in rows]
    assert all(row["active"] == 1 for row in rows)


def test_redo_fail_loud_requires_each_op_restore_all_rewound_ids(monkeypatch, db):
    sid = _make_session(db)
    ids = _seed_three_half_turns(db, sid)
    hermes_undo.undo(sid, 2)
    op = hermes_undo.get_state(sid).undo_stack[-1]
    assert set(op.rewound_ids) == {ids[2], ids[3]}

    calls = []

    def partial_restore(_sid, rewound_ids):
        calls.append(list(rewound_ids))
        return len(rewound_ids) - 1

    monkeypatch.setattr(db, "restore_ids", partial_restore)
    with pytest.raises(RuntimeError, match="restored 1 of 2 rewound rows"):
        hermes_undo.redo(sid, 1)
    assert calls == [[ids[2], ids[3]]]
