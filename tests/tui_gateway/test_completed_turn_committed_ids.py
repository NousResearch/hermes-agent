"""Tests for `_completed_turn_committed_ids` (tui_gateway/server.py).

The desktop renders a sent turn optimistically and stamps its client-minted
row ids to the real committed ids carried on the ``message.complete`` frame.
If that frame omits the ids, the live session-sync poll re-appends every
committed row as a DUPLICATE (each message shown twice, DB clean). This helper
is the server half of the fix: it extracts ``[user_id, assistant_id]`` for the
just-finished turn from ``session["history"]``, keyed off the
``_db_persisted_row_id`` stamped during DB flush.
"""

from __future__ import annotations

import tui_gateway.server as srv


def test_returns_user_then_assistant_committed_ids_in_order():
    history = [
        {"role": "user", "content": "hi", "_db_persisted_row_id": 100},
        {"role": "assistant", "content": "hello", "_db_persisted_row_id": 101},
    ]
    assert srv._completed_turn_committed_ids(history) == [100, 101]


def test_returns_last_user_and_last_assistant_text_across_a_tool_loop():
    # A tool-loop turn: user, assistant(tool_calls), tool result, final
    # assistant text. The client only holds two optimistic rows (the user
    # message and the FINAL assistant text bubble); tool rows arrive via the
    # poll with their own ids. So the frame must carry exactly the user id and
    # the final assistant *text* id.
    history = [
        {"role": "user", "content": "do a thing", "_db_persisted_row_id": 200},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "function": {"name": "x", "arguments": "{}"}}],
            "_db_persisted_row_id": 201,
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result", "_db_persisted_row_id": 202},
        {"role": "assistant", "content": "done", "_db_persisted_row_id": 203},
    ]
    assert srv._completed_turn_committed_ids(history) == [200, 203]


def test_skips_assistant_tool_calls_row_with_no_text():
    # An assistant(tool_calls) row with empty content is NOT an optimistic
    # client text bubble — it must never be returned as the assistant id.
    history = [
        {"role": "user", "content": "go", "_db_persisted_row_id": 10},
        {
            "role": "assistant",
            "content": "   ",
            "tool_calls": [{"id": "t", "function": {"name": "y", "arguments": "{}"}}],
            "_db_persisted_row_id": 11,
        },
    ]
    # Only the user id is a real optimistic row; no final assistant text yet.
    assert srv._completed_turn_committed_ids(history) == [10]


def test_missing_row_ids_are_skipped():
    # Best-effort: a message without a committed row id (not yet flushed, or a
    # mock) is skipped rather than emitting a bogus/None id.
    history = [
        {"role": "user", "content": "hi"},  # no _db_persisted_row_id
        {"role": "assistant", "content": "hello", "_db_persisted_row_id": 101},
    ]
    assert srv._completed_turn_committed_ids(history) == [101]


def test_non_int_row_id_is_ignored():
    history = [
        {"role": "user", "content": "hi", "_db_persisted_row_id": "100"},
        {"role": "assistant", "content": "hello", "_db_persisted_row_id": 101},
    ]
    assert srv._completed_turn_committed_ids(history) == [101]


def test_empty_or_invalid_history_returns_empty():
    assert srv._completed_turn_committed_ids([]) == []
    assert srv._completed_turn_committed_ids(None) == []  # type: ignore[arg-type]
    assert srv._completed_turn_committed_ids(["garbage", 42, None]) == []  # type: ignore[list-item]
