"""The resume display projection ships each row's durable ``message_uid`` next to ``row_id`` (#123294).

A compaction rewrite renumbers row ids but keeps ``message_uid``, so the desktop graft can
recognize a rewritten refreshed page against the cached window only if this projection
forwards the uid on every row it emits — the message branch and the tool branch alike.
"""

from __future__ import annotations

from tui_gateway import server

TIMESTAMP = 1_771_500_000


def _row(role: str, content: str, *, row_id: int, uid: str) -> dict:
    return {
        "role": role,
        "content": content,
        "_row_id": row_id,
        "message_uid": uid,
        "timestamp": TIMESTAMP,
    }


def test_message_rows_carry_message_uid_next_to_row_id():
    history = [
        _row("user", "older prompt", row_id=100, uid="m100"),
        _row("assistant", "look one", row_id=101, uid="m101"),
    ]

    # The rebound copy on server carries the module's globals (method_ctx.bind_module).
    messages = server._history_to_messages(history)

    assert [(m.get("row_id"), m.get("message_uid")) for m in messages] == [
        (100, "m100"),
        (101, "m101"),
    ]


def test_tool_rows_carry_message_uid():
    # The tool branch builds its dict from a copied-key tuple; the uid must be in it,
    # or a rewritten tool row loses the identity the client graft matches by.
    history = [
        _row("tool", "ok", row_id=102, uid="m102")
        | {"tool_call_id": "c1", "tool_name": "terminal"},
    ]

    messages = server._history_to_messages(history)

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["message_uid"] == "m102"


def test_rows_without_message_uid_omit_the_key():
    # Legacy rows predate the uid column; the projection must not invent one.
    history = [
        {
            "role": "user",
            "content": "older prompt",
            "_row_id": 100,
            "timestamp": TIMESTAMP,
        }
    ]

    messages = server._history_to_messages(history)

    assert "message_uid" not in messages[0]
