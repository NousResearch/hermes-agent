"""``session.history`` must hand out the durable ``row_id`` for each message.

``session.history`` is a renderer-facing projection: it returns
``_history_to_messages(history)``, and that helper forwards ``row_id`` only
when the history dicts carry ``_row_id`` (``tui_gateway/server.py``).

``_row_id`` is opt-in (``include_row_ids=True``) so model-facing consumers —
ACP restore, export, inspection, the post-undo live-replay reload — keep the
historical transcript shape. Every *display* projection opts in; this pins
``session.history`` to that side of the split, because ``message.react``
addresses messages by exactly this id and its only fallback (``newest_role``)
can reach the newest row of a role and nothing older.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from tui_gateway import server

SESSION_KEY = "20260730_120000_abc123"


@pytest.fixture
def profile_home(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SESSION_KEY, source="tui")
    db.append_message(SESSION_KEY, "user", "first question")
    db.append_message(SESSION_KEY, "assistant", "first answer")
    db.append_message(SESSION_KEY, "user", "second question")
    db.append_message(SESSION_KEY, "assistant", "second answer")
    return tmp_path


@pytest.fixture
def live_session(profile_home):
    sid = "tui-sid-1"
    server._sessions[sid] = {
        "session_key": SESSION_KEY,
        "profile_home": str(profile_home),
        "history": [],
    }
    try:
        yield sid
    finally:
        server._sessions.pop(sid, None)


def _history(sid):
    return server._methods["session.history"](1, {"session_id": sid})


def test_session_history_carries_row_ids(live_session):
    """Every rendered message is addressable by message.react."""
    resp = _history(live_session)
    messages = resp["result"]["messages"]

    assert messages, "expected the persisted transcript"
    missing = [m for m in messages if m.get("row_id") is None]
    assert not missing, f"{len(missing)}/{len(messages)} messages have no row_id"


def test_row_ids_match_the_durable_message_rows(live_session, profile_home):
    """The ids handed out are the real ``messages.id`` values, in order."""
    db = SessionDB(db_path=profile_home / "state.db")
    expected = [
        m["_row_id"]
        for m in db.get_messages_as_conversation(
            SESSION_KEY, include_ancestors=True, include_row_ids=True
        )
    ]

    messages = _history(live_session)["result"]["messages"]
    assert [m["row_id"] for m in messages] == expected


def test_older_messages_are_addressable_not_just_the_newest(live_session, profile_home):
    """The regression's user-visible shape: reacting to an older message.

    Without row_ids the only way to name a message is ``newest_role``, which
    resolves to the newest row of that role — so every earlier message becomes
    unreactable.
    """
    db = SessionDB(db_path=profile_home / "state.db")
    newest_user = db.latest_message_row_id(SESSION_KEY, role="user")

    messages = _history(live_session)["result"]["messages"]
    user_rows = [m["row_id"] for m in messages if m["role"] == "user"]

    assert len(user_rows) > 1, "fixture should have more than one user turn"
    older = [r for r in user_rows if r != newest_user]
    assert older, "expected at least one message the newest_role fallback cannot reach"

    # The write path accepts them — only the projection was withholding the id.
    assert db.set_message_reaction(SESSION_KEY, older[0], "\N{THUMBS UP SIGN}") is not None


def test_model_facing_projection_keeps_historical_shape(profile_home):
    """Guard the other half of the split: no row ids leak into replay history."""
    db = SessionDB(db_path=profile_home / "state.db")
    replay = db.get_messages_as_conversation(SESSION_KEY, repair_alternation=True)
    assert all("_row_id" not in m for m in replay)
