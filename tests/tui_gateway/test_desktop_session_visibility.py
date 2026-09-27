"""Ordinary Desktop chats stay listable; ``hidden`` is internal plumbing only.

Production class: an active ``source='desktop'`` conversation was persisted ``hidden=1`` because the
server trusted a client ``hidden`` flag at ``session.create``/``session.set_hidden`` and re-applied it
on every row persist, so ``session.list`` dropped it and a manual un-hide was undone on the next
reconnect / resume. Canonical Bot Chat, Agent Inbox, group-room member sessions and internal sources
(kanban/tool/oneshot) are the only rows allowed to leave the default list.
"""

from __future__ import annotations

import pytest

import tui_gateway.methods_session  # noqa: F401  (registers RPC methods)
import tui_gateway.server as srv
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    monkeypatch.setattr(srv, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
    live_before = set(srv._sessions)
    try:
        yield database
    finally:
        for sid in set(srv._sessions) - live_before:
            srv._sessions.pop(sid, None)
        database.close()


def _call(method: str, params: dict) -> dict:
    envelope = srv._methods[method](1, params)
    assert "error" not in envelope, envelope
    return envelope["result"]


def _create_and_persist(db: SessionDB, params: dict) -> tuple[str, dict]:
    """session.create, then what prompt.submit does: persist the row and record the user turn."""
    sid = _call("session.create", params)["session_id"]
    session = srv._sessions[sid]
    assert srv._ensure_session_db_row(session)
    db.append_message(session_id=session["session_key"], role="user", content="first turn")
    return sid, session


def _listed() -> set[str]:
    return {row["id"] for row in _call("session.list", {})["sessions"]}


def test_client_hide_requests_never_drop_an_ordinary_desktop_chat_from_the_list(db):
    sid, session = _create_and_persist(db, {"source": "desktop", "title": "weekly review", "hidden": True})
    key = session["session_key"]
    # Every later hide vector: the live runtime id, the stored key (sweep), the next prompt's persist,
    # and a reconnect that resumes the chat and asks again.
    assert _call("session.set_hidden", {"session_id": sid, "hidden": True})["hidden"] is False
    assert _call("session.set_hidden", {"session_id": key, "hidden": True})["hidden"] is False
    assert srv._ensure_session_db_row(session)
    live_sid = _call("session.resume", {"session_id": key, "omit_messages": True})["session_id"]
    assert _call("session.set_hidden", {"session_id": live_sid, "hidden": True})["hidden"] is False
    assert srv._ensure_session_db_row(srv._sessions[live_sid])
    # A branch of it, created with the same client flag, is born listable too.
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    child_key = _call("session.create", {"source": "desktop", "parent_session_id": key,
                                         "messages": history, "hidden": True})["stored_session_id"]
    # A second "Bot Chat" draft: its queued title collides with the canonical row and is dropped, so the
    # untitled row it would have vouched for stays listable.
    db.create_session("canonical", source="desktop")
    db.set_session_title("canonical", "Bot Chat")
    assert db.set_session_hidden("canonical", True)
    dup_sid, dup = _create_and_persist(db, {"source": "desktop", "title": "Bot Chat", "hidden": True})
    assert _call("session.set_hidden", {"session_id": dup_sid, "hidden": True})["hidden"] is False

    assert {key, child_key, dup["session_key"]} <= _listed()
    assert db.get_session(dup["session_key"])["title"] is None
    assert key in {row["id"] for row in db.list_recent_sessions_bounded(limit=20)}


@pytest.mark.parametrize("params", [
    {"source": "desktop", "title": "Bot Chat", "follow_profile_config": True},
    {"source": "desktop", "title": "Agent Inbox"},
    {"source": "desktop", "title": "Group: room-9 · main", "room_plumbing": True, "follow_profile_config": True},
    {"source": "tool"},
])
def test_plumbing_sessions_are_born_hidden_but_may_be_unhidden(db, params):
    _, session = _create_and_persist(db, {**params, "hidden": True})
    key = session["session_key"]

    assert db.get_session(key)["hidden"] == 1
    assert key not in _listed()
    # The owner can still un-hide it, and the next persist does not re-hide it.
    _call("session.set_hidden", {"session_id": key, "hidden": False})
    assert srv._ensure_session_db_row(session)
    assert db.get_session(key)["hidden"] == 0
    # After compression the untitled tip carries no marker of its own; its plumbing root vouches for the
    # whole lineage, which hides together.
    db.end_session(key, "compression")
    db.create_session("tip", source="desktop", parent_session_id=key)
    db.append_message(session_id="tip", role="user", content="after compression")
    assert _call("session.set_hidden", {"session_id": "tip", "hidden": True})["hidden"] is True
    assert db.get_session(key)["hidden"] == db.get_session("tip")["hidden"] == 1
