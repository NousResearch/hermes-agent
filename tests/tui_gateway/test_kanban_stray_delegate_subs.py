"""Kanban subscriptions bound to a parent's own subagent are reclaimed by the parent (#150198).

Before the reserved child kept its ``_delegate_from`` marker, ``kanban_create`` resolved the parent's
key to the unmarked child and saved the card's subscription under the child's id. v33 re-tags the
session row but never touches kanban.db, so those cards kept notifying the subagent.
"""

import sqlite3
from unittest.mock import patch

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from persistence import SessionId
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope
from tui_gateway import server


def _subscribed_task(chat_id: str) -> str:
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="stray sub", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=chat_id)
        kb.complete_task(conn, tid, summary="shipped")
        return tid
    finally:
        conn.close()


def _chat_ids(tid: str) -> list:
    conn = kbc.connect()
    try:
        return [row["chat_id"] for row in kbn.list_notify_subs(conn, task_id=tid)]
    finally:
        conn.close()


def _create(db, session_id: str, source: str, **kwargs) -> None:
    db.sessions.create(NewSession(session_id=SessionId(session_id), source=source, **kwargs))


def test_parent_reclaims_a_subscription_bound_to_its_own_subagent(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        _create(db, "parent", "desktop")
        _create(db, "child", "subagent", parent_session_id=SessionId("parent"),
                model_config={"_delegate_from": "parent"})
        _create(db, "other", "desktop")
        _create(db, "other-child", "subagent", parent_session_id=SessionId("other"),
                model_config={"_delegate_from": "other"})
        mine, theirs = _subscribed_task("child"), _subscribed_task("other-child")

        with patch.object(server, "_get_db", lambda: db):
            texts = server._collect_kanban_notifications({"session_key": "parent"})

        assert len(texts) == 1 and mine in texts[0]
        assert _chat_ids(mine) == ["parent"]
        assert _chat_ids(theirs) == ["other-child"]  # another session's subagent is left alone
        # A subagent's own tab never claims its parent's (or siblings') subscriptions.
        assert db.lineage.delegate_descendant_ids(SessionId("child")) == ()
    finally:
        db.close()


def test_a_transient_db_error_does_not_end_the_reclaim(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        _create(db, "parent", "desktop")
        _create(db, "child", "subagent", parent_session_id=SessionId("parent"),
                model_config={"_delegate_from": "parent"})
        tid = _subscribed_task("child")
        session = {"session_key": "parent"}
        real_count = kbn.count_notify_subs

        def locked(*args, **kwargs):
            if kwargs.get("chat_id") == "child":
                raise sqlite3.OperationalError("database is locked")
            return real_count(*args, **kwargs)

        with patch.object(server, "_get_db", lambda: db):
            with patch.object(kbn, "count_notify_subs", locked):
                assert server._collect_kanban_notifications(session) == []
            texts = server._collect_kanban_notifications(session)

        assert len(texts) == 1 and tid in texts[0]
        assert _chat_ids(tid) == ["parent"]
    finally:
        db.close()


def test_reclaims_a_subscription_bound_to_a_misrouted_childs_continuation(tmp_path):
    """A child resumed as the chat compressed into an unmarked continuation; the pre-fix resolver subscribed
    the card under that tip, and v33 rightly leaves the continuation unmarked."""
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        _create(db, "parent", "desktop")
        db.messages.append(SessionId("parent"), NewMessage(role="user", content="parent question"))
        _create(db, "child", "subagent", parent_session_id=SessionId("parent"),
                model_config={"_delegate_from": "parent"})
        db.messages.append(SessionId("child"), NewMessage(role="assistant", content="child work"))
        db.lineage.publish_compression_child(
            SessionId("child"), CompressionChild(session_id=SessionId("child-tip"), source="subagent"),
            [NewMessage(role="user", content="typed after misrouted resume")], require_compression_lease=False)
        tid = _subscribed_task("child-tip")

        with patch.object(server, "_get_db", lambda: db):
            texts = server._collect_kanban_notifications({"session_key": "parent"})

        assert len(texts) == 1 and tid in texts[0]
        assert _chat_ids(tid) == ["parent"]
    finally:
        db.close()
