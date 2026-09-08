"""Drain addressed human replies only in the already-existing desktop owner.

The messaging gateway must never reconstruct a GUI agent with messaging tools.
This runs on the same idle poller as heartbeat/kanban, through normal turn admission.
"""
import time

from gateway.notification_replies import (
    _OWNER_SURFACES,
    _REPLY_TTL,
    connect,
    owner_session_lineage,
)


def poll_replies(sid, session, home, submit):
    if not (home / "notification-replies.db").exists():
        return
    with session["history_lock"]:
        if (
            session.get("running")
            or session.get("_finalized")
            or session.get("_closing")
            or not session.get("agent")
        ):
            return
        session_key = session.get("session_key")
        source = session.get("source")
        owner_ids = owner_session_lineage(home, session_key, source)
        with connect(home) as db:
            db.execute("BEGIN IMMEDIATE")
            cutoff = time.time() - _REPLY_TTL
            db.execute(
                "UPDATE notification_reply_queue SET status='expired' WHERE status='queued' "
                "AND message_id IN (SELECT message_id FROM notification_routes WHERE created<?)",
                (cutoff,),
            )
            db.execute(
                "UPDATE notification_routes SET status='expired' WHERE created<? AND message_id IN "
                "(SELECT message_id FROM notification_reply_queue WHERE status='expired')",
                (cutoff,),
            )
            if not owner_ids:
                db.execute(
                    "UPDATE notification_reply_queue SET status='expired' "
                    "WHERE status='queued' AND session_id=?",
                    (session_key,),
                )
                db.execute(
                    "UPDATE notification_routes SET status='expired' WHERE message_id IN "
                    "(SELECT message_id FROM notification_reply_queue "
                    "WHERE status='expired' AND session_id=?)",
                    (session_key,),
                )
                return
            placeholders = ",".join("?" for _ in owner_ids)
            row = db.execute(
                "SELECT q.*, r.created AS notification_created FROM notification_reply_queue q "
                "JOIN notification_routes r ON r.message_id=q.message_id "
                f"WHERE q.status='queued' AND q.session_id IN ({placeholders}) "
                "ORDER BY q.created LIMIT 1",
                owner_ids,
            ).fetchone()
            if row is None:
                return
            if source != row["source"] or row["source"] not in _OWNER_SURFACES:
                return
            db.execute(
                "UPDATE notification_reply_queue SET status='dispatching' WHERE reply_id=?",
                (row["reply_id"],),
            )
            db.execute(
                "UPDATE notification_routes SET status='dispatching' WHERE message_id=?",
                (row["message_id"],),
            )
        session["running"] = True
    # An exception/crash after claim is uncertain, NOT a retryable approval. Keep
    # dispatching durable so another process/poller cannot run it a second time.
    try:
        started = submit(
            "notification:" + row["message_id"],
            sid,
            session,
            row["reply_text"],
            image_paths=[],
        )
    except BaseException:
        with session["history_lock"]:
            session["running"] = False
        raise
    with connect(home) as db:
        status = "dispatched" if started else "queued"
        db.execute(
            "UPDATE notification_reply_queue SET status=? WHERE reply_id=?",
            (status, row["reply_id"]),
        )
        db.execute(
            "UPDATE notification_routes SET status=? WHERE message_id=?",
            (status, row["message_id"]),
        )
    if not started:
        with session["history_lock"]:
            session["running"] = False
