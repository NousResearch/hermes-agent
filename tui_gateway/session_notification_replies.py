"""Drain addressed human replies only in the already-existing desktop owner.

The messaging gateway must never reconstruct a GUI agent with messaging tools.
This runs on the same idle poller as heartbeat/kanban, through normal turn admission.
"""
import time
from gateway.notification_replies import connect, origin_exists, _REPLY_TTL, _OWNER_SURFACES


def poll_replies(sid, session, home, submit):
    if not (home / 'notification-replies.db').exists():
        return
    with session['history_lock']:
        if (session.get('running') or session.get('_finalized') or session.get('_closing')
                or not session.get('agent')):
            return
        with connect(home) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM notification_routes WHERE session_id=? AND status='queued' "
                             "ORDER BY created LIMIT 1", (session.get('session_key'),)).fetchone()
            if row is None:
                return
            if session.get('source') != row['source'] or row['source'] not in _OWNER_SURFACES:
                return
            if (not origin_exists(home, row['session_id'], row['source'])
                    or time.time() - row['created'] > _REPLY_TTL):
                db.execute("UPDATE notification_routes SET status='expired' WHERE message_id=?", (row['message_id'],))
                return
            db.execute("UPDATE notification_routes SET status='dispatching' WHERE message_id=?",
                       (row['message_id'],))
        session['running'] = True
    # An exception/crash after claim is uncertain, NOT a retryable approval. Keep
    # dispatching durable so another process/poller cannot run it a second time.
    try:
        started = submit('notification:' + row['message_id'], sid, session, row['reply_text'], image_paths=[])
    except BaseException:
        with session['history_lock']:
            session['running'] = False
        raise
    with connect(home) as db:
        db.execute('UPDATE notification_routes SET status=? WHERE message_id=?',
                   ('dispatched' if started else 'queued', row['message_id']))
    if not started:
        with session['history_lock']:
            session['running'] = False
