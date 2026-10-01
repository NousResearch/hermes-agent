"""Profile-bound native delegate admissions. Not an OS/global worker inventory.

Only the originating interpreter can attest active work. Persisted active rows
without that attestation are unknown, never inferred dead from a PID or clock.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

OWNER_ID = uuid.uuid4().hex
TERMINAL = {"completed", "failed", "cancelled", "ended"}
STATUS = {"completed": "completed", "failed": "failed", "error": "failed",
          "timeout": "failed", "interrupted": "cancelled", "cancelled": "cancelled",
          "running": "running", "queued": "queued", "waiting": "waiting"}
# Only new roster observations are eligible, never state.db/transcripts or legacy
# rows without a terminal timestamp. Active/unknown observations are never pruned.
TERMINAL_RETENTION_SECONDS = 30 * 24 * 60 * 60
TERMINAL_RETAIN_COUNT = 2000
_lock = threading.RLock()
_live = {}


@contextmanager
def _connect(home):
    path = Path(home) / "worker-roster.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    db.execute("""CREATE TABLE IF NOT EXISTS workers (
        run_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, session_key TEXT NOT NULL,
        subagent_id TEXT NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1)""")
    try:
        with db:
            yield db
    finally:
        db.close()


def _prune(db):
    # Metadata timestamp is additive; existing rows are not migrated/deleted.
    eligible = "status IN ('completed','failed','cancelled','ended') AND json_extract(metadata, '$.retention_v1') = 1"
    db.execute(f"DELETE FROM workers WHERE {eligible} AND json_extract(metadata, '$.finished_at') < ?",
               (time.time() - TERMINAL_RETENTION_SECONDS,))
    db.execute(f"DELETE FROM workers WHERE run_id IN (SELECT run_id FROM workers WHERE {eligible} "
               "ORDER BY json_extract(metadata, '$.finished_at') DESC, run_id DESC LIMIT -1 OFFSET ?)",
               (TERMINAL_RETAIN_COUNT,))


def register(record, home):
    """Persist before publishing admission; failures must reject new native work."""
    if record.get("_roster"):
        return
    key = record.get("root_session_id") or record.get("owner_agent_session_id")
    if not key:
        return
    home = str(Path(home).resolve())
    run_id = uuid.uuid4().hex
    # No prompt/context/output duplication. Only a bounded display label.
    metadata = {k: record.get(k) if isinstance(record.get(k), (str, int, float)) else None
                for k in ("parent_id", "delegation_id", "started_at", "parent_run_id")}
    metadata["goal"] = str(record.get("goal") or "")[:160]
    metadata["retention_v1"] = 1
    state = record.get("status", "queued")
    with _lock:
        with _connect(home) as db:
            _prune(db)
            db.execute("INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?, 1)",
                       (run_id, OWNER_ID, key, record["subagent_id"], state, json.dumps(metadata)))
        record["_roster"] = (home, run_id)
        record["root_session_id"] = key
        _live[(home, run_id)] = record


def admit(parent, tasks, delegation_id):
    from hermes_constants import get_hermes_home
    parent_record = getattr(parent, "_worker_record", None)
    if not isinstance(parent_record, dict):
        parent_record = {}
    home = parent_record.get("_roster", (get_hermes_home(),))[0]
    root = parent_record.get("root_session_id") or str(getattr(parent, "session_id", "") or "")
    records = []
    try:
        for task in tasks:
            record = dict(subagent_id="pending-" + uuid.uuid4().hex, status="queued",
                          goal=task["goal"], root_session_id=root,
                          owner_agent_session_id=str(getattr(parent, "session_id", "") or ""),
                          parent_id=getattr(parent, "_subagent_id", None),
                          parent_run_id=parent_record.get("_roster", (None, None))[1],
                          delegation_id=delegation_id, started_at=time.time())
            register(record, home)
            records.append(record)
        return records
    except Exception:
        for record in records:
            finish(record, "failed")
        raise


def bind(record, child):
    child._worker_record = record
    sid = getattr(child, "_subagent_id", None)
    if isinstance(sid, str) and sid:
        with _lock:
            record["subagent_id"] = sid
            ref = record.get("_roster")
            if ref:
                with _connect(ref[0]) as db:
                    db.execute("UPDATE workers SET subagent_id=?, version=version+1 WHERE run_id=?", (sid, ref[1]))


def claim(child):
    record = getattr(child, "_worker_record", None)
    if not isinstance(record, dict):
        return True
    with _lock:
        if record.get("status") in TERMINAL:
            return False
        transition(record, "running")
        return True


def cancel_queued(child):
    record = getattr(child, "_worker_record", None)
    if isinstance(record, dict):
        with _lock:
            if record.get("status") == "queued":
                finish(record, "cancelled")


def transition(record, status):
    ref = record.get("_roster")
    if not ref:
        return
    with _lock:
        if record.get("status") in TERMINAL:
            return
        with _connect(ref[0]) as db:
            db.execute("UPDATE workers SET status=?, version=version+1 WHERE run_id=? AND status NOT IN ('completed','failed','cancelled','ended')",
                       (status, ref[1]))
        record["status"] = status


@contextmanager
def waiting_for_children(parent):
    """Actual synchronous delegate join, not a tool-name or inactivity guess."""
    record = getattr(parent, "_worker_record", None)
    entered = False
    if isinstance(record, dict):
        with _lock:
            if record.get("status") not in TERMINAL:
                # Publish the count only after persistence accepts entry. A failed
                # entry never executes the join and must not leave a phantom waiter.
                transition(record, "waiting")
                record["_waiters"] = record.get("_waiters", 0) + 1
                entered = True
    try:
        yield
    finally:
        if entered and isinstance(record, dict):
            with _lock:
                record["_waiters"] -= 1
                if not record["_waiters"]:
                    try:
                        transition(record, "running")
                    except BaseException:
                        # The join ended even if its durable exit did not. Keep
                        # recovery possible, but never attest the stale waiting row.
                        record["status"] = "unknown"
                        raise


def finish(record, status):
    try:
        ref = record.get("_roster")
        if not ref:
            return
        state = STATUS.get(status, "ended")
        if state not in TERMINAL:
            state = "ended"
        with _lock:
            with _connect(ref[0]) as db:
                db.execute("UPDATE workers SET status=?, version=version+1, metadata=json_set(metadata, '$.finished_at', ?) "
                           "WHERE run_id=? AND status NOT IN ('completed','failed','cancelled','ended')",
                           (state, time.time(), ref[1]))
            record["status"] = state if record.get("status") not in TERMINAL else record["status"]
            _live.pop(tuple(ref), None)
    except Exception:
        logging.getLogger(__name__).exception("Worker recovery completion unavailable")
        # A lost terminal write cannot leave a positive active attestation.
        if record.get("_roster"):
            with _lock:
                _live.pop(tuple(record["_roster"]), None)


def observe(home, session_keys, live):
    if not session_keys:
        return []
    with _connect(home) as db:
        rows = db.execute("SELECT run_id,owner_id,session_key,subagent_id,status,metadata,version FROM workers WHERE session_key IN ("
                          + ",".join("?" for _ in session_keys) + ") ORDER BY run_id", tuple(session_keys)).fetchall()
    result = []
    for run_id, owner_id, key, sid, status, metadata, version in rows:
        if status not in TERMINAL:
            status = live.get(run_id, "unknown")
        fields = json.loads(metadata)
        fields.pop("retention_v1", None)
        fields.pop("finished_at", None)
        result.append(dict(run_id=run_id, owner_id=owner_id, subagent_id=sid,
                           status=status, version=version, **fields))
    return result


def local_observations(home, session_keys):
    home = str(Path(home).resolve())
    with _lock:
        return {run_id: STATUS.get(str(r.get("status") or ""), "unknown")
                for (profile, run_id), r in _live.items()
                if profile == home and r.get("root_session_id") in session_keys}
