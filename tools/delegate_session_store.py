"""Durable child-conversation ownership and admission in the profile SessionDB.

This is conversation metadata, not an execution checkpoint. A dead process leaves
an idle, resumable conversation; it never causes an assignment to run again.
Tool definitions, credentials and live agent objects are deliberately not stored.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid

from hermes_state_common import (
    _lock_holder_provably_dead, _non_continuation_child_sql, _proc_start_ticks, _sql_session_last_active,
)
from hermes_state_compression import _CHAIN_CAP
from hermes_state_pidns import pid_namespace_id
from tools.delegate_observation import message_metadata, preview_text


class DelegateSessionError(ValueError):
    """A model-actionable refusal, with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS delegate_sessions (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    owner_session_id TEXT NOT NULL,
    current_session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    initial_goal TEXT NOT NULL,
    assignment TEXT NOT NULL,
    context TEXT NOT NULL,
    workspace TEXT,
    forked_context INTEGER NOT NULL DEFAULT 0,
    route_descriptor TEXT,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    role TEXT NOT NULL,
    depth INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'idle', 'closed')),
    created_at REAL NOT NULL,
    last_activity_at REAL NOT NULL,
    run_token TEXT,
    runner TEXT
)
"""

# A compression continuation inherits _delegate_from from its original segment;
# only a marker naming the immediate parent denotes a new delegated conversation.
_NEXT_SEGMENT_SQL = (
    "SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id = parent.id "
    "WHERE parent.id = ? AND parent.end_reason = 'compression' "
    + _non_continuation_child_sql("child.", "parent.id")
    + " ORDER BY CASE WHEN child.end_reason = 'compression' THEN 0 "
    "WHEN child.ended_at IS NULL THEN 1 ELSE 2 END, "
    + _sql_session_last_active("child") + " DESC, child.started_at DESC, child.id DESC LIMIT 1"
)


def _public(row):
    result = {key: value for key, value in dict(row).items() if key not in {"run_token", "runner"}}
    result["forked_context"] = bool(result["forked_context"])
    result["route_descriptor"] = json.loads(result["route_descriptor"]) if result["route_descriptor"] else None
    return result


def _route_json(descriptor):
    """Persist routing identity, never an endpoint/credential connection bundle."""
    if descriptor is None:
        return None
    allowed = {"kind", "difficulty", "endpoint_sha256", "provider"}
    if not isinstance(descriptor, dict) or set(descriptor) - allowed:
        raise DelegateSessionError("invalid_route", "Unsupported retained route fields.")
    if descriptor.get("kind") not in {"named", "default", "difficulty", "parent"}:
        raise DelegateSessionError("invalid_route", "Unknown retained route kind.")
    digest = descriptor.get("endpoint_sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
        raise DelegateSessionError("invalid_route", "The retained endpoint identity must be a SHA-256 digest.")
    return json.dumps(descriptor, sort_keys=True)


def _text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
    return "" if content is None else str(content)


class DelegateSessionStore:
    """Small transactional registry sharing the parent's exact profile database.

    Admission uses SessionDB's BEGIN IMMEDIATE writer, so independent processes
    cannot both resume one child. Tokens fence late cleanup from a previous run.
    The existing turn lease still guards the actual transcript writer.
    """

    def __init__(self, db):
        self.db = db

        def create(conn):
            conn.execute(_SCHEMA)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(delegate_sessions)")}
            for name, declaration in (("workspace", "TEXT"), ("forked_context", "INTEGER NOT NULL DEFAULT 0"),
                                      ("route_descriptor", "TEXT")):
                if name not in columns:
                    conn.execute(f"ALTER TABLE delegate_sessions ADD COLUMN {name} {declaration}")
            conn.execute("CREATE INDEX IF NOT EXISTS delegate_sessions_owner_activity "
                         "ON delegate_sessions(owner_session_id, last_activity_at DESC)")

        db._execute_write(create)

    def _owner(self, conn, session_id):
        if not conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
            raise DelegateSessionError("unavailable", "The parent conversation is unavailable.")
        return self.db._session_turn_lease_key_on_conn(conn, session_id)

    def _chain(self, conn, session_id):
        chain = [session_id]
        seen = {session_id}
        for _ in range(_CHAIN_CAP):
            row = conn.execute(_NEXT_SEGMENT_SQL, (chain[-1],)).fetchone()
            if row is None or row["id"] in seen:
                return chain
            chain.append(row["id"])
            seen.add(row["id"])
        raise DelegateSessionError("unavailable", "The child compression lineage exceeds the safe limit.")

    def _refresh(self, conn, row, *, persist=True):
        """Reconcile process death/compression; observation never renews activity."""
        row = dict(row)
        if not conn.execute("SELECT 1 FROM sessions WHERE id = ?", (row["session_id"],)).fetchone():
            return None
        if not conn.execute("SELECT 1 FROM sessions WHERE id = ?", (row["current_session_id"],)).fetchone():
            return None
        chain = self._chain(conn, row["session_id"])
        tip = conn.execute("SELECT end_reason FROM sessions WHERE id = ?", (chain[-1],)).fetchone()
        if tip is None or tip["end_reason"] == "compression":
            # A missing continuation must not fall back to an older transcript, even
            # if no earlier observer persisted the now-deleted tip in this registry.
            return None
        row["current_session_id"] = chain[-1]
        if row["status"] == "running" and row["runner"]:
            if _lock_holder_provably_dead(json.loads(row["runner"])):
                row.update(status="idle", run_token=None, runner=None)
        # Transcript output (including direct user messages) counts as activity;
        # listing and reading do not. MAX also tolerates regressing wall clocks.
        for sid in chain:
            stamp = conn.execute("SELECT MAX(timestamp) FROM messages WHERE session_id = ?", (sid,)).fetchone()[0]
            if stamp is not None:
                row["last_activity_at"] = max(row["last_activity_at"], float(stamp))
        if persist:
            conn.execute("UPDATE delegate_sessions SET current_session_id = ?, status = ?, "
                         "run_token = ?, runner = ?, last_activity_at = ? WHERE session_id = ?",
                         (row["current_session_id"], row["status"], row["run_token"], row["runner"],
                          row["last_activity_at"], row["session_id"]))
        return row

    def _owned(self, conn, parent_session_id, session_id, *, persist=True):
        owner = self._owner(conn, parent_session_id)
        row = conn.execute("SELECT * FROM delegate_sessions WHERE session_id = ? AND owner_session_id = ?",
                           (session_id, owner)).fetchone()
        row = self._refresh(conn, row, persist=persist) if row else None
        if row is None:
            # Do not disclose whether this ID belongs to a different parent.
            raise DelegateSessionError("unavailable", "No retained child with that session_id belongs to this conversation.")
        return row

    def register(self, parent_session_id, session_id, *, model, provider, assignment,
                 context="", title=None, role="leaf", depth=1, workspace=None, forked_context=False,
                 route_descriptor=None):
        """Register a newly persisted child. Existing records are never overwritten."""
        now = time.time()
        title = str(title or " ".join(assignment.split())[:80]).strip() or "Delegated task"
        route_json = _route_json(route_descriptor)

        def create(conn):
            owner = self._owner(conn, parent_session_id)
            child = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if child is None:
                raise DelegateSessionError("unavailable", "Persist the child session before registering it.")
            if conn.execute("SELECT 1 FROM delegate_sessions WHERE session_id = ?", (session_id,)).fetchone():
                raise DelegateSessionError("already_registered", "This child session is already registered.")
            conn.execute("INSERT INTO delegate_sessions (session_id, owner_session_id, current_session_id, "
                         "title, initial_goal, assignment, context, workspace, forked_context, route_descriptor, "
                         "model, provider, role, depth, status, created_at, last_activity_at) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'idle', ?, ?)",
                         (session_id, owner, session_id, title, assignment, assignment, context, workspace,
                          int(bool(forked_context)), route_json, model, provider,
                          role, depth, now, now))
            return _public(conn.execute("SELECT * FROM delegate_sessions WHERE session_id = ?", (session_id,)).fetchone())

        return self.db._execute_write(create)

    def get(self, parent_session_id, session_id):
        with self.db._read_ctx() as conn:
            return _public(self._owned(conn, parent_session_id, session_id, persist=False))

    def list(self, parent_session_id, *, query=None, limit=20):
        """Recent running/idle children, or a search of all retained own children."""
        limit = max(1, min(int(limit), 100))
        needle = str(query or "").strip().casefold()

        def collect(conn):
            owner = self._owner(conn, parent_session_id)
            records = []
            rows = conn.execute("SELECT * FROM delegate_sessions WHERE owner_session_id = ?", (owner,)).fetchall()
            for original in rows:
                row = self._refresh(conn, original, persist=False)
                if row is None or (not needle and row["status"] == "closed"):
                    continue
                if needle and not self._matches(conn, row, needle):
                    continue
                records.append(_public(row))
            records.sort(key=lambda row: (row["last_activity_at"], row["created_at"], row["session_id"]), reverse=True)
            return records[:limit]

        with self.db._read_ctx() as conn:
            return collect(conn)

    def _matches(self, conn, row, needle):
        if any(needle in str(row[key]).casefold() for key in ("session_id", "title", "assignment")):
            return True
        # SQL scans only owned child transcripts. No FTS dependency or global
        # session-search bypass, and no transcript bodies materialized in Python.
        for sid in self._chain(conn, row["session_id"]):
            if conn.execute("SELECT 1 FROM messages WHERE session_id = ? AND instr(lower(content), ?) > 0 LIMIT 1",
                            (sid, needle)).fetchone():
                return True
        return False

    def acquire(self, parent_session_id, session_id, message, *, model=None):
        if not isinstance(message, str) or not message.strip():
            raise DelegateSessionError("invalid_message", "A nonempty assignment message is required.")
        token = uuid.uuid4().hex
        runner = json.dumps({"pid": os.getpid(), "pidns": pid_namespace_id(),
                             "start_ticks": _proc_start_ticks(os.getpid())})

        def claim(conn):
            row = self._owned(conn, parent_session_id, session_id)
            if row["status"] == "closed":
                raise DelegateSessionError("closed", "This child is closed; start a new delegation.")
            if row["status"] == "running":
                raise DelegateSessionError("busy", "This child is already running; use steer or observe.")
            row.update(status="running", assignment=message, model=model or row["model"],
                       last_activity_at=max(time.time(), row["last_activity_at"]))
            conn.execute("UPDATE delegate_sessions SET status = 'running', assignment = ?, model = ?, "
                         "last_activity_at = ?, run_token = ?, runner = ? WHERE session_id = ?",
                         (row["assignment"], row["model"], row["last_activity_at"], token, runner, session_id))
            return _public(row), token

        return self.db._execute_write(claim)

    def touch(self, session_id, token):
        return bool(self.db._write_rowcount("UPDATE delegate_sessions SET last_activity_at = MAX(last_activity_at, ?) "
                    "WHERE session_id = ? AND run_token = ? AND status = 'running'", (time.time(), session_id, token)))

    def release(self, session_id, token, *, current_session_id=None):
        def settle(conn):
            row = conn.execute("SELECT * FROM delegate_sessions WHERE session_id = ? AND run_token = ? "
                               "AND status = 'running'", (session_id, token)).fetchone()
            if row is None:
                return False
            chain = self._chain(conn, session_id)
            if current_session_id is not None and current_session_id not in chain:
                raise DelegateSessionError("invalid_tip", "The supplied child tip is outside its compression lineage.")
            conn.execute("UPDATE delegate_sessions SET status = 'idle', current_session_id = ?, "
                         "last_activity_at = MAX(last_activity_at, ?), run_token = NULL, runner = NULL "
                         "WHERE session_id = ? AND run_token = ?", (chain[-1], time.time(), session_id, token))
            return True

        return self.db._execute_write(settle)

    def retire_abandoned(self, session_id, token):
        """Permanently fence reuse while a timed-out worker may still write its transcript."""
        return bool(self.db._write_rowcount(
            "UPDATE delegate_sessions SET status = 'closed', run_token = NULL, runner = NULL "
            "WHERE session_id = ? AND run_token = ? AND status = 'running'", (session_id, token)))

    def close(self, parent_session_id, session_id):
        def retire(conn):
            row = self._owned(conn, parent_session_id, session_id)
            if row["status"] == "running":
                raise DelegateSessionError("busy", "Stop the running child before closing it.")
            conn.execute("UPDATE delegate_sessions SET status = 'closed' WHERE session_id = ?", (session_id,))
            row["status"] = "closed"
            return _public(row)

        return self.db._execute_write(retire)

    def observe(self, parent_session_id, session_id, *, limit=20, max_chars=12000):
        row = self.get(parent_session_id, session_id)
        limit = max(1, min(int(limit), 100))
        budget = max(1, min(int(max_chars), 50000))
        messages = self.db.get_messages(row["current_session_id"], include_ancestors=True,
                                       limit=limit + 1, latest=True)
        selected = []
        truncated = len(messages) > limit
        for message in reversed(messages[-limit:]):
            from tools.delegation_redaction import redact_child_text
            content = redact_child_text(_text_content(message.get("content")), force=False)
            metadata, metadata_truncated = message_metadata(message)
            metadata_size = len(json.dumps(metadata, ensure_ascii=False)) if metadata else 0
            if metadata_size > budget:
                metadata, metadata_size, metadata_truncated = {}, 0, True
            budget -= metadata_size
            cap = min(budget, 1500) if message["role"] == "tool" else budget
            truncated = truncated or metadata_truncated or len(content) > cap
            content = preview_text(content, cap)
            selected.append({"role": message["role"], "content": content,
                             "timestamp": message.get("timestamp"), **metadata})
            budget -= len(content)
            if budget <= 0:
                truncated = truncated or len(selected) < len(messages)
                break
        return {"session": row, "messages": list(reversed(selected)), "truncated": truncated}
