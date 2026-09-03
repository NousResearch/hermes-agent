"""Durable gateway clarify interaction storage.

This module stores pending gateway clarify prompts in profile-local SQLite so
Telegram button/text answers can be resolved after a gateway restart. It is
intentionally dependency-free and stores only the routing/question/answer data
needed to resume the user turn.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

_DB_FILENAME = "gateway_interactions.db"
_KIND_CLARIFY = "clarify"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS gateway_interactions (
  interaction_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  session_id TEXT,
  session_key TEXT NOT NULL,
  platform TEXT NOT NULL,
  chat_id TEXT,
  thread_id TEXT,
  user_id TEXT,
  message_id TEXT,
  question TEXT NOT NULL,
  choices_json TEXT,
  answer TEXT,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  resolved_at REAL,
  updated_at REAL NOT NULL,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_gateway_interactions_session
ON gateway_interactions(session_key, status, created_at);
CREATE INDEX IF NOT EXISTS idx_gateway_interactions_expiry
ON gateway_interactions(status, expires_at);
"""


@dataclass(frozen=True)
class ClarifyInteraction:
    interaction_id: str
    kind: str
    status: str
    session_key: str
    platform: str
    question: str
    choices: Optional[list[str]] = None
    answer: Optional[str] = None
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    message_id: Optional[str] = None
    created_at: float = 0.0
    expires_at: float = 0.0
    resolved_at: Optional[float] = None
    updated_at: float = 0.0
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ResolveResult:
    ok: bool
    status: str
    interaction_id: Optional[str] = None
    answer: Optional[str] = None
    interaction: Optional[ClarifyInteraction] = None


def _db_path() -> Path:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / _DB_FILENAME


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _row_to_interaction(row: sqlite3.Row | None) -> Optional[ClarifyInteraction]:
    if row is None:
        return None
    choices = _json_loads(row["choices_json"], None)
    if choices is not None:
        choices = [str(c) for c in choices]
    metadata = _json_loads(row["metadata_json"], None)
    return ClarifyInteraction(
        interaction_id=str(row["interaction_id"]),
        kind=str(row["kind"]),
        status=str(row["status"]),
        session_id=row["session_id"],
        session_key=str(row["session_key"]),
        platform=str(row["platform"]),
        chat_id=row["chat_id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        message_id=row["message_id"],
        question=str(row["question"]),
        choices=choices,
        answer=row["answer"],
        created_at=float(row["created_at"]),
        expires_at=float(row["expires_at"]),
        resolved_at=row["resolved_at"],
        updated_at=float(row["updated_at"]),
        metadata=metadata,
    )


def _same(expected: Optional[str], actual: Optional[str]) -> bool:
    return expected in (None, "") or str(expected) == str(actual or "")


def _authorized(interaction: ClarifyInteraction, *, user_id: Optional[str] = None,
                chat_id: Optional[str] = None, thread_id: Optional[str] = None) -> bool:
    return (
        _same(interaction.user_id, user_id)
        and _same(interaction.chat_id, chat_id)
        and _same(interaction.thread_id, thread_id)
    )


def create_clarify_interaction(
    *,
    session_key: str,
    platform: str,
    question: str,
    choices: Optional[Iterable[Any]] = None,
    session_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    message_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    ttl_seconds: float = 24 * 60 * 60,
    now: Optional[float] = None,
    interaction_id: Optional[str] = None,
) -> ClarifyInteraction:
    now_f = float(time.time() if now is None else now)
    iid = interaction_id or f"cld_{uuid.uuid4().hex[:24]}"
    normalized_choices = [str(c) for c in choices] if choices else None
    expires_at = now_f + float(ttl_seconds)
    status = "pending"
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO gateway_interactions (
              interaction_id, kind, status, session_id, session_key, platform,
              chat_id, thread_id, user_id, message_id, question, choices_json,
              answer, created_at, expires_at, resolved_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?)
            """,
            (
                iid, _KIND_CLARIFY, status, session_id, session_key, platform,
                chat_id, thread_id, user_id, message_id, question,
                _json_dumps(normalized_choices), now_f, expires_at, now_f,
                _json_dumps(metadata),
            ),
        )
    loaded = get_interaction(iid)
    assert loaded is not None
    return loaded


def get_interaction(interaction_id: str) -> Optional[ClarifyInteraction]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM gateway_interactions WHERE interaction_id = ?",
            (interaction_id,),
        ).fetchone()
    return _row_to_interaction(row)


def set_prompt_message_id(interaction_id: str, message_id: Optional[str]) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE gateway_interactions SET message_id = ?, updated_at = ? WHERE interaction_id = ?",
            (message_id, time.time(), interaction_id),
        )
        return cur.rowcount > 0


def _expire_if_needed(conn: sqlite3.Connection, interaction: ClarifyInteraction, now_f: float) -> Optional[ResolveResult]:
    if interaction.expires_at <= now_f and interaction.status in {"pending", "awaiting_text"}:
        conn.execute(
            "UPDATE gateway_interactions SET status = 'expired', updated_at = ? WHERE interaction_id = ?",
            (now_f, interaction.interaction_id),
        )
        return ResolveResult(False, "expired", interaction.interaction_id, interaction=interaction)
    return None


def mark_awaiting_text(
    interaction_id: str,
    *,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    now: Optional[float] = None,
) -> ResolveResult:
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM gateway_interactions WHERE interaction_id = ?", (interaction_id,)).fetchone()
        interaction = _row_to_interaction(row)
        if interaction is None:
            return ResolveResult(False, "missing", interaction_id)
        expired = _expire_if_needed(conn, interaction, now_f)
        if expired:
            return expired
        if interaction.status == "resolved":
            return ResolveResult(False, "already_resolved", interaction_id, interaction=interaction)
        if interaction.status not in {"pending", "awaiting_text"}:
            return ResolveResult(False, interaction.status, interaction_id, interaction=interaction)
        if not _authorized(interaction, user_id=user_id, chat_id=chat_id, thread_id=thread_id):
            return ResolveResult(False, "unauthorized", interaction_id, interaction=interaction)
        conn.execute(
            "UPDATE gateway_interactions SET status = 'awaiting_text', updated_at = ? WHERE interaction_id = ?",
            (now_f, interaction_id),
        )
        updated_row = conn.execute(
            "SELECT * FROM gateway_interactions WHERE interaction_id = ?",
            (interaction_id,),
        ).fetchone()
        updated = _row_to_interaction(updated_row)
        return ResolveResult(True, "awaiting_text", interaction_id, interaction=updated)


def resolve_choice(
    interaction_id: str,
    choice_index: int,
    *,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    now: Optional[float] = None,
) -> ResolveResult:
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM gateway_interactions WHERE interaction_id = ?", (interaction_id,)).fetchone()
        interaction = _row_to_interaction(row)
        if interaction is None:
            return ResolveResult(False, "missing", interaction_id)
        expired = _expire_if_needed(conn, interaction, now_f)
        if expired:
            return expired
        if interaction.status == "resolved":
            return ResolveResult(False, "already_resolved", interaction_id, interaction=interaction)
        if interaction.status not in {"pending", "awaiting_text"}:
            return ResolveResult(False, interaction.status, interaction_id, interaction=interaction)
        if not _authorized(interaction, user_id=user_id, chat_id=chat_id, thread_id=thread_id):
            return ResolveResult(False, "unauthorized", interaction_id, interaction=interaction)
        choices = interaction.choices or []
        if choice_index < 0 or choice_index >= len(choices):
            return ResolveResult(False, "invalid_choice", interaction_id, interaction=interaction)
        answer = choices[choice_index]
        cur = conn.execute(
            """
            UPDATE gateway_interactions
            SET status = 'resolved', answer = ?, resolved_at = ?, updated_at = ?
            WHERE interaction_id = ? AND status IN ('pending', 'awaiting_text') AND expires_at > ?
            """,
            (answer, now_f, now_f, interaction_id, now_f),
        )
        if cur.rowcount != 1:
            return ResolveResult(False, "already_resolved", interaction_id, interaction=interaction)
        updated_row = conn.execute("SELECT * FROM gateway_interactions WHERE interaction_id = ?", (interaction_id,)).fetchone()
        updated = _row_to_interaction(updated_row)
        return ResolveResult(True, "resolved", interaction_id, answer=answer, interaction=updated)


def resolve_text_for_session(
    session_key: str,
    text: str,
    *,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[ClarifyInteraction]:
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT * FROM gateway_interactions
            WHERE session_key = ? AND kind = ? AND status IN ('awaiting_text', 'pending')
            ORDER BY CASE status WHEN 'awaiting_text' THEN 0 ELSE 1 END, created_at ASC
            """,
            (session_key, _KIND_CLARIFY),
        ).fetchall()
        for row in rows:
            interaction = _row_to_interaction(row)
            if interaction is None:
                continue
            expired = _expire_if_needed(conn, interaction, now_f)
            if expired:
                continue
            if not _authorized(interaction, user_id=user_id, chat_id=chat_id, thread_id=thread_id):
                continue
            cur = conn.execute(
                """
                UPDATE gateway_interactions
                SET status = 'resolved', answer = ?, resolved_at = ?, updated_at = ?
                WHERE interaction_id = ? AND status IN ('pending', 'awaiting_text') AND expires_at > ?
                """,
                (text, now_f, now_f, interaction.interaction_id, now_f),
            )
            if cur.rowcount == 1:
                updated_row = conn.execute(
                    "SELECT * FROM gateway_interactions WHERE interaction_id = ?",
                    (interaction.interaction_id,),
                ).fetchone()
                return _row_to_interaction(updated_row)
    return None


def expire_old(now: Optional[float] = None) -> int:
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE gateway_interactions SET status = 'expired', updated_at = ? WHERE status IN ('pending', 'awaiting_text') AND expires_at <= ?",
            (now_f, now_f),
        )
        return int(cur.rowcount or 0)


def cancel_interaction(interaction_id: str, *, reason: str = "cancelled", now: Optional[float] = None) -> bool:
    del reason  # reserved for metadata in a future migration
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE gateway_interactions SET status = 'cancelled', updated_at = ? WHERE interaction_id = ? AND status IN ('pending', 'awaiting_text')",
            (now_f, interaction_id),
        )
        return bool(cur.rowcount == 1)


def cancel_session(session_key: str, *, reason: str = "cancelled", now: Optional[float] = None) -> int:
    del reason  # reserved for metadata in a future migration
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE gateway_interactions SET status = 'cancelled', updated_at = ? WHERE session_key = ? AND status IN ('pending', 'awaiting_text')",
            (now_f, session_key),
        )
        return int(cur.rowcount or 0)


def cancel_all(*, reason: str = "cancelled", now: Optional[float] = None) -> int:
    """Cancel every active durable interaction in the current profile."""
    del reason  # reserved for metadata in a future migration
    now_f = float(time.time() if now is None else now)
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE gateway_interactions SET status = 'cancelled', updated_at = ? WHERE status IN ('pending', 'awaiting_text')",
            (now_f,),
        )
        return int(cur.rowcount or 0)


__all__ = [
    "ClarifyInteraction",
    "ResolveResult",
    "create_clarify_interaction",
    "get_interaction",
    "set_prompt_message_id",
    "mark_awaiting_text",
    "resolve_choice",
    "resolve_text_for_session",
    "expire_old",
    "cancel_interaction",
    "cancel_session",
    "cancel_all",
]
