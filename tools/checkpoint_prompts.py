"""Prompt lookups for checkpoints: read-only questions put to the session database.

A checkpoint records only opaque ids (``tools.checkpoint_manager``: the session id and the user
message's durable ``message_uid``).  These helpers turn them into something to show, or into the
set of prompts a "from this prompt on" question covers.  Nothing here writes, and each lookup
answers empty when the database is missing or unreadable, so a listing never fails because of it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 60
_CHUNK = 200  # bound parameters per query, well under SQLite's variable limit


def _connect(db_path: Optional[Path] = None) -> Optional[sqlite3.Connection]:
    """A ``mode=ro`` connection to the profile's state database, or ``None`` when there is none."""
    from hermes_state import _default_db_path
    from hermes_state_holders import read_only_db_uri

    path = Path(db_path) if db_path else _default_db_path()
    if not path.is_file():
        return None
    try:
        return sqlite3.connect(read_only_db_uri(path), uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        logger.debug("checkpoint prompt lookup: cannot open %s: %s", path, exc)
        return None


def preview_text(content: object) -> str:
    """One line of a stored user message, at most ``PREVIEW_CHARS`` long."""
    from agent.message_content import flatten_message_text

    if isinstance(content, str) and content.lstrip().startswith(("[", "{")):
        try:  # multimodal rows are stored as JSON
            content = json.loads(content)
        except ValueError:
            pass
    text = " ".join(flatten_message_text(content).split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1].rstrip() + "…"


def prompt_previews(
    links: Iterable[tuple[str, str]], db_path: Optional[Path] = None,
) -> dict[tuple[str, str], str]:
    """``{(session_id, prompt_uid): preview}`` for the links whose message is still in the database.

    A link whose session or message has since been deleted is simply absent from the answer."""
    by_session: dict[str, set[str]] = {}
    for session_id, uid in links:
        by_session.setdefault(session_id, set()).add(uid)
    conn = _connect(db_path) if by_session else None
    if conn is None:
        return {}
    previews: dict[tuple[str, str], str] = {}
    try:
        for session_id, uids in by_session.items():
            pending = sorted(uids)
            for start in range(0, len(pending), _CHUNK):
                chunk = pending[start:start + _CHUNK]
                marks = ",".join("?" * len(chunk))
                # Prefer the live copy of a message that compaction or a branch copied.
                rows = conn.execute(
                    f"SELECT message_uid, content FROM messages WHERE session_id = ? AND role = 'user' "
                    f"AND message_uid IN ({marks}) ORDER BY active ASC, id ASC",
                    (session_id, *chunk),
                ).fetchall()
                for uid, content in rows:
                    previews[(session_id, uid)] = preview_text(content)
    except sqlite3.Error as exc:
        logger.debug("checkpoint prompt previews unavailable: %s", exc)
        return {}
    finally:
        conn.close()
    return previews


def prompt_uids_from(session_id: str, row_id: int, db_path: Optional[Path] = None) -> list[str]:
    """``message_uid`` of the user message at ``row_id`` and of every later user message, oldest first.

    ``row_id`` is the stored row id the rewind picker already addresses.  It must be an active user
    row of ``session_id``; anything else (a stale id, another session's row, an assistant row) answers
    ``[]`` so the caller can refuse instead of guessing."""
    conn = _connect(db_path)
    if conn is None:
        return []
    try:
        target = conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND id = ? AND role = 'user' AND active = 1",
            (session_id, int(row_id)),
        ).fetchone()
        if target is None:
            return []
        rows = conn.execute(
            "SELECT message_uid FROM messages WHERE session_id = ? AND role = 'user' AND active = 1 "
            "AND id >= ? AND message_uid IS NOT NULL ORDER BY id ASC",
            (session_id, int(row_id)),
        ).fetchall()
    except (sqlite3.Error, ValueError, TypeError) as exc:
        logger.debug("checkpoint prompt order unavailable: %s", exc)
        return []
    finally:
        conn.close()
    return [uid for (uid,) in rows if uid]


def with_prompt_previews(checkpoints: list[dict], db_path: Optional[Path] = None) -> list[dict]:
    """``checkpoints`` with ``prompt_preview`` added to each linked entry whose prompt is still stored.

    The process that owns the session database calls this before it sends a checkpoint list, so a
    client that cannot open that database (a CLI attached to another gateway) still gets the text."""
    links = [(cp["session_id"], cp["prompt_uid"]) for cp in checkpoints if cp.get("prompt_uid")]
    previews = prompt_previews(links, db_path) if links else {}
    return [
        {**cp, "prompt_preview": previews[key]}
        if cp.get("prompt_uid") and (key := (cp["session_id"], cp["prompt_uid"])) in previews else cp
        for cp in checkpoints
    ]


def checkpoints_from_prompt(mgr, session_id: str, message_id: int, db_path: Optional[Path] = None) -> list[dict] | None:
    """The session's checkpoints taken for the user message at stored row ``message_id`` or after it, per
    working directory, newest first.

    The one function any picker asks "which checkpoints belong to this prompt" through (the
    ``rollback.for_prompt`` RPC is a thin wrapper), so a client has no second implementation to drift.
    ``None`` when ``message_id`` is not an active user message of ``session_id``: the caller refuses
    rather than guess. Each row: ``workdir``, ``hash``, ``short_hash``, ``timestamp``, ``message`` (the
    checkpoint's reason), ``prompt_uid`` and ``files_changed_since`` (files that differ from the working
    tree now, i.e. what restoring it would touch)."""
    if not (prompt_uids := prompt_uids_from(session_id, message_id, db_path)):
        return None
    return [
        {"workdir": cp["workdir"], "hash": cp["hash"], "short_hash": cp["short_hash"],
         "timestamp": cp["timestamp"], "message": cp["reason"], "prompt_uid": cp["prompt_uid"],
         "files_changed_since": cp["files_changed_since"]}
        for cp in mgr.checkpoints_for_prompts(session_id, prompt_uids)
    ]
