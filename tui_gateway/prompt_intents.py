"""Bounded at-most-once claims for client-originated prompt intents."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

_MIN_TTL_S = 6 * 3600
_MAX_REQUEST_ID_CHARS = 256


class PromptIntentClaim(str, Enum):
    ACCEPTED = "accepted"
    CONFLICT = "conflict"
    DUPLICATE = "duplicate"
    INVALID = "invalid"


class PromptIntentLedger:
    """Collapse prompt retries across process and runtime-session replacement.

    Keys and payloads are stored only as fixed-size digests. Accepted claims
    remain protected for the full TTL in SQLite. Expired claims are pruned in
    the same transaction that admits a new claim, so retention is time-bounded
    without a process-wide entry ceiling that can reject unrelated prompts.
    """

    def __init__(
        self,
        *,
        db_path: str | os.PathLike[str] = ":memory:",
        session_ttl_s: float = 0,
    ) -> None:
        self._lock = threading.Lock()
        self._ttl_s = max(float(_MIN_TTL_S), float(session_ttl_s))
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            try:
                os.chmod(self._db_path, 0o600)
            except OSError:
                pass
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_intents (
                profile_scope TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                accepted_at REAL NOT NULL,
                fingerprint TEXT NOT NULL,
                PRIMARY KEY (profile_scope, request_digest)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS prompt_intents_accepted_at_idx "
            "ON prompt_intents (accepted_at)"
        )

    def claim(
        self,
        *,
        profile_scope: str,
        request_id: str,
        route_identity: Any,
        text: Any,
        truncate_ordinal: Any,
    ) -> PromptIntentClaim:
        request_id = str(request_id or "").strip()
        if not request_id:
            return PromptIntentClaim.ACCEPTED
        if len(request_id) > _MAX_REQUEST_ID_CHARS:
            return PromptIntentClaim.INVALID

        request_digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        key = (str(profile_scope), request_digest)
        fingerprint_payload = json.dumps(
            [route_identity, text, truncate_ordinal],
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        now = time.time()
        expired_before = now - self._ttl_s

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "DELETE FROM prompt_intents WHERE accepted_at < ?",
                    (expired_before,),
                )
                existing = self._conn.execute(
                    "SELECT fingerprint FROM prompt_intents "
                    "WHERE profile_scope = ? AND request_digest = ?",
                    key,
                ).fetchone()
                if existing is not None:
                    claim = (
                        PromptIntentClaim.DUPLICATE
                        if existing[0] == fingerprint
                        else PromptIntentClaim.CONFLICT
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO prompt_intents "
                        "(profile_scope, request_digest, accepted_at, fingerprint) "
                        "VALUES (?, ?, ?, ?)",
                        (*key, now, fingerprint),
                    )
                    claim = PromptIntentClaim.ACCEPTED
                self._conn.execute("COMMIT")
                return claim
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM prompt_intents")

    def abort(self, *, profile_scope: str, request_id: str) -> None:
        """Release a reservation that failed before execution could begin."""
        request_id = str(request_id or "").strip()
        if not request_id:
            return

        request_digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        with self._lock:
            self._conn.execute(
                "DELETE FROM prompt_intents "
                "WHERE profile_scope = ? AND request_digest = ?",
                (str(profile_scope), request_digest),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM prompt_intents").fetchone()
            return int(row[0] if row else 0)
