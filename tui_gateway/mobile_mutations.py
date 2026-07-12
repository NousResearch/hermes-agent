"""Durable at-most-once receipts for consequential mobile mutations.

The store is deliberately independent of any live TUI session.  A receipt is
scoped to the authenticated principal plus a client-generated request identity,
while its fingerprint binds the durable Hermes resource and semantic request
parameters.  An unfinished receipt owned by another process is never released
for automatic execution: it becomes ``outcome_unknown`` and stays queryable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_MAX_REQUEST_ID_CHARS = 256


class MutationConflict(ValueError):
    """One client request identity was reused for different semantics."""


class MutationDisposition(str, Enum):
    EXECUTE = "execute"
    IN_PROGRESS = "in_progress"
    REPLAY = "replay"
    OUTCOME_UNKNOWN = "outcome_unknown"


@dataclass(frozen=True)
class MutationClaim:
    disposition: MutationDisposition
    outcome: dict[str, Any] | None = None
    _principal_digest: str = ""
    _request_digest: str = ""
    _fingerprint: str = ""
    _owner_instance_id: str = ""

    @property
    def proof_tag(self) -> str:
        """Opaque identity for binding an in-memory effect to this receipt."""
        if not (
            self._principal_digest
            and self._request_digest
            and self._fingerprint
        ):
            return ""
        return _digest(
            [self._principal_digest, self._request_digest, self._fingerprint]
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class MobileMutationStore:
    """SQLite-backed mutation receipts shared across reconnects and restarts."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        owner_instance_id: str | None = None,
    ) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        elif os.name != "nt":
            os.chmod(self._path, 0o600)

        self._owner_instance_id = owner_instance_id or str(uuid.uuid4())
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._closed = False
        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        self._conn.execute("PRAGMA busy_timeout = 5000")
        # Import lazily so loading the gateway does not freeze SessionDB's
        # profile path before reload-style tests and embedded callers install
        # their Hermes-home override. The receipt store itself is lazy too.
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(
            self._conn,
            db_label=f"mobile mutations ({self._path.name})",
        )
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mobile_mutations (
                principal_digest TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                method TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                state TEXT NOT NULL,
                owner_instance_id TEXT,
                outcome_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (principal_digest, request_digest)
            )
            """
        )
        # A process that disappeared after reservation may already have caused
        # side effects.  Preserve that uncertainty durably instead of making a
        # new process guess that the request is safe to execute again.
        now = time.time()
        self._conn.execute(
            """
            UPDATE mobile_mutations
            SET state = 'outcome_unknown', owner_instance_id = NULL, updated_at = ?
            WHERE state = 'in_progress' AND owner_instance_id <> ?
            """,
            (now, self._owner_instance_id),
        )

    @staticmethod
    def _key(
        *,
        provider: str,
        subject: str,
        client_request_id: str,
    ) -> tuple[str, str, str]:
        request_id = str(client_request_id or "").strip()
        if not request_id or len(request_id) > _MAX_REQUEST_ID_CHARS:
            raise ValueError(
                "client_request_id must contain 1 to 256 non-whitespace characters"
            )
        principal = [str(provider or ""), str(subject or "")]
        return _digest(principal), _digest(request_id), request_id

    @staticmethod
    def _fingerprint(
        *,
        method: str,
        resource_id: str,
        semantic_parameters: dict[str, Any],
    ) -> str:
        return _digest(
            {
                "method": str(method),
                "resource_id": str(resource_id),
                "semantic_parameters": semantic_parameters,
            }
        )

    def reserve(
        self,
        *,
        provider: str,
        subject: str,
        client_request_id: str,
        method: str,
        resource_id: str,
        semantic_parameters: dict[str, Any],
    ) -> MutationClaim:
        principal_digest, request_digest, _ = self._key(
            provider=provider,
            subject=subject,
            client_request_id=client_request_id,
        )
        fingerprint = self._fingerprint(
            method=method,
            resource_id=resource_id,
            semantic_parameters=semantic_parameters,
        )
        now = time.time()

        with self._changed:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """
                    SELECT method, fingerprint, state, owner_instance_id, outcome_json
                    FROM mobile_mutations
                    WHERE principal_digest = ? AND request_digest = ?
                    """,
                    (principal_digest, request_digest),
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        """
                        INSERT INTO mobile_mutations (
                            principal_digest, request_digest, method, fingerprint,
                            state, owner_instance_id, outcome_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'in_progress', ?, NULL, ?, ?)
                        """,
                        (
                            principal_digest,
                            request_digest,
                            str(method),
                            fingerprint,
                            self._owner_instance_id,
                            now,
                            now,
                        ),
                    )
                    disposition = MutationDisposition.EXECUTE
                    outcome = None
                else:
                    _, existing_fingerprint, state, owner, outcome_json = row
                    if existing_fingerprint != fingerprint:
                        raise MutationConflict(
                            "client_request_id was already used for different semantics"
                        )
                    if state == "completed":
                        disposition = MutationDisposition.REPLAY
                        outcome = json.loads(outcome_json) if outcome_json else None
                    elif state == "outcome_unknown":
                        disposition = MutationDisposition.OUTCOME_UNKNOWN
                        outcome = None
                    elif owner == self._owner_instance_id:
                        disposition = MutationDisposition.IN_PROGRESS
                        outcome = None
                    else:
                        self._conn.execute(
                            """
                            UPDATE mobile_mutations
                            SET state = 'outcome_unknown', owner_instance_id = NULL,
                                updated_at = ?
                            WHERE principal_digest = ? AND request_digest = ?
                              AND state = 'in_progress' AND owner_instance_id = ?
                            """,
                            (now, principal_digest, request_digest, owner),
                        )
                        disposition = MutationDisposition.OUTCOME_UNKNOWN
                        outcome = None
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

            return MutationClaim(
                disposition=disposition,
                outcome=outcome,
                _principal_digest=principal_digest,
                _request_digest=request_digest,
                _fingerprint=fingerprint,
                _owner_instance_id=self._owner_instance_id,
            )

    def complete(self, claim: MutationClaim, outcome: dict[str, Any]) -> bool:
        if claim.disposition is not MutationDisposition.EXECUTE:
            return False
        encoded = _canonical_json(outcome)
        with self._changed:
            cursor = self._conn.execute(
                """
                UPDATE mobile_mutations
                SET state = 'completed', outcome_json = ?, updated_at = ?
                WHERE principal_digest = ? AND request_digest = ?
                  AND fingerprint = ? AND state = 'in_progress'
                  AND owner_instance_id = ?
                """,
                (
                    encoded,
                    time.time(),
                    claim._principal_digest,
                    claim._request_digest,
                    claim._fingerprint,
                    claim._owner_instance_id,
                ),
            )
            completed = cursor.rowcount == 1
            self._changed.notify_all()
            return completed

    def release_before_execution(self, claim: MutationClaim) -> bool:
        """Remove this process's reservation before any side effect can run."""
        if claim.disposition is not MutationDisposition.EXECUTE:
            return False
        with self._changed:
            cursor = self._conn.execute(
                """
                DELETE FROM mobile_mutations
                WHERE principal_digest = ? AND request_digest = ?
                  AND fingerprint = ? AND state = 'in_progress'
                  AND owner_instance_id = ?
                """,
                (
                    claim._principal_digest,
                    claim._request_digest,
                    claim._fingerprint,
                    claim._owner_instance_id,
                ),
            )
            released = cursor.rowcount == 1
            self._changed.notify_all()
            return released

    def mark_outcome_unknown(self, claim: MutationClaim) -> bool:
        """Terminalize a reserved request after an unexpected execution failure."""
        with self._changed:
            cursor = self._conn.execute(
                """
                UPDATE mobile_mutations
                SET state = 'outcome_unknown', owner_instance_id = NULL,
                    outcome_json = NULL, updated_at = ?
                WHERE principal_digest = ? AND request_digest = ?
                  AND fingerprint = ? AND state = 'in_progress'
                  AND owner_instance_id = ?
                """,
                (
                    time.time(),
                    claim._principal_digest,
                    claim._request_digest,
                    claim._fingerprint,
                    claim._owner_instance_id,
                ),
            )
            changed = cursor.rowcount == 1
            self._changed.notify_all()
            return changed

    def wait_for_outcome(
        self,
        claim: MutationClaim,
        *,
        timeout: float,
    ) -> MutationClaim:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._changed:
            while True:
                row = self._conn.execute(
                    """
                    SELECT state, outcome_json
                    FROM mobile_mutations
                    WHERE principal_digest = ? AND request_digest = ?
                      AND fingerprint = ?
                    """,
                    (
                        claim._principal_digest,
                        claim._request_digest,
                        claim._fingerprint,
                    ),
                ).fetchone()
                if row is None:
                    return claim
                state, outcome_json = row
                if state == "completed":
                    return MutationClaim(
                        MutationDisposition.REPLAY,
                        json.loads(outcome_json) if outcome_json else None,
                        claim._principal_digest,
                        claim._request_digest,
                        claim._fingerprint,
                        claim._owner_instance_id,
                    )
                if state == "outcome_unknown":
                    return MutationClaim(
                        MutationDisposition.OUTCOME_UNKNOWN,
                        None,
                        claim._principal_digest,
                        claim._request_digest,
                        claim._fingerprint,
                        claim._owner_instance_id,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return claim
                self._changed.wait(remaining)

    def status(
        self,
        *,
        provider: str,
        subject: str,
        client_request_id: str,
    ) -> dict[str, Any] | None:
        principal_digest, request_digest, request_id = self._key(
            provider=provider,
            subject=subject,
            client_request_id=client_request_id,
        )
        with self._lock:
            row = self._conn.execute(
                """
                SELECT method, state, outcome_json
                FROM mobile_mutations
                WHERE principal_digest = ? AND request_digest = ?
                """,
                (principal_digest, request_digest),
            ).fetchone()
        if row is None:
            return None
        method, state, outcome_json = row
        return {
            "client_request_id": request_id,
            "method": method,
            "outcome": json.loads(outcome_json) if outcome_json else None,
            "state": state,
        }

    def close(self) -> None:
        with self._changed:
            if self._closed:
                return
            self._closed = True
            self._conn.close()
            self._changed.notify_all()
