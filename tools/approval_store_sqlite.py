"""SQLite-backed :class:`ApprovalStore` for gateway command approvals.

Stores proposals in a dedicated ``gateway_approvals`` table inside the
existing Hermes ``state.db`` (default ``~/.hermes/state.db``). The table
is self-contained: it is created idempotently via
``CREATE TABLE IF NOT EXISTS`` on every store instantiation, with no
coupling to :data:`hermes_state.SCHEMA_VERSION`. That keeps approval-
storage migrations decoupled from session/message-schema bumps.

Atomicity model
---------------

Every state transition uses an **explicit short transaction** with
``BEGIN IMMEDIATE`` so the writer lock is taken before any read happens.
That, combined with ``UPDATE ... WHERE status='pending' ... RETURNING``,
gives us a true compare-and-set with no TOCTOU window.

Consume / deny semantics:

  - Returns ``None`` if there is no matching row (missing / wrong status /
    expired). The UPDATE simply matches zero rows; the transaction
    commits a no-op. No exception is raised — fail closed, not fail loud.
  - Two concurrent consumers (same process, different threads, or
    different processes entirely) serialise on the SQLite file lock.
    Exactly one's UPDATE matches a pending row; the other matches zero.

Payload model
-------------

The :class:`ApprovalProposal` is serialised verbatim to ``payload_json``
**at submit time** and is never rewritten. Lifecycle columns
(``status``, ``consumed_at``, ``consumed_by``) hold the current FSM
state. On read we deserialise the original payload and overlay the
current lifecycle state — that preserves the pinned-policy invariant
because the payload itself is immutable post-submit.

Connection lifecycle
--------------------

A new ``sqlite3.Connection`` is opened per operation. SQLite handles
file-level locking; WAL allows concurrent readers with a single writer.
For the gateway's approval rate (rare events) the per-operation
connection cost is dominated by the work itself.

NFS / SMB caveat
----------------

SQLite over NFS/SMB has documented locking quirks. ``apply_wal_with_fallback``
(from :mod:`hermes_state`) is reused so the same fallback to journal_mode=
DELETE applies if WAL is unsupported.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from tools.approval_store import (
    ApprovalProposal,
    ApprovalStore,
    ApprovalStoreError,
)


# Split into table-DDL and index-DDL so we can run schema-column
# migration BETWEEN them. The partial index on execution_status would
# otherwise raise "no such column: execution_status" on an upgrade from
# a pre-execution_* DB, because CREATE INDEX is evaluated against the
# current table shape — and CREATE TABLE IF NOT EXISTS is a no-op on
# existing tables. _ensure_schema's ordering:
#     1. CREATE TABLE IF NOT EXISTS  (new DBs get full shape)
#     2. _migrate_existing_schema    (old DBs get missing columns)
#     3. CREATE INDEX IF NOT EXISTS  (all indices are now valid)
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS gateway_approvals (
    approval_id            TEXT PRIMARY KEY,
    created_at             REAL NOT NULL,
    expires_at             REAL,
    status                 TEXT NOT NULL DEFAULT 'pending',
    consumed_at            REAL,
    consumed_by            TEXT,
    -- Post-consume execution outcome (audit-distinct from consume).
    -- 'consumed' status means user clicked /approve; execution_status
    -- means the command actually ran (or didn't) after that.
    execution_status       TEXT NOT NULL DEFAULT 'not_started',
    execution_reason       TEXT,
    execution_recorded_at  REAL,
    payload_json           TEXT NOT NULL
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_gateway_approvals_status
ON gateway_approvals(status);

-- Speeds up expire_due bulk-mark queries; partial index keeps it tiny.
CREATE INDEX IF NOT EXISTS idx_gateway_approvals_pending_expires
ON gateway_approvals(expires_at)
WHERE status = 'pending';

-- Audit slice: "consumed proposals where execution was blocked" — the
-- exact pattern incident review will look for first. Refers to
-- execution_status which is added by _migrate_existing_schema on
-- pre-existing DBs, so this index must be created AFTER migration.
CREATE INDEX IF NOT EXISTS idx_gateway_approvals_blocked_after_consume
ON gateway_approvals(execution_status)
WHERE execution_status = 'blocked_after_consume';
"""

# Back-compat alias for any external callers that import SCHEMA_SQL.
SCHEMA_SQL = CREATE_TABLE_SQL + INDEX_SQL


def _migrate_existing_schema(conn: sqlite3.Connection) -> None:
    """Idempotent ALTER TABLE for execution_* columns on databases that
    were created by an earlier schema version. CREATE TABLE IF NOT EXISTS
    can't add columns to an existing table; we do it explicitly here.
    """
    cur = conn.execute("PRAGMA table_info(gateway_approvals)")
    existing = {row[1] for row in cur.fetchall()}
    for col_def in (
        ("execution_status",
         "ALTER TABLE gateway_approvals ADD COLUMN execution_status TEXT NOT NULL DEFAULT 'not_started'"),
        ("execution_reason",
         "ALTER TABLE gateway_approvals ADD COLUMN execution_reason TEXT"),
        ("execution_recorded_at",
         "ALTER TABLE gateway_approvals ADD COLUMN execution_recorded_at REAL"),
    ):
        col_name, ddl = col_def
        if col_name not in existing:
            conn.execute(ddl)


# Module-level lock taken only briefly during schema-init to avoid two
# threads racing to CREATE TABLE on first construction. Once the schema
# exists this lock is irrelevant — every other operation relies entirely
# on SQLite's own file-level locking.
_schema_init_lock = threading.Lock()
_schema_initialised: set[Path] = set()


def _apply_wal(conn: sqlite3.Connection, db_label: str = "state.db") -> None:
    """Best-effort WAL enable with NFS-aware fallback.

    Reuses the same helper as :mod:`hermes_state` so error messaging
    stays consistent across DBs.
    """
    try:
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(conn, db_label=db_label)
    except ImportError:
        # hermes_state not importable from this context — fall back to
        # straight WAL pragma; failure is non-fatal.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass


def _ensure_schema(db_path: Path) -> None:
    """Idempotent schema create. Cheap on repeat calls (cached path set)."""
    if db_path in _schema_initialised:
        return
    with _schema_init_lock:
        if db_path in _schema_initialised:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=30)
        try:
            _apply_wal(conn)
            # Three-step ordering is required for upgrade-path safety:
            # 1. CREATE TABLE IF NOT EXISTS — new DBs get full shape;
            #    no-op on existing DBs (which may lack the
            #    execution_* columns).
            # 2. _migrate_existing_schema — PRAGMA-checked ALTER TABLE
            #    adds any missing execution_* columns to old DBs.
            # 3. CREATE INDEX IF NOT EXISTS — partial indices reference
            #    execution_status, which must exist by now.
            #
            # Inlining all three in one executescript would fail on old
            # DBs because the partial index on execution_status is
            # evaluated before ALTER TABLE adds the column.
            conn.executescript(CREATE_TABLE_SQL)
            _migrate_existing_schema(conn)
            conn.executescript(INDEX_SQL)
        finally:
            conn.close()
        _schema_initialised.add(db_path)


def _reset_schema_cache_for_tests() -> None:
    """Test-only: clear the schema-init memoisation between tmp_path runs."""
    _schema_initialised.clear()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SqliteApprovalStore:
    """Production :class:`ApprovalStore` backed by a SQLite database file.

    Args:
        db_path: Path to the SQLite database. Default is the same
            ``state.db`` that :mod:`hermes_state` uses. Tests typically
            pass a tmp_path-derived value.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            try:
                from hermes_state import DEFAULT_DB_PATH
                db_path = DEFAULT_DB_PATH
            except ImportError:
                # Fallback — should not happen in normal hermes runtime.
                from pathlib import Path as _P
                db_path = _P.home() / ".hermes" / "state.db"
        self._db_path = Path(db_path)
        _ensure_schema(self._db_path)

    # ----- Connection helper -----

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,   # explicit BEGIN/COMMIT
            timeout=30,
        )
        return conn

    # ----- Lifecycle -----

    def submit(self, proposal: ApprovalProposal) -> None:
        payload = json.dumps(proposal.as_dict())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO gateway_approvals "
                    "(approval_id, created_at, expires_at, status, "
                    " consumed_at, consumed_by, "
                    " execution_status, execution_reason, execution_recorded_at, "
                    " payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        proposal.approval_id,
                        proposal.created_at,
                        proposal.expires_at,
                        proposal.status,
                        proposal.consumed_at,
                        proposal.consumed_by,
                        proposal.execution_status,
                        proposal.execution_reason,
                        proposal.execution_recorded_at,
                        payload,
                    ),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as e:
                conn.execute("ROLLBACK")
                # PRIMARY KEY collision → caller passed duplicate id.
                raise ValueError(
                    f"approval_id collision: {proposal.approval_id!r}"
                ) from e
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def get(self, approval_id: str) -> Optional[ApprovalProposal]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status, consumed_at, consumed_by, "
                "       execution_status, execution_reason, execution_recorded_at, "
                "       payload_json "
                "FROM gateway_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_proposal(*row)

    def consume(self, approval_id: str, *, consumed_by: str,
                now: Optional[float] = None) -> Optional[ApprovalProposal]:
        ts = now if now is not None else time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "UPDATE gateway_approvals "
                    "SET status = 'consumed', consumed_at = ?, consumed_by = ? "
                    "WHERE approval_id = ? "
                    "  AND status = 'pending' "
                    "  AND (expires_at IS NULL OR expires_at > ?) "
                    "RETURNING payload_json",
                    (ts, consumed_by, approval_id, ts),
                )
                row = cur.fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        if row is None:
            return None
        # At consume-time, execution_status is at its default
        # ('not_started'); mark_post_consume() updates it later when
        # the caller knows whether the command actually ran or was
        # blocked post-consume.
        return self._row_to_proposal(
            "consumed", ts, consumed_by,
            "not_started", None, None,
            row[0],
        )

    def deny(self, approval_id: str, *, denied_by: str,
             now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "UPDATE gateway_approvals "
                    "SET status = 'denied', consumed_at = ?, consumed_by = ? "
                    "WHERE approval_id = ? "
                    "  AND status = 'pending' "
                    "  AND (expires_at IS NULL OR expires_at > ?)",
                    (ts, denied_by, approval_id, ts),
                )
                affected = cur.rowcount
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return affected == 1

    def expire_due(self, now: Optional[float] = None) -> int:
        ts = now if now is not None else time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "UPDATE gateway_approvals "
                    "SET status = 'expired' "
                    "WHERE status = 'pending' "
                    "  AND expires_at IS NOT NULL "
                    "  AND expires_at <= ?",
                    (ts,),
                )
                affected = cur.rowcount
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return affected

    def mark_post_consume(self, approval_id: str, *, executed: bool,
                          reason: Optional[str] = None,
                          now: Optional[float] = None) -> bool:
        """Record post-consume execution outcome. See base class docstring."""
        ts = now if now is not None else time.time()
        new_status = "executed" if executed else "blocked_after_consume"
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "UPDATE gateway_approvals "
                    "SET execution_status = ?, execution_reason = ?, "
                    "    execution_recorded_at = ? "
                    "WHERE approval_id = ? AND status = 'consumed'",
                    (new_status, reason, ts, approval_id),
                )
                affected = cur.rowcount
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return affected == 1

    # ----- Helpers -----

    @staticmethod
    def _row_to_proposal(status: str, consumed_at: Optional[float],
                         consumed_by: Optional[str],
                         execution_status: str,
                         execution_reason: Optional[str],
                         execution_recorded_at: Optional[float],
                         payload_json: str) -> ApprovalProposal:
        """Reconstruct the proposal, overlaying current lifecycle columns."""
        try:
            payload: dict[str, Any] = json.loads(payload_json)
        except json.JSONDecodeError as e:
            # Corrupt payload — fail closed. Caller should treat as
            # "no proposal" rather than executing under unknown state.
            raise ApprovalStoreError(
                f"corrupt payload_json for approval_id "
                f"{payload_json[:80]!r}: {e}"
            ) from e
        # Replace lifecycle fields with current column values.
        payload["status"] = status
        payload["consumed_at"] = consumed_at
        payload["consumed_by"] = consumed_by
        payload["execution_status"] = execution_status
        payload["execution_reason"] = execution_reason
        payload["execution_recorded_at"] = execution_recorded_at
        # Filter out any fields not in the dataclass to be forward-compatible
        # if older payloads exist.
        allowed = {f for f in ApprovalProposal.__dataclass_fields__}
        cleaned = {k: v for k, v in payload.items() if k in allowed}
        return ApprovalProposal(**cleaned)


# Conformance assertion.
assert isinstance(
    SqliteApprovalStore.__new__(SqliteApprovalStore), ApprovalStore
), "SqliteApprovalStore must satisfy ApprovalStore Protocol"
