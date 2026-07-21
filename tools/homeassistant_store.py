"""Durable proposal and audit storage for Home Assistant configuration changes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


class ProposalError(RuntimeError):
    """Base class for proposal lifecycle failures."""


class ProposalExpired(ProposalError):
    """The proposal approval window elapsed."""


class ProposalStale(ProposalError):
    """The Home Assistant resource changed after preview."""


class ProposalUnavailable(ProposalError):
    """The proposal or change is not in a claimable state."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for a JSON-compatible value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def structured_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    """Return a deterministic, JSON-pointer-like recursive change list."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(before.keys() | after.keys()):
            child_path = f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}"
            if key not in before:
                changes.append(
                    {"path": child_path, "before": None, "after": after[key], "change": "added"}
                )
            elif key not in after:
                changes.append(
                    {"path": child_path, "before": before[key], "after": None, "change": "removed"}
                )
            else:
                changes.extend(structured_diff(before[key], after[key], child_path))
        return changes
    if before == after:
        return []
    return [{"path": path or "/", "before": before, "after": after, "change": "changed"}]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class HomeAssistantChangeStore:
    """SQLite-backed lifecycle store scoped to the active Hermes profile."""

    def __init__(self, path: str | Path | None = None, *, history_limit: int = 500):
        self.path = Path(path) if path else get_hermes_home() / "state" / "homeassistant_changes.sqlite3"
        self.history_limit = max(1, int(history_limit))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    id TEXT PRIMARY KEY,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    operation TEXT NOT NULL CHECK (operation IN ('create', 'update')),
                    before_json TEXT NOT NULL,
                    desired_json TEXT NOT NULL,
                    before_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS changes (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    operation TEXT NOT NULL,
                    before_json TEXT NOT NULL,
                    after_json TEXT NOT NULL,
                    after_fingerprint TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    created_by_hermes INTEGER NOT NULL,
                    authoritative_id INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    rollback_at TEXT,
                    FOREIGN KEY (proposal_id) REFERENCES proposals(id)
                );
                CREATE INDEX IF NOT EXISTS changes_applied_at ON changes(applied_at DESC);
                """
            )
            change_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(changes)").fetchall()
            }
            if "authoritative_id" not in change_columns:
                conn.execute(
                    "ALTER TABLE changes ADD COLUMN authoritative_id INTEGER NOT NULL DEFAULT 0"
                )
            conn.commit()
        finally:
            conn.close()
        os.chmod(self.path, 0o600)

    @staticmethod
    def _proposal(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "resource_type": row["resource_type"],
            "resource_id": row["resource_id"],
            "operation": row["operation"],
            "before": json.loads(row["before_json"]),
            "desired": json.loads(row["desired_json"]),
            "before_fingerprint": row["before_fingerprint"],
            "created_at": _datetime(row["created_at"]),
            "expires_at": _datetime(row["expires_at"]),
            "status": row["status"],
        }

    @staticmethod
    def _change(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "proposal_id": row["proposal_id"],
            "resource_type": row["resource_type"],
            "resource_id": row["resource_id"],
            "operation": row["operation"],
            "before": json.loads(row["before_json"]),
            "after": json.loads(row["after_json"]),
            "after_fingerprint": row["after_fingerprint"],
            "applied_at": _datetime(row["applied_at"]),
            "created_by_hermes": bool(row["created_by_hermes"]),
            "authoritative_id": bool(row["authoritative_id"]),
            "status": row["status"],
            "rollback_at": _datetime(row["rollback_at"]) if row["rollback_at"] else None,
        }

    def create_proposal(
        self,
        *,
        resource_type: str,
        resource_id: str | None,
        operation: str,
        before: Any,
        desired: Any,
        ttl_seconds: int = 900,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if operation not in {"create", "update"}:
            raise ValueError("operation must be create or update")
        created = _as_utc(now or _utc_now())
        expires = created + timedelta(seconds=max(1, int(ttl_seconds)))
        proposal_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "UPDATE proposals SET status = 'superseded' WHERE status = 'pending'"
            )
            conn.execute(
                "INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    proposal_id,
                    resource_type,
                    resource_id,
                    operation,
                    _canonical_json(before),
                    _canonical_json(desired),
                    canonical_fingerprint(before),
                    _timestamp(created),
                    _timestamp(expires),
                ),
            )
            conn.execute(
                """DELETE FROM proposals
                   WHERE status IN ('expired', 'stale', 'failed', 'superseded')
                     AND id NOT IN (SELECT proposal_id FROM changes)"""
            )
        proposal = self.get_proposal(proposal_id)
        assert proposal is not None
        return proposal

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        return self._proposal(row)

    def mark_proposal_failed(self, proposal_id: str) -> None:
        """Close an in-flight proposal after a remote mutation failure."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE proposals SET status = 'failed' WHERE id = ? AND status = 'applying'",
                (proposal_id,),
            )

    def claim_proposal(
        self,
        proposal_id: str,
        current_fingerprint: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        claimed_at = _as_utc(now or _utc_now())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            proposal = self._proposal(row)
            if proposal is None or proposal["status"] != "pending":
                conn.rollback()
                raise ProposalUnavailable("proposal is not pending")
            if claimed_at > proposal["expires_at"]:
                conn.execute("UPDATE proposals SET status = 'expired' WHERE id = ?", (proposal_id,))
                conn.commit()
                raise ProposalExpired("proposal approval window expired")
            if current_fingerprint != proposal["before_fingerprint"]:
                conn.execute("UPDATE proposals SET status = 'stale' WHERE id = ?", (proposal_id,))
                conn.commit()
                raise ProposalStale("resource changed after preview")
            conn.execute("UPDATE proposals SET status = 'applying' WHERE id = ?", (proposal_id,))
            conn.commit()
        finally:
            conn.close()
        proposal["status"] = "applying"
        return proposal

    def claim_and_begin_apply(
        self,
        proposal_id: str,
        current_fingerprint: str,
        *,
        created_by_hermes: bool,
        resource_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically claim a proposal and persist its pre-mutation audit intent."""
        claimed_at = _as_utc(now or _utc_now())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            proposal = self._proposal(row)
            if proposal is None or proposal["status"] != "pending":
                conn.rollback()
                raise ProposalUnavailable("proposal is not pending")
            if claimed_at > proposal["expires_at"]:
                conn.execute("UPDATE proposals SET status = 'expired' WHERE id = ?", (proposal_id,))
                conn.commit()
                raise ProposalExpired("proposal approval window expired")
            if current_fingerprint != proposal["before_fingerprint"]:
                conn.execute("UPDATE proposals SET status = 'stale' WHERE id = ?", (proposal_id,))
                conn.commit()
                raise ProposalStale("resource changed after preview")
            change_id = uuid.uuid4().hex
            conn.execute("UPDATE proposals SET status = 'applying' WHERE id = ?", (proposal_id,))
            conn.execute(
                """INSERT INTO changes (
                    id, proposal_id, resource_type, resource_id, operation,
                    before_json, after_json, after_fingerprint, applied_at,
                    created_by_hermes, authoritative_id, status, rollback_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'applying', NULL)""",
                (
                    change_id, proposal_id, proposal["resource_type"],
                    resource_id or proposal["resource_id"], proposal["operation"],
                    _canonical_json(proposal["before"]),
                    _canonical_json(proposal["desired"]),
                    canonical_fingerprint(proposal["desired"]),
                    _timestamp(claimed_at), int(created_by_hermes),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        change = self.get_change(change_id)
        assert change is not None
        return change

    def record_applied(
        self,
        proposal_id: str,
        *,
        after: Any,
        created_by_hermes: bool,
        resource_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        change = self.begin_apply(
            proposal_id,
            created_by_hermes=created_by_hermes,
            resource_id=resource_id,
            now=now,
        )
        return self.finalize_applied(
            change["id"], after=after, resource_id=resource_id, now=now
        )

    def begin_apply(
        self,
        proposal_id: str,
        *,
        created_by_hermes: bool,
        resource_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist rollback evidence before making the remote mutation."""
        applied_at = _as_utc(now or _utc_now())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            proposal = self._proposal(row)
            if proposal is None or proposal["status"] != "applying":
                conn.rollback()
                raise ProposalUnavailable("proposal is not being applied")
            existing = conn.execute(
                "SELECT * FROM changes WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
            if existing is not None:
                conn.rollback()
                raise ProposalUnavailable("proposal already has a mutation attempt")
            change_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO changes (
                    id, proposal_id, resource_type, resource_id, operation,
                    before_json, after_json, after_fingerprint, applied_at,
                    created_by_hermes, authoritative_id, status, rollback_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'applying', NULL)""",
                (
                    change_id,
                    proposal_id,
                    proposal["resource_type"],
                    resource_id or proposal["resource_id"],
                    proposal["operation"],
                    _canonical_json(proposal["before"]),
                    _canonical_json(proposal["desired"]),
                    canonical_fingerprint(proposal["desired"]),
                    _timestamp(applied_at),
                    int(created_by_hermes),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        change = self.get_change(change_id)
        assert change is not None
        return change

    def identify_created_resource(self, change_id: str, resource_id: str) -> None:
        """Persist the server-confirmed ID before any further remote read."""
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE changes SET resource_id = ?, authoritative_id = 1
                   WHERE id = ? AND operation = 'create'
                     AND status IN ('applying', 'apply_uncertain')""",
                (resource_id, change_id),
            )
            if cursor.rowcount != 1:
                raise ProposalUnavailable("create attempt is not awaiting identification")

    def finalize_applied(
        self,
        change_id: str,
        *,
        after: Any,
        resource_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        applied_at = _as_utc(now or _utc_now())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM changes WHERE id = ?", (change_id,)).fetchone()
            change = self._change(row)
            if change is None or change["status"] not in {"applying", "apply_uncertain"}:
                conn.rollback()
                raise ProposalUnavailable("change is not awaiting apply completion")
            conn.execute(
                """UPDATE changes SET resource_id = ?, after_json = ?,
                   after_fingerprint = ?, applied_at = ?, status = 'applied'
                   WHERE id = ?""",
                (
                    resource_id or change["resource_id"],
                    _canonical_json(after),
                    canonical_fingerprint(after),
                    _timestamp(applied_at),
                    change_id,
                ),
            )
            conn.execute(
                "UPDATE proposals SET status = 'applied' WHERE id = ?",
                (change["proposal_id"],),
            )
            conn.execute(
                """DELETE FROM changes WHERE id IN (
                    SELECT id FROM changes
                    WHERE status IN ('applied', 'rolled_back', 'failed')
                    ORDER BY applied_at DESC, rowid DESC LIMIT -1 OFFSET ?
                )""",
                (self.history_limit,),
            )
            conn.execute(
                """DELETE FROM proposals
                   WHERE status != 'pending'
                     AND id NOT IN (SELECT proposal_id FROM changes)"""
            )
            conn.commit()
        finally:
            conn.close()
        change = self.get_change(change_id)
        assert change is not None
        return change

    def mark_apply_uncertain(self, change_id: str) -> None:
        """Retain an interrupted apply for later state-based reconciliation."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE changes SET status = 'apply_uncertain' WHERE id = ? AND status = 'applying'",
                (change_id,),
            )

    def mark_apply_not_applied(self, change_id: str) -> None:
        """Close an attempt when the remote resource still matches its before state."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT proposal_id FROM changes WHERE id = ?", (change_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ProposalUnavailable("change not found")
            conn.execute(
                """UPDATE changes SET status = 'failed'
                   WHERE id = ? AND status IN ('applying', 'apply_uncertain')""",
                (change_id,),
            )
            conn.execute(
                "UPDATE proposals SET status = 'failed' WHERE id = ? AND status = 'applying'",
                (row["proposal_id"],),
            )
            conn.commit()
        finally:
            conn.close()

    def list_unfinished(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM changes
                   WHERE status IN ('applying', 'apply_uncertain', 'rolling_back', 'rollback_uncertain')
                   ORDER BY applied_at, rowid"""
            ).fetchall()
        return [self._change(row) for row in rows]

    def get_change(self, change_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM changes WHERE id = ?", (change_id,)).fetchone()
        return self._change(row)

    def list_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        count = min(self.history_limit, max(1, int(limit or self.history_limit)))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM changes ORDER BY applied_at DESC, rowid DESC LIMIT ?", (count,)
            ).fetchall()
        return [self._change(row) for row in rows]

    def claim_rollback(
        self,
        change_id: str,
        current_fingerprint: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        del now  # Reserved for the eventual rollback audit event timestamp.
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM changes WHERE id = ?", (change_id,)).fetchone()
            change = self._change(row)
            if change is None or change["status"] != "applied":
                conn.rollback()
                raise ProposalUnavailable("change is not available for rollback")
            if (
                change["operation"] == "create"
                and (not change["created_by_hermes"] or not change["authoritative_id"])
            ):
                conn.rollback()
                raise ProposalUnavailable(
                    "created resource lacks authoritative Hermes ownership evidence"
                )
            if current_fingerprint != change["after_fingerprint"]:
                conn.rollback()
                raise ProposalStale("resource changed after Hermes applied it")
            conn.execute("UPDATE changes SET status = 'rolling_back' WHERE id = ?", (change_id,))
            conn.commit()
        finally:
            conn.close()
        change["status"] = "rolling_back"
        return change

    def record_rolled_back(
        self, change_id: str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        rolled_back_at = _as_utc(now or _utc_now())
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE changes SET status = 'rolled_back', rollback_at = ?
                   WHERE id = ? AND status IN ('rolling_back', 'rollback_uncertain')""",
                (_timestamp(rolled_back_at), change_id),
            )
            if cursor.rowcount != 1:
                raise ProposalUnavailable("change is not being rolled back")
        change = self.get_change(change_id)
        assert change is not None
        return change

    def mark_rollback_failed(self, change_id: str) -> None:
        """Return a failed rollback claim to its applied, retryable state."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE changes SET status = 'applied'
                   WHERE id = ? AND status IN ('rolling_back', 'rollback_uncertain')""",
                (change_id,),
            )

    def mark_rollback_uncertain(self, change_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE changes SET status = 'rollback_uncertain' WHERE id = ? AND status = 'rolling_back'",
                (change_id,),
            )
