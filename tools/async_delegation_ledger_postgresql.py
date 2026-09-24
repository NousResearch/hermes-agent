"""Dedicated PostgreSQL async-delegation ledger adapter.

Mirrors ``gateway/delivery_ledger_postgresql.py`` architecturally: a tenant-scoped
schema, migration + catalog validation, PG18 minimum, and the same method surface
as the SQLite durable ledger in ``tools/async_delegation.py``. Each operation owns
its transaction and connection; no database handle escapes this adapter. Pure
helpers and policy constants are imported from ``tools.async_delegation`` (the
legacy module is their home), while SQL and event-shaping are re-expressed here —
never a SQLite fallback.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from hermes_constants import get_hermes_home, hermes_home_key, profile_name_for_home
from tools.async_delegation import (
    _CLAIM_LEASE_S,
    _DURABLE_RETENTION_SECONDS,
    _MAX_COMPLETION_REPLAY_AGE_S,
    _MAX_DELIVERY_ATTEMPTS,
    _MAX_DURABLE_PENDING,
    _MAX_RETAINED_COMPLETED,
    _ORPHAN_STALE_S,
    _ROUTING_KEYS,
    _offered,
    _orphan_lock,
    _recovered_results,
)

logger = logging.getLogger(__name__)

_ROOT_SCHEMA = "hermes_async_delegation"
_SCHEMA_RE = re.compile(r"^hermes_async_delegation_tenant_[0-9a-f]{32}$")
_VERSION = 1


class AsyncDelegationLedgerConfigurationError(ValueError):
    pass


def postgresql_async_delegation_schema() -> str:
    """Derive the tenant only from the canonical resolved home/profile."""
    home = get_hermes_home().resolve()
    profile = profile_name_for_home(home)
    if profile == "default":
        return _ROOT_SCHEMA
    digest = hashlib.sha256(f"{home}\0{profile}".encode()).hexdigest()[:32]
    return f"hermes_async_delegation_tenant_{digest}"


@dataclass(frozen=True)
class AsyncDelegationLedgerPostgreSQLConfig:
    connect_timeout_seconds: int = 5
    pool_max_size: int = 4


class PostgreSQLAsyncDelegationLedger:
    """Tenant-scoped PostgreSQL implementation of the async-delegation ledger contract."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str | None = None,
        settings: AsyncDelegationLedgerPostgreSQLConfig | None = None,
    ) -> None:
        try:
            self._psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise AsyncDelegationLedgerConfigurationError(
                "PostgreSQL async-delegation ledger requires psycopg") from exc
        self._schema = schema or postgresql_async_delegation_schema()
        if self._schema != _ROOT_SCHEMA and not _SCHEMA_RE.fullmatch(self._schema):
            raise AsyncDelegationLedgerConfigurationError(
                "PostgreSQL async-delegation ledger received an invalid trusted tenant schema")
        self._dsn, self._settings = dsn, settings or AsyncDelegationLedgerPostgreSQLConfig()
        self._closed = False
        self._lock = threading.Lock()
        connection = self._connect()
        try:
            self._migrate(connection)
        finally:
            connection.close()

    def _connect(self):
        if self._closed:
            raise RuntimeError("PostgreSQL async-delegation ledger is closed")
        return self._psycopg.connect(
            self._dsn, connect_timeout=self._settings.connect_timeout_seconds, autocommit=False)

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'SET LOCAL search_path TO "{self._schema}", pg_catalog')
                yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self, connection: Any) -> None:
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (f"{self._schema}:async-delegation-migration",))
                cur.execute("SHOW server_version_num")
                if int(cur.fetchone()[0]) < 140000:
                    raise AsyncDelegationLedgerConfigurationError(
                        "PostgreSQL async-delegation ledger requires PostgreSQL 14 or newer")
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._schema}.async_delegation_schema_migrations "
                    "(version integer PRIMARY KEY, applied_at double precision NOT NULL)")
                cur.execute(f"SELECT version FROM {self._schema}.async_delegation_schema_migrations")
                versions = {int(row[0]) for row in cur.fetchall()}
                if versions - {_VERSION}:
                    raise AsyncDelegationLedgerConfigurationError(
                        "unsupported PostgreSQL async-delegation migration version")
                if 1 not in versions:
                    cur.execute(f"""CREATE TABLE {self._schema}.async_delegations (
                      delegation_id text PRIMARY KEY,
                      origin_session text NOT NULL,
                      origin_ui_session_id text NOT NULL DEFAULT '',
                      parent_session_id text,
                      state text NOT NULL,
                      dispatched_at double precision NOT NULL,
                      completed_at double precision,
                      updated_at double precision NOT NULL,
                      event_json text,
                      result_json text,
                      delivery_state text NOT NULL DEFAULT 'pending',
                      delivery_attempts integer NOT NULL DEFAULT 0,
                      delivered_at double precision,
                      owner_pid integer,
                      owner_started_at bigint,
                      task_json text,
                      delivery_claim text,
                      delivery_claimed_at double precision,
                      origin_session_id text NOT NULL DEFAULT '')""")
                    cur.execute(
                        f"CREATE INDEX idx_async_delegations_delivery ON "
                        f"{self._schema}.async_delegations (delivery_state, completed_at)")
                    cur.execute(
                        f"INSERT INTO {self._schema}.async_delegation_schema_migrations "
                        "VALUES (1, extract(epoch from clock_timestamp()))")
                self._validate(cur)
            connection.commit()
        except Exception:
            connection.rollback(); raise

    def _validate(self, cur: Any) -> None:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name='async_delegations'", (self._schema,))
        required = {
            'delegation_id', 'origin_session', 'origin_ui_session_id', 'parent_session_id',
            'state', 'dispatched_at', 'completed_at', 'updated_at', 'event_json', 'result_json',
            'delivery_state', 'delivery_attempts', 'delivered_at', 'owner_pid', 'owner_started_at',
            'task_json', 'delivery_claim', 'delivery_claimed_at', 'origin_session_id',
        }
        missing = required - {row[0] for row in cur.fetchall()}
        if missing:
            raise AsyncDelegationLedgerConfigurationError(
                f"PostgreSQL async-delegation schema drift: missing columns {sorted(missing)}")
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname='idx_async_delegations_delivery'",
            (self._schema,))
        if cur.fetchone() is None:
            raise AsyncDelegationLedgerConfigurationError(
                "PostgreSQL async-delegation schema drift: missing idx_async_delegations_delivery")

    # ── Dispatch / completion writes ────────────────────────────────────────
    def persist_dispatch(self, record: dict[str, Any]) -> None:
        now = time.time()
        try:
            from gateway.status import get_process_start_time
            owner_started_at = get_process_start_time(os.getpid())
        except Exception:
            owner_started_at = None
        task_payload = {
            key: record.get(key)
            for key in ("goal", "goals", "context", "toolsets", "role", "model",
                        "is_batch", "task_indexes", "task_transcripts", *_ROUTING_KEYS)
            if key in record}
        try:
            task_payload["owner_cwd"] = os.getcwd()
        except OSError:
            pass
        with self._transaction() as cur:
            cur.execute("""INSERT INTO async_delegations
                (delegation_id, origin_session, origin_ui_session_id, parent_session_id,
                 state, dispatched_at, updated_at, delivery_state, delivery_attempts,
                 owner_pid, owner_started_at, task_json, origin_session_id)
                VALUES (%s, %s, %s, %s, 'running', %s, %s, 'pending', 0, %s, %s, %s, %s)
                ON CONFLICT (delegation_id) DO UPDATE SET
                 origin_session=EXCLUDED.origin_session,
                 origin_ui_session_id=EXCLUDED.origin_ui_session_id,
                 parent_session_id=EXCLUDED.parent_session_id,
                 state='running', dispatched_at=EXCLUDED.dispatched_at,
                 updated_at=EXCLUDED.updated_at, delivery_state='pending',
                 delivery_attempts=0, owner_pid=EXCLUDED.owner_pid,
                 owner_started_at=EXCLUDED.owner_started_at, task_json=EXCLUDED.task_json,
                 origin_session_id=EXCLUDED.origin_session_id""",
                (record["delegation_id"], record.get("session_key", ""),
                 record.get("origin_ui_session_id", ""), record.get("parent_session_id"),
                 record["dispatched_at"], now, os.getpid(), owner_started_at,
                 json.dumps(task_payload), record.get("origin_session_id", "")))

    def persist_completion(self, event: dict[str, Any], result: dict[str, Any]) -> None:
        now = time.time()
        with self._transaction() as cur:
            cur.execute("""UPDATE async_delegations SET state=%s, completed_at=%s, updated_at=%s,
                event_json=%s, result_json=%s, delivery_state='pending'
                WHERE delegation_id=%s""",
                (event.get("status", "completed"), event.get("completed_at", now), now,
                 json.dumps(event), json.dumps(result), event["delegation_id"]))

    def prune_durable_records(self) -> None:
        cutoff = time.time() - _DURABLE_RETENTION_SECONDS
        with self._transaction() as cur:
            cur.execute(
                "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < %s",
                (cutoff,))
            cur.execute(
                "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')")
            terminal_count = int(cur.fetchone()[0])
            if terminal_count > _MAX_RETAINED_COMPLETED:
                cur.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT %s)""",
                    (terminal_count - _MAX_RETAINED_COMPLETED,))
            cur.execute("""SELECT COUNT(*) FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'""")
            pending_count = int(cur.fetchone()[0])
            if pending_count > _MAX_DURABLE_PENDING:
                cur.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     ORDER BY updated_at ASC LIMIT %s)""",
                    (pending_count - _MAX_DURABLE_PENDING,))

    def record_unit_child(self, delegation_id: str, entry: dict[str, Any]) -> None:
        with self._transaction() as cur:
            cur.execute(
                "SELECT result_json FROM async_delegations WHERE delegation_id=%s AND state='running'",
                (delegation_id,))
            row = cur.fetchone()
            if row is None:
                return
            partial = json.loads(row[0] or "{}") or {}
            results = [r for r in partial.get("results") or []
                       if r.get("task_index") != entry.get("task_index")]
            results.append(entry)
            cur.execute(
                "UPDATE async_delegations SET result_json=%s, updated_at=%s "
                "WHERE delegation_id=%s AND state='running'",
                (json.dumps({"results": results, "partial": True}), time.time(), delegation_id))

    # ── Recovery ────────────────────────────────────────────────────────────
    def _owner_alive(self, pid, started) -> bool:
        try:
            from gateway.status import _pid_exists, get_process_start_time, start_time_fingerprints_match
        except Exception:
            return False
        if not pid:
            return False
        try:
            return _pid_exists(int(pid)) and (
                started is None
                or start_time_fingerprints_match(started, get_process_start_time(int(pid)) or 0)
            )
        except (TypeError, ValueError):
            return False

    def recover_abandoned_delegations(self) -> int:
        now, recovered = time.time(), 0
        with self._transaction() as cur:
            cur.execute("""SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id, result_json, state
               FROM async_delegations WHERE state IN ('running','finalizing')""")
            rows = cur.fetchall()
            for (delegation_id, session_key, origin_ui, parent_id, dispatched_at,
                 pid, started, task_json, origin_sid, result_json, last_state) in rows:
                if self._owner_alive(pid, started):
                    continue
                task = json.loads(task_json or "{}")
                error = "Delegation owner exited before recording a terminal result; outcome unknown."
                recovered_results = _recovered_results(task, result_json, error)
                if recovered_results:
                    done = sum(1 for r in recovered_results if r.get("status") != "unknown")
                    error = (f"Delegation owner exited before the unit finished; "
                             f"{done}/{len(recovered_results)} child results were recorded and are "
                             f"included below, the rest are unknown.")
                diagnostics = {
                    "last_known_status": last_state,
                    "task_transcripts": task.get("task_transcripts") or {},
                }
                from tools.async_delegation_recovery_hints import git_state_hint, transcript_tails
                if tails := transcript_tails(diagnostics["task_transcripts"]):
                    diagnostics["transcript_tails"] = tails
                if hint := git_state_hint(task.get("owner_cwd")):
                    diagnostics["git_state_hint"] = hint
                event = {
                    "type": "async_delegation", "delegation_id": delegation_id,
                    "session_key": session_key, "origin_ui_session_id": origin_ui,
                    "origin_session_id": origin_sid or "", "parent_session_id": parent_id,
                    "goal": task.get("goal", ""), "goals": task.get("goals"),
                    "context": task.get("context"), "toolsets": task.get("toolsets"),
                    "role": task.get("role"), "model": task.get("model"),
                    "is_batch": bool(task.get("is_batch")), "status": "unknown",
                    "summary": None, "error": error, **diagnostics,
                    **({"results": recovered_results} if recovered_results else {}),
                    "dispatched_at": dispatched_at, "completed_at": now,
                    **{k: task[k] for k in _ROUTING_KEYS if task.get(k)}}
                result = {
                    "status": "unknown", "summary": None, "error": event["error"], **diagnostics,
                    **({"results": recovered_results} if recovered_results else {}),
                }
                cur.execute("""UPDATE async_delegations SET state='unknown', completed_at=%s,
                       updated_at=%s, event_json=%s, result_json=%s, delivery_state='pending'
                       WHERE delegation_id=%s""",
                    (now, now, json.dumps(event), json.dumps(result), delegation_id))
                recovered += 1
        return recovered

    def restore_undelivered_completions(self, target_queue) -> int:
        self.recover_abandoned_delegations()
        now = time.time()
        with self._transaction() as cur:
            cur.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
                   ORDER BY completed_at, delegation_id""")
            return self._replay_pending(cur, cur.fetchall(), target_queue, now)

    def _replay_pending(self, cur, rows, target_queue, now: float) -> int:
        home, restored = hermes_home_key(get_hermes_home()), 0
        for delegation_id, payload, completed_at, dispatched_at in rows:
            age_basis = completed_at or dispatched_at
            if age_basis and (now - age_basis) > _MAX_COMPLETION_REPLAY_AGE_S:
                cur.execute("""UPDATE async_delegations SET delivery_state='dropped',
                              delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=%s
                       WHERE delegation_id=%s AND delivery_state='pending'""",
                    (now, delegation_id))
                logger.warning(
                    "Async delegation %s: pending completion is %.1fh old (cap %.1fh); "
                    "terminally dropping the replay (result remains queryable).",
                    delegation_id, (now - age_basis) / 3600.0,
                    _MAX_COMPLETION_REPLAY_AGE_S / 3600.0)
                continue
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
            target_queue.put(evt)
            with _orphan_lock:
                _offered.add((home, delegation_id))
            restored += 1
        return restored

    def sweep_orphaned_completions(self, target_queue, *, now: float | None = None) -> int:
        self.recover_abandoned_delegations()
        now = time.time() if now is None else now
        home = hermes_home_key(get_hermes_home())
        with _orphan_lock:
            offered = {delegation_id for key, delegation_id in _offered if key == home}
        with self._transaction() as cur:
            cur.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at,
                          owner_pid, owner_started_at, delivery_attempts
                   FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < %s
                     AND (delivery_claim IS NULL OR delivery_claimed_at < %s)
                   ORDER BY completed_at, delegation_id""",
                (now - _ORPHAN_STALE_S, now - _CLAIM_LEASE_S))
            orphans = []
            for delegation_id, payload, completed_at, dispatched_at, pid, started, attempts in cur.fetchall():
                if delegation_id in offered or self._owner_alive(pid, started):
                    continue
                if (attempts or 0) >= _MAX_DELIVERY_ATTEMPTS:
                    cur.execute("""UPDATE async_delegations SET delivery_state='dropped',
                                  delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=%s
                           WHERE delegation_id=%s AND delivery_state='pending'""",
                        (now, delegation_id))
                    logger.warning(
                        "Async delegation %s exhausted its %d delivery attempts; "
                        "marking terminally dropped (result remains queryable).",
                        delegation_id, _MAX_DELIVERY_ATTEMPTS)
                    continue
                orphans.append((delegation_id, payload, completed_at, dispatched_at))
            return self._replay_pending(cur, orphans, target_queue, now)

    # ── Delivery claim transitions ──────────────────────────────────────────
    def mark_completion_delivered(self, delegation_id: str) -> bool:
        now = time.time()
        with self._transaction() as cur:
            cur.execute("""UPDATE async_delegations SET delivery_state='delivered',
                          delivered_at=%s, updated_at=%s
                   WHERE delegation_id=%s AND delivery_state!='delivered'""",
                (now, now, delegation_id))
            return cur.rowcount == 1

    def claim_completion_delivery(self, delegation_id: str, claim_id: str) -> bool:
        now = time.time()
        with self._transaction() as cur:
            cur.execute(
                "SELECT delivery_state FROM async_delegations WHERE delegation_id=%s",
                (delegation_id,))
            row = cur.fetchone()
            if row is None:
                return True  # legacy event created before durable dispatch
            cur.execute("""UPDATE async_delegations SET delivery_claim=%s, delivery_claimed_at=%s,
                      delivery_attempts=delivery_attempts+1, updated_at=%s
               WHERE delegation_id=%s AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < %s)""",
                (claim_id, now, now, delegation_id, now - _CLAIM_LEASE_S))
            return cur.rowcount == 1

    def release_completion_delivery(self, delegation_id: str, claim_id: str) -> bool:
        now = time.time()
        with self._transaction() as cur:
            cur.execute("""UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=%s
               WHERE delegation_id=%s AND delivery_state='pending'
                 AND delivery_claim=%s AND delivery_attempts>=%s""",
                (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS))
            if cur.rowcount == 1:
                logger.warning("Async delegation %s exhausted its %d delivery attempts; "
                               "marking terminally dropped (result remains queryable).",
                               delegation_id, _MAX_DELIVERY_ATTEMPTS)
                return True
            cur.execute("""UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=%s
               WHERE delegation_id=%s AND delivery_state='pending'
                 AND delivery_claim=%s""", (now, delegation_id, claim_id))
            return cur.rowcount == 1

    def defer_completion_delivery(self, delegation_id: str, claim_id: str) -> bool:
        with self._transaction() as cur:
            cur.execute("""UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, delivery_attempts=GREATEST(0, delivery_attempts-1),
                      updated_at=%s
               WHERE delegation_id=%s AND delivery_state='pending' AND delivery_claim=%s""",
                (time.time(), delegation_id, claim_id))
            return cur.rowcount == 1

    def drop_completion_delivery(self, delegation_id: str, claim_id: str) -> bool:
        with self._transaction() as cur:
            cur.execute("""UPDATE async_delegations SET delivery_state='dropped',
                      updated_at=%s, delivery_claim=NULL, delivery_claimed_at=NULL
               WHERE delegation_id=%s AND delivery_state='pending' AND delivery_claim=%s""",
                (time.time(), delegation_id, claim_id))
            return cur.rowcount == 1

    def complete_completion_delivery(self, delegation_id: str, claim_id: str) -> bool:
        now = time.time()
        with self._transaction() as cur:
            cur.execute("""UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=%s, updated_at=%s, delivery_claim=NULL, delivery_claimed_at=NULL
               WHERE delegation_id=%s AND delivery_state='pending' AND delivery_claim=%s""",
                (now, now, delegation_id, claim_id))
            return cur.rowcount == 1

    # ── Reads ───────────────────────────────────────────────────────────────
    def get_durable_delegation(self, delegation_id: str) -> dict[str, Any] | None:
        with self._transaction() as cur:
            cur.execute("""SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts, origin_session_id
               FROM async_delegations WHERE delegation_id=%s""", (delegation_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "delegation_id": delegation_id, "origin_session": row[0], "state": row[1],
            "dispatched_at": row[2], "completed_at": row[3],
            "result": json.loads(row[4]) if row[4] else None, "delivery_state": row[5],
            "delivery_attempts": row[6], "origin_session_id": row[7] or ""}

    def debug_rows(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._transaction() as cur:
            cur.execute(
                "SELECT delegation_id, state, delivery_state, delivery_attempts "
                "FROM async_delegations ORDER BY updated_at DESC LIMIT %s", (limit,))
            return [dict(zip(('delegation_id', 'state', 'delivery_state', 'delivery_attempts'), row))
                    for row in cur.fetchall()]

    def close(self) -> None:
        self._closed = True
