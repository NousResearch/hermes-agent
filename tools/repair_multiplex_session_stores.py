#!/usr/bin/env python3
"""Repair multiplexed Hermes sessions into their canonical per-profile stores.

This is a fail-closed, dry-run-first repair utility for issue #88715 and the
physical-store invariant selected by merged PR #88734:

* the default profile owns ``$HERMES_HOME/state.db``;
* a named profile owns ``$HERMES_HOME/profiles/<name>/state.db``;
* gateway session rows are physically stored with their runtime profile;
* keyless CLI/TUI/subagent rows stay in the store where they were created.

The tool never guesses through contradictory evidence.  It derives profile
ownership from ``session_key``, ``profile_name``, and ``origin_json.profile``;
propagates one unambiguous profile across a parent/child lineage component; and
blocks a component when two authoritative claims disagree.

Apply mode is intentionally copy-before-delete.  Destination copies are
committed and verified before source rows are removed.  A crash between those
steps can leave an identical duplicate, but cannot lose the only copy.  A rerun
recognises identical duplicates and completes the cleanup idempotently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

DEFAULT_PROFILE = "default"
LEGACY_NAMESPACE = "main"
SESSION_REFERENCE_NAMES = frozenset(
    {
        "session_id",
        "parent_session_id",
        "child_session_id",
        "old_session_id",
        "new_session_id",
        "source_session_id",
        "target_session_id",
        "origin_session_id",
    }
)
SESSION_KEY_TABLES = ("gateway_routing", "gateway_hygiene_state")


class RepairError(RuntimeError):
    """Base class for repair failures."""


class RepairBlocked(RepairError):
    """The proposed repair is ambiguous or unsafe and must not run."""


@dataclass(frozen=True, slots=True)
class Store:
    profile: str
    path: Path


@dataclass(frozen=True, slots=True)
class Evidence:
    source: str
    profile: str


@dataclass(slots=True)
class SessionRecord:
    store: Store
    session_id: str
    row: dict[str, Any]
    explicit_evidence: list[Evidence] = field(default_factory=list)
    target_profile: Optional[str] = None

    @property
    def parent_id(self) -> Optional[str]:
        value = self.row.get("parent_session_id")
        return str(value) if value not in (None, "") else None


@dataclass(frozen=True, slots=True)
class TablePlan:
    name: str
    columns: tuple[str, ...]
    reference_columns: tuple[str, ...]
    primary_key_columns: tuple[str, ...]
    surrogate_pk_column: Optional[str] = None

    @property
    def identity_columns(self) -> tuple[str, ...]:
        if self.surrogate_pk_column is None:
            return self.primary_key_columns
        return tuple(column for column in self.columns if column != self.surrogate_pk_column)


@dataclass(slots=True)
class Move:
    source: Store
    destination: Store
    session_ids: list[str]
    component_id: str


@dataclass(slots=True)
class Plan:
    hermes_home: Path
    stores: dict[str, Store]
    records: dict[tuple[str, str], SessionRecord]
    moves: list[Move]
    blocked: list[str]
    notes: list[str]

    @property
    def safe(self) -> bool:
        return not self.blocked

    @property
    def session_count(self) -> int:
        return sum(len(move.session_ids) for move in self.moves)


def normalize_profile(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in {DEFAULT_PROFILE, LEGACY_NAMESPACE}:
        return DEFAULT_PROFILE
    return text


def profile_from_session_key(session_key: object) -> Optional[str]:
    text = str(session_key or "").strip()
    if not text:
        return None
    parts = text.split(":", 2)
    if len(parts) < 3 or parts[0] != "agent":
        return None
    return normalize_profile(parts[1])


def profile_from_origin_json(origin_json: object) -> Optional[str]:
    if not origin_json:
        return None
    try:
        value = json.loads(str(origin_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    return normalize_profile(value.get("profile"))


def _row_dict(cursor: sqlite3.Cursor, row: Sequence[Any]) -> dict[str, Any]:
    return {
        description[0]: row[index]
        for index, description in enumerate(cursor.description or ())
    }


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
    else:
        conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote_ident(table)})"))


def _table_primary_key_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = list(conn.execute(f"PRAGMA table_info({quote_ident(table)})"))
    return tuple(
        str(row[1])
        for row in sorted(
            (row for row in rows if int(row[5] or 0) > 0),
            key=lambda row: int(row[5]),
        )
    )


def _surrogate_integer_pk(conn: sqlite3.Connection, table: str) -> Optional[str]:
    rows = list(conn.execute(f"PRAGMA table_info({quote_ident(table)})"))
    pk_rows = [row for row in rows if int(row[5] or 0) > 0]
    if len(pk_rows) != 1:
        return None
    row = pk_rows[0]
    declared = str(row[2] or "").strip().upper()
    column = str(row[1])
    # SQLite aliases a one-column INTEGER PRIMARY KEY to rowid.  Those values
    # are local to one physical database and are expected to collide across
    # profile stores; the repair remaps them unless another normal table has a
    # foreign key to the surrogate.
    if declared == "INTEGER" and column not in SESSION_REFERENCE_NAMES:
        return column
    return None


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def discover_stores(hermes_home: Path) -> dict[str, Store]:
    stores: dict[str, Store] = {}
    root_db = hermes_home / "state.db"
    if root_db.exists():
        stores[DEFAULT_PROFILE] = Store(DEFAULT_PROFILE, root_db)
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir():
                continue
            db_path = child / "state.db"
            if db_path.exists():
                profile = normalize_profile(child.name)
                if profile is None or profile == DEFAULT_PROFILE:
                    continue
                stores[profile] = Store(profile, db_path)
    return stores


def _session_rows(store: Store) -> list[SessionRecord]:
    conn = _connect(store.path, read_only=True)
    try:
        if not _table_exists(conn, "sessions"):
            raise RepairBlocked(f"{store.path}: missing sessions table")
        cursor = conn.execute("SELECT * FROM sessions")
        records: list[SessionRecord] = []
        for raw in cursor.fetchall():
            row = dict(raw)
            sid = str(row.get("id") or "")
            if not sid:
                raise RepairBlocked(f"{store.path}: sessions row has an empty id")
            evidence: list[Evidence] = []
            for source, profile in (
                ("session_key", profile_from_session_key(row.get("session_key"))),
                ("profile_name", normalize_profile(row.get("profile_name"))),
                ("origin_json.profile", profile_from_origin_json(row.get("origin_json"))),
            ):
                if profile is not None:
                    evidence.append(Evidence(source, profile))
            records.append(
                SessionRecord(
                    store=store,
                    session_id=sid,
                    row=row,
                    explicit_evidence=evidence,
                )
            )
        return records
    finally:
        conn.close()


def load_records(stores: Mapping[str, Store]) -> dict[tuple[str, str], SessionRecord]:
    records: dict[tuple[str, str], SessionRecord] = {}
    for store in stores.values():
        for record in _session_rows(store):
            records[(store.profile, record.session_id)] = record
    return records


def _resolve_explicit_target(record: SessionRecord) -> Optional[str]:
    profiles = {e.profile for e in record.explicit_evidence}
    if len(profiles) > 1:
        detail = ", ".join(f"{e.source}={e.profile}" for e in record.explicit_evidence)
        raise RepairBlocked(
            f"session {record.session_id!r} in {record.store.path} has conflicting "
            f"profile evidence: {detail}"
        )
    return next(iter(profiles), None)


def _component_records(records: Sequence[SessionRecord]) -> list[list[SessionRecord]]:
    by_id: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in records:
        by_id[record.session_id].append(record)

    # Build lineage only within one physical store.  Cross-store duplicate IDs
    # are handled separately; linking them here would falsely merge two copies.
    by_store_id = {(r.store.profile, r.session_id): r for r in records}
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        if not record.parent_id:
            continue
        parent_key = (record.store.profile, record.parent_id)
        child_key = (record.store.profile, record.session_id)
        if parent_key in by_store_id:
            adjacency[parent_key].add(child_key)
            adjacency[child_key].add(parent_key)

    seen: set[tuple[str, str]] = set()
    components: list[list[SessionRecord]] = []
    for key, record in by_store_id.items():
        if key in seen:
            continue
        queue = deque([key])
        seen.add(key)
        component: list[SessionRecord] = []
        while queue:
            current = queue.popleft()
            component.append(by_store_id[current])
            for neighbor in adjacency.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _schema_version(path: Path) -> tuple[int, int]:
    conn = _connect(path, read_only=True)
    try:
        return (
            int(conn.execute("PRAGMA application_id").fetchone()[0]),
            int(conn.execute("PRAGMA user_version").fetchone()[0]),
        )
    finally:
        conn.close()


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store_fingerprint(path: Path) -> dict[str, Any]:
    """Pin the exact SQLite bytes reviewed by the dry run.

    WAL mode may hold committed pages outside ``state.db``. Hash both the main
    file and ``-wal`` (when present); omit ``-shm`` because it is transient
    reader-coordination state rather than durable database content. A harmless
    checkpoint can therefore invalidate the reviewed fingerprint, which is an
    intentional fail-closed outcome: rerun the dry run against the new bytes.
    """

    wal = path.with_name(path.name + "-wal")
    return {
        "path": str(path),
        "state_db_sha256": _file_sha256(path),
        "state_db_size": path.stat().st_size if path.exists() else None,
        "wal_sha256": _file_sha256(wal),
        "wal_size": wal.stat().st_size if wal.exists() else None,
    }


def build_plan(hermes_home: Path) -> Plan:
    stores = discover_stores(hermes_home)
    blocked: list[str] = []
    notes: list[str] = []
    if DEFAULT_PROFILE not in stores:
        blocked.append(f"default store does not exist: {hermes_home / 'state.db'}")
        return Plan(hermes_home, stores, {}, [], blocked, notes)

    try:
        records = load_records(stores)
    except RepairBlocked as exc:
        return Plan(hermes_home, stores, {}, [], [str(exc)], notes)

    values = list(records.values())
    explicit: dict[tuple[str, str], Optional[str]] = {}
    for record in values:
        try:
            explicit[(record.store.profile, record.session_id)] = _resolve_explicit_target(record)
        except RepairBlocked as exc:
            blocked.append(str(exc))

    moves: list[Move] = []
    for component in _component_records(values):
        claims = {
            explicit[(r.store.profile, r.session_id)]
            for r in component
            if explicit.get((r.store.profile, r.session_id)) is not None
        }
        component_label = f"{component[0].store.profile}:{min(r.session_id for r in component)}"
        if len(claims) > 1:
            blocked.append(
                f"lineage component {component_label} has conflicting profile claims: "
                + ", ".join(sorted(str(c) for c in claims))
            )
            continue
        target = next(iter(claims), None)
        if target is None:
            # Keyless/local rows carry no gateway ownership evidence.  Keep them
            # in their physical store rather than guessing from the process.
            for record in component:
                record.target_profile = record.store.profile
            continue
        if target not in stores:
            blocked.append(
                f"lineage component {component_label} targets profile {target!r}, "
                f"but {hermes_home / 'profiles' / target / 'state.db'} does not exist"
            )
            continue
        source_profiles = {record.store.profile for record in component}
        if len(source_profiles) != 1:
            blocked.append(
                f"lineage component {component_label} spans physical stores unexpectedly"
            )
            continue
        source_profile = next(iter(source_profiles))
        for record in component:
            record.target_profile = target
        if source_profile != target:
            moves.append(
                Move(
                    source=stores[source_profile],
                    destination=stores[target],
                    session_ids=sorted(r.session_id for r in component),
                    component_id=component_label,
                )
            )

    # Duplicate session IDs across stores are allowed only when every row that
    # would be copied is byte-for-byte equal; apply mode then removes the
    # misplaced duplicate after destination verification.
    locations: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in values:
        locations[record.session_id].append(record)
    for sid, copies in sorted(locations.items()):
        if len(copies) <= 1:
            continue
        targets = {copy.target_profile or copy.store.profile for copy in copies}
        if len(targets) > 1:
            blocked.append(
                f"duplicate session id {sid!r} has conflicting target stores: "
                + ", ".join(sorted(str(v) for v in targets))
            )
        else:
            notes.append(
                f"duplicate session id {sid!r} exists in "
                + ", ".join(str(copy.store.path) for copy in copies)
                + "; row-level equality will be checked before apply"
            )

    # All moving stores must be on compatible schema revisions.  Detailed table
    # compatibility is checked again before each copy.
    for move in moves:
        try:
            src_version = _schema_version(move.source.path)
            dst_version = _schema_version(move.destination.path)
        except sqlite3.Error as exc:
            blocked.append(f"schema probe failed for {move.component_id}: {exc}")
            continue
        if src_version != dst_version:
            blocked.append(
                f"schema version mismatch for {move.component_id}: "
                f"{move.source.path}={src_version}, "
                f"{move.destination.path}={dst_version}"
            )

    return Plan(hermes_home, stores, records, moves, blocked, notes)


def _normal_tables(conn: sqlite3.Connection) -> list[str]:
    shadow_names: set[str] = set()
    try:
        for row in conn.execute("PRAGMA table_list"):
            # columns: schema, name, type, ncol, wr, strict
            if len(row) >= 3 and str(row[2]).lower() in {"shadow", "virtual"}:
                shadow_names.add(str(row[1]))
    except sqlite3.Error:
        pass
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    names: list[str] = []
    for name, sql in rows:
        name = str(name)
        if name in shadow_names:
            continue
        if isinstance(sql, str) and sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE"):
            continue
        names.append(name)
    return sorted(names)


def _table_plan(conn: sqlite3.Connection, table: str) -> Optional[TablePlan]:
    columns = _table_columns(conn, table)
    if not columns:
        return None
    refs: set[str] = {column for column in columns if column in SESSION_REFERENCE_NAMES}
    try:
        for fk in conn.execute(f"PRAGMA foreign_key_list({quote_ident(table)})"):
            # id, seq, table, from, to, on_update, on_delete, match
            if str(fk[2]) == "sessions" and str(fk[4]) == "id":
                refs.add(str(fk[3]))
    except sqlite3.Error:
        pass
    if not refs:
        return None
    return TablePlan(
        name=table,
        columns=columns,
        reference_columns=tuple(sorted(refs)),
        primary_key_columns=_table_primary_key_columns(conn, table),
        surrogate_pk_column=_surrogate_integer_pk(conn, table),
    )


def related_table_plans(conn: sqlite3.Connection) -> list[TablePlan]:
    plans: list[TablePlan] = []
    for table in _normal_tables(conn):
        if table == "sessions":
            continue
        plan = _table_plan(conn, table)
        if plan is not None:
            plans.append(plan)
    return plans


def _schema_signature(
    conn: sqlite3.Connection,
    table: str,
) -> tuple[tuple[str, str, int, Any, int], ...]:
    return tuple(
        (str(row[1]), str(row[2] or ""), int(row[3] or 0), row[4], int(row[5] or 0))
        for row in conn.execute(f"PRAGMA table_info({quote_ident(table)})")
    )


def _assert_table_compatible(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    table: str,
) -> None:
    if not _table_exists(destination, table):
        raise RepairBlocked(f"destination is missing table {table!r}")
    if _schema_signature(source, table) != _schema_signature(destination, table):
        raise RepairBlocked(f"table schema mismatch for {table!r}")


def _select_session_rows(
    conn: sqlite3.Connection, session_ids: Sequence[str]
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in session_ids)
    rows = conn.execute(
        f"SELECT * FROM sessions WHERE id IN ({placeholders})", tuple(session_ids)
    ).fetchall()
    result = [dict(row) for row in rows]
    found = {str(row["id"]) for row in result}
    missing = sorted(set(session_ids).difference(found))
    if missing:
        raise RepairBlocked("source sessions disappeared during repair: " + ", ".join(missing))
    return result


def _rows_for_plan(
    conn: sqlite3.Connection,
    plan: TablePlan,
    session_ids: Sequence[str],
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in session_ids)
    predicates = [f"{quote_ident(column)} IN ({placeholders})" for column in plan.reference_columns]
    params: list[str] = []
    for _ in plan.reference_columns:
        params.extend(session_ids)
    rows = conn.execute(
        f"SELECT * FROM {quote_ident(plan.name)} WHERE " + " OR ".join(predicates),
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def _external_session_refs(
    rows: Sequence[Mapping[str, Any]],
    plan: TablePlan,
    moving_ids: set[str],
) -> set[str]:
    external: set[str] = set()
    for row in rows:
        for column in plan.reference_columns:
            value = row.get(column)
            if value in (None, ""):
                continue
            text = str(value)
            if text not in moving_ids:
                external.add(text)
    return external


def _rows_equal(a: Mapping[str, Any], b: Mapping[str, Any], columns: Sequence[str]) -> bool:
    return all(a.get(column) == b.get(column) for column in columns)


def _existing_rows_by_pk(
    conn: sqlite3.Connection,
    plan: TablePlan,
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not plan.primary_key_columns or not rows:
        return {}
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in plan.primary_key_columns)
        predicates = " AND ".join(
            f"{quote_ident(column)} IS ?" for column in plan.primary_key_columns
        )
        found = conn.execute(
            f"SELECT * FROM {quote_ident(plan.name)} WHERE {predicates}", key
        ).fetchone()
        if found is not None:
            result[key] = dict(found)
    return result


def _identity_key(plan: TablePlan, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in plan.identity_columns)


def _find_identical_surrogate_row(
    conn: sqlite3.Connection,
    plan: TablePlan,
    row: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if plan.surrogate_pk_column is None:
        return None
    columns = plan.identity_columns
    predicates = " AND ".join(f"{quote_ident(column)} IS ?" for column in columns)
    found = conn.execute(
        f"SELECT * FROM {quote_ident(plan.name)} WHERE {predicates} LIMIT 1",
        tuple(row.get(column) for column in columns),
    ).fetchone()
    return dict(found) if found is not None else None


def _incoming_foreign_key_tables(
    conn: sqlite3.Connection, table: str
) -> list[tuple[str, str, str]]:
    incoming: list[tuple[str, str, str]] = []
    for candidate in _normal_tables(conn):
        if candidate == table:
            continue
        try:
            rows = conn.execute(
                f"PRAGMA foreign_key_list({quote_ident(candidate)})"
            ).fetchall()
        except sqlite3.Error:
            continue
        for fk in rows:
            if str(fk[2]) == table:
                incoming.append((candidate, str(fk[3]), str(fk[4])))
    return incoming


def _insert_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    if not rows:
        return 0, 0
    quoted = ",".join(quote_ident(column) for column in columns)
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO {quote_ident(table)} ({quoted}) VALUES ({placeholders})"
    inserted = 0
    skipped = 0
    for row in rows:
        values = tuple(row.get(column) for column in columns)
        try:
            conn.execute(sql, values)
            inserted += 1
        except sqlite3.IntegrityError:
            # Caller must have prevalidated any duplicate.  Treating an unknown
            # conflict as a skip would hide divergent data.
            raise
    return inserted, skipped


def _copy_system_prompts(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    session_rows: Sequence[Mapping[str, Any]],
) -> int:
    if not (
        _table_exists(source, "system_prompts")
        and _table_exists(destination, "system_prompts")
    ):
        return 0
    if "system_prompt_hash" not in _table_columns(source, "sessions"):
        return 0
    hashes = sorted(
        {
            str(row.get("system_prompt_hash"))
            for row in session_rows
            if row.get("system_prompt_hash")
        }
    )
    if not hashes:
        return 0
    _assert_table_compatible(source, destination, "system_prompts")
    columns = _table_columns(source, "system_prompts")
    pk = _table_primary_key_columns(source, "system_prompts")
    key_column = "hash" if "hash" in columns else (pk[0] if pk else None)
    if key_column is None:
        raise RepairBlocked("system_prompts has no stable key column")
    placeholders = ",".join("?" for _ in hashes)
    rows = [
        dict(row)
        for row in source.execute(
            f"SELECT * FROM system_prompts WHERE {quote_ident(key_column)} IN ({placeholders})",
            tuple(hashes),
        ).fetchall()
    ]
    inserted = 0
    for row in rows:
        existing = destination.execute(
            f"SELECT * FROM system_prompts WHERE {quote_ident(key_column)} IS ?",
            (row.get(key_column),),
        ).fetchone()
        if existing is not None:
            if not _rows_equal(row, dict(existing), columns):
                raise RepairBlocked(
                    f"conflicting system prompt {row.get(key_column)!r} in destination"
                )
            continue
        _insert_rows(destination, "system_prompts", columns, [row])
        inserted += 1
    return inserted


def _backup_database(path: Path, backup_dir: Path, stamp: str) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{path.parent.name or 'root'}-{path.name}.{stamp}.bak"
    source = _connect(path, read_only=True)
    target = sqlite3.connect(dest)
    try:
        source.backup(target)
        target.commit()
    finally:
        target.close()
        source.close()
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_id_from_gateway_routing_entry(value: object) -> Optional[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    session_id = parsed.get("session_id")
    return str(session_id) if session_id not in (None, "") else None


def _key_table_rows(
    conn: sqlite3.Connection,
    table: str,
    *,
    session_keys: Sequence[str],
    session_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    columns = _table_columns(conn, table)
    if "session_key" not in columns:
        raise RepairBlocked(f"{table!r} no longer has a session_key column")
    keys = tuple(key for key in session_keys if key)
    rows: list[dict[str, Any]] = []
    if keys:
        placeholders = ",".join("?" for _ in keys)
        rows.extend(
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM {quote_ident(table)} "
                f"WHERE session_key IN ({placeholders})",
                keys,
            ).fetchall()
        )
    if table == "gateway_routing" and "entry_json" in columns:
        wanted = set(session_ids)
        seen = {
            tuple(row.get(column) for column in _table_primary_key_columns(conn, table))
            for row in rows
        }
        for raw in conn.execute("SELECT * FROM gateway_routing").fetchall():
            row = dict(raw)
            if _session_id_from_gateway_routing_entry(row.get("entry_json")) not in wanted:
                continue
            key = tuple(
                row.get(column)
                for column in _table_primary_key_columns(conn, table)
            )
            if key not in seen:
                rows.append(row)
                seen.add(key)
    return rows


def _key_table_payloads(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    session_rows: Sequence[Mapping[str, Any]],
    session_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    session_keys = sorted(
        {str(row.get("session_key")) for row in session_rows if row.get("session_key")}
    )
    payloads: dict[str, dict[str, Any]] = {}
    for table in SESSION_KEY_TABLES:
        source_exists = _table_exists(source, table)
        destination_exists = _table_exists(destination, table)
        if source_exists != destination_exists:
            raise RepairBlocked(
                f"keyed routing table {table!r} exists in only one store"
            )
        if not source_exists:
            continue
        _assert_table_compatible(source, destination, table)
        columns = _table_columns(source, table)
        pk = _table_primary_key_columns(source, table)
        if not pk:
            raise RepairBlocked(f"keyed routing table {table!r} has no primary key")
        rows = _key_table_rows(
            source,
            table,
            session_keys=session_keys,
            session_ids=session_ids,
        )
        plan = TablePlan(
            name=table,
            columns=columns,
            reference_columns=("session_key",),
            primary_key_columns=pk,
        )
        existing = _existing_rows_by_pk(destination, plan, rows)
        for row in rows:
            key = tuple(row.get(column) for column in pk)
            found = existing.get(key)
            if found is not None and not _rows_equal(row, found, columns):
                raise RepairBlocked(
                    f"destination has conflicting {table} row for {key!r}"
                )
        payloads[table] = {"plan": plan, "rows": rows}
    return payloads


def _preflight_move(move: Move) -> dict[str, Any]:
    source = _connect(move.source.path, read_only=True)
    destination = _connect(move.destination.path, read_only=True)
    try:
        _assert_table_compatible(source, destination, "sessions")
        session_rows = _select_session_rows(source, move.session_ids)
        moving = set(move.session_ids)
        for row in session_rows:
            parent = row.get("parent_session_id")
            if parent not in (None, "") and str(parent) not in moving:
                exists = destination.execute(
                    "SELECT 1 FROM sessions WHERE id=?", (str(parent),)
                ).fetchone()
                if exists is None:
                    raise RepairBlocked(
                        f"session {row['id']!r} references parent {parent!r} outside "
                        f"component {move.component_id} and destination does not contain it"
                    )

        table_payloads: dict[str, list[dict[str, Any]]] = {}
        table_plans = related_table_plans(source)
        for table_plan in table_plans:
            _assert_table_compatible(source, destination, table_plan.name)
            rows = _rows_for_plan(source, table_plan, move.session_ids)
            external = _external_session_refs(rows, table_plan, moving)
            missing_external = {
                sid
                for sid in external
                if destination.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone()
                is None
            }
            if missing_external:
                raise RepairBlocked(
                    f"table {table_plan.name!r} has references outside component "
                    f"{move.component_id}: {', '.join(sorted(missing_external))}"
                )
            table_payloads[table_plan.name] = rows

        # Validate destination duplicates before any write.
        session_columns = _table_columns(source, "sessions")
        for row in session_rows:
            existing = destination.execute(
                "SELECT * FROM sessions WHERE id=?", (row["id"],)
            ).fetchone()
            if existing is not None and not _rows_equal(row, dict(existing), session_columns):
                raise RepairBlocked(
                    f"destination has conflicting session id {row['id']!r} for "
                    f"component {move.component_id}"
                )

        for table_plan in table_plans:
            rows = table_payloads[table_plan.name]
            if not rows:
                continue
            if not table_plan.primary_key_columns:
                # Without a stable identity the tool cannot distinguish an
                # idempotent rerun from a legitimate duplicate row.
                raise RepairBlocked(
                    f"table {table_plan.name!r} has session references but no primary key"
                )
            if table_plan.surrogate_pk_column is not None:
                incoming = _incoming_foreign_key_tables(source, table_plan.name)
                for row in rows:
                    identical = _find_identical_surrogate_row(
                        destination, table_plan, row
                    )
                    if identical is not None:
                        continue
                    pk_value = row.get(table_plan.surrogate_pk_column)
                    collision = destination.execute(
                        f"SELECT 1 FROM {quote_ident(table_plan.name)} "
                        f"WHERE {quote_ident(table_plan.surrogate_pk_column)} IS ?",
                        (pk_value,),
                    ).fetchone()
                    if collision is not None and incoming:
                        refs = ", ".join(
                            f"{name}.{column}->{target}"
                            for name, column, target in incoming
                        )
                        raise RepairBlocked(
                            f"surrogate primary-key collision in {table_plan.name!r} "
                            f"cannot be remapped because of incoming foreign keys: {refs}"
                        )
                continue
            existing_by_pk = _existing_rows_by_pk(destination, table_plan, rows)
            for row in rows:
                key = tuple(row.get(column) for column in table_plan.primary_key_columns)
                existing = existing_by_pk.get(key)
                if existing is not None and not _rows_equal(row, existing, table_plan.columns):
                    raise RepairBlocked(
                        f"destination has conflicting row in {table_plan.name!r} "
                        f"for primary key {key!r}"
                    )

        key_table_payloads = _key_table_payloads(
            source,
            destination,
            session_rows=session_rows,
            session_ids=move.session_ids,
        )

        return {
            "session_rows": session_rows,
            "session_columns": session_columns,
            "table_plans": table_plans,
            "table_payloads": table_payloads,
            "key_table_payloads": key_table_payloads,
        }
    finally:
        destination.close()
        source.close()


def _copy_move(move: Move, payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _connect(move.source.path, read_only=True)
    destination = _connect(move.destination.path, read_only=False)
    try:
        destination.execute("BEGIN IMMEDIATE")
        destination.execute("PRAGMA defer_foreign_keys = ON")
        prompts = _copy_system_prompts(source, destination, payload["session_rows"])

        session_inserted = 0
        for row in payload["session_rows"]:
            existing = destination.execute(
                "SELECT * FROM sessions WHERE id=?", (row["id"],)
            ).fetchone()
            if existing is not None:
                continue
            _insert_rows(destination, "sessions", payload["session_columns"], [row])
            session_inserted += 1

        related_inserted: dict[str, int] = {}
        for table_plan in payload["table_plans"]:
            rows = payload["table_payloads"][table_plan.name]
            inserted = 0
            if table_plan.surrogate_pk_column is not None:
                for row in rows:
                    if _find_identical_surrogate_row(destination, table_plan, row) is not None:
                        continue
                    pk = table_plan.surrogate_pk_column
                    collision = destination.execute(
                        f"SELECT 1 FROM {quote_ident(table_plan.name)} "
                        f"WHERE {quote_ident(pk)} IS ?",
                        (row.get(pk),),
                    ).fetchone()
                    if collision is None:
                        _insert_rows(
                            destination, table_plan.name, table_plan.columns, [row]
                        )
                    else:
                        remapped_columns = tuple(
                            column for column in table_plan.columns if column != pk
                        )
                        _insert_rows(
                            destination, table_plan.name, remapped_columns, [row]
                        )
                    inserted += 1
                related_inserted[table_plan.name] = inserted
                continue

            existing_by_pk = _existing_rows_by_pk(destination, table_plan, rows)
            new_rows = []
            for row in rows:
                key = tuple(row.get(column) for column in table_plan.primary_key_columns)
                if key in existing_by_pk:
                    continue
                new_rows.append(row)
            if new_rows:
                _insert_rows(destination, table_plan.name, table_plan.columns, new_rows)
            related_inserted[table_plan.name] = len(new_rows)

        keyed_inserted: dict[str, int] = {}
        for table, keyed in payload["key_table_payloads"].items():
            table_plan = keyed["plan"]
            rows = keyed["rows"]
            existing = _existing_rows_by_pk(destination, table_plan, rows)
            new_rows = []
            for row in rows:
                key = tuple(
                    row.get(column) for column in table_plan.primary_key_columns
                )
                if key not in existing:
                    new_rows.append(row)
            if new_rows:
                _insert_rows(
                    destination, table_plan.name, table_plan.columns, new_rows
                )
            keyed_inserted[table] = len(new_rows)

        destination.commit()
    except Exception:
        destination.rollback()
        raise
    finally:
        destination.close()
        source.close()

    return {
        "sessions_inserted": session_inserted,
        "system_prompts_inserted": prompts,
        "related_rows_inserted": related_inserted,
        "keyed_rows_inserted": keyed_inserted,
    }


def _verify_destination(move: Move, payload: Mapping[str, Any]) -> None:
    source = _connect(move.source.path, read_only=True)
    destination = _connect(move.destination.path, read_only=True)
    try:
        for row in payload["session_rows"]:
            found = destination.execute(
                "SELECT * FROM sessions WHERE id=?", (row["id"],)
            ).fetchone()
            if found is None or not _rows_equal(row, dict(found), payload["session_columns"]):
                raise RepairError(f"destination verification failed for session {row['id']!r}")
        for table_plan in payload["table_plans"]:
            rows = payload["table_payloads"][table_plan.name]
            if not rows:
                continue
            if table_plan.surrogate_pk_column is not None:
                for row in rows:
                    found = _find_identical_surrogate_row(
                        destination, table_plan, row
                    )
                    if found is None:
                        raise RepairError(
                            f"destination verification failed for "
                            f"{table_plan.name} surrogate row"
                        )
                continue
            existing = _existing_rows_by_pk(destination, table_plan, rows)
            for row in rows:
                key = tuple(row.get(column) for column in table_plan.primary_key_columns)
                found = existing.get(key)
                if found is None or not _rows_equal(row, found, table_plan.columns):
                    raise RepairError(
                        f"destination verification failed for {table_plan.name} {key!r}"
                    )
        for table, keyed in payload["key_table_payloads"].items():
            table_plan = keyed["plan"]
            rows = keyed["rows"]
            existing = _existing_rows_by_pk(destination, table_plan, rows)
            for row in rows:
                key = tuple(
                    row.get(column) for column in table_plan.primary_key_columns
                )
                found = existing.get(key)
                if found is None or not _rows_equal(row, found, table_plan.columns):
                    raise RepairError(
                        f"destination verification failed for {table} {key!r}"
                    )
    finally:
        destination.close()
        source.close()


def _delete_source(move: Move, payload: Mapping[str, Any]) -> dict[str, int]:
    source = _connect(move.source.path, read_only=False)
    try:
        source.execute("BEGIN IMMEDIATE")
        source.execute("PRAGMA defer_foreign_keys = ON")
        deleted: dict[str, int] = {}
        moving = tuple(move.session_ids)
        placeholders = ",".join("?" for _ in moving)

        # Delete routing/session-key metadata only after its destination copy
        # has been committed and verified.
        for table, keyed in payload["key_table_payloads"].items():
            table_plan = keyed["plan"]
            rows = keyed["rows"]
            count = 0
            for row in rows:
                predicates = " AND ".join(
                    f"{quote_ident(column)} IS ?"
                    for column in table_plan.primary_key_columns
                )
                values = tuple(
                    row.get(column) for column in table_plan.primary_key_columns
                )
                cursor = source.execute(
                    f"DELETE FROM {quote_ident(table)} WHERE {predicates}", values
                )
                count += max(0, int(cursor.rowcount or 0))
            deleted[table] = count

        # Delete leaf tables first. Tables are ordered by name for deterministic
        # receipts; FK cascades can make a later explicit delete a harmless zero.
        for table_plan in reversed(payload["table_plans"]):
            predicates = [
                f"{quote_ident(column)} IN ({placeholders})"
                for column in table_plan.reference_columns
            ]
            params: list[str] = []
            for _ in table_plan.reference_columns:
                params.extend(moving)
            cursor = source.execute(
                f"DELETE FROM {quote_ident(table_plan.name)} WHERE "
                + " OR ".join(predicates),
                tuple(params),
            )
            deleted[table_plan.name] = max(0, int(cursor.rowcount or 0))

        # Null a self-parent edge only when SQLite's sessions schema is not
        # deferred/self-FK friendly. All parents inside the component are being
        # deleted anyway; this avoids ordering dependency without affecting the
        # destination copy.
        if "parent_session_id" in payload["session_columns"]:
            source.execute(
                f"UPDATE sessions SET parent_session_id=NULL "
                f"WHERE id IN ({placeholders}) AND parent_session_id IN ({placeholders})",
                moving + moving,
            )
        cursor = source.execute(
            f"DELETE FROM sessions WHERE id IN ({placeholders})", moving
        )
        deleted["sessions"] = max(0, int(cursor.rowcount or 0))
        source.commit()
        return deleted
    except Exception:
        source.rollback()
        raise
    finally:
        source.close()


def plan_fingerprint(plan: Plan) -> str:
    """Stable SHA-256 of the exact reviewed move plan."""

    encoded = json.dumps(
        plan_to_dict(plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execute_plan(
    plan: Plan,
    *,
    apply: bool,
    backup_dir: Optional[Path] = None,
) -> dict[str, Any]:
    if not plan.safe:
        raise RepairBlocked("repair plan is blocked; inspect the manifest")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: dict[str, Any] = {
        "format": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dry_run": not apply,
        "safe": True,
        "hermes_home": str(plan.hermes_home),
        "plan_sha256": plan_fingerprint(plan),
        "moves": [],
        "blocked": list(plan.blocked),
        "notes": list(plan.notes),
    }

    # A dry run must prove table/schema compatibility too, not merely identify
    # candidate session IDs. Preflight every component before backups or writes
    # so the reviewed manifest and the apply gate describe the same operation.
    preflight: list[tuple[Move, dict[str, Any], dict[str, Any]]] = []
    for move in plan.moves:
        entry: dict[str, Any] = {
            "component": move.component_id,
            "source_profile": move.source.profile,
            "source_db": str(move.source.path),
            "destination_profile": move.destination.profile,
            "destination_db": str(move.destination.path),
            "session_ids": list(move.session_ids),
        }
        try:
            payload = _preflight_move(move)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            entry["status"] = "blocked"
            entry["error"] = reason
            manifest["blocked"].append(
                f"component {move.component_id} preflight failed: {reason}"
            )
            manifest["moves"].append(entry)
            continue

        entry["preflight"] = {
            "sessions": len(payload["session_rows"]),
            "related_rows": {
                table: len(rows)
                for table, rows in sorted(payload["table_payloads"].items())
            },
            "routing_rows": {
                table: len(value["rows"])
                for table, value in sorted(payload["key_table_payloads"].items())
            },
        }
        entry["status"] = "planned" if not apply else "preflight_passed"
        manifest["moves"].append(entry)
        preflight.append((move, payload, entry))

    if manifest["blocked"]:
        manifest["safe"] = False
        if apply:
            raise RepairBlocked(
                "one or more move components failed preflight; no database was modified"
            )
        return manifest

    if not apply:
        return manifest

    backup_dir = backup_dir or plan.hermes_home / "repair-backups" / f"88715-{stamp}"
    touched = sorted(
        {move.source.path for move in plan.moves}.union(
            move.destination.path for move in plan.moves
        )
    )
    backups: dict[Path, Path] = {}
    for path in touched:
        backups[path] = _backup_database(path, backup_dir, stamp)

    manifest["backups"] = {
        str(path): {
            "path": str(backup),
            "sha256": _sha256(backup),
        }
        for path, backup in backups.items()
    }

    for move, payload, entry in preflight:
        try:
            entry["copy"] = _copy_move(move, payload)
            _verify_destination(move, payload)
            entry["source_delete"] = _delete_source(move, payload)
            entry["status"] = "completed"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise

    manifest["database_sha256_after"] = {
        str(path): _sha256(path) for path in touched if path.exists()
    }
    return manifest


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "format": 1,
        "hermes_home": str(plan.hermes_home),
        "safe": plan.safe,
        "blocked": list(plan.blocked),
        "notes": list(plan.notes),
        "stores": {
            profile: _store_fingerprint(store.path)
            for profile, store in sorted(plan.stores.items())
        },
        "moves": [
            {
                "component": move.component_id,
                "source_profile": move.source.profile,
                "source_db": str(move.source.path),
                "destination_profile": move.destination.profile,
                "destination_db": str(move.destination.path),
                "session_ids": list(move.session_ids),
            }
            for move in plan.moves
        ],
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser(),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write the plan/result JSON here",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the repair; without this flag the command is read-only",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required with --apply to acknowledge the destructive source cleanup",
    )
    parser.add_argument(
        "--gateway-stopped",
        action="store_true",
        help="required with --apply; confirms every Hermes writer is stopped",
    )
    parser.add_argument(
        "--plan-sha256",
        help="required with --apply; must match the reviewed dry-run manifest",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="directory for SQLite backup snapshots (apply mode)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    home = args.hermes_home.expanduser().resolve()
    plan = build_plan(home)
    initial = plan_to_dict(plan)
    initial["plan_sha256"] = plan_fingerprint(plan)
    manifest_path = args.manifest or home / "repair-manifests" / (
        "88715-apply.json" if args.apply else "88715-dry-run.json"
    )

    apply_blocks: list[str] = []
    if args.apply and not args.yes:
        apply_blocks.append("--apply requires --yes")
    if args.apply and not args.gateway_stopped:
        apply_blocks.append("--apply requires --gateway-stopped")
    if args.apply and not args.plan_sha256:
        apply_blocks.append("--apply requires --plan-sha256 from the dry-run manifest")
    if (
        args.apply
        and args.plan_sha256
        and args.plan_sha256.lower() != initial["plan_sha256"]
    ):
        apply_blocks.append(
            "--plan-sha256 does not match the current plan; rerun dry-run review"
        )
    if apply_blocks:
        initial["blocked"] = list(initial["blocked"]) + apply_blocks
        initial["safe"] = False
        write_json(manifest_path, initial)
        print(json.dumps(initial, indent=2, sort_keys=True))
        return 2

    if not plan.safe:
        write_json(manifest_path, initial)
        print(json.dumps(initial, indent=2, sort_keys=True))
        return 2

    try:
        result = execute_plan(
            plan,
            apply=args.apply,
            backup_dir=args.backup_dir.expanduser().resolve()
            if args.backup_dir
            else None,
        )
    except Exception as exc:
        result = initial
        result["execution_error"] = f"{type(exc).__name__}: {exc}"
        result["safe"] = False
        write_json(manifest_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    write_json(manifest_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("blocked"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
