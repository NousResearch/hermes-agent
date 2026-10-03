"""Profile-scoped, read-only-observable leases for live delegated children."""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from hermes_cli.sqlite_util import open_db, transaction

# Ten one-second heartbeats tolerate ordinary scheduler pauses without hiding a live child.
_LEASE_SECONDS = 10.0
_READER_BUSY_TIMEOUT_MS = 200
# A roster writer must fail long before another profile's ten-second lease expires.
_WRITER_BUSY_TIMEOUT_MS = 200
_ALLOWLIST = (
    "subagent_id",
    "parent_id",
    "owner_agent_session_id",
    "delegation_id",
    "depth",
    "goal",
    "model",
    "started_at",
    "status",
    "tool_count",
    "last_tool",
)
_PERSISTED_KEYS = _ALLOWLIST + ("owner_pid", "lease_expires_at")
_schema_init_lock = threading.Lock()
_initialized_db_identities: set[tuple[str, int, int]] = set()


def _db_identity(path: Path) -> tuple[str, int, int] | None:
    """Stable identity for schema replay suppression; replaced DB files get a new identity."""
    try:
        stat = path.stat()
    except OSError:
        return None
    if not stat.st_dev or not stat.st_ino:
        return None
    return (str(path.resolve()), stat.st_dev, stat.st_ino)


@contextlib.contextmanager
def _record_profile_scope(record: dict[str, Any]) -> Iterator[None]:
    """Temporarily bind the record owner while projecting profile-specific redactions."""
    home = record.get("_roster_home")
    if not home:
        yield
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(Path(home).resolve())
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _redact(value: Any) -> str | None:
    if value is None:
        return None
    from agent.redact import redact_sensitive_text

    return (redact_sensitive_text(str(value), force=True) or "")[:500]


def _snapshot(record: dict[str, Any]) -> dict[str, Any] | None:
    """Project a local record under its owner profile before it reaches disk or observers."""
    with _record_profile_scope(record):
        subagent_id = str(record.get("subagent_id") or "")
        if not subagent_id:
            return None
        now = time.time()
        return {
            "subagent_id": subagent_id,
            "parent_id": _redact(record.get("parent_id")),
            "owner_agent_session_id": _redact(record.get("owner_agent_session_id")),
            "delegation_id": _redact(record.get("delegation_id")),
            "depth": int(record.get("depth") or 0),
            "goal": _redact(record.get("goal")),
            "model": _redact(record.get("model")),
            "started_at": float(record.get("started_at") or now),
            "status": "running",
            "tool_count": max(0, int(record.get("tool_count") or 0)),
            "last_tool": _redact(record.get("last_tool")),
            "owner_pid": os.getpid(),
            "lease_expires_at": now + _LEASE_SECONDS,
        }


def observable_snapshot(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the one allowlisted, redacted shape permitted to local roster observers."""
    snapshot = _snapshot(record)
    if snapshot is None:
        return None
    return {key: snapshot[key] for key in _ALLOWLIST}


def _initialize_schema_once(conn: sqlite3.Connection, path: Path) -> None:
    """Initialize only once per file identity; a replacement file is initialized again."""
    identity = _db_identity(path)
    with _schema_init_lock:
        if identity is not None and identity in _initialized_db_identities:
            return
        from tools.async_delegation import _initialize_schema

        _initialize_schema(conn)
        if (identity := _db_identity(path)) is not None:
            _initialized_db_identities.add(identity)


def _open_writer(path: Path) -> sqlite3.Connection:
    return open_db(
        path,
        db_label="state.db (delegation_roster)",
        busy_timeout_ms=_WRITER_BUSY_TIMEOUT_MS,
        wal=False,
        initialize=lambda conn: _initialize_schema_once(conn, path),
    )


def publish(record: dict[str, Any]) -> None:
    """Upsert one process-owned, sanitized live-child lease in its profile state DB."""
    publish_many([record])


def publish_many(records: list[dict[str, Any]]) -> None:
    """Upsert each profile's sanitized leases in one transaction per profile."""
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        snapshot = _snapshot(record)
        home = record.get("_roster_home")
        if snapshot is None or not home:
            continue
        grouped.setdefault(Path(home) / "state.db", []).append(snapshot)
    if not grouped:
        return
    from hermes_state import _secure_state_db_files

    first_error: Exception | None = None
    for path, snapshots in grouped.items():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _secure_state_db_files(path, create_main=True)
            conn = _open_writer(path)
            with transaction(conn) as db:
                db.execute(
                    "DELETE FROM delegation_live_subagents WHERE lease_expires_at < ?",
                    (time.time(),),
                )
                db.executemany(
                    """INSERT INTO delegation_live_subagents
                (subagent_id, parent_id, owner_agent_session_id, delegation_id, depth, goal, model,
                 started_at, status, tool_count, last_tool, owner_pid, lease_expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subagent_id) DO UPDATE SET
                  parent_id=excluded.parent_id,
                  owner_agent_session_id=excluded.owner_agent_session_id,
                  delegation_id=excluded.delegation_id,
                  depth=excluded.depth, goal=excluded.goal, model=excluded.model,
                  started_at=excluded.started_at, status=excluded.status,
                  tool_count=excluded.tool_count, last_tool=excluded.last_tool,
                  owner_pid=excluded.owner_pid, lease_expires_at=excluded.lease_expires_at""",
                    [
                        tuple(snapshot[key] for key in _PERSISTED_KEYS)
                        for snapshot in snapshots
                    ],
                )
        except Exception as exc:
            if first_error is None:
                first_error = exc
        finally:
            try:
                _secure_state_db_files(path)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error


def remove(record: dict[str, Any]) -> None:
    """Delete this process's completed child lease; a foreign process cannot remove it."""
    home, subagent_id = record.get("_roster_home"), str(record.get("subagent_id") or "")
    if not home or not subagent_id:
        return
    path = Path(home) / "state.db"
    if not path.exists():
        return
    from hermes_state import _secure_state_db_files

    try:
        _secure_state_db_files(path)
        conn = open_db(
            path,
            db_label="state.db (delegation_roster)",
            busy_timeout_ms=_WRITER_BUSY_TIMEOUT_MS,
            wal=False,
        )
        with transaction(conn) as db:
            db.execute(
                "DELETE FROM delegation_live_subagents WHERE lease_expires_at < ?",
                (time.time(),),
            )
            db.execute(
                "DELETE FROM delegation_live_subagents WHERE subagent_id=? AND owner_pid=?",
                (subagent_id, os.getpid()),
            )
    finally:
        _secure_state_db_files(path)


def list_live(home: Path) -> list[dict[str, Any]]:
    """Read unexpired projected leases only; a locked or malformed DB fails closed without mutation."""
    path = Path(home) / "state.db"
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=_READER_BUSY_TIMEOUT_MS / 1000,
        )
    except sqlite3.Error:
        return []
    try:
        conn.execute(f"PRAGMA busy_timeout={_READER_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            """SELECT subagent_id, parent_id, owner_agent_session_id, delegation_id, depth, goal, model,
                      started_at, status, tool_count, last_tool
               FROM delegation_live_subagents
               WHERE status='running' AND lease_expires_at >= ?
               ORDER BY started_at, subagent_id""",
            (time.time(),),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    # Persisted columns are already the owner-profile redacted allowlist.  Never re-run them through
    # redaction in the reader's profile context, which could leak a secondary vault secret on heartbeat.
    return [dict(zip(_ALLOWLIST, row)) for row in rows]
