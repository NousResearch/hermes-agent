"""Canonical Facebook SQLite location and connection factory."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def canonical_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "facebook_crm.db"


def connect(
    path: str | Path | None = None,
    *,
    readonly: bool = False,
    timeout: float = 30.0,
) -> sqlite3.Connection:
    db_path = Path(path) if path is not None else canonical_db_path()
    if readonly:
        connection = sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=timeout,
        )
    else:
        connection = sqlite3.connect(str(db_path), timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    if not readonly:
        # All runtime containers mount the same database directory, so WAL and
        # SHM sidecars are shared along with the canonical database file.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection
