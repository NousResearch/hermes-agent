from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from hermes_constants import get_hermes_home


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root TEXT NOT NULL UNIQUE,
    root_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    indexed_at REAL NOT NULL,
    UNIQUE(repo_id, path)
);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT,
    docstring TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    source_kind TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    files_seen INTEGER NOT NULL DEFAULT 0,
    files_indexed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_repo_path ON files(repo_id, path);
CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols(repo_id, name);
CREATE INDEX IF NOT EXISTS idx_symbols_repo_qualname ON symbols(repo_id, qualname);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(repo_id, file_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(repo_id, source_kind, source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(repo_id, target_kind, target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(repo_id, file_id);
"""


def root_hash(root: Path) -> str:
    return hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:24]


def cache_path_for_root(root: Path) -> Path:
    return get_hermes_home() / "cache" / "code_graph" / f"{root_hash(root)}.sqlite"


class CodeGraphStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def table_names(self) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {row["name"] for row in rows}

    def ensure_repo(self, root: Path) -> int:
        root_text = str(Path(root).resolve())
        repo_hash = root_hash(Path(root))
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO repos(root, root_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(root) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (root_text, repo_hash, now, now),
            )
            row = conn.execute(
                "SELECT id FROM repos WHERE root = ?",
                (root_text,),
            ).fetchone()
        return int(row["id"])

    def get_repo_id(self, root: Path) -> int | None:
        if not self.db_path.exists():
            return None
        root_text = str(Path(root).resolve())
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM repos WHERE root = ?",
                (root_text,),
            ).fetchone()
        return int(row["id"]) if row else None

    def clear_repo_index(self, repo_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM symbols WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM edges WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))

