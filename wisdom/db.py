"""SQLite storage for Hermes Wisdom Kernel."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from wisdom.models import (
    ApplicationRecord,
    CaptureRecord,
    Classification,
    InterpretationRecord,
    StatusSnapshot,
    WisdomConfig,
)


SCHEMA_VERSION = 2


class WisdomDB:
    def __init__(self, db_path: str | Path, *, force_no_fts: bool = False):
        self.path = Path(db_path).expanduser()
        self.force_no_fts = force_no_fts
        self._conn: sqlite3.Connection | None = None
        self._fts_failed = False

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "WisdomDB":
        self.init()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            _enable_wal_with_fallback(self._conn)
        return self._conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.conn
        with self.transaction():
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version(version) VALUES (0)")
                version = 0
            else:
                version = int(row["version"])
            stored_version = version
            if version < 1:
                self._migrate_v1(conn)
                version = 1
            if version < 2:
                self._migrate_v2(conn)
                version = 2
            if stored_version < SCHEMA_VERSION:
                conn.execute("DELETE FROM schema_version")
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        self._ensure_fts()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _migrate_v1(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_events (
                id INTEGER PRIMARY KEY,
                created_at REAL NOT NULL,
                channel TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                session_key_hash TEXT,
                message_ref_hash TEXT,
                original_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                processing_state TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS captures (
                id INTEGER PRIMARY KEY,
                raw_event_id INTEGER NOT NULL REFERENCES raw_events(id),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                title TEXT NOT NULL,
                cleaned_text TEXT,
                category TEXT NOT NULL,
                source_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance_score REAL,
                novelty_score REAL,
                actionability_score REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS interpretations (
                id INTEGER PRIMARY KEY,
                capture_id INTEGER NOT NULL REFERENCES captures(id),
                created_at REAL NOT NULL,
                summary TEXT NOT NULL,
                insight TEXT,
                why_it_matters TEXT,
                possible_application TEXT,
                counterpoint TEXT,
                confidence REAL NOT NULL,
                method TEXT NOT NULL,
                model_used TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY,
                capture_id INTEGER NOT NULL REFERENCES captures(id),
                created_at REAL NOT NULL,
                application_type TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_captures_status_created
                ON captures(status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_raw_events_created
                ON raw_events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_interpretations_capture
                ON interpretations(capture_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_applications_capture
                ON applications(capture_id, created_at DESC);
            """
        )

    def _migrate_v2(self, conn: sqlite3.Connection) -> None:
        _add_column_if_missing(conn, "captures", "review_status", "TEXT NOT NULL DEFAULT 'unreviewed'")
        _add_column_if_missing(conn, "captures", "reviewed_at", "REAL")
        _add_column_if_missing(conn, "captures", "accepted_at", "REAL")
        _add_column_if_missing(conn, "captures", "dismissed_at", "REAL")
        _add_column_if_missing(conn, "captures", "applied_at", "REAL")
        conn.execute(
            """
            UPDATE captures
            SET review_status = 'archived'
            WHERE status = 'archived' AND review_status != 'archived'
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_captures_review_status
                ON captures(review_status, created_at DESC)
            """
        )

    def _ensure_fts(self) -> bool:
        if self.force_no_fts or self._fts_failed:
            return False
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS wisdom_fts USING fts5(
                    capture_id UNINDEXED,
                    original_text,
                    cleaned_text,
                    title,
                    interpretation_text,
                    application_text
                )
                """
            )
            self.conn.commit()
            return True
        except sqlite3.Error:
            self._fts_failed = True
            return False

    def fts_available(self) -> bool:
        if self.force_no_fts or self._fts_failed:
            return False
        self.init_if_needed()
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='wisdom_fts'"
        ).fetchone()
        return row is not None

    def init_if_needed(self) -> None:
        if self._conn is None or not self.path.exists():
            self.init()

    def create_capture(
        self,
        *,
        original_text: str,
        cleaned_text: str | None,
        classification: Classification,
        channel: str,
        source_kind: str,
        session_key_hash: str | None,
        message_ref_hash: str | None,
        raw_metadata: dict[str, Any] | None = None,
        capture_metadata: dict[str, Any] | None = None,
    ) -> CaptureRecord:
        self.init()
        now = time.time()
        with self.transaction() as conn:
            raw_cur = conn.execute(
                """
                INSERT INTO raw_events (
                    created_at, channel, source_kind, session_key_hash, message_ref_hash,
                    original_text, metadata_json, processing_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    channel,
                    source_kind,
                    session_key_hash,
                    message_ref_hash,
                    original_text,
                    _json(raw_metadata or {}),
                    "captured",
                ),
            )
            raw_id = int(raw_cur.lastrowid)
            cap_cur = conn.execute(
                """
                INSERT INTO captures (
                    raw_event_id, created_at, updated_at, title, cleaned_text, category,
                    source_type, status, confidence, importance_score, novelty_score,
                    actionability_score, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw_id,
                    now,
                    now,
                    classification.title,
                    cleaned_text,
                    classification.category,
                    classification.source_type,
                    "active",
                    classification.confidence,
                    classification.importance_score,
                    classification.novelty_score,
                    classification.actionability_score,
                    _json(capture_metadata or {}),
                ),
            )
            capture_id = int(cap_cur.lastrowid)
            self._refresh_fts_unlocked(conn, capture_id)
        record = self.get_capture(capture_id)
        if record is None:
            raise RuntimeError("capture was inserted but could not be read")
        return record

    def get_capture(self, capture_id: int) -> CaptureRecord | None:
        self.init()
        row = self.conn.execute(
            """
            SELECT c.*, r.original_text
            FROM captures c
            JOIN raw_events r ON r.id = c.raw_event_id
            WHERE c.id = ?
            """,
            (capture_id,),
        ).fetchone()
        return _capture_from_row(row) if row else None

    def list_captures(self, *, limit: int = 5, include_archived: bool = False) -> list[CaptureRecord]:
        self.init()
        if include_archived:
            rows = self.conn.execute(
                """
                SELECT c.*, r.original_text
                FROM captures c JOIN raw_events r ON r.id = c.raw_event_id
                ORDER BY c.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT c.*, r.original_text
                FROM captures c JOIN raw_events r ON r.id = c.raw_event_id
                WHERE c.status != 'archived'
                ORDER BY c.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_capture_from_row(row) for row in rows]

    def archive_capture(self, capture_id: int) -> bool:
        self.init()
        now = time.time()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE captures
                SET status = 'archived',
                    review_status = 'archived',
                    updated_at = ?,
                    reviewed_at = COALESCE(reviewed_at, ?)
                WHERE id = ?
                """,
                (now, now, capture_id),
            )
            changed = cur.rowcount > 0
            if changed:
                self._refresh_fts_unlocked(conn, capture_id)
        return changed

    def set_review_status(self, capture_id: int, review_status: str) -> bool:
        self.init()
        if review_status not in {"unreviewed", "reviewed", "accepted", "dismissed", "applied", "archived"}:
            return False
        now = time.time()
        timestamp_updates = {
            "reviewed": "reviewed_at = COALESCE(reviewed_at, ?)",
            "accepted": "reviewed_at = COALESCE(reviewed_at, ?), accepted_at = ?",
            "dismissed": "reviewed_at = COALESCE(reviewed_at, ?), dismissed_at = ?",
            "applied": "reviewed_at = COALESCE(reviewed_at, ?), applied_at = ?",
            "archived": "reviewed_at = COALESCE(reviewed_at, ?)",
            "unreviewed": "reviewed_at = NULL, accepted_at = NULL, dismissed_at = NULL, applied_at = NULL",
        }
        status_clause = ", status = 'archived'" if review_status == "archived" else ""
        timestamp_clause = timestamp_updates[review_status]
        if review_status == "unreviewed":
            params: tuple[Any, ...] = (review_status, now, capture_id)
        elif review_status in {"reviewed", "archived"}:
            params = (review_status, now, now, capture_id)
        else:
            params = (review_status, now, now, now, capture_id)
        with self.transaction() as conn:
            cur = conn.execute(
                f"""
                UPDATE captures
                SET review_status = ?,
                    updated_at = ?,
                    {timestamp_clause}
                    {status_clause}
                WHERE id = ?
                """,
                params,
            )
            changed = cur.rowcount > 0
            if changed:
                self._refresh_fts_unlocked(conn, capture_id)
        return changed

    def insert_interpretation(
        self,
        *,
        capture_id: int,
        summary: str,
        insight: str | None,
        why_it_matters: str | None,
        possible_application: str | None,
        counterpoint: str | None,
        confidence: float,
        method: str,
        model_used: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InterpretationRecord:
        self.init()
        now = time.time()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO interpretations (
                    capture_id, created_at, summary, insight, why_it_matters,
                    possible_application, counterpoint, confidence, method, model_used,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    now,
                    summary,
                    insight,
                    why_it_matters,
                    possible_application,
                    counterpoint,
                    confidence,
                    method,
                    model_used,
                    _json(metadata or {}),
                ),
            )
            interpretation_id = int(cur.lastrowid)
            self._refresh_fts_unlocked(conn, capture_id)
        record = self.get_interpretation(capture_id)
        if record is None or record.id != interpretation_id:
            raise RuntimeError("interpretation was inserted but could not be read")
        return record

    def get_interpretation(self, capture_id: int) -> InterpretationRecord | None:
        self.init()
        row = self.conn.execute(
            """
            SELECT * FROM interpretations
            WHERE capture_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (capture_id,),
        ).fetchone()
        return _interpretation_from_row(row) if row else None

    def insert_applications(
        self,
        *,
        capture_id: int,
        applications: list[dict[str, Any]],
    ) -> list[ApplicationRecord]:
        self.init()
        now = time.time()
        ids: list[int] = []
        with self.transaction() as conn:
            for app in applications:
                cur = conn.execute(
                    """
                    INSERT INTO applications (
                        capture_id, created_at, application_type, title, body, status,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capture_id,
                        now,
                        app["application_type"],
                        app["title"],
                        app["body"],
                        app.get("status", "proposed"),
                        _json(app.get("metadata") or {}),
                    ),
                )
                ids.append(int(cur.lastrowid))
            self._refresh_fts_unlocked(conn, capture_id)
        return self.list_applications(capture_id, ids=ids)

    def list_applications(self, capture_id: int, *, ids: list[int] | None = None) -> list[ApplicationRecord]:
        self.init()
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = self.conn.execute(
                f"SELECT * FROM applications WHERE id IN ({placeholders}) ORDER BY id",
                tuple(ids),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM applications
                WHERE capture_id = ?
                ORDER BY id
                """,
                (capture_id,),
            ).fetchall()
        return [_application_from_row(row) for row in rows]

    def search(self, query: str, *, limit: int = 5) -> list[CaptureRecord]:
        self.init()
        query = query.strip()
        if not query:
            return []
        if self.fts_available():
            try:
                terms = _fts_query(query)
                if terms:
                    rows = self.conn.execute(
                        """
                        SELECT c.*, r.original_text
                        FROM wisdom_fts f
                        JOIN captures c ON c.id = f.capture_id
                        JOIN raw_events r ON r.id = c.raw_event_id
                        WHERE wisdom_fts MATCH ? AND c.status != 'archived'
                        GROUP BY c.id
                        ORDER BY bm25(wisdom_fts), c.created_at DESC
                        LIMIT ?
                        """,
                        (terms, limit),
                    ).fetchall()
                    if rows:
                        return [_capture_from_row(row) for row in rows]
            except sqlite3.Error:
                pass
        return self._like_search(query, limit=limit)

    def _like_search(self, query: str, *, limit: int) -> list[CaptureRecord]:
        pattern = f"%{_escape_like(query)}%"
        rows = self.conn.execute(
            """
            SELECT c.*, r.original_text
            FROM captures c
            JOIN raw_events r ON r.id = c.raw_event_id
            LEFT JOIN (
                SELECT capture_id,
                       group_concat(summary || ' ' || IFNULL(insight, '') || ' ' ||
                                    IFNULL(why_it_matters, '') || ' ' ||
                                    IFNULL(possible_application, '') || ' ' ||
                                    IFNULL(counterpoint, ''), ' ') AS interpretation_text
                FROM interpretations GROUP BY capture_id
            ) i ON i.capture_id = c.id
            LEFT JOIN (
                SELECT capture_id, group_concat(title || ' ' || body, ' ') AS application_text
                FROM applications GROUP BY capture_id
            ) a ON a.capture_id = c.id
            WHERE c.status != 'archived'
              AND (
                r.original_text LIKE ? ESCAPE '\\'
                OR IFNULL(c.cleaned_text, '') LIKE ? ESCAPE '\\'
                OR c.title LIKE ? ESCAPE '\\'
                OR IFNULL(i.interpretation_text, '') LIKE ? ESCAPE '\\'
                OR IFNULL(a.application_text, '') LIKE ? ESCAPE '\\'
              )
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, pattern, limit),
        ).fetchall()
        return [_capture_from_row(row) for row in rows]

    def set_setting(self, key: str, value: str) -> None:
        self.init()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, time.time()),
            )

    def get_setting(self, key: str) -> str | None:
        self.init()
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def status_snapshot(self, config: WisdomConfig) -> StatusSnapshot:
        self.init()
        enabled_setting = self.get_setting("enabled")
        capture_mode_setting = self.get_setting("capture_mode")
        enabled = config.enabled if enabled_setting is None else enabled_setting.lower() in {"1", "true", "yes", "on"}
        capture_mode = capture_mode_setting or config.capture_mode
        return StatusSnapshot(
            enabled=enabled,
            capture_mode=capture_mode,
            db_path=self.path,
            fts_available=self.fts_available(),
            counts=self.counts(),
            last_capture_at=self.last_capture_at(),
        )

    def counts(self) -> dict[str, int]:
        self.init()
        tables = ("raw_events", "captures", "interpretations", "applications")
        result = {}
        for table in tables:
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"]) if row else 0
        return result

    def count_by_category(self) -> dict[str, int]:
        self.init()
        rows = self.conn.execute(
            """
            SELECT category, COUNT(*) AS n
            FROM captures
            WHERE status != 'archived'
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
        return {str(row["category"]): int(row["n"]) for row in rows}

    def unapplied_captures(self, *, limit: int = 5) -> list[CaptureRecord]:
        self.init()
        rows = self.conn.execute(
            """
            SELECT c.*, r.original_text
            FROM captures c
            JOIN raw_events r ON r.id = c.raw_event_id
            LEFT JOIN applications a ON a.capture_id = c.id
            WHERE c.status != 'archived' AND a.id IS NULL
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_capture_from_row(row) for row in rows]

    def last_capture_at(self) -> float | None:
        self.init()
        row = self.conn.execute("SELECT MAX(created_at) AS ts FROM captures").fetchone()
        return float(row["ts"]) if row and row["ts"] is not None else None

    def _refresh_fts_unlocked(self, conn: sqlite3.Connection, capture_id: int) -> None:
        if self.force_no_fts or self._fts_failed:
            return
        try:
            if not _fts_table_exists(conn):
                return
            row = conn.execute(
                """
                SELECT c.id AS capture_id, c.cleaned_text, c.title, r.original_text,
                       (
                           SELECT group_concat(summary || ' ' || IFNULL(insight, '') || ' ' ||
                                               IFNULL(why_it_matters, '') || ' ' ||
                                               IFNULL(possible_application, '') || ' ' ||
                                               IFNULL(counterpoint, ''), ' ')
                           FROM interpretations WHERE capture_id = c.id
                       ) AS interpretation_text,
                       (
                           SELECT group_concat(title || ' ' || body, ' ')
                           FROM applications WHERE capture_id = c.id
                       ) AS application_text
                FROM captures c
                JOIN raw_events r ON r.id = c.raw_event_id
                WHERE c.id = ?
                """,
                (capture_id,),
            ).fetchone()
            conn.execute("DELETE FROM wisdom_fts WHERE capture_id = ?", (capture_id,))
            if row:
                conn.execute(
                    """
                    INSERT INTO wisdom_fts (
                        capture_id, original_text, cleaned_text, title,
                        interpretation_text, application_text
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["capture_id"],
                        row["original_text"] or "",
                        row["cleaned_text"] or "",
                        row["title"] or "",
                        row["interpretation_text"] or "",
                        row["application_text"] or "",
                    ),
                )
        except sqlite3.Error:
            self._fts_failed = True


def _enable_wal_with_fallback(conn: sqlite3.Connection) -> str:
    try:
        row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        mode = str(row[0]).lower() if row and row[0] else ""
        if mode == "wal":
            conn.execute("PRAGMA synchronous=NORMAL")
            return "wal"
    except sqlite3.Error:
        pass
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.Error:
        pass
    return "delete"


def _fts_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='wisdom_fts'"
    ).fetchone()
    return row is not None


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _capture_from_row(row: sqlite3.Row) -> CaptureRecord:
    return CaptureRecord(
        id=int(row["id"]),
        raw_event_id=int(row["raw_event_id"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        title=str(row["title"]),
        original_text=str(row["original_text"]),
        cleaned_text=row["cleaned_text"],
        category=row["category"],
        source_type=row["source_type"],
        status=row["status"],
        review_status=_row_value(row, "review_status", "unreviewed"),
        reviewed_at=_row_value(row, "reviewed_at"),
        accepted_at=_row_value(row, "accepted_at"),
        dismissed_at=_row_value(row, "dismissed_at"),
        applied_at=_row_value(row, "applied_at"),
        confidence=float(row["confidence"]),
        importance_score=row["importance_score"],
        novelty_score=row["novelty_score"],
        actionability_score=row["actionability_score"],
        metadata=_loads(row["metadata_json"]),
    )


def _interpretation_from_row(row: sqlite3.Row) -> InterpretationRecord:
    return InterpretationRecord(
        id=int(row["id"]),
        capture_id=int(row["capture_id"]),
        created_at=float(row["created_at"]),
        summary=str(row["summary"]),
        insight=row["insight"],
        why_it_matters=row["why_it_matters"],
        possible_application=row["possible_application"],
        counterpoint=row["counterpoint"],
        confidence=float(row["confidence"]),
        method=str(row["method"]),
        model_used=row["model_used"],
        metadata=_loads(row["metadata_json"]),
    )


def _application_from_row(row: sqlite3.Row) -> ApplicationRecord:
    return ApplicationRecord(
        id=int(row["id"]),
        capture_id=int(row["capture_id"]),
        created_at=float(row["created_at"]),
        application_type=row["application_type"],
        title=str(row["title"]),
        body=str(row["body"]),
        status=row["status"],
        metadata=_loads(row["metadata_json"]),
    )


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _fts_query(value: str) -> str:
    terms = re.findall(r"[\w']+", value, flags=re.UNICODE)
    return " ".join(terms[:20])
