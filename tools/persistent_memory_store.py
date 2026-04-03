#!/usr/bin/env python3
"""SQLite-backed persistent memory store with markdown export compatibility."""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

ENTRY_DELIMITER = "\n§\n"
DEFAULT_MEMORY_DIR = get_hermes_home() / "memories"
DEFAULT_DB_PATH = get_hermes_home() / "memory.db"


class PersistentMemoryStore:
    def __init__(
        self,
        db_path: Path | None = None,
        memory_dir: Path | None = None,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.memory_dir = Path(memory_dir or DEFAULT_MEMORY_DIR)
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    scope_value TEXT,
                    source TEXT NOT NULL DEFAULT 'manual',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_used_at REAL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    supersedes_id TEXT,
                    fingerprint TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_target_status ON memory_entries(target, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_kind_status ON memory_entries(kind, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_scope_status ON memory_entries(scope, scope_value, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_entries(updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_fingerprint ON memory_entries(fingerprint)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    detail TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _normalize_content(self, content: str) -> str:
        return " ".join(content.strip().split())

    def _fingerprint(self, target: str, content: str) -> str:
        normalized = f"{target}:{self._normalize_content(content).lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _write_event(self, conn: sqlite3.Connection, entry_id: Optional[str], action: str, target: str, detail: str = ""):
        conn.execute(
            "INSERT INTO memory_events(entry_id, action, target, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (entry_id, action, target, detail, time.time()),
        )

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    def list_entries(self, target: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memory_entries WHERE target = ?"
        params: list[Any] = [target]
        if not include_inactive:
            query += " AND status = 'active'"
        query += " ORDER BY updated_at DESC, created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _active_total_chars(self, conn: sqlite3.Connection, target: str, extra_content: str | None = None, replacing_id: str | None = None) -> int:
        rows = conn.execute(
            "SELECT id, content FROM memory_entries WHERE target = ? AND status = 'active' ORDER BY created_at ASC",
            (target,),
        ).fetchall()
        contents = []
        for row in rows:
            if replacing_id and row["id"] == replacing_id:
                continue
            contents.append(row["content"])
        if extra_content is not None:
            contents.append(extra_content)
        if not contents:
            return 0
        return len(ENTRY_DELIMITER.join(contents))

    def _export_markdown_locked(self, conn: sqlite3.Connection, target: str):
        rows = conn.execute(
            "SELECT content FROM memory_entries WHERE target = ? AND status = 'active' ORDER BY importance DESC, updated_at DESC",
            (target,),
        ).fetchall()
        content = ENTRY_DELIMITER.join(row["content"] for row in rows)
        path = self.memory_dir / ("USER.md" if target == "user" else "MEMORY.md")
        path.write_text(content, encoding="utf-8")

    def add_entry(
        self,
        target: str,
        content: str,
        *,
        kind: str = "lesson",
        scope: str = "global",
        scope_value: str | None = None,
        source: str = "manual",
        confidence: float = 1.0,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        fingerprint = self._fingerprint(target, content)
        with self._connect() as conn:
            duplicate = conn.execute(
                "SELECT * FROM memory_entries WHERE target = ? AND fingerprint = ? AND status = 'active'",
                (target, fingerprint),
            ).fetchone()
            if duplicate:
                return {
                    "success": True,
                    "message": "Entry already exists (no duplicate added).",
                    "entry": self._row_to_dict(duplicate),
                    "entries": self.list_entries(target),
                }
            total = self._active_total_chars(conn, target, extra_content=content)
            limit = self._char_limit(target)
            if total > limit:
                return {
                    "success": False,
                    "error": f"Memory at {self._active_total_chars(conn, target):,}/{limit:,} chars. Adding this entry ({len(content)} chars) would exceed the limit. Replace or remove existing entries first.",
                }
            now = time.time()
            entry_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO memory_entries(
                    id, target, kind, content, status, scope, scope_value, source,
                    confidence, importance, created_at, updated_at, last_used_at,
                    use_count, supersedes_id, fingerprint
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, NULL, 0, NULL, ?)
                """,
                (
                    entry_id, target, kind, content, scope, scope_value, source,
                    confidence, importance, now, now, fingerprint,
                ),
            )
            self._write_event(conn, entry_id, "add", target, content)
            self._export_markdown_locked(conn, target)
            row = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (entry_id,)).fetchone()
            conn.commit()
        return {
            "success": True,
            "message": "Entry added.",
            "entry": self._row_to_dict(row),
            "entries": self.list_entries(target),
        }

    def _find_active_matches(self, conn: sqlite3.Connection, target: str, substring: str) -> List[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM memory_entries WHERE target = ? AND status = 'active' AND content LIKE ? ORDER BY created_at ASC",
            (target, f"%{substring}%"),
        ).fetchall()

    def replace_entry(
        self,
        target: str,
        old_text: str,
        new_content: str,
        *,
        kind: str = "lesson",
        scope: str = "global",
        scope_value: str | None = None,
        source: str = "manual",
        confidence: float = 1.0,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}
        with self._connect() as conn:
            matches = self._find_active_matches(conn, target, old_text)
            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}
            unique_contents = {row["content"] for row in matches}
            if len(unique_contents) > 1:
                previews = [row["content"][:80] + ("..." if len(row["content"]) > 80 else "") for row in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}
            old_row = matches[0]
            limit = self._char_limit(target)
            total = self._active_total_chars(conn, target, extra_content=new_content, replacing_id=old_row["id"])
            if total > limit:
                return {"success": False, "error": f"Replacement would put memory at {total:,}/{limit:,} chars. Shorten the new content or remove other entries first."}
            now = time.time()
            conn.execute("UPDATE memory_entries SET status = 'superseded', updated_at = ? WHERE id = ?", (now, old_row["id"]))
            new_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO memory_entries(
                    id, target, kind, content, status, scope, scope_value, source,
                    confidence, importance, created_at, updated_at, last_used_at,
                    use_count, supersedes_id, fingerprint
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?)
                """,
                (
                    new_id, target, kind, new_content, scope, scope_value, source,
                    confidence, importance, now, now, old_row["id"], self._fingerprint(target, new_content),
                ),
            )
            self._write_event(conn, old_row["id"], "supersede", target, new_content)
            self._export_markdown_locked(conn, target)
            row = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (new_id,)).fetchone()
            conn.commit()
        return {"success": True, "message": "Entry replaced.", "entry": self._row_to_dict(row), "entries": self.list_entries(target)}

    def forget_entry(self, target: str, old_text: str) -> Dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        with self._connect() as conn:
            matches = self._find_active_matches(conn, target, old_text)
            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}
            unique_contents = {row["content"] for row in matches}
            if len(unique_contents) > 1:
                previews = [row["content"][:80] + ("..." if len(row["content"]) > 80 else "") for row in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}
            row = matches[0]
            now = time.time()
            conn.execute("UPDATE memory_entries SET status = 'forgotten', updated_at = ? WHERE id = ?", (now, row["id"]))
            self._write_event(conn, row["id"], "forget", target, row["content"])
            self._export_markdown_locked(conn, target)
            conn.commit()
        return {"success": True, "message": "Entry removed.", "entries": self.list_entries(target)}

    def retrieve_for_prompt(self, target: str) -> List[Dict[str, Any]]:
        target_bias = 0.2 if target == "user" else 0.0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_entries WHERE target = ? AND status = 'active'",
                (target,),
            ).fetchall()
        items = []
        now = time.time()
        for row in rows:
            age_days = max(0.0, (now - row["updated_at"]) / 86400.0)
            recency = max(0.0, 0.3 - min(age_days / 365.0, 0.3))
            kind_bonus = 0.0
            if row["kind"] in {"preference", "instruction", "constraint", "identity"}:
                kind_bonus = 0.25
            score = float(row["importance"] or 0) + target_bias + kind_bonus + recency + 0.05 * int(row["use_count"] or 0)
            item = self._row_to_dict(row)
            item["_score"] = score
            items.append(item)
        items.sort(key=lambda x: (x["_score"], x["updated_at"]), reverse=True)
        return items

    def render_prompt_block(self, target: str, char_limit: Optional[int] = None) -> Optional[str]:
        limit = char_limit or self._char_limit(target)
        entries = self.retrieve_for_prompt(target)
        if not entries:
            return None
        separator = "═" * 46
        header_name = "USER PROFILE (who the user is)" if target == "user" else "MEMORY (your personal notes)"
        selected: List[str] = []
        body = ""
        for entry in entries:
            candidate_entries = selected + [entry["content"]]
            candidate_body = ENTRY_DELIMITER.join(candidate_entries)
            header = f"{separator}\n{header_name}\n{separator}\n"
            candidate = header + candidate_body
            if len(candidate) <= limit:
                selected.append(entry["content"])
                body = candidate_body
        if not selected:
            # Keep at least the top memory, clipped to fit.
            top = entries[0]["content"]
            header = f"{separator}\n{header_name}\n{separator}\n"
            remaining = max(0, limit - len(header))
            body = top[:remaining]
        return f"{separator}\n{header_name}\n{separator}\n{body}"
