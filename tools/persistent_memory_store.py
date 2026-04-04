#!/usr/bin/env python3
"""SQLite-backed persistent memory store with markdown export compatibility."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

ENTRY_DELIMITER = "\n§\n"
DEFAULT_MEMORY_DIR = get_hermes_home() / "memories"
DEFAULT_DB_PATH = get_hermes_home() / "memory.db"


def get_default_memory_dir() -> Path:
    """Return the live profile-scoped memories directory."""
    return get_hermes_home() / "memories"


def get_default_db_path() -> Path:
    """Return the live profile-scoped SQLite memory database path."""
    return get_hermes_home() / "memory.db"


class PersistentMemoryStore:
    _ENTRY_TYPE_BONUS = {
        "prohibition": 0.45,
        "user_preference": 0.28,
        "workflow_rule": 0.24,
        "project_convention": 0.22,
        "constraint": 0.22,
        "instruction": 0.20,
        "user_identity": 0.12,
        "identity": 0.12,
        "environment_fact": 0.08,
        "project": 0.08,
    }

    _STRENGTH_BONUS = {
        "hard_rule": 0.65,
        "strong_pref": 0.24,
        "soft_pref": 0.10,
        "contextual": 0.0,
    }

    _SOURCE_BONUS = {
        "user_explicit": 0.12,
        "user_implicit": 0.05,
        "manual": 0.0,
        "migration": -0.02,
        "imported": -0.02,
        "agent_learned": -0.03,
    }

    def __init__(
        self,
        db_path: Path | None = None,
        memory_dir: Path | None = None,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        resolved_db_path = db_path
        if resolved_db_path is None:
            resolved_db_path = get_default_db_path()

        resolved_memory_dir = memory_dir
        if resolved_memory_dir is None:
            resolved_memory_dir = get_default_memory_dir()

        self.db_path = Path(resolved_db_path)
        self.memory_dir = Path(resolved_memory_dir)
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
                    strength TEXT NOT NULL DEFAULT 'contextual',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_used_at REAL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    supersedes_id TEXT,
                    created_in_session_id TEXT,
                    replaced_by TEXT,
                    forgotten_by TEXT,
                    fingerprint TEXT NOT NULL
                )
                """
            )
            self._ensure_memory_entry_columns(conn)
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

    def _ensure_memory_entry_columns(self, conn: sqlite3.Connection):
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_entries)").fetchall()}
        required_columns = {
            "strength": "TEXT NOT NULL DEFAULT 'contextual'",
            "created_in_session_id": "TEXT",
            "replaced_by": "TEXT",
            "forgotten_by": "TEXT",
        }
        for name, ddl in required_columns.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE memory_entries ADD COLUMN {name} {ddl}")

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
        data = dict(row)
        data.setdefault("strength", "contextual")
        data.setdefault("created_in_session_id", None)
        data.setdefault("replaced_by", None)
        data.setdefault("forgotten_by", None)
        data.setdefault("entry_type", data.get("kind", "lesson"))
        return data

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
        entry_type: str | None = None,
        scope: str = "global",
        scope_value: str | None = None,
        source: str = "manual",
        strength: str = "contextual",
        confidence: float = 1.0,
        importance: float = 0.5,
        created_in_session_id: str | None = None,
    ) -> Dict[str, Any]:
        kind = entry_type or kind
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
                    strength, confidence, importance, created_at, updated_at, last_used_at,
                    use_count, supersedes_id, created_in_session_id, replaced_by, forgotten_by, fingerprint
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, NULL, ?, NULL, NULL, ?)
                """,
                (
                    entry_id, target, kind, content, scope, scope_value, source,
                    strength, confidence, importance, now, now, created_in_session_id, fingerprint,
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
        entry_type: str | None = None,
        scope: str = "global",
        scope_value: str | None = None,
        source: str = "manual",
        strength: str = "contextual",
        confidence: float = 1.0,
        importance: float = 0.5,
        created_in_session_id: str | None = None,
    ) -> Dict[str, Any]:
        kind = entry_type or kind
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
            conn.execute(
                "UPDATE memory_entries SET status = 'superseded', updated_at = ?, replaced_by = ? WHERE id = ?",
                (now, new_id := str(uuid.uuid4()), old_row["id"]),
            )
            conn.execute(
                """
                INSERT INTO memory_entries(
                    id, target, kind, content, status, scope, scope_value, source,
                    strength, confidence, importance, created_at, updated_at, last_used_at,
                    use_count, supersedes_id, created_in_session_id, replaced_by, forgotten_by, fingerprint
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, NULL, NULL, ?)
                """,
                (
                    new_id, target, kind, new_content, scope, scope_value, source,
                    strength, confidence, importance, now, now, old_row["id"], created_in_session_id,
                    self._fingerprint(target, new_content),
                ),
            )
            self._write_event(conn, old_row["id"], "supersede", target, new_content)
            self._export_markdown_locked(conn, target)
            row = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (new_id,)).fetchone()
            conn.commit()
        return {"success": True, "message": "Entry replaced.", "entry": self._row_to_dict(row), "entries": self.list_entries(target)}

    def forget_entry(self, target: str, old_text: str, *, forgotten_by: str | None = None) -> Dict[str, Any]:
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
            conn.execute(
                "UPDATE memory_entries SET status = 'forgotten', updated_at = ?, forgotten_by = ? WHERE id = ?",
                (now, forgotten_by, row["id"]),
            )
            self._write_event(conn, row["id"], "forget", target, row["content"])
            self._export_markdown_locked(conn, target)
            conn.commit()
        return {"success": True, "message": "Entry removed.", "entries": self.list_entries(target)}

    def _score_prompt_entry(self, row: sqlite3.Row | Dict[str, Any], target: str, now: float | None = None) -> tuple[float, str]:
        data = self._row_to_dict(row) if isinstance(row, sqlite3.Row) else dict(row)
        now = now or time.time()
        age_days = max(0.0, (now - float(data.get("updated_at") or now)) / 86400.0)
        recency = max(0.0, 0.3 - min(age_days / 365.0, 0.3))
        entry_type = data.get("entry_type") or data.get("kind") or "lesson"
        strength = data.get("strength") or "contextual"
        source = data.get("source") or "manual"

        target_bias = 0.2 if target == "user" else 0.0
        kind_bonus = 0.0
        if data.get("kind") in {"preference", "instruction", "constraint", "identity"}:
            kind_bonus = 0.25
        entry_type_bonus = self._ENTRY_TYPE_BONUS.get(entry_type, 0.0)
        strength_bonus = self._STRENGTH_BONUS.get(strength, 0.0)
        source_bonus = self._SOURCE_BONUS.get(source, 0.0)
        use_bonus = 0.05 * int(data.get("use_count") or 0)
        importance = float(data.get("importance") or 0)

        score = importance + target_bias + kind_bonus + entry_type_bonus + strength_bonus + source_bonus + recency + use_bonus
        reason = (
            f"importance={importance:.2f}; entry_type={entry_type}; strength={strength}; "
            f"source={source}; recency={recency:.2f}"
        )
        return score, reason

    def retrieve_for_prompt(self, target: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_entries WHERE target = ? AND status = 'active'",
                (target,),
            ).fetchall()
        items = []
        now = time.time()
        for row in rows:
            item = self._row_to_dict(row)
            score, reason = self._score_prompt_entry(item, target, now=now)
            item["_score"] = score
            item["_selection_reason"] = reason
            items.append(item)
        items.sort(key=lambda x: (x["_score"], x["updated_at"]), reverse=True)
        return items

    def explain_prompt_selection(self, target: str, char_limit: Optional[int] = None) -> Dict[str, Any]:
        limit = char_limit or self._char_limit(target)
        entries = self.retrieve_for_prompt(target)
        selected: List[Dict[str, Any]] = []
        selected_contents: List[str] = []
        separator = "═" * 46
        header_name = "USER PROFILE (who the user is)" if target == "user" else "MEMORY (your personal notes)"
        header = f"{separator}\n{header_name}\n{separator}\n"

        for entry in entries:
            candidate_contents = selected_contents + [entry["content"]]
            candidate = header + ENTRY_DELIMITER.join(candidate_contents)
            if len(candidate) <= limit:
                selected_contents.append(entry["content"])
                selected.append(
                    {
                        "id": entry["id"],
                        "content": entry["content"],
                        "score": entry["_score"],
                        "reason": entry.get("_selection_reason", ""),
                        "path": "hot_memory",
                    }
                )

        if not selected and entries:
            top = entries[0]
            selected.append(
                {
                    "id": top["id"],
                    "content": top["content"],
                    "score": top["_score"],
                    "reason": top.get("_selection_reason", ""),
                    "path": "hot_memory",
                }
            )

        return {"target": target, "char_limit": limit, "selected": selected}

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

    def export_snapshot(self) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_entries ORDER BY created_at ASC, updated_at ASC"
            ).fetchall()
        entries = [self._row_to_dict(r) for r in rows]
        return {
            "format": "hermes-memory-snapshot-v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": entries,
        }

    def export_snapshot_to_file(self, output_path: Path | str) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.export_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def import_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if snapshot.get("format") != "hermes-memory-snapshot-v1":
            return {"success": False, "error": "Unsupported snapshot format."}
        imported = 0
        updated = 0
        with self._connect() as conn:
            for entry in snapshot.get("entries", []):
                existing = conn.execute("SELECT updated_at FROM memory_entries WHERE id = ?", (entry["id"],)).fetchone()
                payload = (
                    entry["id"], entry["target"], entry.get("kind") or entry.get("entry_type", "lesson"), entry["content"], entry["status"],
                    entry.get("scope", "global"), entry.get("scope_value"), entry.get("source", "manual"),
                    entry.get("strength", "contextual"), float(entry.get("confidence", 1.0)), float(entry.get("importance", 0.5)),
                    float(entry.get("created_at", time.time())), float(entry.get("updated_at", time.time())),
                    entry.get("last_used_at"), int(entry.get("use_count", 0)), entry.get("supersedes_id"),
                    entry.get("created_in_session_id"), entry.get("replaced_by"), entry.get("forgotten_by"),
                    entry.get("fingerprint") or self._fingerprint(entry["target"], entry["content"]),
                )
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO memory_entries(
                            id, target, kind, content, status, scope, scope_value, source,
                            strength, confidence, importance, created_at, updated_at, last_used_at,
                            use_count, supersedes_id, created_in_session_id, replaced_by, forgotten_by, fingerprint
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        payload,
                    )
                    imported += 1
                elif float(entry.get("updated_at", 0)) > float(existing["updated_at"] or 0):
                    conn.execute(
                        """
                        UPDATE memory_entries
                        SET target = ?, kind = ?, content = ?, status = ?, scope = ?, scope_value = ?,
                            source = ?, strength = ?, confidence = ?, importance = ?, created_at = ?, updated_at = ?,
                            last_used_at = ?, use_count = ?, supersedes_id = ?, created_in_session_id = ?,
                            replaced_by = ?, forgotten_by = ?, fingerprint = ?
                        WHERE id = ?
                        """,
                        (
                            entry["target"], entry.get("kind") or entry.get("entry_type", "lesson"), entry["content"], entry["status"],
                            entry.get("scope", "global"), entry.get("scope_value"), entry.get("source", "manual"),
                            entry.get("strength", "contextual"), float(entry.get("confidence", 1.0)), float(entry.get("importance", 0.5)),
                            float(entry.get("created_at", time.time())), float(entry.get("updated_at", time.time())),
                            entry.get("last_used_at"), int(entry.get("use_count", 0)), entry.get("supersedes_id"),
                            entry.get("created_in_session_id"), entry.get("replaced_by"), entry.get("forgotten_by"),
                            entry.get("fingerprint") or self._fingerprint(entry["target"], entry["content"]), entry["id"],
                        ),
                    )
                    updated += 1
            self._export_markdown_locked(conn, "memory")
            self._export_markdown_locked(conn, "user")
            conn.commit()
        return {"success": True, "imported": imported, "updated": updated, "entry_count": snapshot.get("entry_count", 0)}

    def import_snapshot_from_file(self, input_path: Path | str) -> Dict[str, Any]:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        return self.import_snapshot(payload)
