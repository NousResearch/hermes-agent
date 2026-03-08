#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import json
import logging
import os
import re
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Where memory files live
MEMORY_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "memories"

ENTRY_DELIMITER = "\n§\n"
MEMORY_DB_PATH = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "state.db"
DEFAULT_TOKEN_BUDGET = 8000  # ~2000 tokens


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


class MemoryStore:
    """
    SQLite-backed curated memory. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - In-memory cache (memory_entries / user_entries): live state for tool responses.

    On first run, auto-migrates existing MEMORY.md / USER.md entries to SQLite.
    Old files kept as read-only backups.
    """

    def __init__(self, db_path: Path = None, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.db_path = db_path or MEMORY_DB_PATH
        self.token_budget = token_budget
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self, conn: sqlite3.Connection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                target TEXT NOT NULL DEFAULT 'memory',
                scope TEXT NOT NULL DEFAULT '/',
                categories TEXT DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed_at REAL,
                source TEXT,
                forgotten INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_target ON memories(target)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)")
        conn.commit()

    # ------------------------------------------------------------------
    # Migration: flat files -> SQLite (runs once)
    # ------------------------------------------------------------------

    def _migrate_from_files(self, conn: sqlite3.Connection):
        """Import MEMORY.md and USER.md into SQLite if not already done."""
        for target, filename in [("memory", "MEMORY.md"), ("user", "USER.md")]:
            path = MEMORY_DIR / filename
            if not path.exists():
                continue
            backup = MEMORY_DIR / (filename + ".bak")
            if backup.exists():
                continue  # Already migrated
            entries = self._read_file(path)
            now = time.time()
            for entry in entries:
                if not entry.strip():
                    continue
                # Skip if already in DB
                row = conn.execute(
                    "SELECT id FROM memories WHERE content = ? AND target = ? AND forgotten = 0",
                    (entry, target)
                ).fetchone()
                if row:
                    continue
                conn.execute(
                    """INSERT INTO memories (id, content, target, scope, importance, created_at, updated_at)
                       VALUES (?, ?, ?, '/', 0.5, ?, ?)""",
                    (str(uuid.uuid4()), entry, target, now, now)
                )
            conn.commit()
            # Rename original to .bak (keep as backup)
            try:
                path.rename(backup)
                logger.info("Migrated %s to SQLite, backup at %s", filename, backup)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load_from_disk(self):
        """Load entries from SQLite (with auto-migration), capture system prompt snapshot."""
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_conn() as conn:
            self._ensure_table(conn)
            self._migrate_from_files(conn)
            self.memory_entries = self._load_entries(conn, "memory")
            self.user_entries = self._load_entries(conn, "user")

        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    def _load_entries(self, conn: sqlite3.Connection, target: str) -> List[str]:
        """Load active entries ordered by importance DESC."""
        rows = conn.execute(
            "SELECT content FROM memories WHERE target = ? AND forgotten = 0 ORDER BY importance DESC, created_at ASC",
            (target,)
        ).fetchall()
        return [r["content"] for r in rows]

    def _save_entry(self, target: str, content: str, entry_id: str = None,
                    scope: str = "/", importance: float = 0.5):
        """Insert a new entry into SQLite."""
        now = time.time()
        with self._get_conn() as conn:
            self._ensure_table(conn)
            conn.execute(
                """INSERT INTO memories (id, content, target, scope, importance, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entry_id or str(uuid.uuid4()), content, target, scope, importance, now, now)
            )
            conn.commit()

    def _update_entry(self, old_content: str, new_content: str, target: str):
        """Update an existing entry by content match."""
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE content = ? AND target = ? AND forgotten = 0",
                (new_content, now, old_content, target)
            )
            conn.commit()

    def _soft_delete_entry(self, content: str, target: str):
        """Soft-delete by setting forgotten=1."""
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE memories SET forgotten = 1, updated_at = ? WHERE content = ? AND target = ? AND forgotten = 0",
                (now, content, target)
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Core operations (same API as before)
    # ------------------------------------------------------------------

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def add(self, target: str, content: str, scope: str = "/", importance: float = 0.5) -> Dict[str, Any]:
        """Append a new entry to SQLite."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        entries = self._entries_for(target)

        if content in entries:
            return self._success_response(target, "Entry already exists (no duplicate added).")

        self._save_entry(target, content, scope=scope, importance=importance)
        entries.append(content)
        self._set_entries(target, entries)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

        if len(matches) == 0:
            return {"success": False, "error": f"No entry matched '{old_text}'."}

        if len(matches) > 1:
            unique_texts = set(e for _, e in matches)
            if len(unique_texts) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}

        idx, old_entry = matches[0]
        self._update_entry(old_entry, new_content, target)
        entries[idx] = new_content
        self._set_entries(target, entries)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Soft-delete the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

        if len(matches) == 0:
            return {"success": False, "error": f"No entry matched '{old_text}'."}

        if len(matches) > 1:
            unique_texts = set(e for _, e in matches)
            if len(unique_texts) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}

        idx, old_entry = matches[0]
        self._soft_delete_entry(old_entry, target)
        entries.pop(idx)
        self._set_entries(target, entries)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Return frozen snapshot for system prompt injection."""
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        total_chars = sum(len(e) for e in entries)
        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "entry_count": len(entries),
            "total_chars": total_chars,
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header, respecting token budget."""
        if not entries:
            return ""

        if target == "user":
            header = "USER PROFILE (who the user is)"
        else:
            header = "MEMORY (your personal notes)"

        separator = "═" * 46
        lines = []
        budget = self.token_budget
        for entry in entries:
            chunk = entry + ENTRY_DELIMITER
            budget -= len(chunk)
            if budget < 0:
                break
            lines.append(entry)

        if not lines:
            return ""

        content = ENTRY_DELIMITER.join(lines)
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a legacy memory file and split into entries (migration only)."""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return json.dumps({"success": False, "error": "Memory is not available. It may be disabled in config or this environment."}, ensure_ascii=False)

    if target not in ("memory", "user"):
        return json.dumps({"success": False, "error": f"Invalid target '{target}'. Use 'memory' or 'user'."}, ensure_ascii=False)

    if action == "add":
        if not content:
            return json.dumps({"success": False, "error": "Content is required for 'add' action."}, ensure_ascii=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return json.dumps({"success": False, "error": "old_text is required for 'replace' action."}, ensure_ascii=False)
        if not content:
            return json.dumps({"success": False, "error": "content is required for 'replace' action."}, ensure_ascii=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return json.dumps({"success": False, "error": "old_text is required for 'remove' action."}, ensure_ascii=False)
        result = store.remove(target, old_text)

    else:
        return json.dumps({"success": False, "error": f"Unknown action '{action}'. Use: add, replace, remove"}, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save important information to persistent memory that survives across sessions. "
        "Your memory appears in your system prompt at session start -- it's how you "
        "remember things about the user and your environment between conversations.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You completed something - log it like a diary entry\n"
        "- After completing a complex task, save a brief note about what was done\n\n"
        "- If you've discovered a new way to do something, solved a problem that could be necessary later, save it as a skill with the skill tool\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n"
        "Capacity shown in system prompt. When >80%, consolidate entries before adding new ones.\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
)




