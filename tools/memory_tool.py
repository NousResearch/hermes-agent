#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory (DB-backed)

Provides bounded, SQLite-backed memory that persists across sessions. Two stores:
  - memory: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - user: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a snapshot at session start.
Mid-session writes update the DB immediately and refresh the snapshot.

Design:
- Single `memory` tool with action parameter: add, replace, remove
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- DB-backed with auto-eviction: when adding would exceed the char limit, the
  lowest-value entries (access_count ASC, last_accessed ASC) are evicted first.
- Access-count tracking: every format_for_system_prompt() call touches entries
  so frequently-used memories survive eviction.
- Thread-safe: uses threading.Lock to serialize DB writes.
"""

import json
import logging
import re
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENTRY_DELIMITER = "\n§\n"

# Default category for entries (used for future multi-category queries)
_DEFAULT_CATEGORY = "default"


# -------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# -------------------------------------------------------------------------

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
    (r'\$HOME/\.ssh|\~/.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."
    return None


class MemoryStore:
    """
    DB-backed curated memory with auto-eviction. One instance per AIAgent.

    Thread-safe: all DB writes are serialized via threading.Lock.

    Maintains two parallel states:
      - _system_prompt_snapshot: refreshed on load_from_db() and after every
        write. Used for system prompt injection.
      - _live_entries: live state, mutated by tool calls. Tool responses always
        reflect this live state.
    """

    def __init__(
        self,
        session_db,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        self._db = session_db
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._live_entries: Dict[str, List[str]] = {"memory": [], "user": []}
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Lock for thread-safe DB access
    # -------------------------------------------------------------------------

    @contextmanager
    def _db_lock(self):
        """Acquire the DB write lock. All mutating DB operations must run under this lock."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    # -------------------------------------------------------------------------
    # Persistence (DB-only, no .md files)
    # -------------------------------------------------------------------------

    def load_from_db(self):
        """Load entries from DB, capture system prompt snapshot."""
        for section in ("memory", "user"):
            rows = self._db.memory_get_active(section)
            self._live_entries[section] = [r["value"] for r in rows]
        self._refresh_snapshot()

    def refresh(self):
        """Rebuild the snapshot after a mid-session write."""
        self.load_from_db()

    def _refresh_snapshot(self):
        """Rebuild _system_prompt_snapshot from current _live_entries."""
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self._live_entries["memory"]),
            "user": self._render_block("user", self._live_entries["user"]),
        }

    # -------------------------------------------------------------------------
    # Read helpers
    # -------------------------------------------------------------------------

    def _entries_for(self, target: str) -> List[str]:
        return self._live_entries.get(target, [])

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the current snapshot for system prompt injection.

        Also touches all entries in the section to increment their access_count,
        so frequently-used memories survive auto-eviction.
        """
        block = self._system_prompt_snapshot.get(target, "")
        if not block:
            return None

        # Touch all entries for this section to update access counts (thread-safe)
        with self._db_lock():
            rows = self._db.memory_get_active(target)
            for row in rows:
                self._db.memory_touch(row["id"])

        return block

    # -------------------------------------------------------------------------
    # Mutations (thread-safe via _db_lock)
    # -------------------------------------------------------------------------

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Auto-evicts lowest-value entries if over limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        entries = self._entries_for(target)
        # Reject exact duplicates
        if content in entries:
            return self._success_response(target, "Entry already exists (no duplicate added).")

        limit = self._char_limit(target)
        new_chars = len(content)

        with self._db_lock():
            # Evict lowest-value entries until we have room
            evicted = self._db.memory_evict_for_section(target, new_chars, limit)
            if evicted > 0:
                logger.debug("memory add: evicted %d entries from %s", evicted, target)

            # Persist new entry
            entry_id = str(uuid.uuid4())
            self._db.memory_upsert(
                id=entry_id,
                section=target,
                category=_DEFAULT_CATEGORY,
                key=entry_id,
                value=content,
            )

            # Update live state directly (no DB re-read)
            self._live_entries[target].append(content)

        self._refresh_snapshot()
        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        # Find matching entries
        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}

        if len(matches) > 1:
            unique_texts = set(e for _, e in matches)
            if len(unique_texts) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}

        idx, old_value = matches[0]
        limit = self._char_limit(target)
        new_chars = len(new_content)
        old_chars = len(old_value)
        delta = new_chars - old_chars

        with self._db_lock():
            # If replacing makes it bigger, evict if needed.
            # Eviction removes LOWEST priority entries (access_count ASC, last_accessed ASC).
            # The target entry's index is stable — eviction never removes entries
            # with higher access_count than the target, so idx remains valid.
            if delta > 0:
                self._db.memory_evict_for_section(target, delta, limit)

            # Find the DB row for the matching entry
            rows = self._db.memory_get_active(target)
            match_row = None
            for row in rows:
                if row["value"] == old_value:
                    match_row = row
                    break
            if not match_row:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            # Update in place (same id, same key)
            self._db.memory_upsert(
                id=match_row["id"],
                section=target,
                category=match_row["category"],
                key=match_row["key"],
                value=new_content,
            )

            # Update live state directly using stable idx.
            self._live_entries[target][idx] = new_content

        self._refresh_snapshot()
        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}

        if len(matches) > 1:
            unique_texts = set(e for _, e in matches)
            if len(unique_texts) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}

        idx, old_value = matches[0]

        with self._db_lock():
            # Find the DB row for the matching entry
            rows = self._db.memory_get_active(target)
            for row in rows:
                if row["value"] == old_value:
                    self._db.memory_delete(target, row["category"], row["key"])
                    break
            else:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            # Re-read after mutation: eviction may have shifted indices.
            # Find current position by value (stable identity), then delete.
            current_entries = [r["value"] for r in rows]
            for i, v in enumerate(current_entries):
                if v == old_value:
                    del self._live_entries[target][i]
                    break

        self._refresh_snapshot()
        return self._success_response(target, "Entry removed.")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"


# -------------------------------------------------------------------------
# Tool entry point
# -------------------------------------------------------------------------

def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """Single entry point for the memory tool. Dispatches to MemoryStore methods."""
    from tools.registry import tool_error
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)
    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)
    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)
    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# -------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# -------------------------------------------------------------------------

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
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
from tools.registry import registry, tool_error

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
    emoji="🧠",
)
