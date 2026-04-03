#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory.

Public API stays stable: add / replace / remove.
Storage now uses a SQLite-backed persistent memory store with markdown export
compatibility so older tooling still sees MEMORY.md and USER.md.
"""

import json
import logging
import re
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

import tools.persistent_memory_store as pm

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected. The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"


# Backward-compatible alias — some callers/tests still reference MEMORY_DIR
# directly, but new code should prefer get_memory_dir().
MEMORY_DIR = get_memory_dir()
ENTRY_DELIMITER = "\n§\n"

_MEMORY_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."
    return None


class MemoryStore:
    """Compatibility wrapper over PersistentMemoryStore."""

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        self._backend = None

    def _memory_dir(self) -> Path:
        return get_memory_dir()

    def _db_path(self) -> Path:
        memory_dir = self._memory_dir()
        if memory_dir.name == "memories":
            return memory_dir.parent / "memory.db"
        return memory_dir / "memory.db"

    def _backend_store(self) -> pm.PersistentMemoryStore:
        if self._backend is None:
            self._backend = pm.PersistentMemoryStore(
                db_path=self._db_path(),
                memory_dir=self._memory_dir(),
                memory_char_limit=self.memory_char_limit,
                user_char_limit=self.user_char_limit,
            )
        return self._backend

    def _entry_kind(self, target: str, content: str) -> str:
        lower = content.lower()
        if target == "user":
            if "prefer" in lower or "dislike" in lower or "hate" in lower:
                return "preference"
            if "timezone" in lower or "name" in lower or "role" in lower:
                return "identity"
            return "instruction"
        if "secret" in lower or "constraint" in lower or "must" in lower or "do not" in lower or "don't" in lower:
            return "constraint"
        if "project" in lower or "repo" in lower or "prod" in lower:
            return "project"
        if "workflow" in lower or "command" in lower:
            return "workflow"
        return "lesson"

    def _load_target_from_backend(self, target: str) -> List[str]:
        rows = self._backend_store().list_entries(target)
        return [row["content"] for row in rows]

    @staticmethod
    def _read_markdown_file(path: Path) -> List[str]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]

    def _bootstrap_from_markdown_if_needed(self):
        backend = self._backend_store()
        if backend.list_entries("memory", include_inactive=True) or backend.list_entries("user", include_inactive=True):
            return
        memory_dir = self._memory_dir()
        for target, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
            for entry in self._read_markdown_file(memory_dir / filename):
                backend.add_entry(target, entry, kind=self._entry_kind(target, entry), source="migration")

    def load_from_disk(self):
        """Load entries from the SQLite store, bootstrapping legacy markdown if needed."""
        memory_dir = self._memory_dir()
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._backend = pm.PersistentMemoryStore(
            db_path=self._db_path(),
            memory_dir=memory_dir,
            memory_char_limit=self.memory_char_limit,
            user_char_limit=self.user_char_limit,
        )
        self._bootstrap_from_markdown_if_needed()
        self.memory_entries = self._load_target_from_backend("memory")
        self.user_entries = self._load_target_from_backend("user")
        self._system_prompt_snapshot = {
            "memory": self._backend_store().render_prompt_block("memory", char_limit=self.memory_char_limit) or "",
            "user": self._backend_store().render_prompt_block("user", char_limit=self.user_char_limit) or "",
        }

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _refresh_live_entries(self, target: str):
        self._set_entries(target, self._load_target_from_backend(target))

    def add(self, target: str, content: str) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        result = self._backend_store().add_entry(target, content, kind=self._entry_kind(target, content))
        self._refresh_live_entries(target)
        if not result.get("success"):
            current = self._char_count(target)
            limit = self._char_limit(target)
            result.setdefault("current_entries", self._entries_for(target))
            result.setdefault("usage", f"{current:,}/{limit:,}")
            return result
        return self._success_response(target, result.get("message", "Entry added."))

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}
        result = self._backend_store().replace_entry(target, old_text, new_content, kind=self._entry_kind(target, new_content))
        self._refresh_live_entries(target)
        if not result.get("success"):
            return result
        return self._success_response(target, result.get("message", "Entry replaced."))

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        result = self._backend_store().forget_entry(target, old_text)
        self._refresh_live_entries(target)
        if not result.get("success"):
            return result
        return self._success_response(target, result.get("message", "Entry removed."))

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    def render_live_for_system_prompt(self, target: str) -> Optional[str]:
        return self._backend_store().render_prompt_block(target, char_limit=self._char_limit(target))

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
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        header = (
            f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
            if target == "user"
            else f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"
        )
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
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
    return True


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
            "action": {"type": "string", "enum": ["add", "replace", "remove"], "description": "The action to perform."},
            "target": {"type": "string", "enum": ["memory", "user"], "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."},
            "content": {"type": "string", "description": "The entry content. Required for 'add' and 'replace'."},
            "old_text": {"type": "string", "description": "Short unique substring identifying the entry to replace or remove."},
        },
        "required": ["action", "target"],
    },
}

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
    emoji="🧠",
)
