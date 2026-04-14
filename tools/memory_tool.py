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

import fcntl
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"
DEFAULT_MEMORY_TYPE = "uncategorized"
EXPLICIT_MEMORY_TYPES = ("user", "feedback", "project", "reference")
ALL_MEMORY_TYPES = EXPLICIT_MEMORY_TYPES + (DEFAULT_MEMORY_TYPE,)
MEMORY_TYPE_PRIORITY = {
    "user": 0,
    "feedback": 1,
    "project": 2,
    "reference": 3,
    DEFAULT_MEMORY_TYPE: 4,
}


def _normalize_memory_type(
    memory_type: Optional[str],
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """Normalize and validate a memory type."""
    if memory_type is None:
        return default

    normalized = memory_type.strip().lower()
    if not normalized:
        return default
    if normalized not in ALL_MEMORY_TYPES:
        allowed = ", ".join(ALL_MEMORY_TYPES)
        raise ValueError(f"Invalid type '{memory_type}'. Use one of: {allowed}.")
    return normalized


def _infer_memory_type(content: str) -> str:
    """Infer a memory type from content using simple keyword heuristics."""
    text = content.lower()

    feedback_markers = (
        "prefer",
        "don't",
        "do not",
        "stop doing",
        "keep doing",
        "always",
        "never",
    )
    project_markers = (
        "deadline",
        "sprint",
        "release",
        "milestone",
        "freeze",
    )
    reference_markers = (
        "url",
        "http",
        "dashboard",
        "link",
        "wiki",
        "docs",
    )
    user_markers = (
        "i am",
        "my role",
        "i work",
        "my team",
    )

    if any(marker in text for marker in feedback_markers):
        return "feedback"
    if any(marker in text for marker in project_markers):
        return "project"
    if any(marker in text for marker in reference_markers):
        return "reference"
    if any(marker in text for marker in user_markers):
        return "user"
    return DEFAULT_MEMORY_TYPE


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
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_entry_types: Dict[str, str] = {}
        self.user_entry_types: Dict[str, str] = {}
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot."""
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))
        self.memory_entry_types = self._read_types_file(
            self._types_path_for("memory"), self.memory_entries
        )
        self.user_entry_types = self._read_types_file(
            self._types_path_for("user"), self.user_entries
        )

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(lock_path, "w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    @classmethod
    def _types_path_for(cls, target: str) -> Path:
        base = cls._path_for(target)
        return base.with_name(f"{base.stem}.types.json")

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        """
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        self._set_types(target, self._read_types_file(self._types_path_for(target), fresh))

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))
        self._write_types_file(self._types_path_for(target), self._entries_for(target), self._types_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _types_for(self, target: str) -> Dict[str, str]:
        if target == "user":
            return self.user_entry_types
        return self.memory_entry_types

    def _set_types(self, target: str, entry_types: Dict[str, str]):
        if target == "user":
            self.user_entry_types = entry_types
        else:
            self.memory_entry_types = entry_types

    def _entry_type(self, target: str, content: str) -> str:
        return self._types_for(target).get(content, DEFAULT_MEMORY_TYPE)

    def _typed_entries(self, target: str, entries: Optional[List[str]] = None) -> List[Dict[str, str]]:
        selected = self._entries_for(target) if entries is None else entries
        return [
            {
                "type": self._entry_type(target, entry),
                "content": entry,
                "display": f"[{self._entry_type(target, entry)}] {entry}",
            }
            for entry in selected
        ]

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(
        self,
        target: str,
        content: str,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        try:
            resolved_type = _normalize_memory_type(memory_type, default=None)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        if resolved_type is None:
            resolved_type = _infer_memory_type(content)

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions
            self._reload_target(target)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            entry_types = self._types_for(target).copy()
            entry_types[content] = resolved_type
            self._set_types(target, entry_types)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(
        self,
        target: str,
        old_text: str,
        new_content: str,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        try:
            resolved_type = _normalize_memory_type(memory_type, default=None)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)
            old_entry = entries[idx]
            final_type = resolved_type or _infer_memory_type(new_content)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            entry_types = self._types_for(target).copy()
            if old_entry not in entries:
                entry_types.pop(old_entry, None)
            entry_types[new_content] = final_type
            self._set_types(target, entry_types)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            removed_entry = entries.pop(idx)
            self._set_entries(target, entries)
            entry_types = self._types_for(target).copy()
            if removed_entry not in entries:
                entry_types.pop(removed_entry, None)
            self._set_types(target, entry_types)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def read(
        self,
        target: str,
        memory_type: Optional[str] = None,
        group_by_type: bool = False,
    ) -> Dict[str, Any]:
        """Return current entries, optionally filtered by type."""
        try:
            resolved_type = _normalize_memory_type(memory_type, default=None)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            if resolved_type is None:
                entries = None
            else:
                entries = [
                    entry for entry in self._entries_for(target)
                    if self._entry_type(target, entry) == resolved_type
                ]

        return self._success_response(
            target,
            "Entries read.",
            entries=entries,
            type_filter=resolved_type,
            include_rendered=True,
            group_by_type=group_by_type,
        )

    def recall_by_types(self, types: List[str], limit: int = 20) -> List[Dict[str, str]]:
        """Return matching memories across stores, ordered by type priority."""
        normalized_types: List[str] = []
        for memory_type in types:
            normalized = _normalize_memory_type(memory_type, default=None)
            if normalized is not None:
                normalized_types.append(normalized)

        if not normalized_types or limit <= 0:
            return []

        allowed_types = set(normalized_types)
        recalled: List[Dict[str, str]] = []
        seen = set()

        for target in ("user", "memory"):
            for item in self._typed_entries(target):
                if item["type"] not in allowed_types:
                    continue
                key = (item["type"], item["content"])
                if key in seen:
                    continue
                seen.add(key)
                recalled.append(
                    {
                        "type": item["type"],
                        "title": self._entry_title(item["content"]),
                        "content": item["content"],
                    }
                )

        recalled.sort(key=lambda item: (MEMORY_TYPE_PRIORITY[item["type"]], item["title"]))
        return recalled[:limit]

    def bulk_update_types(self, target: str, updates: Dict[str, str]) -> Dict[str, Any]:
        """Update entry types by exact content match. Used by migration tooling."""
        normalized_updates: Dict[str, str] = {}
        try:
            for content, memory_type in updates.items():
                normalized_updates[content] = _normalize_memory_type(
                    memory_type,
                    default=DEFAULT_MEMORY_TYPE,
                )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = self._entries_for(target)
            entry_types = self._types_for(target).copy()
            applied = 0
            skipped = []

            for content, memory_type in normalized_updates.items():
                if content not in entries:
                    skipped.append(content)
                    continue
                if entry_types.get(content) != memory_type:
                    entry_types[content] = memory_type
                    applied += 1

            self._set_types(target, entry_types)
            self.save_to_disk(target)

        return {
            "success": True,
            "target": target,
            "applied": applied,
            "skipped": skipped,
        }

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    def _success_response(
        self,
        target: str,
        message: str = None,
        entries: Optional[List[str]] = None,
        type_filter: Optional[str] = None,
        include_rendered: bool = False,
        group_by_type: bool = False,
    ) -> Dict[str, Any]:
        live_entries = self._entries_for(target)
        result_entries = live_entries if entries is None else entries
        typed_entries = self._typed_entries(target, result_entries)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": result_entries,
            "typed_entries": typed_entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(result_entries),
        }
        if entries is not None:
            resp["total_entry_count"] = len(live_entries)
        if type_filter is not None:
            resp["type_filter"] = type_filter
        if include_rendered:
            resp["group_by_type"] = group_by_type
            resp["rendered"] = self._render_read_entries(typed_entries, group_by_type=group_by_type)
        if message:
            resp["message"] = message
        return resp

    @staticmethod
    def _entry_title(content: str, max_length: int = 80) -> str:
        """Build a short title from the first non-empty line of content."""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0] if lines else content.strip()
        if len(title) <= max_length:
            return title
        return title[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _render_read_entries(typed_entries: List[Dict[str, str]], *, group_by_type: bool = False) -> str:
        """Render read results either flat or grouped by memory type."""
        if not typed_entries:
            return ""

        if not group_by_type:
            return "".join(f"- {item['content']}\n" for item in typed_entries)

        grouped: Dict[str, List[str]] = {memory_type: [] for memory_type in ALL_MEMORY_TYPES}
        for item in typed_entries:
            grouped[item["type"]].append(item["content"])

        sections: List[str] = []
        for memory_type in sorted(ALL_MEMORY_TYPES, key=lambda name: MEMORY_TYPE_PRIORITY[name]):
            contents = grouped[memory_type]
            if not contents:
                continue
            section_lines = [f"## [Type] {memory_type}"]
            section_lines.extend(f"- {content}" for content in contents)
            sections.append("\n".join(section_lines))

        return "\n\n".join(sections) + "\n"

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

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _read_types_file(path: Path, entries: List[str]) -> Dict[str, str]:
        """Read the sidecar type mapping for the current entries."""
        if not entries:
            return {}

        raw_types: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw_types = loaded
            except (OSError, IOError, json.JSONDecodeError, TypeError):
                raw_types = {}

        normalized: Dict[str, str] = {}
        for entry in entries:
            try:
                normalized[entry] = _normalize_memory_type(
                    raw_types.get(entry),
                    default=DEFAULT_MEMORY_TYPE,
                )
            except ValueError:
                normalized[entry] = DEFAULT_MEMORY_TYPE
        return normalized

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))  # Atomic on same filesystem
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")

    @staticmethod
    def _write_types_file(path: Path, entries: List[str], entry_types: Dict[str, str]):
        """Write entry type metadata to a sidecar JSON file."""
        payload = {
            entry: entry_types.get(entry, DEFAULT_MEMORY_TYPE)
            for entry in entries
        }
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_types_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory type file {path}: {e}")


_SMART_RECALL_CLASS = None


def _load_smart_recall_class():
    global _SMART_RECALL_CLASS
    if _SMART_RECALL_CLASS is not None:
        return _SMART_RECALL_CLASS

    plugin_path = Path(__file__).resolve().parent.parent / "plugins" / "hongxing-enhancements" / "smart_recall.py"
    if not plugin_path.is_file():
        raise RuntimeError(f"smart_recall plugin not found at {plugin_path}")

    module_name = "hongxing_smart_recall_memory_tool"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load smart_recall plugin module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    _SMART_RECALL_CLASS = module.SmartRecall
    return _SMART_RECALL_CLASS


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    memory_type: Optional[str] = None,
    group_by_type: bool = False,
    query: Optional[str] = None,
    top_k: int = 5,
    types: Optional[List[str]] = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if action == "smart_recall":
        if not query or not query.strip():
            return tool_error("query is required for 'smart_recall' action.", success=False)
        if types is not None and not isinstance(types, list):
            return tool_error("types must be a list of strings for 'smart_recall' action.", success=False)

        try:
            resolved_top_k = 5 if top_k is None else int(top_k)
        except (TypeError, ValueError):
            return tool_error("top_k must be an integer for 'smart_recall' action.", success=False)

        try:
            smart_recall = _load_smart_recall_class()(store)
        except Exception as exc:
            return tool_error(f"smart_recall is unavailable: {exc}", success=False)

        result = {
            "success": True,
            "results": smart_recall.recall(query, top_k=resolved_top_k, types=types),
        }
        return json.dumps(result, ensure_ascii=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content, memory_type=memory_type)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content, memory_type=memory_type)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    elif action == "read":
        result = store.read(target, memory_type=memory_type, group_by_type=group_by_type)

    else:
        return tool_error(
            f"Unknown action '{action}'. Use: add, replace, remove, read, smart_recall",
            success=False,
        )

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
        "remove (delete -- old_text identifies it), read (list current entries, optionally filtered by type), "
        "smart_recall (recall relevant memories by query, optionally filtered by types).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "read", "smart_recall"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile. Required for add, replace, remove, and read."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
            "type": {
                "type": "string",
                "enum": list(ALL_MEMORY_TYPES),
                "description": (
                    "Optional memory category. Use user, feedback, project, or reference for explicit typing. "
                    "If omitted on add or replace, Hermes infers a type from the content. "
                    "On read, this acts as an optional type filter."
                ),
            },
            "group_by_type": {
                "type": "boolean",
                "description": (
                    "Optional read-only flag. When true, the rendered output is grouped into "
                    "type-based sections. Defaults to false for backward compatibility."
                ),
            },
            "query": {
                "type": "string",
                "description": "Required for 'smart_recall'. Free-text query used to recall relevant memories.",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Optional result limit for 'smart_recall'. Defaults to 5.",
            },
            "types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(ALL_MEMORY_TYPES),
                },
                "description": "Optional type filter list for 'smart_recall'.",
            },
        },
        "required": ["action"],
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
        memory_type=args.get("type"),
        group_by_type=args.get("group_by_type", False),
        query=args.get("query"),
        top_k=args.get("top_k", 5),
        types=args.get("types"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
    allowed_in_plan_mode_default=False,
)
