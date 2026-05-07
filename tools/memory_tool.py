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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"


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

        if fcntl is None and msvcrt is None:
            yield
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        fd = open(lock_path, "r+" if msvcrt else "a+")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        """
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

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
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
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
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).

        Special target ``"warm_status"`` returns a one-line status string
        for the warm tier (e.g. ``"WARM MEMORY: 247 facts indexed..."``)
        that's safe to include in the system prompt every turn — it's a
        small constant string that only changes when the warm-tier count
        crosses a turn boundary.
        """
        if target == "warm_status":
            return self._format_warm_status()
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    @staticmethod
    def _format_warm_status() -> Optional[str]:
        """Return a one-line warm-tier status block for the system prompt.

        Returns ``None`` when the warm tier is empty or unavailable —
        callers append the result conditionally so the system prompt
        stays clean for users who haven't migrated.
        """
        try:
            from tools.memory_warm import get_warm_store
            store = get_warm_store()
            n = store.count()
        except Exception:
            return None
        if n <= 0:
            return None
        return (
            "══════════════════════════════════════════════\n"
            f"WARM MEMORY: {n} facts indexed (search-only)\n"
            "══════════════════════════════════════════════\n"
            "Search via memory(action=\"recall\", query=\"...\") when the "
            "user references something cross-session, you suspect related "
            "context exists, or you're debugging a system covered in prior "
            "notes. ~50 tokens per call. Default tier for new entries is "
            "\"warm\" — use tier=\"hot\" only for facts that must influence "
            "every turn (user preferences, recurring corrections)."
        )

    # -- Internal helpers --

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
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def _get_warm_store_or_error():
    """Return the warm-tier store, or a ``tool_error`` JSON if unavailable."""
    try:
        from tools.memory_warm import get_warm_store
        return get_warm_store(), None
    except Exception as e:
        return None, tool_error(
            f"Warm-tier memory unavailable: {e}. Hot-tier writes still work.",
            success=False,
        )


def _handle_warm_action(
    action: str,
    args_query: Optional[str],
    args_content: Optional[str],
    args_old_text: Optional[str],
    args_top_k: Optional[int],
    args_category: Optional[str],
    args_tags: Optional[str],
    args_fact_id: Optional[int],
    args_helpful: Optional[bool],
    args_target: Optional[str],
    hot_store: Optional[MemoryStore],
) -> str:
    """Dispatch warm-tier actions. Always returns a JSON string."""
    warm, err = _get_warm_store_or_error()
    if err is not None:
        return err

    if action == "add":
        if not args_content:
            return tool_error("content is required for warm add.", success=False)
        # Warm-tier content is also injected as recall results, so the same
        # injection-safety scan applies.
        scan_error = _scan_memory_content(args_content)
        if scan_error:
            return tool_error(scan_error, success=False)
        result = warm.add(
            content=args_content,
            category=args_category or "general",
            tags=args_tags or "",
        )

    elif action == "recall":
        if not args_query:
            return tool_error("query is required for recall.", success=False)
        rows = warm.recall(
            query=args_query,
            top_k=int(args_top_k) if args_top_k else 5,
            category=args_category,
        )
        if not rows:
            result = {
                "success": True,
                "results": [],
                "count": 0,
                "message": (
                    "No matches in warm tier. Try different keywords, or "
                    "memory(action=\"read\", tier=\"warm\") to browse."
                ),
            }
        else:
            result = {
                "success": True,
                "results": rows,
                "count": len(rows),
            }

    elif action == "recall_related":
        seed = args_query or args_content or ""
        if not seed and args_fact_id:
            row = warm.get(int(args_fact_id))
            if row is None:
                return tool_error(
                    f"No warm fact with id {args_fact_id}.", success=False,
                )
            seed = row["content"]
        if not seed:
            return tool_error(
                "recall_related requires query, content, or fact_id.",
                success=False,
            )
        rows = warm.recall_related(
            seed=seed, top_k=int(args_top_k) if args_top_k else 5,
        )
        result = {"success": True, "results": rows, "count": len(rows)}

    elif action == "read":
        rows = warm.list_facts(
            category=args_category,
            limit=int(args_top_k) if args_top_k else 50,
        )
        result = {
            "success": True,
            "results": rows,
            "count": len(rows),
            "total_indexed": warm.count(),
        }

    elif action == "remove":
        if args_fact_id is None:
            return tool_error(
                "fact_id is required for warm remove.", success=False,
            )
        result = warm.remove(int(args_fact_id))

    elif action == "replace":
        if args_fact_id is None:
            return tool_error(
                "fact_id is required for warm replace.", success=False,
            )
        if not args_content:
            return tool_error(
                "content is required for warm replace.", success=False,
            )
        scan_error = _scan_memory_content(args_content)
        if scan_error:
            return tool_error(scan_error, success=False)
        result = warm.update(
            fact_id=int(args_fact_id),
            content=args_content,
            tags=args_tags,
            category=args_category,
        )

    elif action == "feedback":
        if args_fact_id is None:
            return tool_error(
                "fact_id is required for feedback.", success=False,
            )
        if args_helpful is None:
            return tool_error(
                "helpful (true/false) is required for feedback.", success=False,
            )
        result = warm.record_feedback(
            fact_id=int(args_fact_id), helpful=bool(args_helpful),
        )

    elif action == "promote":
        # Move a warm fact to the hot tier. Fetch the row, write it to hot,
        # delete from warm only if hot write succeeded.
        #
        # Destination hot target is taken from ``target`` ('memory' or
        # 'user'), defaulting to 'memory'. Earlier versions overloaded
        # ``old_text`` for this — callers passing old_text='user' will
        # still get user-target promotion via the back-compat shim
        # below, but new code should use target=.
        if hot_store is None:
            return tool_error(
                "Hot tier is not available; cannot promote.", success=False,
            )
        if args_fact_id is None:
            return tool_error(
                "fact_id is required for promote.", success=False,
            )
        row = warm.get(int(args_fact_id))
        if row is None:
            return tool_error(
                f"No warm fact with id {args_fact_id}.", success=False,
            )
        # Resolve destination target. Prefer the new explicit ``target``
        # arg; fall back to the legacy ``old_text`` overload only when
        # target wasn't explicitly set to a valid value.
        if args_target in ("memory", "user"):
            hot_target = args_target
        elif args_old_text in ("memory", "user"):
            hot_target = args_old_text  # legacy behavior — preserved
        else:
            hot_target = "memory"
        hot_result = hot_store.add(hot_target, row["content"])
        if not hot_result.get("success"):
            return json.dumps(hot_result, ensure_ascii=False)
        # Hot write succeeded — drop from warm.
        warm.remove(int(args_fact_id))
        result = {
            "success": True,
            "message": f"Promoted warm fact {args_fact_id} to hot tier.",
            "hot_target": hot_target,
            "hot_state": hot_result,
        }

    elif action == "demote":
        # Move a hot entry to warm. Identified by old_text substring (same
        # rules as hot remove).
        #
        # Source hot target is taken from ``target`` ('memory' or 'user'),
        # defaulting to 'memory'. Earlier versions overloaded ``category``
        # for this, which clashed with category's documented meaning
        # ("warm-tier category for the new fact"). New code should use
        # target= for the source and category= for the new warm fact's
        # category. The legacy category-as-target overload is preserved
        # only when ``target`` wasn't explicitly set to a valid value
        # AND ``category`` happens to be 'memory'/'user'.
        if hot_store is None:
            return tool_error(
                "Hot tier is not available; cannot demote.", success=False,
            )
        if not args_old_text:
            return tool_error(
                "old_text is required for demote.", success=False,
            )
        if args_target in ("memory", "user"):
            hot_target = args_target
            warm_category = args_category or "general"
        elif args_category in ("memory", "user"):
            # Legacy overload — category was the source target. Preserved
            # for back-compat; new code should use target=.
            hot_target = args_category
            warm_category = "general"
        else:
            hot_target = "memory"
            warm_category = args_category or "general"
        # Find the hot entry first (without removing it), so we don't
        # delete-without-write if warm add fails.
        with hot_store._file_lock(hot_store._path_for(hot_target)):  # type: ignore[attr-defined]
            hot_store._reload_target(hot_target)  # type: ignore[attr-defined]
            entries = hot_store._entries_for(hot_target)  # type: ignore[attr-defined]
            matches = [e for e in entries if args_old_text in e]
        if not matches:
            return tool_error(
                f"No hot entry matched '{args_old_text}'.", success=False,
            )
        if len(set(matches)) > 1:
            return tool_error(
                f"Multiple hot entries matched '{args_old_text}'. Be more specific.",
                success=False,
            )
        content = matches[0]
        warm_result = warm.add(
            content=content,
            category=warm_category,
            tags=args_tags or "demoted-from-hot",
        )
        if not warm_result.get("success"):
            return json.dumps(warm_result, ensure_ascii=False)
        # Warm write OK — drop from hot.
        hot_store.remove(hot_target, args_old_text)
        result = {
            "success": True,
            "message": f"Demoted hot entry to warm fact {warm_result.get('fact_id')}.",
            "warm_state": warm_result,
            "hot_target": hot_target,
            "warm_category": warm_category,
        }

    else:
        return tool_error(
            f"Unknown warm action '{action}'. Use: add, recall, recall_related, "
            f"read, replace, remove, feedback, promote, demote",
            success=False,
        )

    return json.dumps(result, ensure_ascii=False, default=str)


def memory_tool(
    action: str,
    target: Optional[str] = None,
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
    # Warm-tier extension (Phase 1 dynamic memory recall):
    tier: str = "hot",
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    fact_id: Optional[int] = None,
    helpful: Optional[bool] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore (hot
    tier) or WarmStore (warm tier) based on the ``tier`` arg or action.

    Hot tier: ``add``/``replace``/``remove`` — small, always-loaded,
    file-backed (MEMORY.md / USER.md), bounded by char_limit.

    Warm tier: ``add``/``recall``/``recall_related``/``read``/``replace``/
    ``remove``/``feedback`` — unbounded, search-only via FTS5 + BM25,
    SQLite-backed at ``$HERMES_HOME/memory_store.db``. Plus cross-tier:
    ``promote`` (warm → hot), ``demote`` (hot → warm).

    Returns JSON string with results.
    """
    # Warm-tier-only actions route directly regardless of tier param.
    WARM_ONLY_ACTIONS = {
        "recall", "recall_related", "feedback", "promote", "demote",
    }
    is_warm_action = (tier == "warm") or (action in WARM_ONLY_ACTIONS)

    if is_warm_action:
        return _handle_warm_action(
            action=action,
            args_query=query,
            args_content=content,
            args_old_text=old_text,
            args_top_k=top_k,
            args_category=category,
            args_tags=tags,
            args_fact_id=fact_id,
            args_helpful=helpful,
            args_target=target,
            hot_store=store,
        )

    # Hot-tier path (legacy behavior — unchanged for backward compat).
    if store is None:
        return tool_error(
            "Memory is not available. It may be disabled in config or this environment.",
            success=False,
        )

    # Default target for hot-tier ops is 'memory' (the personal-notes file).
    # The warm path handles its own target resolution (None means "not
    # specified" — see _handle_warm_action's promote/demote branches).
    if target is None:
        target = "memory"
    if target not in ("memory", "user"):
        return tool_error(
            f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False,
        )

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

    elif action == "read":
        # New explicit hot-tier read action — returns the live entries.
        result = store._success_response(target, "Hot tier entries returned.")

    else:
        return tool_error(
            f"Unknown action '{action}'. Hot-tier actions: add, replace, remove, read. "
            f"Warm-tier actions (use tier='warm' or these names): "
            f"recall, recall_related, feedback, promote, demote",
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
        "Save durable information to persistent memory and recall it across sessions. "
        "TWO TIERS, ONE TOOL — pick the right tier per fact.\n\n"
        "HOT TIER (tier='hot', the default for add/replace/remove with target='memory' or 'user'):\n"
        "  - Always loaded into the system prompt at session start. Costs tokens every turn forever.\n"
        "  - SMALL CAP (~600+400 chars combined). Use only for facts that MUST influence every turn.\n"
        "  - Best fit: user preferences, recurring corrections, routing rules, hot environment quirks.\n"
        "  - Targets: 'memory' (your notes) or 'user' (who the user is).\n\n"
        "WARM TIER (tier='warm' on add, or use any warm-only action):\n"
        "  - Searchable via memory(action='recall', query='...'). Not in the prompt by default.\n"
        "  - UNBOUNDED. SQLite + FTS5 keyword search with trust scoring.\n"
        "  - Best fit: factual reference (TDS internals, MCP procedures), debugging notes, "
        "project conventions, lessons learned. Anything you'd otherwise jam into hot tier and run out of room.\n"
        "  - DEFAULT for new content unless it genuinely belongs in hot tier — when in doubt, warm.\n\n"
        "WHEN TO SAVE (proactively, don't wait):\n"
        "- User corrects you or says 'remember this'\n"
        "- User shares a preference / personal detail → HOT (target='user')\n"
        "- You discover environment / project / API quirks → WARM\n"
        "- You learn a stable fact useful in future sessions → WARM unless it's a recurring correction\n\n"
        "ACTIONS:\n"
        "  HOT-TIER: add (target+content), replace (target+old_text+content), "
        "remove (target+old_text), read (target).\n"
        "  WARM-TIER: add (content [+category +tags]), recall (query [+top_k +category]), "
        "recall_related (query OR fact_id), read ([+category +top_k]), "
        "replace (fact_id+content), remove (fact_id), "
        "feedback (fact_id+helpful) — train trust scores by rating retrieved facts.\n"
        "  CROSS-TIER: promote (fact_id [+target]) — move warm fact to hot tier "
        "(target='memory' or 'user', defaults to 'memory'); "
        "demote (old_text [+target +category]) — move hot entry to warm "
        "(target picks the source hot tier; category sets the new warm category).\n\n"
        "RECALL: use memory(action='recall', query='...') when the user references something cross-session, "
        "you suspect related context exists from prior work, or you're debugging a system covered in older notes. "
        "It's keyword search (BM25), so use exact terms / proper nouns when possible. ~50 tokens per call.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO state. "
        "Use session_search for those. If you've solved a non-trivial problem worth reusing, save it as a skill.\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "add", "replace", "remove", "read",
                    "recall", "recall_related",
                    "feedback", "promote", "demote",
                ],
                "description": "The action to perform.",
            },
            "tier": {
                "type": "string",
                "enum": ["hot", "warm"],
                "description": (
                    "Which tier to write/read. Defaults to 'hot' for backward compat. "
                    "Use 'warm' for new content unless it must be always-loaded. "
                    "Warm-only actions (recall, recall_related, feedback, promote, demote) "
                    "ignore this param."
                ),
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": (
                    "'memory' for personal notes, 'user' for user profile. "
                    "Hot tier (add/replace/remove/read): which file to operate on. "
                    "Cross-tier promote: which hot file the fact lands in. "
                    "Cross-tier demote: which hot file the fact comes from. "
                    "Defaults to 'memory'. Ignored for warm-tier-only actions "
                    "(add/recall/recall_related/read/replace/remove/feedback)."
                ),
            },
            "content": {
                "type": "string",
                "description": "Entry content. Required for 'add' and 'replace' (both tiers).",
            },
            "old_text": {
                "type": "string",
                "description": (
                    "Hot tier: short unique substring identifying the entry to replace, remove, "
                    "or demote. Ignored for warm tier (warm uses fact_id)."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Warm-tier search query. Required for 'recall'. Used as the seed for "
                    "'recall_related' if no fact_id is given. Plain text — keyword search."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Warm-tier max results (default 5, max 25). For 'read' max 200.",
            },
            "category": {
                "type": "string",
                "description": (
                    "Warm-tier category filter / assignment. Free-form string "
                    "(e.g. 'tanium', 'debugging', 'preferences'). Defaults to 'general' on add."
                ),
            },
            "tags": {
                "type": "string",
                "description": (
                    "Warm-tier tags on add/replace. Comma-separated free-form (e.g. 'tds,mcp,review')."
                ),
            },
            "fact_id": {
                "type": "integer",
                "description": (
                    "Warm-tier fact id. Required for 'replace'/'remove'/'feedback'/'promote'. "
                    "Returned by 'add'/'recall'."
                ),
            },
            "helpful": {
                "type": "boolean",
                "description": (
                    "Warm-tier feedback flag. True → trust+0.05, helpful_count+1. "
                    "False → trust-0.10. Helps the recall ranker prefer reliable facts."
                ),
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
        # target=None means "not specified" — memory_tool defaults it
        # to 'memory' on hot-tier ops and treats None as a signal to
        # the warm dispatcher's promote/demote branches.
        target=args.get("target"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store"),
        tier=args.get("tier", "hot"),
        query=args.get("query"),
        top_k=args.get("top_k"),
        category=args.get("category"),
        tags=args.get("tags"),
        fact_id=args.get("fact_id"),
        helpful=args.get("helpful"),
    ),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




