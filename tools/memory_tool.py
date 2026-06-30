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
- Single `memory` tool with action parameter: add, replace, remove
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional, Tuple

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
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


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
        # Continuous consolidation
        self._write_count: Dict[str, int] = {"memory": 0, "user": 0}
        # Hybrid retrieval indexes (FTS5 + embedding) — lazy initialized
        self._fts_index = None
        self._embedding_index = None
        self._consolidation_interval = 5  # hint after this many writes

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so the user can still SEE poisoned entries via
        see poisoned entries by inspecting the source files directly, and remove them — silently dropping them would hide the attack from the user.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Sanitize entries for the system-prompt snapshot only.  Live state
        # (memory_entries / user_entries) keeps the raw text so the user
        # can see + remove poisoned entries via the memory tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

        # Rebuild hybrid retrieval indexes
        self._rebuild_indexes()

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

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

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
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

    def _reload_target(self, target: str) -> Optional[str]:
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.
        """
        path = self._path_for(target)
        bak = self._detect_external_drift(target)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return bak

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
            # Re-read from disk under lock to pick up writes from other sessions.
            # If external drift was detected, the file was backed up to .bak.<ts>
            # — refuse the mutation so we don't clobber the un-roundtrippable
            # content the patch tool / shell append / sister session wrote.
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

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
                        f"Consolidate now: use 'replace' to merge overlapping entries into "
                        f"shorter ones or 'remove' stale or less important entries (see "
                        f"current_entries below), then retry this add — all in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)
            self._increment_write_count(target)

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
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
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
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content, or 'remove' other stale or less important "
                        f"entries to make room (see current_entries below), then retry — all "
                        f"in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)
            self._increment_write_count(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
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
            self._increment_write_count(target)

        return self._success_response(target, "Entry removed.")

    def apply_batch(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply a sequence of add/replace/remove ops to one target atomically.

        All operations are validated and applied against the FINAL budget --
        intermediate overflow is irrelevant. This lets the model free space
        (remove/replace) and add new entries in a SINGLE tool call instead of
        the multi-turn consolidate-then-retry dance that re-sends the whole
        conversation context several times.

        Semantics: all-or-nothing. If any op is malformed, doesn't match, or
        the net result would exceed the char limit, NOTHING is written and an
        error is returned describing the first failure plus the live state.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content for injection/exfil BEFORE touching
        # disk -- a single poisoned op rejects the whole batch.
        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content")
            if act in {"add", "replace"} and new_content:
                scan_error = _scan_memory_content(new_content)
                if scan_error:
                    return {"success": False, "error": f"Operation {i + 1}: {scan_error}"}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            # Work on a copy; only commit if the whole batch validates.
            working: List[str] = list(self._entries_for(target))
            limit = self._char_limit(target)

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    # Apply context and signal tags from the operation
                    ctx = op.get("context")
                    sig = op.get("signal")
                    if ctx and not content.startswith("[context:"):
                        content = f"[context: {ctx}] {content}"
                    if sig is not None and not content.startswith("[signal:"):
                        # Insert signal after context tag if present
                        sig_tag = f"[signal: {sig:.1f}]"
                        if content.startswith("[context:"):
                            # content is now "[context: type] actual text"
                            idx = content.index("]") + 1
                            content = content[:idx] + f" {sig_tag}" + content[idx:]
                        else:
                            content = f"{sig_tag} {content}"
                    if content in working:
                        continue  # idempotent -- skip duplicate, don't fail the batch
                    working.append(content)

                elif act == "replace":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working[matches[0]] = content

                elif act == "remove":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working.pop(matches[0])

                else:
                    return self._batch_error(
                        target,
                        f"{pos}: unknown action. Use add, replace, or remove.",
                    )

            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"After applying all {len(operations)} operations, memory would be at "
                        f"{new_total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                        f"entries in the same batch (see current_entries below), then retry."
                    ),
                    "current_entries": self._entries_for(target),
                    "usage": f"{current:,}/{limit:,}",
                }

            # Commit.
            self._set_entries(target, working)
            self.save_to_disk(target)
            self._increment_write_count(target, len(operations))

        return self._success_response(target, f"Applied {len(operations)} operation(s).")

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return {
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
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

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # The success response is intentionally TERMINAL: it confirms the write
        # landed and tells the model to stop. We do NOT echo the full entries
        # list here -- dumping it invites the model to "find more to fix" and
        # re-issue the same operations (observed thrash: the correct batch on
        # call 1, then 5 redundant repeats). Entries are only shown on the
        # error/over-budget paths, where the model genuinely needs them to
        # decide what to consolidate.
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        # Consolidation hint: when write count passes interval, suggest it
        if self._consolidation_hint(target):
            resp["consolidation_hint"] = True
            resp["consolidation"] = (
                f"Consolidation recommended ({self._write_count[target]} writes since last pass). "
                f"Call memory(action='consolidate', target='{target}') to merge/prune/decay "
                f"or handle it manually via operations=[remove low-signal entries]."
            )
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    # ── Hybrid retrieval indexes (FTS5 + Embedding) ───────────────────────

    def _ensure_indexes(self) -> bool:
        """Lazy-init FTS5 and embedding indexes if not already created."""
        if self._fts_index is None:
            from agent.memory.fts_index import FTS5Index, default_fts_path
            try:
                self._fts_index = FTS5Index(default_fts_path())
            except Exception as e:
                logger.debug("FTS5 index init failed: %s", e)
                self._fts_index = None

        if self._embedding_index is None:
            from agent.memory.embedding_index import EmbeddingIndex, default_embedding_cache_dir
            try:
                self._embedding_index = EmbeddingIndex(default_embedding_cache_dir())
            except Exception as e:
                logger.debug("Embedding index init failed: %s", e)
                self._embedding_index = None

        return self._fts_index is not None or self._embedding_index is not None

    def _rebuild_indexes(self) -> None:
        """Rebuild FTS5 and embedding indexes from current entries.

        Called after load_from_disk() and after significant content changes.
        Failures are non-fatal — indexes gracefully degrade.
        """
        if not self._ensure_indexes():
            return

        # FTS5 rebuild
        if self._fts_index is not None:
            try:
                combined = list(self.memory_entries) + list(self.user_entries)
                self._fts_index.rebuild(combined)
            except Exception as e:
                logger.debug("FTS5 rebuild failed: %s", e)

        # Embedding rebuild (only for memory entries — user profile is less
        # suited to semantic search)
        if self._embedding_index is not None:
            try:
                self._embedding_index.rebuild(self.memory_entries)
            except Exception as e:
                logger.debug("Embedding rebuild failed: %s", e)

    def close_indexes(self) -> None:
        """Close index resources on agent shutdown."""
        if self._fts_index is not None:
            try:
                self._fts_index.close()
            except Exception:
                pass
        if self._embedding_index is not None:
            try:
                self._embedding_index.close()
            except Exception:
                pass

    # ── Context-tag support ──────────────────────────────────────────────
    # Entries can carry a [context: <type>] prefix to associate them with
    # specific workflow types. _render_block groups by this tag, and the
    # frozen snapshot guidance tells the agent to prioritise context-matched
    # entries for the current task.

    _CONTEXT_TAG_RE = re.compile(r'^\[context:\s*(\w[\w-]*)\]\s*(.*)', re.DOTALL)

    # ── Signal-strength tag ──────────────────────────────────────────────
    # Each entry can carry a [signal: 0.0-1.0] score indicating confidence /
    # importance. Higher signal = less likely to be pruned during char-limit
    # consolidation. Default is 0.5 (neutral).

    _SIGNAL_TAG_RE = re.compile(r'\[signal:\s*([0-9]*\.?[0-9]+)\]\s*', re.DOTALL)
    DEFAULT_SIGNAL = 0.5

    # ── Relevance scoring ─────────────────────────────────────────────────
    # Methods for scoring memory entries by relevance to the current task.
    # Used by the scored prefetch pipeline (agent/memory/).

    @classmethod
    def scored_prefetch(
        cls,
        entries: List[str],
        task_type: Optional[str] = None,
        task_confidence: float = 0.0,
        keywords: Optional[List[Tuple[str, float]]] = None,
        expanded_query: str = "",
        fts_scores: Optional[Dict[int, float]] = None,
        embedding_scores: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score all entries against a task context, return ranked scored list.

        Hybrid scoring pipeline:
          1. FTS5 BM25 score (if ``fts_scores`` provided — maps entry_index -> score)
          2. Embedding semantic similarity (if ``embedding_scores`` provided)
          3. Context-tag match score
          4. Signal strength
          5. Keyword overlap (fallback when FTS5 unavailable)

        Each result dict contains:
            {content, score, provider, context_tag, signal_strength,
             score_semantic, score_keyword, score_context, score_signal}
        """
        from agent.memory.scored_memory import (
            ScoredMemory,
            context_tag_match,
            HYBRID_WEIGHTS,
        )

        if not entries:
            return []

        scored: List[Dict[str, Any]] = []
        for i, raw in enumerate(entries):
            sm = ScoredMemory.from_raw_entry(
                raw, provider="builtin", entry_index=i,
            )

            # Hybrid dimension 1: FTS5 BM25 score (keyword relevance)
            if fts_scores and i in fts_scores:
                sm.score_keyword = fts_scores[i]
            else:
                # Fallback: keyword overlap heuristic
                if keywords and expanded_query:
                    query_lower = expanded_query.lower()
                    clean_content = sm.content.lower()
                    from agent.memory.scored_memory import _PROJECT_KEYWORDS
                    matched = sum(w for kw, w in _PROJECT_KEYWORDS.items()
                                  if kw in query_lower and kw in clean_content)
                    max_kw = 4.0
                    sm.score_keyword = min(matched / max_kw, 1.0)
                else:
                    sm.score_keyword = 0.0

            # Hybrid dimension 2: Embedding semantic similarity
            if embedding_scores and i in embedding_scores:
                sm.score_semantic = embedding_scores[i]
            else:
                # Fallback to 0.0 — no semantic signal without embeddings
                sm.score_semantic = 0.0

            # Hybrid dimension 3: Context-tag match
            sm.score_context = context_tag_match(sm.context_tag, task_type)

            # Hybrid dimension 4: Signal strength (normalised [0.3-1.0])
            norm_signal = (sm.signal_strength - 0.3) / 0.7 if sm.signal_strength > 0.3 else sm.signal_strength / 0.3
            sm.score_signal = max(0.0, min(1.0, norm_signal))

            # Compute composite using configurable weights
            sm.compute_score()

            scored.append({
                "content": sm.content,
                "score": sm.score,
                "provider": sm.provider,
                "source_file": sm.source_file,
                "entry_index": sm.entry_index,
                "context_tag": sm.context_tag,
                "signal_strength": sm.signal_strength,
                "score_semantic": sm.score_semantic,
                "score_keyword": sm.score_keyword,
                "score_context": sm.score_context,
                "score_entity": sm.score_entity,
                "score_temporal": sm.score_temporal,
                "score_signal": sm.score_signal,
            })

        # Sort by composite score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def scored_prefetch_instance(
        self,
        task_type: Optional[str] = None,
        task_confidence: float = 0.0,
        keywords: Optional[List[Tuple[str, float]]] = None,
        expanded_query: str = "",
    ) -> List[Dict[str, Any]]:
        """Instance method: scored_prefetch using this store's FTS5+embedding indexes.

        Pre-computes FTS5 and embedding scores from the live indexes and
        delegates to the classmethod.
        """
        fts_scores: Dict[int, float] = {}
        embedding_scores: Dict[int, float] = {}
        entries = self.memory_entries

        if not entries or not expanded_query:
            return self.scored_prefetch(
                entries,
                task_type=task_type,
                task_confidence=task_confidence,
                keywords=keywords,
                expanded_query=expanded_query,
            )

        self._ensure_indexes()

        # FTS5 search
        if self._fts_index is not None:
            try:
                fts_results = self._fts_index.search(expanded_query, limit=len(entries) * 2)
                for r in fts_results:
                    # Map back to entry index by content matching
                    content = r.get("content", "")
                    for i, e in enumerate(entries):
                        if content in e or e in content:
                            fts_scores[i] = max(fts_scores.get(i, 0), r.get("score", 0))
                            break
            except Exception:
                logger.debug("FTS5 search in scored_prefetch failed", exc_info=True)

        # Embedding search
        if self._embedding_index is not None:
            try:
                emb_results = self._embedding_index.search(expanded_query, entries, limit=len(entries))
                for r in emb_results:
                    idx = r.get("entry_index")
                    if idx is not None:
                        embedding_scores[idx] = r.get("score", 0.5)
            except Exception:
                logger.debug("Embedding search in scored_prefetch failed", exc_info=True)

        return self.scored_prefetch(
            entries,
            task_type=task_type,
            task_confidence=task_confidence,
            keywords=keywords,
            expanded_query=expanded_query,
            fts_scores=fts_scores if fts_scores else None,
            embedding_scores=embedding_scores if embedding_scores else None,
        )

    # ── Context descriptions ──────────────────────────────────────────────
    _CONTEXT_DESCRIPTIONS = {
        'deploy': 'infrastructure, deployments, configuration changes',
        'feature': 'building features (branch+PR, TDD, testing)',
        'debug': 'root cause analysis, bug fixing, proving causation',
        'review': 'code review, PR review, security audit',
        'research': 'investigating options, competitor analysis, exploration',
        'planning': 'strategy, architecture decisions, roadmap',
        'content': 'writing, documentation, social media, changelogs',
        'maintenance': 'repo org, cleanup, dep updates, caching',
        'kanban': 'kanban board operations, task routing, orchestrator',
        'design': 'UI/UX, mockups, visual design, component architecture',
    }

    @classmethod
    def _group_by_context(cls, entries: List[str]) -> Dict[str, List[str]]:
        """Group entries by [context: <type>] tag. Untagged → 'general'."""
        groups: Dict[str, List[str]] = {}
        for entry in entries:
            m = cls._CONTEXT_TAG_RE.match(entry)
            if m:
                ctx = m.group(1).lower()
                # Strip the tag from display (it's the grouping key, not helpful prose)
                display = (m.group(2) or entry).strip()
                # Also strip any signal tag from the display text
                signal, display = cls._parse_signal(display)
                groups.setdefault(ctx, []).append(display)
            else:
                groups.setdefault('general', []).append(entry)
        return groups

    @classmethod
    def _parse_signal(cls, entry: str) -> Tuple[float, str]:
        """Extract [signal: X.X] from an entry (may appear anywhere).

        Returns (signal_value, remaining_content with tag stripped).
        Defaults to 0.5 if no signal tag found.
        """
        m = cls._SIGNAL_TAG_RE.search(entry)
        if m:
            # Remove just the [signal: X.X] tag from the entry
            cleaned = (entry[:m.start()] + entry[m.end():]).strip()
            return (float(m.group(1)), cleaned)
        return (cls.DEFAULT_SIGNAL, entry)

    @classmethod
    def _by_signal(cls, entries: List[str], reverse: bool = False) -> List[Tuple[str, float]]:
        """Sort entries by signal strength (ascending = lowest first).

        Returns list of (entry, signal) tuples. Useful for consolidation
        decisions: remove from the front (lowest signal) first.
        """
        with_signal = [(e, cls._parse_signal(e)[0]) for e in entries]
        return sorted(with_signal, key=lambda x: x[1], reverse=reverse)

    def _signal_range(self, entries: List[str]) -> str:
        """Get a compact signal display string for a list of FULL entries (with tags intact)."""
        signals = [self._parse_signal(e)[0] for e in entries]
        if not signals:
            return ""
        low = min(signals)
        high = max(signals)
        if low == high:
            return f"signal {low:.1f}"
        return f"signal {low:.1f}-{high:.1f}"

    def _signal_range_for_context(self, entries: List[str], ctx: str) -> str:
        """Signal range for entries belonging to a specific context type."""
        ctx_tag = f"[context: {ctx}]"
        matching = [e for e in entries if ctx_tag in e]
        if not matching:
            # Fallback: check entries without explicit context
            if ctx == 'general':
                matching = [e for e in entries if not e.startswith("[context:")]
            else:
                return ""
        return self._signal_range(matching)

    # ── Continuous consolidation ─────────────────────────────────────────
    # Consolidation mini-batches run every N writes or on-demand.
    # They prune decayed/low-signal entries, merge overlapping content,
    # apply signal decay/boost, and keep char usage in a healthy range.

    _CONSOLIDATE_TARGET_PCT = 75    # aim to stay at or below this % after consolidation
    _CONSOLIDATE_PRUNE_PCT = 65     # hard floor — never prune below this
    _SIGNAL_DECAY_PER_PASS = 0.03   # signal reduction per entry per consolidation
    _SIGNAL_BOOST_PER_PASS = 0.02   # signal increase for survivors
    _SIGNAL_FLOOR = 0.30            # lowest signal decays to; entries below get pruned
    _SIGNAL_CEILING = 1.0

    # ── Multi-tier depth table ───────────────────────────────────────────
    # Defines which memory tiers each task type should access, at what depth.
    # Static here; designed to be extracted to a config/service later.
    #
    # Keys: l0 (session buffer), l1_depth_days (session_search), l2_context
    #       (which context to prioritise), l3 (whether to query fleet graph).
    #
    # l1_depth_days = None means "full depth" (no limit).
    # NOTE: this table is byte-stable in the frozen snapshot. When extracted
    # to a service, the service would inject it at prompt-build time.

    TIER_DEPTH: Dict[str, Dict[str, Any]] = {
        "debug": {
            "l0": True,
            "l1_depth_days": 7,
            "l2_context": "debug",
            "l3": False,
        },
        "review": {
            "l0": True,
            "l1_depth_days": 1,
            "l2_context": "review",
            "l3": False,
        },
        "feature": {
            "l0": False,
            "l1_depth_days": 3,
            "l2_context": "feature",
            "l3": False,
        },
        "deploy": {
            "l0": False,
            "l1_depth_days": 1,
            "l2_context": "deploy",
            "l3": True,
        },
        "planning": {
            "l0": False,
            "l1_depth_days": None,
            "l2_context": "planning",
            "l3": True,
        },
        "research": {
            "l0": False,
            "l1_depth_days": None,
            "l2_context": "research",
            "l3": True,
        },
        "content": {
            "l0": False,
            "l1_depth_days": 1,
            "l2_context": "content",
            "l3": False,
        },
        "maintenance": {
            "l0": False,
            "l1_depth_days": 1,
            "l2_context": "maintenance",
            "l3": False,
        },
    }

    @classmethod
    def _tier_depth_guidance(cls) -> str:
        """Build the tier-depth guidance block for the system prompt.
        
        Tells the agent which memory depth to access per detected task type.
        """
        lines = [
            "── Multi-tier memory depths ──",
            "Memory has 4 tiers:",
            "  L0 = current session (your context window)",
            "  L1 = recent sessions (session_search tool)",
            "  L2 = [context: X] entries above (injected in this prompt)",
            "  L3 = fleet graph (mcp_graph_api_graph_list_decisions tool)",
            "",
            "Depth by detected task type:",
        ]
        # Sort by name for stable output
        for task in sorted(cls.TIER_DEPTH.keys()):
            cfg = cls.TIER_DEPTH[task]
            parts = []
            if cfg["l0"]:
                parts.append("L0")
            d = cfg["l1_depth_days"]
            if d is not None:
                parts.append(f"L1({d}d)")
            elif cfg["l1_depth_days"] is None and task in cls.TIER_DEPTH:
                # None means full depth — but only mention L1 if it's used
                # For planning/research, L1 isn't primary, so skip
                pass
            parts.append(f"L2({cfg['l2_context']})")
            if cfg["l3"]:
                parts.append("L3(query decisions)")
            lines.append(f"  {task:12s} → {' + '.join(parts)}")

        lines.append("")
        lines.append(
            "When you detect the task type, use session_search with the "
            "corresponding depth_days and limit. "
        )
        lines.append(
            "For L3 query: mcp_graph_api_graph_list_decisions(tags='context:<type>') "
            "to surface relevant strategic decisions."
        )
        return "\n".join(lines)

    def _consolidate_mini_batch(self, target: str) -> Dict[str, Any]:
        """Run one mini-batch consolidation pass on the given target store.

        Flow:
          1. Strip signal decoys (entries below signal floor)
          2. Merge same-context entries with high word overlap
          3. If still over target %, prune lowest-signal entries
          4. Apply signal decay to survivors, boost to consolidated entries
          5. Write back to disk, reset write counter

        Returns a summary dict (not a tool response).
        """
        entries = self._entries_for(target)
        if not entries:
            self._write_count[target] = 0
            return {"status": "empty", "removed": 0, "merged": 0}

        limit = self._char_limit(target)
        before_count = len(entries)
        before_chars = self._char_count(target)

        # --- Step 1: Strip entries below signal floor ---
        survivors: List[str] = []
        pruned_count = 0
        for e in entries:
            sig, _ = self._parse_signal(e)
            if sig < self._SIGNAL_FLOOR:
                pruned_count += 1
            else:
                survivors.append(e)
        pruned_signal = pruned_count

        # --- Step 2: Merge same-context entries with high word overlap ---
        merged_count = 0
        # Group by context
        groups = self._group_by_context(survivors)
        merged: List[str] = []

        # Build reverse mapping: context tag text → all entries for that context
        # We need to merge within each context group
        for ctx, display_entries in groups.items():
            # We have the display text, but need original entries.
            # Re-match from survivors.
            ctx_tag = f"[context: {ctx}]"
            ctx_entries = [e for e in survivors if ctx_tag in e] \
                          if ctx != 'general' else \
                          [e for e in survivors if not e.startswith("[context:")]

            if len(ctx_entries) <= 1:
                merged.extend(ctx_entries)
                continue

            # Sort by signal ascending (lowest first = consolidation targets)
            sorted_ctx = self._by_signal(ctx_entries)

            # Greedy merge: walk entries, merge adjacent ones that overlap
            merged_ctx: List[str] = []
            skip = set()
            for i, (entry_i, sig_i) in enumerate(sorted_ctx):
                if i in skip:
                    continue
                best_j = None
                best_overlap = 0
                _, text_i = self._parse_signal(entry_i)
                words_i = set(text_i.lower().split())
                for j, (entry_j, sig_j) in enumerate(sorted_ctx):
                    if j <= i or j in skip:
                        continue
                    _, text_j = self._parse_signal(entry_j)
                    words_j = set(text_j.lower().split())
                    if not words_i or not words_j:
                        continue
                    overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                    if overlap >= 0.40 and overlap > best_overlap:
                        best_overlap = overlap
                        best_j = j

                if best_j is not None:
                    # Merge entry_j into entry_i, take higher signal
                    _, text_j = self._parse_signal(sorted_ctx[best_j][0])
                    _, text_i_clean = self._parse_signal(entry_i)
                    # Short form: keep the longer/more detailed entry content
                    merged_text = text_i_clean if len(text_i_clean) >= len(text_j) else text_j
                    new_sig = min(self._SIGNAL_CEILING, max(sig_i, sig_j) + self._SIGNAL_BOOST_PER_PASS)

                    # Reconstruct with tags — preserve context from entry_i
                    if ctx_tag in entry_i:
                        _, after_ctx = entry_i.split("]", 1)
                        # Remove old signal tag if present
                        _, clean_content = self._parse_signal(after_ctx)
                        new_entry = f"{ctx_tag} [signal: {new_sig:.1f}] {merged_text}"
                    else:
                        new_entry = f"[signal: {new_sig:.1f}] {merged_text}"
                    merged_ctx.append(new_entry)
                    skip.add(best_j)
                    merged_count += 1
                else:
                    merged_ctx.append(entry_i)
            merged.extend(merged_ctx)

        entries = merged if merged_count > 0 else survivors

        # --- Step 3: Prune if over target char % ---
        current_chars = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        target_chars = int(limit * self._CONSOLIDATE_TARGET_PCT / 100)
        pruned_threshold = 0

        if current_chars > target_chars:
            # Sort by signal ascending, remove lowest until under threshold
            by_signal = self._by_signal(entries)
            keep: List[str] = []
            keep_chars = 0
            hard_floor = int(limit * self._CONSOLIDATE_PRUNE_PCT / 100)
            for entry, sig in reversed(by_signal):  # highest signal first
                if keep_chars + len(entry) <= target_chars or keep_chars <= hard_floor:
                    keep.append(entry)
                    keep_chars += len(entry) + (1 if keep else 0)
                else:
                    pruned_threshold += 1
            entries = list(reversed(keep)) if keep else entries

        # --- Step 4: Apply signal decay/boost ---
        final: List[str] = []
        decay_applied = 0
        for e in entries:
            sig, rest = self._parse_signal(e)
            # Decay: reduce signal slightly (memories weaken if not reinforced)
            new_sig = max(self._SIGNAL_FLOOR, sig - self._SIGNAL_DECAY_PER_PASS)
            if new_sig != sig:
                decay_applied += 1
            # Rebuild entry with new signal
            # Restore original tags but update signal value
            e_updated = self._update_signal_in_entry(e, new_sig)
            final.append(e_updated)

        # --- Step 5: Write back ---
        self._set_entries(target, final)
        self.save_to_disk(target)
        self._write_count[target] = 0

        after_chars = self._char_count(target)
        after_count = len(final)

        return {
            "status": "consolidated",
            "target": target,
            "removed_pruned": pruned_count,
            "removed_threshold": pruned_threshold,
            "merged": merged_count,
            "decayed": decay_applied,
            "entries_before": before_count,
            "entries_after": after_count,
            "chars_before": before_chars,
            "chars_after": after_chars,
            "usage": f"{after_chars:,}/{limit:,} ({int(after_chars/limit*100)}%)",
        }

    @classmethod
    def _update_signal_in_entry(cls, entry: str, new_signal: float) -> str:
        """Replace the [signal: X.X] tag in an entry with a new value.
        If no signal tag exists, append one.
        """
        m = cls._SIGNAL_TAG_RE.search(entry)
        new_tag = f"[signal: {new_signal:.1f}]"
        if m:
            return entry[:m.start()] + new_tag + entry[m.end():]
        # No existing signal tag — insert after context tag or at start
        if entry.startswith("[context:"):
            idx = entry.index("]") + 1
            return entry[:idx] + f" {new_tag}" + entry[idx:]
        return f"{new_tag} {entry}"

    def _increment_write_count(self, target: str, count: int = 1):
        """Increment write counter for the given target."""
        self._write_count[target] = self._write_count.get(target, 0) + count

    def _consolidation_hint(self, target: str) -> bool:
        """Return True if consolidation is recommended (write count >= interval)."""
        return self._write_count.get(target, 0) >= self._consolidation_interval

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with context-grouped sections.

        Entries are grouped by ``[context: <type>]`` tag so the agent
        can prioritise entries relevant to the current task type.
        Untagged entries go under a "General" section.
        """
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

        # Group by context tag for task-type relevance
        groups = self._group_by_context(entries)
        sections = []

        # General (untagged) first
        if 'general' in groups:
            sig_display = self._signal_range_for_context(entries, 'general')
            label = "[General — applies to all tasks]"
            if sig_display:
                label += f" ({sig_display})"
            sections.append(label)
            block = ENTRY_DELIMITER.join(groups['general'])
            sections.append(block)

        # Known contexts, sorted alphabetically
        for ctx in sorted(k for k in groups if k != 'general'):
            desc = self._CONTEXT_DESCRIPTIONS.get(ctx, '')
            ctx_entries = groups[ctx]
            sig_display = self._signal_range_for_context(entries, ctx)
            label = f"[context: {ctx}]"
            if sig_display:
                label += f" ({sig_display})"
            if desc:
                label += f" — {desc}"
            sections.append(label)
            sections.append(ENTRY_DELIMITER.join(groups[ctx]))

        body = "\n\n".join(sections)

        # Context-awareness + signal-strength + tier-depth guidance
        known_types = ", ".join(sorted(k for k in self._CONTEXT_DESCRIPTIONS if k != 'general'))
        tier_block = self._tier_depth_guidance()
        guidance = (
            "\n\n── Context-aware memory ──\n"
            "Entries grouped by [context: <type>]; each carries a [signal: 0.0-1.0] strength.\n"
            "Prioritise entries matching the current task type. "
            "Continuous consolidation: every ~5 writes, consider calling "
            "memory(action='consolidate', target=...) to auto-merge/prune/decay. "
            "When consolidating manually, remove lowest-signal entries first.\n"
            f"Known context types: {known_types}.\n"
            "Use the 'signal' parameter on memory() save to set importance; "
            "default 0.5. Higher = keep longer under consolidation pressure.\n"
            f"\n{tier_block}"
        )

        return f"{separator}\n{header}\n{separator}\n{body}{guidance}"

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

    def _detect_external_drift(self, target: str) -> Optional[str]:
        """Return a backup-path string if on-disk content shows external drift.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds the
           store's whole-file char limit. The tool budgets the ENTIRE store
           against that limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Returns the absolute path of the .bak file when drift was found and
        backed up; returns None when the file looks tool-shaped.

        Note: this is an INSTANCE method (not static) because we need the
        per-target char_limit for signal #2.
        """
        path = self._path_for(target)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return None
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # Drift confirmed — snapshot the file so the operator can recover
        # whatever the external writer added, then return the .bak path so
        # the caller can refuse the mutation.
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

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


def _apply_write_gate(action: str, target: str, content: Optional[str],
                      old_text: Optional[str]) -> Optional[str]:
    """Evaluate the memory write gate. Returns a JSON tool-result string when
    the write should NOT proceed normally (blocked or staged), or None when the
    caller should perform the real write.

    Only the mutating actions (add/replace/remove) are gated.
    """
    if action not in {"add", "replace", "remove"}:
        return None

    try:
        from tools import write_approval as wa
    except Exception:
        # If the gate module can't load, fail open (current behaviour) rather
        # than blocking all memory writes.
        return None

    # Build a small inline summary/detail for the foreground approval prompt.
    label = "user profile" if target == "user" else "memory"
    if action == "add":
        summary = f"add to {label}"
        detail = content or ""
    elif action == "replace":
        summary = f"replace in {label}"
        detail = f"old: {old_text}\nnew: {content}"
    else:  # remove
        summary = f"remove from {label}"
        detail = old_text or ""

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    # stage
    payload = {
        "action": action,
        "target": target,
        "content": content,
        "old_text": old_text,
    }
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _apply_batch_write_gate(target: str, operations: List[Dict[str, Any]]) -> Optional[str]:
    """Evaluate the write gate for a batch of memory operations.

    Returns a JSON tool-result string when the batch should NOT proceed
    (blocked or staged), or None when the caller should perform the real
    batch write. The whole batch is gated as a single unit.
    """
    try:
        from tools import write_approval as wa
    except Exception:
        return None

    label = "user profile" if target == "user" else "memory"
    summary = f"apply {len(operations)} op(s) to {label}"
    detail_lines = []
    for op in operations:
        op = op or {}
        act = op.get("action", "?")
        if act == "remove":
            detail_lines.append(f"- remove: {op.get('old_text', '')}")
        elif act == "replace":
            detail_lines.append(f"- replace: {op.get('old_text', '')} -> {op.get('content', '')}")
        else:
            detail_lines.append(f"- {act}: {op.get('content', '')}")
    detail = "\n".join(detail_lines)

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    payload = {"action": "batch", "target": target, "operations": operations}
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """Build a recoverable error for a replace/remove call that arrived without
    ``old_text``.

    ``replace``/``remove`` are inherently targeted -- without ``old_text`` there
    is no entry to act on, so we cannot fulfil the call. But returning a bare
    "old_text is required" is a dead-end: some structured-output clients omit the
    optional ``old_text`` field (it isn't, and can't be, schema-required without
    a top-level combinator the Codex backend rejects -- see
    tests/tools/test_memory_tool_schema.py). So instead we return the current
    entry inventory plus an explicit retry instruction, letting the model reissue
    the call with ``old_text`` set to a unique substring of the entry it means.
    Mirrors the batch path's ``_batch_error`` shape. (issues #43412, #49466)
    """
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- a short unique substring of the entry "
                f"to {action}. None was provided. Reissue the {action} with old_text "
                f"set to part of one of the current_entries below."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )


def memory_tool(
    action: str = None,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    context: str = None,
    signal: float = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Two shapes:
      - Single op: action + (content / old_text) [+ optional context + signal].
      - Batch:     operations=[{action, content?, old_text?, context?, signal?}, ...] applied
                   atomically against the final char budget in ONE call.

    The optional ``context`` parameter tags an entry with a workflow type
    (e.g. ``deploy``, ``debug``, ``feature``, ``review``, ``research``).
    Tagged entries are grouped by context in the system prompt so the agent
    can prioritise task-relevant memories.

    The optional ``signal`` parameter (float 0.0-1.0) sets the entry's signal
    strength — higher values mean the entry should be preserved longer during
    consolidation when near the char limit. Default 0.5.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    # --- Apply context/signal tags to content -----------------------------
    def _tag(content: str, ctx: Optional[str], sig: Optional[float]) -> str:
        if not content:
            return content
        parts = []
        if ctx and not content.startswith("[context:"):
            parts.append(f"[context: {ctx}]")
        if sig is not None and not content.startswith("[signal:"):
            parts.append(f"[signal: {sig:.1f}]")
        if parts:
            return " ".join(parts) + " " + content
        return content

    # --- Batch path -------------------------------------------------------
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
        gate_result = _apply_batch_write_gate(target, operations)
        if gate_result is not None:
            return gate_result
        result = store.apply_batch(target, operations)
        return json.dumps(result, ensure_ascii=False)

    # --- Consolidate action (no write gate needed) -------------------------
    if action == "consolidate":
        result = store._consolidate_mini_batch(target)
        return json.dumps(result, ensure_ascii=False)

    # --- Single-op path ---------------------------------------------------
    # Validate required params BEFORE the gate so an invalid write is rejected
    # immediately instead of being staged and only failing at approve time.
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)

    # Apply context/signal tag to single-op content
    tagged_content = _tag(content, context, signal)

    if action == "replace" and (not old_text or not content):
        missing = "old_text" if not old_text else "content"
        if not old_text:
            # The client/model omitted old_text. Replace is inherently targeted
            # -- we can't guess which entry. Return the current inventory plus a
            # retry instruction so the model can reissue with old_text set,
            # instead of hitting a dead-end error. (issues #43412, #49466)
            return _missing_old_text_error(store, target, "replace")
        return tool_error(f"{missing} is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    # Approval gate: when on, stages the write (background/gateway) or prompts
    # inline (interactive CLI); when off (default) passes straight through.
    gate_result = _apply_write_gate(action, target, tagged_content if action == "add" else content, old_text)
    if gate_result is not None:
        return gate_result

    if action == "add":
        result = store.add(target, tagged_content)

    elif action == "replace":
        result = store.replace(target, old_text, content)

    elif action == "remove":
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


def apply_memory_pending(payload: Dict[str, Any], store: "MemoryStore") -> Dict[str, Any]:
    """Replay a staged memory write directly against the store, bypassing the
    write gate. Called by the /memory approve handler.

    Returns the store's result dict.
    """
    action = payload.get("action")
    target = payload.get("target", "memory")
    content = payload.get("content") or ""
    old_text = payload.get("old_text") or ""
    if action == "batch":
        return store.apply_batch(target, payload.get("operations") or [])
    if action == "add":
        return store.add(target, content)
    if action == "replace":
        return store.replace(target, old_text, content)
    if action == "remove":
        return store.remove(target, old_text)
    return {"success": False, "error": f"Unknown staged action '{action}'."}
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory is "
        "injected into every future turn, so keep entries compact and high-signal.\n\n"
        "SIGNAL STRENGTH: Use the 'signal' parameter (float 0.0-1.0) to set an entry's "
        "importance. Higher signal = preserved longer during consolidation when near the "
        "char limit. Default 0.5. Signal + context combine: "
        "save preferences with context='deploy' signal=0.9 for deployment-critical facts.\n\n"
        "CONSOLIDATION: When near the char limit, prune in one batch call: "
        "remove lowest-signal entries, shorten verbose ones, keep high-signal entries. "
        "Signal values guide what to keep vs discard.\n\n"
        "CONTEXT TAGGING: Use the 'context' parameter to tag entries with a workflow type "
        "(e.g. 'deploy', 'feature', 'debug', 'review', 'research', 'planning', 'content', "
        "'maintenance'). Tagged entries are grouped by context in the system prompt so the "
        "agent can prioritise task-relevant memories.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?, context?, signal?}). The batch applies atomically and "
        "the char limit is checked only on the FINAL result — so a single call can remove/replace "
        "stale entries to free room AND add new ones, even when an add alone would overflow. "
        "The response reports current/limit chars and confirms completion; one batch call "
        "finishes the update, so don't repeat it.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that "
        "removes low-signal or stale entries and adds the new one together.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your "
        "notes (environment, conventions, tool quirks, lessons).\n\n"
        "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, "
        "completed-work logs, temporary TODO state (use session_search for those). Reusable "
        "procedures belong in a skill, not memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "consolidate"],
                "description": "The action to perform. 'add'/'replace'/'remove' for single ops (omit when using 'operations'). 'consolidate' runs a mini-batch pass: strips decayed entries (signal<0.3), merges overlapping same-context content, applies signal decay/boost, and prunes to stay within char limits."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape)."
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
            },
            "context": {
                "type": "string",
                "description": "OPTIONAL workflow context tag. Groups this entry under [context: <type>] so it surfaces for relevant tasks. Known types: deploy, feature, debug, review, research, planning, content, maintenance, kanban, design."
            },
            "signal": {
                "type": "number",
                "description": "OPTIONAL signal strength (0.0-1.0). Higher = preserved longer during consolidation when near char limit. Default 0.5."
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?, context?, signal?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                        "context": {"type": "string", "description": "Optional workflow context tag (deploy, feature, debug, review, etc.)."},
                        "signal": {"type": "number", "description": "Optional signal strength (0.0-1.0) for consolidation prioritization."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
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
        context=args.get("context"),
        signal=args.get("signal"),
        operations=args.get("operations"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




