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
the system prompt — this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read, prune
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
|- Entries can carry an optional metadata tag for scoring/eviction:
|    [260522|t]      ← created 2026-05-22, source=tool (default)
|    [260522|3|u]    ← ref_count=3, source=user
|    [260522|a]      ← source=auto
|  Field encoding: YYMMDD[|count][|source]  (1 bracket pair, pipe-separated)
|  Source codes: u(user), t(tool), a(auto), x(archive)
|  Auto-pruning uses tag data to score and evict low-value entries.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any

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

# ── Entry metadata tags ────────────────────────────────────────────────────
#
# Each entry can optionally begin with a tag like:
#   [260522|t]             ← default: r=1, source=tool
#   [260522|3|u]           ← ref_count=3, source=user
#
# Single-bracket pipe-delimited format: [YYMMDD[|count][|source]]
# Fields:
#   YYMMDD  = creation date
#   count   = reference count (omitted when 1)
#   source  = single char: u(user), t(tool), a(auto), x(archive)
#
# Entries without tags are treated as untagged (score = 0) for pruning
# purposes but otherwise work identically.

TAG_RE = re.compile(
    r"^\[(\d{6})(?:\|(\d+))?\|(\w)\]\s+"
)

def _parse_tag(entry: str) -> dict | None:
    """Extract metadata tag from an entry. Returns None if no tag present."""
    m = TAG_RE.match(entry)
    if not m:
        return None
    return {
        "created": datetime.strptime(m.group(1), "%y%m%d").replace(tzinfo=timezone.utc),
        "ref_count": int(m.group(2)) if m.group(2) else 1,
        "source": m.group(3),
    }

def _format_tag(created: date | None = None, ref_count: int = 1,
                source: str = "tool") -> str:
    """Build a compact metadata tag prefix string.
    
    Format: [YYMMDD[|count][|source]]  (9-13 chars)
    Examples: [260522|t]  [260522|3|u]
    """
    c = (created or date.today()).strftime("%y%m%d")
    r = f"|{ref_count}" if ref_count > 1 else ""
    s_map = {"user": "u", "tool": "t", "auto": "a", "archive": "x"}
    s = s_map.get(source, source[:1])
    return f"[{c}{r}|{s}] "

def _strip_tag(entry: str) -> str:
    """Remove the leading tag from an entry, returning just the content."""
    m = TAG_RE.match(entry)
    if m:
        return entry[m.end():]
    return entry

def _tag_or_none(entry: str) -> str | None:
    """Return the raw tag if present, None otherwise."""
    m = TAG_RE.match(entry)
    return m.group(0) if m else None

# ── Scoring for auto-pruning ───────────────────────────────────────────────

_SOURCE_WEIGHTS = {
    "user": 1.0,
    "tool": 0.8,
    "auto": 0.5,
    "archive": 0.2,
}

_PRUNE_PROTECT_DAYS = int(os.environ.get("HERMES_MEMORY_PRUNE_PROTECT_DAYS", "7"))
_PRUNE_SOFT_THRESHOLD = float(os.environ.get("HERMES_MEMORY_PRUNE_THRESHOLD", "0.80"))
_PRUNE_TARGET_USAGE = float(os.environ.get("HERMES_MEMORY_PRUNE_TARGET", "0.70"))
_PRUNE_MIN_ENTRIES = int(os.environ.get("HERMES_MEMORY_PRUNE_MIN_ENTRIES", "1"))


def _score_entry(
    tag: dict | None,
    now: datetime,
) -> float:
    """Score an entry's value for pruning decisions. 0 = lowest value, 1 = highest.

    Uses three signals, each contributing equally:
      - Freshness: how recently the entry was created
      - Frequency: how often the entry has been referenced
      - Source authority: user > tool > auto > archive
    """
    if tag is None:
        return 0.0  # untagged entries are always pruned first

    days_since = (now - tag["created"]).days
    freshness = max(0.0, 1.0 - days_since / 60.0)
    freq = min(tag["ref_count"], 10) / 10.0
    sw = _SOURCE_WEIGHTS.get(tag["source"], 0.3)

    return 0.3 * freshness + 0.4 * freq + 0.3 * sw


def _bump_ref_count(entry: str) -> str:
    """Increment the 'r' field in an entry's tag. Returns the entry unchanged
    if no tag is present."""
    tag = _parse_tag(entry)
    if tag is None:
        return entry
    prefix = _format_tag(
        created=tag["created"].date(),
        ref_count=tag["ref_count"] + 1,
        source=tag["source"],
    )
    return prefix + _strip_tag(entry)


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


def _scan_memory_content(content: str) -> str | None:
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
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: dict[str, str] = {"memory": "", "user": ""}

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

    def _reload_target(self, target: str, *, skip_drift: bool = False) -> Optional[str]:
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.

        When *skip_drift* is True the round-trip / entry-size check is
        bypassed.  Used by the ``add`` action which appends without
        rewriting, so existing content is never clobbered.
        """
        path = self._path_for(target)
        bak = None if skip_drift else self._detect_external_drift(target)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return bak

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> list[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: list[str]):
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

    # ── Pruning ────────────────────────────────────────────────────────────

    def _prune_if_needed(self, target: str) -> dict | None:
        """Check current usage and prune low-value entries if above threshold.

        Returns a summary dict if pruning happened, None otherwise.
        Called under file lock.
        """
        entries = self._entries_for(target)
        limit = self._char_limit(target)
        current = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        usage = current / limit if limit > 0 else 0

        if usage < _PRUNE_SOFT_THRESHOLD:
            return None

        now = datetime.now(timezone.utc)
        cutoff = now.replace(tzinfo=timezone.utc) - __import__("datetime").timedelta(
            days=_PRUNE_PROTECT_DAYS
        )
        target_usage = current * _PRUNE_TARGET_USAGE

        # Score entries; protect recent ones
        scored: list[tuple[float, str, int]] = []
        protected: list[tuple[float, str, int]] = []
        for idx, entry in enumerate(entries):
            tag = _parse_tag(entry)
            score = _score_entry(tag, now)
            if tag and tag["created"] >= cutoff:
                protected.append((score, entry, idx))
            else:
                scored.append((score, entry, idx))

        # Sort by score ascending (worst first)
        scored.sort(key=lambda x: x[0])
        removed: list[str] = []
        chars_freed = 0

        for score, entry, idx in scored:
            if len(entries) - len(removed) <= _PRUNE_MIN_ENTRIES:
                break
            new_total = current - chars_freed - len(entry)
            new_usage = new_total / limit if limit > 0 else 0
            if new_usage <= target_usage:
                break
            removed.append(entry)
            chars_freed += len(entry)

        if not removed:
            return None

        remaining = [e for i, e in enumerate(entries) if i not in {r[2] for r in
                      [(s, e, i) for s, e, i in scored[:len(removed)]]}]
        # Actually, let me just do it simply:
        remove_indices = {entry[2] for entry in scored[:len(removed)]}
        surviving = [e for i, e in enumerate(entries) if i not in remove_indices]
        self._set_entries(target, surviving)
        self.save_to_disk(target)

        return {
            "pruned": len(removed),
            "chars_freed": chars_freed,
            "remaining": len(surviving),
            "usage_before": f"{current:,}/{limit:,} ({usage*100:.0f}%)",
            "usage_after": f"{(current - chars_freed):,}/{limit:,} ({(current - chars_freed)/limit*100:.0f}%)" if limit else "N/A",
        }

    # ── Mutation methods ───────────────────────────────────────────────────

    def add(self, target: str, content: str) -> dict[str, Any]:
        """Append a new entry. Auto-prunes low-value entries if space is tight."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Tag new entries with creation metadata
        tagged_content = _format_tag() + content

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # For add (append-only), we skip the drift guard — appending never
            # clobbers existing content, so round-trip mismatches from prior
            # tool-written entries in the same session are harmless.  The drift
            # guard remains active for replace/remove where full-file rewrite
            # would discard un-roundtrippable content (issue #26045).
            self._reload_target(target, skip_drift=True)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries or tagged_content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_total = len(ENTRY_DELIMITER.join(entries + [tagged_content])) if entries else len(tagged_content)

            if new_total > limit:
                # Instead of hard-error, try auto-pruning
                prune_result = self._prune_if_needed(target)
                if prune_result:
                    # Retry after pruning - re-read entries since prune modified them
                    entries = self._entries_for(target)
                    new_total = len(ENTRY_DELIMITER.join(entries + [tagged_content])) if entries else len(tagged_content)

                if new_total > limit:
                    current = len(ENTRY_DELIMITER.join(entries)) if entries else 0
                    return {
                        "success": False,
                        "error": (
                            f"Memory at {current:,}/{limit:,} chars. "
                            f"Adding this entry ({len(tagged_content)} chars) would exceed the limit. "
                            f"Auto-prune freed {prune_result.get('pruned', 0) if prune_result else 0} "
                            f"entries but space is still insufficient. "
                            f"Replace or remove existing entries first."
                        ),
                        "current_entries": entries,
                        "usage": f"{current:,}/{limit:,}",
                    }

            entries.append(tagged_content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> dict[str, Any]:
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

            # Preserve the existing tag if present; tag the new content otherwise
            existing_tag = _tag_or_none(entries[idx])
            if existing_tag:
                tagged_new = existing_tag + new_content
            else:
                tagged_new = _format_tag() + new_content

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = tagged_new
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

            entries[idx] = tagged_new
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> dict[str, Any]:
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

    def format_for_system_prompt(self, target: str) -> str | None:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None


    def prune(self, target: str, dry_run: bool = False) -> dict[str, Any]:
        """Explicitly trigger pruning. Returns summary of what would be / was removed.

        Args:
            target: 'memory' or 'user'
            dry_run: if True, only report without modifying entries.
        """
        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = self._entries_for(target)
            limit = self._char_limit(target)
            current = len(ENTRY_DELIMITER.join(entries)) if entries else 0
            usage = current / limit if limit > 0 else 0

            now = datetime.now(timezone.utc)
            cutoff = now - __import__("datetime").timedelta(days=_PRUNE_PROTECT_DAYS)

            scored: list[tuple[float, str, int]] = []
            protected_count = 0
            for idx, entry in enumerate(entries):
                tag = _parse_tag(entry)
                score = _score_entry(tag, now)
                if tag and tag["created"] >= cutoff:
                    protected_count += 1
                scored.append((score, entry, idx))

            scored.sort(key=lambda x: x[0])

            result = {
                "target": target,
                "total_entries": len(entries),
                "usage": f"{current:,}/{limit:,} ({usage*100:.0f}%)" if limit else "N/A",
                "protected_entries": protected_count,
                "prune_suggested": usage >= _PRUNE_SOFT_THRESHOLD,
            }

            if usage >= _PRUNE_SOFT_THRESHOLD:
                # Show bottom 20% scores
                n_show = max(1, len(scored) // 5)
                candidates = []
                for score, entry, idx in scored[:n_show]:
                    tag = _parse_tag(entry)
                    preview = _strip_tag(entry)[:60].replace("\n", " ")
                    candidates.append({
                        "index": idx,
                        "score": round(score, 3),
                        "tag": {
                            "created": tag["created"].strftime("%Y-%m-%d") if tag else None,
                            "ref_count": tag["ref_count"] if tag else 0,
                            "source": tag["source"] if tag else "untagged",
                        } if tag else None,
                        "preview": preview,
                    })
                result["candidates"] = candidates

                if not dry_run:
                    pruned = self._prune_if_needed(target)
                    if pruned:
                        result["pruned"] = pruned
                    else:
                        result["pruned"] = {"pruned": 0, "reason": "all entries protected or high-value"}

            return result

    # ── System prompt integration ───────────────────────────────────────────

    def format_for_system_prompt(self, target: str) -> str | None:
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

    def _success_response(self, target: str, message: str = None) -> dict[str, Any]:
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
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: list[str]) -> str:
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
    def _read_file(path: Path) -> list[str]:
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
    def _write_file(path: Path, entries: list[str]):
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


def load_on_disk_store() -> "MemoryStore":
    """Build a fresh on-disk :class:`MemoryStore`, honoring configured char limits.

    Use this from any context that has no live agent (the messaging gateway, the
    Desktop GUI, the bare CLI ``/memory`` handler) but still needs to read or
    apply approved memory writes. Mirrors how the live agent constructs its store
    in ``agent/agent_init.py`` — including the user's ``memory.memory_char_limit``
    / ``memory.user_char_limit`` overrides — so an approval applied without a live
    agent enforces the SAME caps as one applied with one.

    Falls back to the built-in defaults if config can't be loaded, so this can
    never raise on a missing/unreadable config.
    """
    memory_char_limit = 2200
    user_char_limit = 1375
    try:
        from hermes_cli.config import load_config

        mem_cfg = (load_config() or {}).get("memory", {}) or {}
        memory_char_limit = int(mem_cfg.get("memory_char_limit", memory_char_limit))
        user_char_limit = int(mem_cfg.get("user_char_limit", user_char_limit))
    except Exception:
        pass  # config optional — fall back to defaults rather than break /memory

    store = MemoryStore(
        memory_char_limit=memory_char_limit,
        user_char_limit=user_char_limit,
    )
    store.load_from_disk()
    return store


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
    operations: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
    store: Any = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Two shapes:
      - Single op: action + (content / old_text).
      - Batch:     operations=[{action, content?, old_text?}, ...] applied
                   atomically against the final char budget in ONE call.

    Returns JSON string with results.
    """
    from tools.registry import tool_error

    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    # --- Batch path -------------------------------------------------------
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
        gate_result = _apply_batch_write_gate(target, operations)
        if gate_result is not None:
            return gate_result
        result = store.apply_batch(target, operations)
        return json.dumps(result, ensure_ascii=False)

    # --- Single-op path ---------------------------------------------------
    # Validate required params BEFORE the gate so an invalid write is rejected
    # immediately instead of being staged and only failing at approve time.
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
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
    gate_result = _apply_write_gate(action, target, content, old_text)
    if gate_result is not None:
        return gate_result

    if action == "add":
        result = store.add(target, content)

    elif action == "replace":
        result = store.replace(target, old_text, content)

    elif action == "remove":
        result = store.remove(target, old_text)

    elif action == "prune":
        result = store.prune(target, dry_run=dry_run)

    else:
        return tool_error(f"Invalid action '{action}'. Use 'add', 'replace', 'remove', or 'prune'.", success=False)

    return json.dumps(result, ensure_ascii=False)


# ---- Required env vars / dependencies ----

def check_memory_requirements() -> bool:
    """Memory is always available — no external dependencies."""
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
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that "
        "removes or shortens enough stale entries and adds the new one together.\n\n"
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
                "enum": ["add", "replace", "remove", "prune"],
                "description": (
                    "The action to perform. 'add' appends a new entry (auto-tagged with metadata). "
                    "'replace' finds an entry by substring and replaces it. "
                    "'remove' finds an entry by substring and deletes it. "
                    "'prune' scores all entries and removes low-value ones when usage exceeds 80%."
                ),
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile.",
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'. For 'add', a metadata tag is prepended automatically.",
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                }

            },

            "dry_run": {
                "type": "boolean",
                "description": "For 'prune' action only: if true, only report candidates without removing anything.",
            },
        },
        "required": ["target"],
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
        operations=args.get("operations"),
        dry_run=args.get("dry_run", False),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)
