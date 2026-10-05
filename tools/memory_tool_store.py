"""MemoryStore — bounded, file-backed curated memory (MEMORY.md / USER.md).
Entries are joined by ``ENTRY_DELIMITER``; budgets are in chars (model-independent).
Module state that tests monkeypatch (``get_memory_dir``, ``fcntl``/``msvcrt``) stays
in ``tools.memory_tool`` and is read lazily."""

import logging
import os
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.threat_patterns import first_threat_message as _first_threat_message
from tools.memory_entry_identity import (deduplicate_entries, id_fields, inherit_identity,
                                         new_entry, parse_entries)
from tools.memory_store_io import (apply_change, detect_external_drift, prepare_mutation,
                                   read_file, read_raw_checked, write_file)

logger = logging.getLogger("tools.memory_tool")

# Block header prefixes rendered by _render_block; agent/conversation_compression.py
# matches them to detect a leftover block for an emptied target — keep in lockstep.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)", "user": "USER PROFILE (who the user is)"}

ENTRY_DELIMITER = "\n§\n"


def _scan_memory_content(content: str) -> Optional[str]:
    """Error string if *content* matches injection/exfil patterns. Strict scope:
    memory enters the system prompt, so a poisoned entry persists across sessions."""
    return _first_threat_message(content, scope="strict")


def _error(message: str, **extra) -> Dict[str, Any]:
    return {"success": False, "error": message, **extra}


def _drift_error(path: Path, bak_path: str) -> Dict[str, Any]:
    """External drift: the file wouldn't round-trip, so flushing would discard content."""
    return _error((
        f"Refusing to write {path.name}: file on disk has content that wouldn't round-trip "
        f"through the memory tool (likely added by the patch tool, a shell append, a manual edit, "
        f"or a concurrent session). A snapshot was saved to {bak_path}. Resolve the drift first — "
        f"either rewrite the file as a clean §-delimited list of entries, or move the extra "
        f"content out — then retry. This guard exists to prevent silent data loss (issue #26045)."
    ), drift_backup=bak_path, remediation=(
        "Open the .bak file, integrate the missing entries into the memory tool one at a time via "
        "memory(action=add, content=...), then remove or rewrite the original file to a clean state."))


def _read_failed_error(path: Path) -> Dict[str, Any]:
    """Existing-but-unreadable file: saving from an assumed-empty view would wipe it."""
    return _error(
        f"Refusing to write {path.name}: the file exists on disk but could not be read right now "
        f"(temporarily locked by another program, a permission change, invalid/corrupt text encoding, "
        f"or a filesystem error). Treating an unreadable file as empty and saving would wipe existing "
        f"memory, so the write is refused. Nothing was changed — retry in a moment.")


def _find_unique_match(entries: List[str], old_text: str) -> Tuple[Optional[int], bool]:
    """``(index, ambiguous)`` for entries matching *old_text*. A whole-entry
    EXACT match (``old_text == entry``) takes absolute priority — substring
    matches are only considered when no entry equals *old_text*, so a short
    entry stays addressable even when its full text is contained inside a
    longer sibling entry (remove('test') vs '...tests pass...'). Exact-duplicate
    matches are safe (first wins); distinct matches → ``(None, True)``."""
    exact = [i for i, e in enumerate(entries) if e == old_text]
    matches = exact if exact else [i for i, e in enumerate(entries) if old_text in e]
    if len({getattr(entries[i], "entry_id", entries[i]) for i in matches}) > 1:
        return None, True
    return (matches[0] if matches else None), False


def _pinned_index(entries: List[str], matched_entry: str) -> Optional[int]:
    """Index of the exact entry a staged write was reviewed against; None once it is gone (stale)."""
    index, ambiguous = _find_unique_match(entries, matched_entry)
    return None if ambiguous or index is None or entries[index] != matched_entry else index


def _stale_entry_message(entry: str) -> str:
    return (f"Entry changed since it was staged, so this write was not applied: '{entry}' is no longer "
            f"in memory as reviewed. Recreate the change against the current entry or reject it; the "
            f"pending record has been preserved.")


# Optional third value of an _apply closure: a dict merged into the success payload —
# e.g. the full text an entry-level replace overwrote, so a whole-entry write is never
# silent about the loss (#117952). Same convention as _error(**extra).


class MemoryStore:
    """Bounded curated memory with file persistence; one instance per AIAgent.
    ``_system_prompt_snapshot`` is frozen at load time (prefix-cache stable);
    ``memory_entries`` / ``user_entries`` are live state persisted to disk."""

    # Failed consolidation attempts (overflow / zero-match) allowed per turn before
    # a TERMINAL "save skipped" result, so a fragile replace/add can't loop the turn
    # to budget exhaustion and suppress the user's reply.
    # See #42405.
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375, *,
                 memory_enabled: bool = True, user_profile_enabled: bool = True):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit, self.user_char_limit = memory_char_limit, user_char_limit
        self.memory_enabled, self.user_profile_enabled = memory_enabled, user_profile_enabled
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        self._identity_enabled = {"memory": False, "user": False}
        self._consolidation_failures = 0  # per turn; reset by reset_consolidation_failures()

    # Per-turn counter of failed at-capacity consolidation attempts; reset at each turn boundary by
    # reset_consolidation_failures() (#42405).
    def target_enabled(self, target: str) -> bool:
        return self.user_profile_enabled if target == "user" else self.memory_enabled

    def reset_consolidation_failures(self) -> None:
        """Call at turn start."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count a consolidation failure: under the per-turn cap return ``response``
        (it says how to retry); past it a TERMINAL result so the model stops looping.

        Once the cap is exceeded, drop the retry instruction and return a TERMINAL result so the model stops
        looping memory calls and proceeds to answer the user — a failed memory side effect must never block
        the turn's reply (#42405).
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {"success": False, "done": True, "error": (
            f"Memory consolidation failed {self._consolidation_failures} times this turn. Stop retrying "
            "memory calls — leave memory unchanged for now and continue with your reply to the user. "
            "The fact can be saved in a later turn.")}

    def load_from_disk(self):
        """Load MEMORY.md / USER.md and capture the frozen system-prompt snapshot.
        Threat hits are replaced by a ``[BLOCKED: …]`` placeholder in the SNAPSHOT only;
        live lists keep the raw text so the user can see and remove poisoned entries
        (dropping them silently would hide the attack)."""
        from tools.threat_patterns import scan_for_threats

        def _sanitize(entry, filename):
            # Strict scope, same as writes; empty / already-blocked entries pass through.
            findings = scan_for_threats(entry, scope="strict") if entry and not entry.startswith("[BLOCKED:") else None
            if not findings:
                return entry
            logger.warning("Memory entry from %s blocked at load time: %s", filename, ", ".join(findings))
            return (f"[BLOCKED: {filename} entry contained threat pattern(s): {', '.join(findings)}. "
                    f"Removed from system prompt; use memory(action=remove) to delete the original.]")

        for target in ("memory", "user"):
            path = self._path_for(target)
            from hermes_constants import mkdir_under_hermes_home

            mkdir_under_hermes_home(path.parent)
            # Deduplicate (order-preserving, first occurrence wins).
            entries = deduplicate_entries(self._read_file(path))
            self._set_entries(target, entries)
            # External writers (MCP bridges, hand edits) can exceed the cap; the limit only fires on
            # add/replace, so the oversized block would silently ride in the prompt while every later
            # add is refused with no visible cause (#10877). Warn; never truncate a user's memories.
            if (count := self._char_count(target)) > (limit := self._char_limit(target)):
                logger.warning("%s exceeds its char limit on load: %d/%d chars. Entries stay loaded; "
                               "further additions are blocked until it is back under the limit.",
                               path.name, count, limit)
            self._system_prompt_snapshot[target] = self._render_block(target, [_sanitize(e, path.name) for e in entries])

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Exclusive lock on a separate .lock file so the memory file itself can
        still be atomically replaced."""
        from tools import memory_tool as _mt  # fcntl/msvcrt live (and are patched) there
        fcntl, msvcrt = _mt.fcntl, _mt.msvcrt
        lock_path = path.with_suffix(path.suffix + ".lock")
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(lock_path.parent)
        if fcntl is None and msvcrt is None:
            yield
            return
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        raw_fd = os.open(lock_path, flags, 0o600)
        try:
            # The creation mode is filtered through the process umask and does
            # not repair a lock left loose by an older Hermes process. Tighten
            # the opened inode before acquiring the lock so both cases are
            # owner-only. Operating on the fd avoids a path-swap window.
            if hasattr(os, "fchmod"):
                os.fchmod(raw_fd, 0o600)
            fd = os.fdopen(raw_fd, "r+", encoding="utf-8")
        except Exception:
            os.close(raw_fd)
            raise
        with fd:
            def _flock(unlock: bool):
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX)
                else:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_LOCK, 1)
            _flock(False)
            try:
                yield
            finally:
                with suppress(OSError):
                    _flock(True)

    @staticmethod
    def _path_for(target: str) -> Path:
        from tools import memory_tool  # get_memory_dir is monkeypatched there
        return memory_tool.get_memory_dir() / ("USER.md" if target == "user" else "MEMORY.md")

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        setattr(self, "user_entries" if target == "user" else "memory_entries", entries)

    def _char_count(self, target: str) -> int:
        return len(ENTRY_DELIMITER.join(self._entries_for(target)))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _usage(self, target: str) -> str:
        return f"{self._char_count(target):,}/{self._char_limit(target):,}"

    def _usage_pct(self, target: str, current: int) -> str:
        limit = self._char_limit(target)
        return f"{min(100, int((current / limit) * 100)) if limit > 0 else 0}% — {current:,}/{limit:,} chars"

    def _failure_with_entries(self, target: str, message: str) -> Dict[str, Any]:
        """Consolidation failure carrying the live entries so the model can consolidate."""
        return self._consolidation_failure(
            _error(message, current_entries=self._entries_for(target), usage=self._usage(target)))

    def _batch_failure(self, target: str, message: str) -> Dict[str, Any]:
        """Batch-abort failure WITHOUT ``current_entries``: the store did not change and the
        caller already holds the inventory, so echoing it made each consolidation retry
        grow the context it was invoked to shrink (#97316)."""
        return self._consolidation_failure(
            _error(message + " No operations were applied (batch is all-or-nothing).", usage=self._usage(target)))

    def _mutate(self, target: str, mutate, *, skip_drift: bool = False) -> Dict[str, Any]:
        """Lock, re-read from disk, run ``mutate(entries, limit)`` -> ``(new_entries, message)``
        or an error dict, then persist and return the success response. The reload aborts
        on an existing-but-unreadable file (even append-only ``add`` rewrites the whole
        file) and, unless *skip_drift*, on external drift (flushing would discard
        un-roundtrippable content). Drift check and parse use the SAME raw snapshot —
        a failed second read used to count as "no drift". The closure may return a
        third value, a dict merged into the success payload (``_error``'s ``**extra``
        convention) — e.g. the full text a replace overwrote (#117952). ANY dict the closure
        returns is passed through verbatim and nothing is persisted: error dicts, or the
        success payload of a read-only closure (``resolve_entry``, ``resolve_batch_entries``)."""
        path = self._path_for(target)
        with self._file_lock(path):
            raw = prepare_mutation(self, target, path, skip_drift=skip_drift)
            if isinstance(raw, dict):
                return raw
            return apply_change(self, target, path, raw, mutate)

    def add(self, target: str, content: str, *, scope: Optional[str] = None) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return _error("Content cannot be empty.")
        if scan_error := _scan_memory_content(content):
            return _error(scan_error)

        def _add(entries, limit):
            existing = [e for e in entries if e == content and getattr(e, "scope", None) == scope]
            if existing:
                return self._success_response(target, "Entry already exists (no duplicate added).", **id_fields(existing[0]))
            if scope is not None and not self._identity_enabled[target]:
                return _error("Scoped writes require explicit memory identity migration first.")
            if len(ENTRY_DELIMITER.join(entries + [content])) > limit:
                return self._failure_with_entries(target, (
                    f"Memory at {self._char_count(target):,}/{limit:,} chars. Adding this entry "
                    f"({len(content)} chars) would exceed the limit. Consolidate now: use 'replace' to merge "
                    f"overlapping entries into shorter ones or 'remove' stale or less important entries (see "
                    f"current_entries below), then retry this add — all in this turn."))
            item = new_entry(content, target, scope) if self._identity_enabled[target] else content
            return entries + [item], "Entry added.", {"_identity_index": len(entries)}
        # Append-only: skip the drift guard (appending never clobbers foreign
        # content) but still refuse a failed read — add rewrites the WHOLE file.
        return self._mutate(target, _add, skip_drift=True)

    def replace(self, target: str, old_text: str, new_content: str,
                matched_entry: Optional[str] = None) -> Dict[str, Any]:
        """Find the entry containing old_text (whole-entry exact match first) and
        replace the WHOLE entry with new_content — old_text only locates the entry;
        the matched span is not spliced into it."""
        new_content = new_content.strip()
        if not old_text.strip():
            return _error("old_text cannot be empty.")
        if not new_content:
            return _error("new_content cannot be empty. Use 'remove' to delete entries.")
        if scan_error := _scan_memory_content(new_content):
            return _error(scan_error)
        return self._edit(target, old_text.strip(), new_content, matched_entry)

    def remove(self, target: str, old_text: str, matched_entry: Optional[str] = None) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        if not old_text.strip():
            return _error("old_text cannot be empty.")
        return self._edit(target, old_text.strip(), None, matched_entry)

    def _locate(self, entries: List[str], old_text: str, verb: str, matched_entry: Optional[str] = None):
        """Index of the entry *old_text* selects, or the error dict the edit returns. A write
        staged for approval carries the FULL entry it was reviewed against (*matched_entry*):
        only that exact entry qualifies, so replay never hits a newer entry that still
        contains old_text."""
        if matched_entry is not None:
            idx = _pinned_index(entries, matched_entry)
            return idx if idx is not None else _error(_stale_entry_message(matched_entry))
        idx, ambiguous = _find_unique_match(entries, old_text)
        if ambiguous:
            return _error(f"Multiple entries matched '{old_text}'. Be more specific.",
                          matches=[e[:80] + ("..." if len(e) > 80 else "") for e in entries if old_text in e])
        if idx is None:
            return self._consolidation_failure(_error(
                f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text "
                f"of the entry you want to {verb}.", current_entries=entries))
        return idx

    def resolve_entry(self, target: str, old_text: str, verb: str) -> Dict[str, Any]:
        """``{"success": True, "matched_entry": <full entry>}`` for the entry *old_text* selects
        now, read under the lock, or the error the direct edit would return."""
        def _resolve(entries, limit):
            idx = self._locate(entries, old_text.strip(), verb)
            return idx if isinstance(idx, dict) else {"success": True, "matched_entry": entries[idx], **id_fields(entries[idx])}
        return self._mutate(target, _resolve, skip_drift=True)

    def _edit(self, target: str, old_text: str, new_content: Optional[str],
              matched_entry: Optional[str] = None) -> Dict[str, Any]:
        """Locked replace (``new_content`` set) or remove (None) of the entry matching *old_text*."""
        def _apply(entries, limit):
            idx = self._locate(entries, old_text, "replace" if new_content else "remove", matched_entry)
            if isinstance(idx, dict):
                return idx
            replaced = entries[:idx] + ([] if new_content is None else [inherit_identity(new_content, entries[idx])]) + entries[idx + 1:]
            if new_content is None:
                return replaced, "Entry removed.", {"removed_entry": entries[idx], **id_fields(entries[idx])}
            new_total = len(ENTRY_DELIMITER.join(replaced))
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"Replacement would put memory at {new_total:,}/{limit:,} chars. Shorten the new content, "
                    f"or 'remove' other stale or less important entries to make room (see current_entries "
                    f"below), then retry — all in this turn."))
            return replaced, "Entry replaced.", {"replaced_entry": entries[idx], "_identity_index": idx}
        return self._mutate(target, _apply)

    def apply_batch(self, target: str, operations: List[Dict[str, Any]], *, replay: bool = False) -> Dict[str, Any]:
        """Apply add/replace/remove ops atomically against the FINAL budget, so one call
        can free space and add entries. All-or-nothing: any malformed / unmatched op or
        an over-limit result writes NOTHING and returns the first failure. Aborts do not
        echo ``current_entries`` — the store is unchanged and the model already has it."""
        return self._batch(target, operations, commit=True, replay=replay)

    def resolve_batch_entries(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dry-run ``apply_batch`` under the lock without persisting: the same content scan,
        op walk, empty-store and budget checks, so it fails exactly where the direct batch
        would; on success ``{"success": True, "matched_entries": [...]}`` — per op, the FULL
        entry its replace/remove selects now (None for add), in batch order."""
        return self._batch(target, operations, commit=False)

    def _batch(self, target: str, operations: List[Dict[str, Any]], *, commit: bool,
               replay: bool = False) -> Dict[str, Any]:
        from tools.memory_tool_batch import apply_batch
        return apply_batch(self, target, operations, commit=commit, replay=replay)

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Frozen load-time snapshot (NOT live state — mid-session writes don't touch
        it, preserving the prefix cache); None if empty."""
        return self._system_prompt_snapshot.get(target, "") or None

    def _success_response(self, target: str, message: str = None, **extra) -> Dict[str, Any]:
        """TERMINAL and WITHOUT the entries list: echoing entries invites the model to
        "find more to fix" and re-issue the same ops. A successful write resets the
        per-turn failure budget. ``**extra`` mirrors ``_error``'s convention — e.g. the
        full text a replace overwrote (#117952), deliberately visible despite the
        no-entries rule: silent data loss is the failure this field exists to prevent."""
        # A successful write means the consolidation loop made progress, so the per-turn failure budget
        # resets (the cap counts consecutive failures, not lifetime ones within a turn) (#42405).
        self._consolidation_failures = 0
        return {"success": True, "done": True, "target": target,
                "usage": self._usage_pct(target, self._char_count(target)),
                "entry_count": len(self._entries_for(target)), **({"message": message} if message else {}),
                **extra,
                "note": "Write saved. This update is complete — do not repeat it."}

    def _render_block(self, target: str, entries: List[str]) -> str:
        """System prompt block: header + usage indicator + entries ("" when empty)."""
        if not entries:
            return ""
        content, sep = ENTRY_DELIMITER.join(entries), "═" * 46
        title = MEMORY_BLOCK_HEADERS["user" if target == "user" else "memory"]
        return f"{sep}\n{title} [{self._usage_pct(target, len(content))}]\n{sep}\n{content}"

    _read_raw_checked = staticmethod(read_raw_checked)
    _parse_entries = staticmethod(parse_entries)
    _read_file = staticmethod(read_file)
    _write_file = staticmethod(write_file)
    _detect_external_drift = detect_external_drift
