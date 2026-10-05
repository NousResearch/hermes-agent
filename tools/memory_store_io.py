"""Persistence and identity-aware parsing for the existing bounded memory store."""
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from utils import atomic_write_text
from tools.memory_entry_identity import (DELIMITER as ENTRY_DELIMITER, IdentityError, deduplicate_entries,
                                         enabled, encode_entries, id_fields, parse_entries)

logger = logging.getLogger("tools.memory_tool")


def read_raw_checked(path: Path) -> Tuple[str, bool]:
    """``(raw, read_ok)``; ``read_ok`` is False ONLY when the file EXISTS but can't be
    read. Decoding stays STRICT (``errors="replace"`` would hand callers a lossy view
    a save then persists); ``utf-8-sig`` strips a Notepad BOM off the first entry."""
    if not path.exists():
        return "", True
    try:
        # utf-8-sig strips a leading UTF-8 BOM (Notepad-edited memory files on Windows) and is
        # byte-identical to utf-8 otherwise. Plain utf-8 kept U+FEFF glued to the first entry,
        # corrupting matching/dedup for that entry forever (#10878 / PR #10888). Decode errors stay
        # STRICT on purpose: errors="replace" would hand read-modify-write callers a lossy view that a
        # subsequent save persists over the real bytes — the wipe class documented above. Undecodable
        # bytes must surface as read_ok=False.
        return path.read_text(encoding="utf-8-sig"), True
    except (OSError, UnicodeDecodeError):
        return "", False

def write_file(path: Path, entries: List[str], *, identity_enabled: bool = False):
    """Write prose plus opt-in metadata atomically under the existing memory file lock."""
    target = "user" if path.name == "USER.md" else "memory"
    content = encode_entries(entries, identity_enabled=identity_enabled, target=target)
    try:
        atomic_write_text(path, content, tmp_prefix=".mem_")
    except OSError as exc:
        raise RuntimeError(f"Failed to write memory file {path}: {exc}") from exc


def detect_external_drift(self, target: str, raw: str) -> Optional[str]:
    """``.bak.<ts>`` snapshot path if *raw* shows external drift, else None. Signals:
    round-trip mismatch, or one entry over the whole-file limit (no tool-written
    entry can be — an external writer appended free-form text)."""
    parsed = self._parse_entries(raw, target=target)
    if not raw.strip() or (raw.strip() == encode_entries(parsed, identity_enabled=enabled(raw), target=target)
                           and max(map(len, parsed), default=0) <= self._char_limit(target)):
        return None
    path = self._path_for(target)
    bak_path = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    try:
        bak_path.write_text(raw, encoding="utf-8")
    except OSError:
        return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
    return str(bak_path)


def read_file(path: Path) -> List[str]:
    from tools.memory_tool_store import MemoryStore
    try:
        target = "user" if path.name == "USER.md" else "memory"
        return parse_entries(MemoryStore._read_raw_checked(path)[0], target=target)
    except IdentityError as exc:
        logger.warning("Cannot load memory identities from %s: %s", path.name, exc)
        return []


def prepare_mutation(store, target, path, *, skip_drift):
    from tools.memory_tool_store import _drift_error, _error, _read_failed_error
    raw, read_ok = store._read_raw_checked(path)
    if not read_ok:
        return _read_failed_error(path)
    try:
        entries = parse_entries(raw, target=target)
        bak = None if skip_drift else store._detect_external_drift(target, raw)
    except IdentityError as exc:
        return _error(str(exc))
    store._identity_enabled[target] = enabled(raw)
    store._set_entries(target, deduplicate_entries(entries))
    return _drift_error(path, bak) if bak else raw


def apply_change(store, target, path, raw, mutate):
    from tools.memory_tool_store import _error
    from hermes_constants import mkdir_under_hermes_home
    try:
        result = mutate(store._entries_for(target), store._char_limit(target))
        if isinstance(result, dict):
            return result
        entries = result[0]
        mkdir_under_hermes_home(path.parent)
        if enabled(raw):
            store._write_file(path, entries, identity_enabled=True)
        else:
            store._write_file(path, entries)
    except IdentityError as exc:
        return _error(str(exc))
    store._set_entries(target, entries)
    extras = dict(result[2]) if len(result) > 2 else {}
    index = extras.pop("_identity_index", None)
    if index is not None:
        extras.update(id_fields(entries[index]))
    return store._success_response(target, result[1], **extras)
