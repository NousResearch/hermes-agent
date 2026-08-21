"""Reader for the legacy interrupted-turn sidecar (migration only).

Interrupted-turn records live in ``state.db``'s ``interrupted_turns`` table
(see ``SessionDB.record_interrupted_turn``). Before that they lived in this
JSON sidecar, one file per ``HERMES_HOME``, rewritten in full on every turn
start. This module is what remains of it: a reader the gateway calls once per
home to import surviving entries into the table, after which the file is
renamed and never read again.

Nothing writes the sidecar any more. The whole-file rewrite it used was
synchronized only by a lock local to the writing process, so two processes
sharing a home could each load the map, change one key, and store the result,
silently dropping the other's record; and any process could delete any entry,
including one belonging to a turn still running elsewhere. Both are structural
properties of the file format, which is why the record moved to a table with
per-row writes and an owner stamp.

Reads are best-effort: an unreadable or corrupt file degrades to "no entries"
instead of raising, so a bad sidecar cannot block a resume.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MARKER_DIR = "desktop"
_MARKER_FILE = "interrupted_turns.json"
_MIGRATED_SUFFIX = ".migrated"


def _marker_path(home: Path | str) -> Path:
    return Path(home) / _MARKER_DIR / _MARKER_FILE


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s; ignoring", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _coerce_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    prompt = str(entry.get("prompt") or "")
    if not prompt.strip():
        return None
    try:
        started_at = float(entry.get("started_at") or 0)
        attempts = max(0, int(entry.get("attempts") or 0))
    except (TypeError, ValueError):
        return None
    return {"attempts": attempts, "prompt": prompt, "started_at": started_at}


def sidecar_exists(home: Path | str) -> bool:
    """Whether this home still has a sidecar left to import."""
    try:
        return _marker_path(home).is_file()
    except Exception:
        return False


def retire_sidecar(home: Path | str) -> None:
    """Rename the sidecar aside once its entries are in the table.

    Raises on failure so the caller can leave the file where it is and retry
    on the next resume; the rename is what makes the import one-shot.
    """
    path = _marker_path(home)
    os.replace(path, path.with_name(path.name + _MIGRATED_SUFFIX))


def read_turn_markers(home: Path | str) -> dict[str, dict[str, Any]]:
    """Every usable entry in the legacy sidecar, keyed as the file keyed them.

    Keys are the session key the turn was running under when it was recorded,
    which is the compression segment rather than the lineage root the table is
    keyed on — the importer resolves that.
    """
    try:
        raw = _load(_marker_path(home))
    except Exception:
        return {}
    entries = {}
    for session_key, entry in raw.items():
        coerced = _coerce_entry(entry)
        if session_key and coerced is not None:
            entries[str(session_key)] = coerced
    return entries
