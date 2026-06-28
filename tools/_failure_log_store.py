"""
Shared storage primitives for the tool failure log.

Used by both ``tools/tool_failure_log.py`` (the agent-facing tool) and
``model_tools.py`` (the auto-log hook in ``handle_function_call``).

All mutations acquire an exclusive lock where available (``fcntl.LOCK_EX``
on POSIX; no-op on native Windows) to prevent TOCTOU races and lost-update
conflicts between concurrent writers.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from hermes_constants import get_hermes_home

# fcntl is Unix-only — native Windows has no equivalent advisory lock.
# We degrade gracefully: concurrent writes are rare for a failure log
# and the worst case (duplicate id) is non-destructive.
try:
    import fcntl
    _HAS_FCNTL = True
except (ImportError, NotImplementedError):
    _HAS_FCNTL = False

_MAX_ERROR_LEN = 256
_MAX_ARGS_LEN = 200


def _data_dir() -> Path:
    p = get_hermes_home() / "tool_failures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_path() -> Path:
    return _data_dir() / "failures.jsonl"


def _next_id(records: List[dict]) -> int:
    """Derive the next id from the highest id across all records."""
    max_id = 0
    for r in records:
        rid = r.get("id", 0)
        if isinstance(rid, int) and rid > max_id:
            max_id = rid
    return max_id + 1


def _read_lines_unlocked(fh) -> List[dict]:
    """Read all valid JSONL records from an already-open file handle.

    Caller must hold the lock.  Handle is rewound and read from the start.
    Returns records in insertion order (oldest first)."""
    fh.seek(0)
    records: List[dict] = []
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def read_all() -> List[dict]:
    """Return every valid record from the log, newest first (lock-free snapshot)."""
    lp = _log_path()
    if not lp.exists():
        return []
    with open(lp, "r", encoding="utf-8") as fh:
        records = _read_lines_unlocked(fh)
    records.reverse()  # newest first
    return records


def write_records(records: List[dict]) -> None:
    """Atomically rewrite the entire log file under an exclusive lock."""
    lp = _log_path()
    tmp = lp.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        if _HAS_FCNTL:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if _HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    os.replace(tmp, lp)


def append_record(rec: dict) -> dict:
    """Append one record with auto-assigned id under an exclusive lock.

    The caller should NOT set ``id`` on *rec* — it is assigned atomically
    inside the lock to prevent TOCTOU races between concurrent writers.
    """
    lp = _log_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a+", encoding="utf-8") as fh:
        if _HAS_FCNTL:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            existing = _read_lines_unlocked(fh)
            rec["id"] = _next_id(existing)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if _HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return rec


def auto_log(
    tool_name: str,
    error_msg: str,
    args: Optional[dict] = None,
    session_id: str = "",
) -> Optional[int]:
    """Called from ``handle_function_call`` and ``registry.dispatch`` error paths.

    Returns the record id, or None if logging failed (never raises).
    """
    try:
        args_summary = ""
        if args:
            args_summary = json.dumps(args, ensure_ascii=False)[:_MAX_ARGS_LEN]

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "t": tool_name,
            "e": error_msg[:_MAX_ERROR_LEN],
            "a": args_summary,
            "s": session_id or "",
            "f": "",
            "l": [],
            "r": "pending",
        }
        # id is assigned inside append_record under the lock
        append_record(rec)
        return rec["id"]
    except Exception:
        # Never let failure logging break the agent loop
        return None
