"""
Shared storage primitives for the tool failure log.

Used by both ``tools/tool_failure_log.py`` (the agent-facing tool) and
``model_tools.py`` (the auto-log hook in ``handle_function_call``).

All mutations acquire an exclusive advisory lock (``fcntl.LOCK_EX`` on POSIX;
a threading.Lock no-op on native Windows) so read-modify-write sequences
(resolve/update/link) and concurrent appends cannot lose updates.

Security: any text persisted here (error messages, tool arguments) is run
through ``agent.redact.redact_sensitive_text`` before it is written, so a
durable JSONL journal can never retain credentials.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from hermes_constants import get_hermes_home

# fcntl is Unix-only — native Windows has no equivalent advisory lock.
# We fall back to an in-process threading.Lock so that read-modify-write
# sequences within the same process still cannot lose updates; cross-process
# races on native Windows are rare for a failure log and the worst case
# (a lost update on a low-frequency journal) is non-destructive.
try:
    import fcntl
    _HAS_FCNTL = True
except (ImportError, NotImplementedError):
    fcntl = None
    _HAS_FCNTL = False

_MAX_ERROR_LEN = 256
_MAX_ARGS_LEN = 200


def _flock_ex(fh) -> None:
    """Acquire an exclusive advisory lock on *fh* (POSIX only)."""
    if _HAS_FCNTL:
        import fcntl  # local import → typed as the module, not Optional

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)


def _flock_un(fh) -> None:
    """Release the advisory lock held on *fh* (POSIX only)."""
    if _HAS_FCNTL:
        import fcntl  # local import → typed as the module, not Optional

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# Module-level lock guarding every read-modify-write transaction and the
# temporary-file rename dance.  This is the single shared lock the sweeper
# review asked for: resolve/update/link all take it for the whole transaction,
# so an intervening append_record() can never be lost.
_RW_LOCK = threading.Lock()


def _data_dir() -> Path:
    p = get_hermes_home() / "tool_failures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_path() -> Path:
    return _data_dir() / "failures.jsonl"


def _redact(text: str) -> str:
    """Redact secrets from a string before it is persisted durably.

    Never raises — if redaction is unavailable the raw value still goes
    through, but the common path masks credentials per agent/redact.py.
    """
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        return text


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
    """Atomically rewrite the entire log file under the shared RW lock."""
    with _RW_LOCK:
        lp = _log_path()
        tmp = lp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            _flock_ex(fh)
            try:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                _flock_un(fh)
        os.replace(tmp, lp)


def append_record(rec: dict) -> dict:
    """Append one record with auto-assigned id under the shared RW lock.

    The caller should NOT set ``id`` on *rec* — it is assigned atomically
    inside the lock to prevent TOCTOU races between concurrent writers.
    """
    with _RW_LOCK:
        lp = _log_path()
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a+", encoding="utf-8") as fh:
            _flock_ex(fh)
            try:
                existing = _read_lines_unlocked(fh)
                rec["id"] = _next_id(existing)
                rec["e"] = _redact(str(rec.get("e", "")))[:_MAX_ERROR_LEN]
                rec["a"] = _redact(str(rec.get("a", "")))[:_MAX_ARGS_LEN]
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                _flock_un(fh)
    return rec


def mutate_records(matcher, mutator) -> int:
    """Single-transaction read-modify-write over the whole log file.

    *matcher(rec)*: return True if *rec* should be mutated.
    *mutator(rec)*: apply the in-place change to *rec*.

    Both run inside the shared RW lock so that the read, the edits, and the
    rewrite are atomic with respect to concurrent append_record() calls.
    Returns the number of records mutated.
    """
    with _RW_LOCK:
        lp = _log_path()
        if not lp.exists():
            return 0
        with open(lp, "r", encoding="utf-8") as fh:
            _flock_ex(fh)
            try:
                records = _read_lines_unlocked(fh)
            finally:
                _flock_un(fh)

        updated = 0
        for r in records:
            if matcher(r):
                mutator(r)
                updated += 1

        if updated:
            tmp = lp.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                _flock_ex(fh)
                try:
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    _flock_un(fh)
            os.replace(tmp, lp)
        return updated


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
            try:
                args_summary = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_summary = str(args)

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "t": tool_name,
            "e": error_msg,
            "a": args_summary,
            "s": session_id or "",
            "f": "",
            "l": [],
            "r": "pending",
        }
        # id is assigned inside append_record under the lock; args/error are
        # redacted inside append_record before the line is written.
        append_record(rec)
        return rec["id"]
    except Exception:
        # Never let failure logging break the agent loop
        return None
