"""Append-only metadata-only JSONL query log.

Append-only by construction: the file is opened with O_APPEND and we
never seek/truncate. Writes are guarded by an intra-process lock; an
OS-level file lock (``flock``) is held for the duration of each write
to keep concurrent processes from interleaving bytes within one line.
This is the only persistence side of the plugin that touches disk.

Acceptance contract:
- May contain only: timestamp, task/session/turn IDs, opaque query ID,
  project/topic labels, SHA-256 digest of normalized search terms
  (never raw terms), consulted relative path/range, source freshness,
  character counts, outcome, redaction counts, DATA_HOLD reason.
- Must never contain: extracted content, snippets, credentials, tokens,
  raw PII, raw user prompts, raw search terms.
- File mode 0o600; parent dir mode 0o700.
- If ``query_log_enabled`` is true and the log cannot be written, fail
  closed — return ``QueryLogWriteError`` and let the caller convert it
  to ``data_hold``. (Acceptance test #15.)
- If ``query_log_enabled`` is false, also fail closed — the audit trail
  is part of the contract; "logging disabled" is not an excuse to skip.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional


class QueryLogWriteError(RuntimeError):
    """Raised when an enabled query log cannot be appended to."""


# Forbidden keys (recursively dropped). The intent: never log anything
# the spec forbids us from logging.
_FORBIDDEN_KEYS = frozenset({
    "query", "raw_query", "prompt", "raw_prompt", "search_terms_raw",
    "extracted_text", "extracted_content", "content", "snippet", "snippets",
    "secret", "token", "password", "api_key", "credential", "private_key",
    "raw_pii", "email", "phone",
})


def normalize_terms(terms: Iterable[str]) -> List[str]:
    """Lowercase, strip, dedupe (preserving first-occurrence order)."""
    seen: set[str] = set()
    out: List[str] = []
    for t in terms or ():
        if not isinstance(t, str):
            continue
        s = t.strip().lower()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def terms_sha256(terms: Iterable[str]) -> str:
    """Stable SHA-256 hex of the normalised terms joined by ``\\x1f`` (unit
    separator — never appears in a real query, so no concatenation
    ambiguity)."""
    norm = normalize_terms(terms)
    payload = "\x1f".join(norm).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scrub_forbidden(obj: Any) -> Any:
    """Recursively drop forbidden keys from a log entry. Defence in depth:
    callers should not pass these in the first place, but if they do, we
    never write them.
    """
    if isinstance(obj, dict):
        return {
            k: _scrub_forbidden(v)
            for k, v in obj.items()
            if k not in _FORBIDDEN_KEYS
        }
    if isinstance(obj, list):
        return [_scrub_forbidden(v) for v in obj]
    return obj


@dataclass(frozen=True)
class QueryLogEntry:
    query_id: str
    queried_at: str  # ISO-8601
    task_id: Optional[str]
    run_id: Optional[str]
    project_topic: str
    decision_horizon: str
    search_terms: List[str]  # raw terms — will be hashed, never logged
    consulted: List[Dict[str, Any]]
    total_characters_read: int
    budget_status: str
    outcome: str
    data_hold_reason: Optional[str]
    redaction_counts: Dict[str, int]
    sensitive_payload_logged: bool = False  # always False; asserted on read


class QueryLogger:
    """Profile-scoped append-only JSONL query logger."""

    def __init__(self, state_dir: Path, *, query_log_enabled: bool):
        self._dir = Path(state_dir)
        self._path = self._dir / "query-log.jsonl"
        self._enabled = bool(query_log_enabled)
        self._lock = Lock()  # intra-process
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self._dir, 0o700)
        except OSError as exc:
            raise QueryLogWriteError(f"cannot create state dir: {exc}") from exc

    @property
    def path(self) -> Path:
        return self._path

    def _build_record(self, entry: QueryLogEntry) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "query_id": entry.query_id,
            "queried_at": entry.queried_at,
            "task_id": entry.task_id,
            "run_id": entry.run_id,
            "project_topic": entry.project_topic,
            "decision_horizon": entry.decision_horizon,
            "search_terms_sha256": terms_sha256(entry.search_terms),
            "consulted": _scrub_forbidden(entry.consulted),
            "total_characters_read": entry.total_characters_read,
            "budget_status": entry.budget_status,
            "outcome": entry.outcome,
            "data_hold_reason": entry.data_hold_reason,
            "redaction_counts": _scrub_forbidden(entry.redaction_counts),
            "sensitive_payload_logged": False,
        }
        return _scrub_forbidden(record)

    def append(self, entry: QueryLogEntry) -> None:
        """Append one entry as a single JSONL line.

        Fails closed: when ``query_log_enabled`` is False OR the OS write
        fails, raises :class:`QueryLogWriteError`. The vault_context
        handler converts that exception to a ``data_hold`` result.
        """
        if not self._enabled:
            raise QueryLogWriteError("query log is disabled (fail-closed)")

        record = self._build_record(entry)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        encoded = line.encode("utf-8")

        with self._lock:
            # Open O_APPEND | O_CREAT | O_WRONLY with 0o600 from the start.
            # We don't truncate, we never seek.
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            fd = os.open(self._path, flags, 0o600)
            try:
                # Ensure the file is at least mode 0o600 — operators may
                # have tweaked it.
                try:
                    st = os.fstat(fd)
                    if stat.S_IMODE(st.st_mode) != 0o600:
                        os.fchmod(fd, 0o600)
                except OSError:
                    pass
                # Acquire an exclusive flock for the duration of the write
                # so concurrent processes can't interleave bytes within a
                # single JSONL line. Best-effort: if flock is not available
                # (e.g. on a filesystem that doesn't support it), fall back
                # to the open-with-O_APPEND behaviour which still gives
                # line-atomicity for writes <= PIPE_BUF.
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    os.write(fd, encoded)
                except (OSError, AttributeError):
                    os.write(fd, encoded)
                finally:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except (OSError, AttributeError):
                        pass
            except OSError as exc:
                raise QueryLogWriteError(f"cannot write query log: {exc}") from exc
            finally:
                os.close(fd)
