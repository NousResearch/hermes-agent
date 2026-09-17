"""
Artifact action invocation ledger.

Append-only JSONL at ``~/.hermes/artifacts/invocations.jsonl``.
One line per invoke/confirm phase transition (terminal outcomes only for
invoke; every phase for confirm).

Why
---
(a) Durable idempotency: the in-memory ``_idempotency_cache`` in
    ``artifact_actions`` is lost on gateway restart. A retry with the same
    idempotency key after a restart would re-execute a destructive action.
    The ledger gives durable terminal-outcome lookup that survives restarts.

(b) Audit trail: "what did I click last Tuesday and did it land?" is
    answerable via ``artifact.action.log`` RPC or a grep on the JSONL.

(c) Native badge re-hydration: when the artifact pane opens after an app
    restart, native calls ``artifact.action.log`` to restore ✓/⚠ badge
    state from the ledger rather than showing blank.

Schema (one JSON object per line)
----------------------------------
{
  "ts":              ISO-8601 UTC,
  "artifact_id":     str,
  "rev":             int,
  "binding_id":      str,
  "entity_ref":      str,
  "intent":          str,
  "idempotency_key": str,
  "request_scope":   str,            # optional; scoped idempotency digest
  "phase":           "invoke" | "confirm",
  "outcome":         "succeeded" | "failed" | "conflict" | "unsupported"
                   | "needs_confirmation" | "running",
  "reason":          str | null,
  "duration_ms":     int | null,
  "actor":           str,
  "result":          object           # optional; complete replay payload
}

Rotation
--------
File is capped at ``MAX_LEDGER_BYTES``. When the cap is exceeded on append
the current file is renamed to ``invocations.jsonl.1`` (overwriting any
prior backup) and a new file starts. Simple, same class as MAX_REVISIONS.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

MAX_LEDGER_BYTES = 4 * 1024 * 1024  # 4 MB
MAX_QUERY_ROWS = 200
_LEDGER_FILE_NAME = "invocations.jsonl"

_lock = threading.Lock()


# ── Paths ─────────────────────────────────────────────────────────────────────


def _ledger_path() -> Path:
    return Path(get_hermes_home()) / "artifacts" / _LEDGER_FILE_NAME


# ── Write ─────────────────────────────────────────────────────────────────────


def append(
    *,
    artifact_id: str,
    rev: int,
    binding_id: str,
    entity_ref: str,
    intent: str,
    idempotency_key: str,
    phase: str,
    outcome: str,
    reason: Optional[str] = None,
    duration_ms: Optional[int] = None,
    actor: str = "",
    request_scope: str = "",
    result: Optional[dict] = None,
) -> None:
    """Append one invocation record to the ledger."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "artifact_id": artifact_id,
        "rev": rev,
        "binding_id": binding_id,
        "entity_ref": entity_ref,
        "intent": intent,
        "idempotency_key": idempotency_key,
        "phase": phase,
        "outcome": outcome,
        "reason": reason,
        "duration_ms": duration_ms,
        "actor": actor,
    }
    if request_scope:
        record["request_scope"] = request_scope
    if result is not None:
        record["result"] = result
    line = json.dumps(record, ensure_ascii=False) + "\n"
    path = _ledger_path()

    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            size = path.stat().st_size if path.exists() else 0
        except OSError:
            size = 0

        if size > MAX_LEDGER_BYTES:
            backup = path.with_suffix(".jsonl.1")
            try:
                os.replace(str(path), str(backup))
            except OSError as exc:
                logger.warning("ledger rotation failed: %s", exc)

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            logger.warning("ledger append failed: %s", exc)


# ── Read (tail index) ─────────────────────────────────────────────────────────


def _read_all() -> list[dict]:
    """Read the ledger file, newest first."""
    path = _ledger_path()
    if not path.exists():
        return []
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    records.reverse()
    return records


def lookup_terminal(
    idempotency_key: str,
    *,
    request_scope: Optional[str] = None,
    allow_legacy_scope: bool = False,
) -> Optional[dict]:
    """Return the most recent terminal outcome record for the given key,
    or None if no terminal record exists in the ledger.

    Terminal outcomes: succeeded, failed, conflict, unsupported.
    (needs_confirmation and running are non-terminal.)
    """
    TERMINAL = {"succeeded", "failed", "conflict", "unsupported"}
    for record in _read_all():
        if record.get("idempotency_key") != idempotency_key:
            continue
        if record.get("outcome") not in TERMINAL:
            continue
        if request_scope is not None:
            stored_scope = record.get("request_scope")
            if stored_scope != request_scope:
                if not (allow_legacy_scope and not stored_scope):
                    continue
        return record
    return None


def query(
    artifact_id: str,
    binding_id: Optional[str] = None,
    entity_ref: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Query ledger entries for an artifact, newest first.

    Used by native to re-hydrate badge state on pane open and to show
    per-artifact action history.
    """
    limit = min(limit, MAX_QUERY_ROWS)
    results = []
    for record in _read_all():
        if record.get("artifact_id") != artifact_id:
            continue
        if binding_id is not None and record.get("binding_id") != binding_id:
            continue
        if entity_ref is not None and record.get("entity_ref") != entity_ref:
            continue
        results.append(record)
        if len(results) >= limit:
            break
    return results
