"""Exact-args commit protocol for tools that declare ``apply`` at registration.

A tool registered with ``preview=`` and ``apply=`` is an approval tool by its own declaration: the
approval gate (:mod:`tools.approval_tool_gate`) renders the card from ``preview(args)``, and once a
human (or a bypass the user configured) approves, it mints a *decision* bound to a digest of the
exact approved arguments. Dispatch then runs ``apply(args, decision_id=...)`` through
:func:`apply_approved`, never the plain handler, and only when the dispatched args hash to the
approved digest.

The decision journal lives in ``<hermes home>/approvals/decisions.db`` so it survives the process:

- ``approved``: minted by the gate, not yet started.
- ``applying``: claimed by one process (pid + start time recorded) and running now.
- ``applied`` / ``failed``: finished; the result is stored and a replay returns it without
  running ``apply`` again.
- ``unconfirmed``: an ``applying`` record whose owner process is gone. The change may or may not
  have happened, so it is never re-run; the model is told to check the target instead.

A decision id is derived from the session, the tool call id, the tool and the args digest when a
tool call id exists, so a retried or replayed tool call finds its earlier decision instead of
prompting and executing twice.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Optional
import uuid

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_RETENTION_S = 30 * 24 * 3600  # finished results are dropped after this; the decision row is kept for good
_SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, tool TEXT NOT NULL, args_digest TEXT NOT NULL, status TEXT NOT NULL,
    approved_by TEXT, owner_pid INTEGER, owner_start INTEGER, result TEXT,
    created REAL NOT NULL, updated REAL NOT NULL)"""

# Decisions the gate approved for a call that has not been dispatched yet, keyed by
# ``(hermes_home, session_key, tool_call_id, tool, args_digest)``. Dispatch takes the id from here;
# the digest in the key is what makes it exact-args (a call whose args changed after approval finds
# nothing), and the profile home and session keep one process's profiles and sessions from taking
# each other's decisions (execute_code calls carry no tool call id).
_pending: dict[tuple[str, str, str, str, str], list[str]] = {}
_pending_lock = threading.Lock()


def args_digest(args: Any) -> str:
    canonical = json.dumps(args if isinstance(args, dict) else {}, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _db_path() -> Path:
    return Path(get_hermes_home()) / "approvals" / "decisions.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    return con


def decision_id_for(session_key: str, tool_call_id: str, tool: str, digest: str) -> str:
    """Stable id for a tool call that has an id (a replay finds it again), else a fresh one."""
    if not tool_call_id:
        return uuid.uuid4().hex
    return hashlib.sha256(f"{session_key}\0{tool_call_id}\0{tool}\0{digest}".encode()).hexdigest()[:32]


def get(decision_id: str) -> Optional[dict]:
    with closing(_connect()) as con:
        row = con.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    return None if row is None else _with_liveness(dict(row))


def mint(decision_id: str, tool: str, digest: str, approved_by: str) -> None:
    """Record an approved decision (no-op when the id already exists: a replay keeps its record).
    Old finished results are emptied, never their rows: a deleted row would let a replay of that
    call mint it again as ``approved`` and run ``apply`` a second time."""
    now = time.time()
    with closing(_connect()) as con, con:
        con.execute("UPDATE decisions SET result = NULL WHERE updated < ? AND status IN ('applied', 'failed') "
                    "AND result IS NOT NULL", (now - _RETENTION_S,))
        con.execute("INSERT OR IGNORE INTO decisions (id, tool, args_digest, status, approved_by, created, updated) "
                    "VALUES (?, ?, ?, 'approved', ?, ?, ?)", (decision_id, tool, digest, approved_by, now, now))


def _pending_key(tool_call_id: str, tool: str, digest: str) -> tuple[str, str, str, str, str]:
    from tools.approval_context import get_current_session_key
    return str(get_hermes_home()), get_current_session_key(), tool_call_id, tool, digest


def queue_for_dispatch(tool_call_id: str, tool: str, digest: str, decision_id: str) -> None:
    key = _pending_key(tool_call_id, tool, digest)
    with _pending_lock:
        _pending.setdefault(key, []).append(decision_id)


def _take_queued(tool_call_id: str, tool: str, digest: str) -> Optional[str]:
    key = _pending_key(tool_call_id, tool, digest)
    with _pending_lock:
        queued = _pending.get(key)
        if not queued:
            return None
        decision_id = queued.pop(0)
        if not queued:
            _pending.pop(key, None)
        return decision_id


def _owner_alive(pid: Any, started: Any) -> bool:
    from gateway.status import _pid_exists, get_process_start_time
    if not isinstance(pid, int) or pid <= 0 or not _pid_exists(pid):
        return False
    return started is None or get_process_start_time(pid) == started


def _with_liveness(record: dict) -> dict:
    if record["status"] == "applying" and not _owner_alive(record["owner_pid"], record["owner_start"]):
        record["status"] = "unconfirmed"
    return record


def _claim(decision_id: str, digest: str) -> Optional[dict]:
    """``None`` when this process now owns the decision and must run ``apply``; otherwise the
    record to answer from (finished, in progress elsewhere, or unconfirmed)."""
    from gateway.status import get_process_start_time
    pid = os.getpid()
    with closing(_connect()) as con, con:
        claimed = con.execute(
            "UPDATE decisions SET status = 'applying', owner_pid = ?, owner_start = ?, updated = ? "
            "WHERE id = ? AND status = 'approved' AND args_digest = ?",
            (pid, get_process_start_time(pid), time.time(), decision_id, digest)).rowcount
        if claimed:
            return None
        row = con.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    if row is None:
        return {"status": "missing"}
    record = _with_liveness(dict(row))
    if record["args_digest"] != digest:
        record["status"] = "args_mismatch"
    return record


def _finish(decision_id: str, status: str, result: str | dict) -> None:
    """Store the outcome losslessly (JSON of the string or the multimodal envelope), so a replay
    returns exactly what the first execution returned."""
    with closing(_connect()) as con, con:
        con.execute("UPDATE decisions SET status = ?, result = ?, updated = ? WHERE id = ?",
                    (status, json.dumps(result, ensure_ascii=False), time.time(), decision_id))


def _replay(tool: str, decision_id: str, record: dict) -> str | dict:
    from tools.registry import tool_error
    status = record["status"]
    if status in ("applied", "failed"):
        if record.get("result") is None:
            return tool_error(f"{tool} decision {decision_id} was already {status}; its stored result has "
                              "expired. It was not run again.", decision_id=decision_id, decision_status=status)
        return json.loads(record["result"])
    messages = {
        "applying": f"{tool} decision {decision_id} is already being applied; it was not run again.",
        "unconfirmed": (f"UNCONFIRMED: {tool} decision {decision_id} started applying but the process stopped "
                        "before it finished. The change may or may not have taken effect. Do NOT retry; check "
                        "the target and tell the user."),
        "args_mismatch": f"BLOCKED: the arguments differ from the ones approved for decision {decision_id}.",
    }
    return tool_error(messages.get(status, f"BLOCKED: no approved decision {decision_id} for {tool}."),
                      decision_id=decision_id, decision_status=status)


def run_once(tool: str, args: dict, decision_id: str, apply: Callable[[], Any],
             normalize: Callable[[Any], str | dict]) -> str | dict:
    """Run ``apply()`` at most once for ``decision_id``; every later call returns the stored outcome."""
    digest = args_digest(args)
    record = _claim(decision_id, digest)
    if record is not None:
        logger.info("approval decision %s for %s not re-run (status %s)", decision_id, tool, record["status"])
        return _replay(tool, decision_id, record)
    try:
        result = normalize(apply())
    except BaseException as exc:
        from tools.registry import tool_error
        _finish(decision_id, "failed", tool_error(f"{tool} apply failed: {type(exc).__name__}: {exc}",
                                                  decision_id=decision_id))
        raise
    _finish(decision_id, "applied", result)
    return result


def apply_approved(entry, args: dict, kwargs: dict, normalize: Callable[[Any], str | dict]) -> str | dict:
    """Dispatch for a tool that declared ``apply``: only from a decision the gate approved for
    these exact args, and at most once per decision."""
    from tools.approval_context import _approval_tool_call_id
    from tools.registry import _kwargs_accepted_by, tool_error
    decision_id = _take_queued(_approval_tool_call_id.get(), entry.name, args_digest(args))
    if decision_id is None:
        return tool_error(f"BLOCKED: {entry.name} runs only from an approved decision for these exact "
                          "arguments, and none was approved. Do NOT retry; ask the user.")
    call_kwargs = _kwargs_accepted_by(entry.apply, {**kwargs, "decision_id": decision_id})

    def apply():
        if entry.is_async:
            from model_tools import _run_async
            return _run_async(entry.apply(args, **call_kwargs))
        return entry.apply(args, **call_kwargs)

    return run_once(entry.name, args, decision_id, apply, normalize)
