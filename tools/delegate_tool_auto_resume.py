"""``delegation.auto_resume``: continue stopped subagents without a person, and only while the parent still wants them.

The gateway runs :func:`tick` every few seconds for each profile it serves, inside that profile's scope. A tick reads the
profile's ``resumable_subagents`` records (``delegate_tool_resumable``) and continues the ones that are due through the
same claim + ``continue`` path as ``delegate_task(action='continue')``:

- ``AUTO_AT`` once ``next_attempt_at`` (the limit's reset) has passed;
- ``AUTO_ON_BOOT`` (gateway shutdown, crash) once the process that ran it is gone and this one is up.

Never: ``ON_FIX`` / ``HANDOFF`` / ``REPAIR_THEN_CONTINUE`` records, user stops, expired records, records whose pin the
cap dropped, nested children (their parent is a subagent that joined them inside its own turn).

The parent wins. A due record is ``superseded`` (kept, pinned and still continuable by hand until its TTL) when, after
it stopped, its parent conversation moved on: the parent session is gone or ended deliberately (``/new``, deleted,
closed), or the same parent lineage started another delegation in the same repository. ``/stop`` and session ends mark
it the same way at once (:func:`supersede_for_session`), and ``delegate_task(action='stop')`` on a resumable child
dismisses it (:func:`dismiss`).

Bounded: at most ``MAX_AUTO_ATTEMPTS`` automatic attempts per record, counted in the record BEFORE the attempt runs so
a resume that kills the gateway cannot loop across boots; at most ``delegation.max_concurrent_children`` running async
units; one resume per endpoint per tick, so children that share one reset are staggered.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tools.delegate_tool_resumable import AUTO_AT, AUTO_ON_BOOT

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

MAX_AUTO_ATTEMPTS = 2
SUPERSEDED = "superseded"
# A continue refused for one of these reasons ran nothing and can succeed on a later tick: the attempt is refunded.
_TRANSIENT_REFUSALS = ("still running", "being continued", "still stopping", "paused", "capacity", "depth limit")


def enabled() -> bool:
    """``delegation.auto_resume`` of the bound profile (opt-in, default off)."""
    from tools.delegate_tool_config import _load_config
    from utils import is_truthy_value
    return is_truthy_value(_load_config().get("auto_resume", False))


def promise(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """What the stop notice may promise for a just-stopped record: ``{"at": epoch}`` or ``{"on_boot": True}``, or
    None when this record will not be resumed automatically."""
    if not enabled() or int(record.get("auto_attempts") or 0) >= MAX_AUTO_ATTEMPTS:
        return None
    if record.get("resume_mode") == AUTO_AT and isinstance(record.get("next_attempt_at"), (int, float)):
        return {"at": float(record["next_attempt_at"])}
    return {"on_boot": True} if record.get("resume_mode") == AUTO_ON_BOOT else None


def notice_line(sid: str, auto: dict[str, Any]) -> str:
    """The sentence a stop notice carries for a child the scheduler will continue."""
    when = (f"at {time.strftime('%H:%M', time.localtime(auto['at']))} (when the limit resets)" if auto.get("at")
            else "as soon as the gateway is back up")
    return (f"Subagent {sid} WILL resume automatically {when}, on its own conversation and worktree (at most "
            f"{MAX_AUTO_ATTEMPTS} automatic attempts). Do not re-dispatch this task: its result will arrive here. To "
            f"cancel the automatic resume, call delegate_task(action='stop', subagent_id='{sid}'); a new subagent "
            "you start in the same repository also cancels it.")


def boot_notice(sids: list[str], records: dict[str, dict[str, Any]]) -> str:
    """Boot recovery's sentence for crashed children the scheduler will continue ('' when none will)."""
    lines = [notice_line(sid, auto) for sid in sids if (record := records.get(sid)) and record.get("unit_id")
             and not record.get("auto_off") and (auto := promise({**record, "resume_mode": AUTO_ON_BOOT}))]
    return (" " + " ".join(lines)) if lines else ""


# ── storage ────────────────────────────────────────────────────────────────
def _db() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _write(fn) -> Any:
    from tools.delegate_tool_resumable import _write as write
    return write(_db(), fn)


def _note(conn, sid: str, *, state: Optional[str], note: str, from_states=("resumable",), unpin=False) -> bool:
    """Record why the scheduler will not (or no longer) run this record; *state* moves it when given."""
    marks = ",".join("?" * len(from_states))
    sets = "record_json=json_set(record_json, '$.auto_note', ?), updated_at=?"
    if state:
        sets += ", state=?" + (", pinned=0" if unpin else "")
    params = [note, time.time(), *([state] if state else []), sid, *from_states]
    return conn.execute(f"UPDATE resumable_subagents SET {sets} WHERE subagent_id=? AND state IN ({marks})",
                        params).rowcount == 1


def _candidates(now: float) -> list[dict[str, Any]]:
    def read(conn):
        rows = conn.execute(
            "SELECT subagent_id, record_json, pinned, stop_cause FROM resumable_subagents WHERE state='resumable' "
            "AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at", (now,)).fetchall()
        # The owner fields come from the JSON (the process that RAN it); a claim moves only the columns.
        return [{**json.loads(raw or "{}"), "subagent_id": sid, "pinned": pinned, "stop_cause": cause}
                for sid, raw, pinned, cause in rows]
    return [r for r in (_write(read) or []) if r.get("resume_mode") in (AUTO_AT, AUTO_ON_BOOT)]


def _unit_row(unit_id: Any) -> Optional[dict[str, Any]]:
    """The async ledger row the child's background unit reported on (its completion route), or None."""
    if not unit_id:
        return None
    cols = ("origin_session", "origin_ui_session_id", "parent_session_id", "origin_session_id", "task_json")
    got = _write(lambda conn: conn.execute(f"SELECT {', '.join(cols)} FROM async_delegations WHERE delegation_id=?",
                                           (str(unit_id),)).fetchone())
    return dict(zip(cols, got)) if got else None


def _records_since(stopped_at: float, exclude: str) -> list[dict[str, Any]]:
    """Every OTHER record a parent SPAWNED after *stopped_at* (a continue of another child, or this scheduler's,
    is not new work)."""
    def read(conn):
        rows = conn.execute("SELECT subagent_id, record_json FROM resumable_subagents WHERE subagent_id != ?",
                            (exclude,)).fetchall()
        return [json.loads(raw or "{}") for _sid, raw in rows]
    return [r for r in (_write(read) or []) if r.get("run_origin", "spawn") == "spawn"
            and float(r.get("run_started_at") or 0) > stopped_at]


# ── verdicts ───────────────────────────────────────────────────────────────
def _lineage(parent_sid: str) -> tuple[set[str], Optional[dict[str, Any]]]:
    """``(ids of the parent's compression lineage, the live tip's session row or None)``."""
    from hermes_state_registry import acquire, release_or_close
    db = acquire(_db())
    try:
        chain = db.get_compression_chain(parent_sid) or [parent_sid]
        return set(chain), db.get_session(chain[-1])
    finally:
        release_or_close(db)


def _repo_of(path: Any) -> Optional[str]:
    """The repository a path is in, named by its main worktree (so every isolation worktree of one repository
    names the same one), else the path itself."""
    if not isinstance(path, str) or not path:
        return None
    from tools.delegate_tool_resumable import _worktrees_of
    trees = _worktrees_of(path)
    return trees[0] if trees else os.path.realpath(path)


def _repos(record: dict[str, Any]) -> set[str]:
    """Every repository a record's run started in, moved to, was isolated in or pins. An isolation worktree names
    its repository itself (``repo_root``): a finished child's tree may already be pruned."""
    worktree = record.get("worktree") or {}
    paths = [record.get("cwd"), record.get("start_cwd"), worktree.get("path"), *(record.get("pin_paths") or ())]
    repos = {repo for repo in map(_repo_of, paths) if repo}
    return repos | ({_repo_of(worktree["repo_root"])} if worktree.get("repo_root") else set())


def _parent_ended(row: Optional[dict[str, Any]]) -> Optional[str]:
    if row is None:
        return "its parent session no longer exists"
    reason = row.get("end_reason")
    if row.get("ended_at") is None or not reason:
        return None
    from hermes_state_common import is_automatic_end_reason
    return None if is_automatic_end_reason(reason) else f"its parent session ended ({reason})"


def superseded_reason(record: dict[str, Any], parent_sid: str) -> Optional[str]:
    """Why the parent no longer wants *record* resumed, or None while it still does."""
    lineage, tip_row = _lineage(parent_sid)
    if ended := _parent_ended(tip_row):
        return ended
    stopped_at = float(record.get("stopped_at") or 0)
    mine = None
    for other in _records_since(stopped_at, record["subagent_id"]):
        if not lineage & set(other.get("owners") or ()):
            continue
        mine = _repos(record) if mine is None else mine
        if shared := mine & _repos(other):
            return (f"its parent started subagent {other.get('subagent_id')} in {min(shared)} after it "
                    "stopped")
    return None


def _due(record: dict[str, Any], now: float) -> bool:
    if record.get("resume_mode") == AUTO_AT:
        return isinstance(record.get("next_attempt_at"), (int, float)) and now >= float(record["next_attempt_at"])
    # AUTO_ON_BOOT: the process that ran it is gone (a pid this process reuses still has another start time).
    from tools.delegate_tool_resumable import _owner_identity
    me = _owner_identity()
    return (record.get("owner_pid"), record.get("owner_started_at")) != (me["owner_pid"], me["owner_started_at"])


def _blocked(record: dict[str, Any]) -> Optional[str]:
    """A permanent reason this record is never run automatically, else None."""
    if record.get("auto_off"):
        return f"not resumed automatically: {record['auto_off']}"
    if int(record.get("auto_attempts") or 0) >= MAX_AUTO_ATTEMPTS:
        return (f"{MAX_AUTO_ATTEMPTS} automatic attempts used (last stop: "
                f"{record.get('failure_reason') or record.get('outcome') or 'unknown'}); continue it by hand")
    if record.get("pin_paths") and not record.get("pinned"):
        return "its worktree pin was dropped (delegation.max_resumable)"
    return None


# ── run ────────────────────────────────────────────────────────────────────
def _cause(record: dict[str, Any]) -> str:
    if record.get("resume_mode") == AUTO_AT:
        return f"a provider limit ({record.get('failure_reason') or 'rate_limit'})"
    if record.get("stop_cause") == "gateway_shutdown":
        return "a gateway restart"
    return "a crash of the process that ran you (outcome unknown)"


def resume_message(record: dict[str, Any]) -> str:
    return (f"Your run was stopped by {_cause(record)}; continue where you left off. Check git status and the effect "
            "of your last tool call before re-running anything with side effects.")


def _count_attempt(sid: str, attempts: int) -> bool:
    """Spend one attempt BEFORE it runs (a resume that kills the process still counts); False when the record moved."""
    return bool(_write(lambda conn: conn.execute(
        "UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_attempts', ?, '$.auto_last_at', ?),"
        " updated_at=? WHERE subagent_id=? AND state='resumable' "
        "AND COALESCE(json_extract(record_json, '$.auto_attempts'), 0)=?",
        (attempts + 1, time.time(), time.time(), sid, attempts)).rowcount))


def _settle_refusal(sid: str, attempts: int, error: str) -> None:
    """A continue refused before anything ran: a transient refusal refunds the attempt, any other one (a changed
    route, a missing transcript) ends automatic resumes of this record."""
    note = f"automatic continue refused: {error}"[:500]
    if any(word in error for word in _TRANSIENT_REFUSALS):
        sql, params = "json_set(record_json, '$.auto_attempts', ?, '$.auto_note', ?)", (attempts, note)
    else:
        sql, params = "json_set(record_json, '$.auto_off', ?, '$.auto_note', ?)", (note, note)
    _write(lambda conn: conn.execute(f"UPDATE resumable_subagents SET record_json={sql} WHERE subagent_id=?",
                                     (*params, sid)))


def _resume(record: dict[str, Any], unit: dict[str, Any], start: Callable[..., Optional[str]]) -> bool:
    sid, attempts = record["subagent_id"], int(record.get("auto_attempts") or 0)
    if not _count_attempt(sid, attempts):
        return False
    try:
        error = start(record, unit, resume_message(record))
    except Exception as exc:  # health: allow BLE001 -- one record's failure must not stop the tick
        logger.warning("automatic resume of subagent %s failed", sid, exc_info=True)
        error = f"{type(exc).__name__}: {exc}"
    if error:
        _settle_refusal(sid, attempts, error)
        logger.info("automatic resume of subagent %s refused: %s", sid, error)
        return False
    logger.info("Automatically resumed subagent %s (attempt %d/%d)", sid, attempts + 1, MAX_AUTO_ATTEMPTS)
    return True


def _verdict(record: dict[str, Any], now: float) -> tuple[str, Any]:
    """``(kind, payload)``: ``wait``, ``never`` (note), ``supersede`` (note) or ``run`` (its async ledger row)."""
    if not _due(record, now):
        return "wait", None
    if blocked := _blocked(record):
        return "never", blocked
    unit = _unit_row(record.get("unit_id"))
    if unit is None or not unit.get("parent_session_id"):
        return "never", "no background completion route (a nested or joined child)"
    if reason := superseded_reason(record, str(unit["parent_session_id"])):
        return "supersede", reason
    return "run", unit


def tick(start: Callable[..., Optional[str]], *, capacity: int, now: Optional[float] = None) -> list[str]:
    """One pass over the bound profile's records; returns the subagent ids it resumed. ``start(record, unit_row,
    message)`` performs the continue and returns an error string, or None once the child was dispatched."""
    if not enabled() or capacity <= 0:
        return []
    now = now or time.time()
    resumed, endpoints = [], set()
    for record in _candidates(now):
        kind, payload = _verdict(record, now)
        sid = record["subagent_id"]
        if kind == "never":
            if record.get("auto_note") != payload:
                _write(lambda conn, n=payload: _note(conn, sid, state=None, note=n))
            continue
        if kind == "supersede":
            _write(lambda conn, n=payload: _note(conn, sid, state=SUPERSEDED, note=n))
            logger.info("Subagent %s will not resume automatically: %s", sid, payload)
            continue
        endpoint = json.dumps(record.get("endpoint"))
        if kind != "run" or endpoint in endpoints or len(resumed) >= capacity:
            continue  # one resume per endpoint per tick: children sharing one reset are staggered
        endpoints.add(endpoint)
        if _resume(record, payload, start):
            resumed.append(sid)
    return resumed


# ── parent actions ─────────────────────────────────────────────────────────
def supersede_for_session(*, session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "",
                          reason: str = "") -> int:
    """A person stopped, reset or closed a conversation: its resumable children must not wake up later. Automatic
    cleanup (a gateway restart, an orphan reap) is not a decision about them and changes nothing."""
    from hermes_state_common import is_automatic_end_reason
    if is_automatic_end_reason(reason.rsplit(":", 1)[-1]) or not enabled():
        return 0
    selectors = [(col, val) for col, val in (("origin_session", session_key), ("origin_ui_session_id",
                 origin_ui_session_id), ("parent_session_id", parent_session_id)) if val]
    if not selectors:
        return 0

    def do(conn):
        where = " OR ".join(f"{col}=?" for col, _ in selectors)
        units = {row[0] for row in conn.execute(f"SELECT delegation_id FROM async_delegations WHERE {where}",
                                                [val for _, val in selectors])}
        rows = conn.execute("SELECT subagent_id, json_extract(record_json, '$.unit_id') FROM resumable_subagents "
                            "WHERE state='resumable'").fetchall()
        note = f"its parent conversation was stopped or closed ({reason or 'session end'})"
        return sum(_note(conn, sid, state=SUPERSEDED, note=note) for sid, unit in rows if unit in units)
    return int(_write(do) or 0)


def dismiss(parent_agent: Any, sid: str) -> bool:
    """``delegate_task(action='stop')`` on a child that is not running but waits to resume: it never resumes on its
    own and its worktree pin is released (its transcript stays). False when there is no such record of this parent."""
    if not enabled():
        return False
    from tools.delegate_tool_registry import _owns_subagent_record
    from tools.delegate_tool_resumable import _db_path_for, _row, _write as write
    path = _db_path_for(parent_agent)
    record = write(path, lambda conn: _row(conn, sid))
    if not record or record.get("state") not in ("resumable", SUPERSEDED):
        return False
    if not _owns_subagent_record({"ancestor_owner_session_ids": record.get("owners") or ()}, parent_agent):
        return False
    return bool(write(path, lambda conn: _note(conn, sid, state="stopped", note="dismissed by its parent",
                                               from_states=("resumable", SUPERSEDED), unpin=True)))
