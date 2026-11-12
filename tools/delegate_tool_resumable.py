"""Durable subagent records in ``state.db`` (``resumable_subagents``): what ``delegate_task(action='continue')``
rebuilds a child from after a restart or crash, how a stopped child may resume, and which worktrees the startup
sweep must keep for it.

One row per subagent_id, written by the run itself (``record_start`` when its workspace is seeded, ``checkpoint`` on
each heartbeat while its terminal cwd moves, ``record_stop`` after its cleanup ran) and by boot recovery for a run
whose owner process died (``recover_abandoned``). The row's ``pinned`` flag is the worktree pin: set for a child that
did not complete and was not stopped by a person, for ``delegation.resumable_ttl_hours`` (newest
``delegation.max_resumable`` per profile; the oldest beyond the cap loses its pin, the row stays). Automatic
resumption is NOT done here: ``resume_mode`` / ``next_attempt_at`` are what a scheduler gated on
``delegation.auto_resume`` will read.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

AUTO_AT, AUTO_ON_BOOT, ON_FIX = "AUTO_AT", "AUTO_ON_BOOT", "ON_FIX"
REPAIR_THEN_CONTINUE, HANDOFF = "REPAIR_THEN_CONTINUE", "HANDOFF"
GATEWAY_SHUTDOWN = "gateway_shutdown"
# A record ``action='continue'`` can rebuild from (``claimed`` while a continue is starting, ``running`` live).
_FINISHED = ("resumable", "completed", "stopped")
_DEFAULT_TTL_HOURS, _DEFAULT_MAX_RESUMABLE = 48.0, 32
_EXPIRED_RETENTION_S = 7 * 24 * 3600.0

# DESIGN.md per-reason table (agent/error_classifier.py::FailoverReason): mode, and the default delay before an
# AUTO_AT attempt when the failure carried no reset time.
_MODE_BY_REASON: dict[str, tuple[str, Optional[float]]] = {
    "rate_limit": (AUTO_AT, 15 * 60.0), "upstream_rate_limit": (AUTO_AT, 15 * 60.0),
    "timeout": (AUTO_AT, 5 * 60.0), "overloaded": (AUTO_AT, 5 * 60.0), "server_error": (AUTO_AT, 5 * 60.0),
    "session_busy": (AUTO_AT, 30.0), "interpreter_shutdown": (AUTO_ON_BOOT, None),
    "content_policy_blocked": (HANDOFF, None),
    **dict.fromkeys(("context_overflow", "payload_too_large", "image_too_large", "image_corrupt",
                     "thinking_signature", "invalid_encrypted_content"), (REPAIR_THEN_CONTINUE, None)),
}


def _settings() -> tuple[float, int]:
    from tools.delegate_tool_config import _load_config
    cfg = _load_config()
    try:
        ttl = max(0.0, float(cfg.get("resumable_ttl_hours", _DEFAULT_TTL_HOURS)))
        cap = max(0, int(cfg.get("max_resumable", _DEFAULT_MAX_RESUMABLE)))
    except (TypeError, ValueError):
        ttl, cap = _DEFAULT_TTL_HOURS, _DEFAULT_MAX_RESUMABLE
    return ttl, cap


def classify(entry: dict[str, Any], stop_cause: Optional[str]) -> tuple[str, Optional[str], Optional[float]]:
    """``(state, resume_mode, next_attempt_at)`` for a stopped child's result entry. ``state`` is ``resumable``
    (pinned), ``completed`` or ``stopped`` (a person or the parent stopped it: no pin)."""
    status, exit_reason = entry.get("status"), entry.get("exit_reason")
    if status == "completed" and exit_reason == "completed":
        return "completed", None, None
    if status == "interrupted":
        return ("resumable", AUTO_ON_BOOT, None) if stop_cause == GATEWAY_SHUTDOWN else ("stopped", None, None)
    if status == "unknown":
        return "resumable", AUTO_ON_BOOT, None
    reason = str(entry.get("failure_reason") or "")
    resets_at = entry.get("failure_resets_at")
    if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool) and reason in (
            "billing", "rate_limit", "upstream_rate_limit"):
        return "resumable", AUTO_AT, float(resets_at)
    mode, delay = _MODE_BY_REASON.get(reason, (ON_FIX, None))
    return "resumable", mode, (time.time() + delay if delay else None)


# ── storage ────────────────────────────────────────────────────────────────
def _db_path_for(parent_agent: Any = None) -> Optional[Path]:
    """The profile store the parent's children write their transcripts to (else this scope's ``state.db``);
    None in a managed worker, which owns no ledger."""
    from agent.runtime_session_store import is_worker_process
    if is_worker_process():
        return None
    path = getattr(getattr(parent_agent, "_session_db", None), "db_path", None)
    if isinstance(path, (str, Path)):
        return Path(path)
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _connect(path: Path):
    from hermes_cli.sqlite_util import open_db, transaction
    from tools.async_delegation import _initialize_schema
    return transaction(open_db(path, db_label="state.db (resumable_subagents)", busy_timeout_ms=10_000, wal=False,
                               row_factory=None, initialize=_initialize_schema))


def _write(path: Optional[Path], fn) -> Any:
    if path is None:
        return None
    try:
        with _connect(path) as conn:
            return fn(conn)
    except Exception:
        logger.warning("resumable subagent record write failed (%s)", path, exc_info=True)
        return None


def _row(conn, sid: str) -> Optional[dict[str, Any]]:
    got = conn.execute("SELECT record_json, state, stop_cause FROM resumable_subagents WHERE subagent_id=?",
                       (sid,)).fetchone()
    if got is None:
        return None
    record = json.loads(got[0] or "{}")
    return {**record, "state": got[1], "stop_cause": got[2]}


def _upsert(conn, sid: str, record: dict[str, Any], *, state: str, pinned: bool, expires_at: Optional[float],
            stop_cause: Optional[str] = None) -> None:
    now = time.time()
    record = {k: v for k, v in record.items() if k not in ("state", "stop_cause")}
    conn.execute(
        """INSERT INTO resumable_subagents (subagent_id, state, pinned, stop_cause, owner_pid, owner_started_at,
               record_json, created_at, updated_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(subagent_id) DO UPDATE SET state=excluded.state, pinned=excluded.pinned,
               stop_cause=excluded.stop_cause, owner_pid=excluded.owner_pid,
               owner_started_at=excluded.owner_started_at, record_json=excluded.record_json,
               updated_at=excluded.updated_at, expires_at=excluded.expires_at""",
        (sid, state, int(pinned), stop_cause, record.get("owner_pid"), record.get("owner_started_at"),
         json.dumps(record, ensure_ascii=False, default=str), now, now, expires_at))


def _apply_limits(conn) -> None:
    """TTL un-pins and retires a record; beyond the cap the oldest pins drop (the row stays continuable)."""
    ttl, cap = _settings()
    now = time.time()
    conn.execute("UPDATE resumable_subagents SET pinned=0, state='expired' WHERE expires_at IS NOT NULL "
                 "AND expires_at <= ? AND state IN ('resumable','completed','stopped')", (now,))
    conn.execute("DELETE FROM resumable_subagents WHERE state='expired' AND updated_at < ?",
                 (now - _EXPIRED_RETENTION_S - ttl * 3600,))
    conn.execute("""UPDATE resumable_subagents SET pinned=0 WHERE pinned=1 AND subagent_id NOT IN (
                        SELECT subagent_id FROM resumable_subagents WHERE pinned=1
                        ORDER BY updated_at DESC LIMIT ?)""", (cap,))


# ── worktrees ──────────────────────────────────────────────────────────────
def _git_lines(args: list[str], cwd: str) -> list[str]:
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    env = noninteractive_repo_git_env(cwd)
    if env is None:
        return []
    try:
        res = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=10,
                             stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return res.stdout.splitlines() if res.returncode == 0 else []


def _worktrees_of(cwd: Optional[str]) -> list[str]:
    """Every worktree of the repository *cwd* is in (``git worktree list``), or ``[]``."""
    if not cwd or not os.path.isdir(cwd):
        return []
    return [os.path.realpath(line[9:]) for line in _git_lines(["worktree", "list", "--porcelain"], cwd)
            if line.startswith("worktree ")]


def _toplevel(cwd: Optional[str]) -> Optional[str]:
    if not cwd or not os.path.isdir(cwd):
        return None
    lines = _git_lines(["rev-parse", "--show-toplevel"], cwd)
    return os.path.realpath(lines[0]) if lines else None


def _pin_paths(record: dict[str, Any]) -> list[str]:
    """Where the child was when it STOPPED: its last terminal cwd's worktree, its isolation worktree, and every
    worktree its repository gained during the run (a ``git worktree add`` it ran without cd-ing into it)."""
    cwd = record.get("cwd")
    paths = [p for p in (_toplevel(cwd), (record.get("worktree") or {}).get("path")) if p]
    baseline = set(record.get("worktrees_at_start") or ())
    if baseline or cwd:
        paths += [p for p in _worktrees_of(cwd or record.get("start_cwd")) if p not in baseline]
    return sorted({os.path.realpath(p) for p in paths})


# ── run lifecycle ──────────────────────────────────────────────────────────
def _owner_identity() -> dict[str, Any]:
    try:
        from gateway.status import get_process_start_time
        started = get_process_start_time(os.getpid())
    except Exception:
        started = None
    return {"owner_pid": os.getpid(), "owner_started_at": started}


def _spec(child: Any) -> Optional[dict[str, Any]]:
    """The rebuild spec without live objects or secrets: the route is kept as the request that re-routes it."""
    build = getattr(child, "_delegate_resume", None)
    session_id = getattr(child, "session_id", None)
    if not isinstance(build, dict) or not isinstance(session_id, str) or not session_id:
        return None
    route = build.get("route")
    return {"goal": build.get("goal"), "context": build.get("context"), "toolsets": build.get("toolsets"),
            "endpoint": list(build.get("endpoint") or ()), "model": getattr(child, "model", None),
            "session_id": session_id, "route_request": getattr(route, "request", None)}


def record_start(run: Any) -> None:
    """A child began (or resumed) running: durable identity, rebuild spec, owner and starting workspace."""
    child = run.child
    spec = _spec(child)
    sid = run.subagent_id
    if spec is None or not sid:
        return
    path = _db_path_for(run.parent_agent)
    cwd = _child_cwd(run)
    from tools.delegate_tool_registry import _recent_subagents, _active_subagents_lock
    owners = {str(getattr(child, "_parent_session_id", "") or "") or str(getattr(run.parent_agent, "session_id", "")
                                                                         or "")}

    def _do(conn):
        prev = _row(conn, sid) or {}
        with _active_subagents_lock:
            owners.update((_recent_subagents.get(sid) or {}).get("ancestor_owner_session_ids") or ())
        owners.update(prev.get("owners") or ())
        owners.update(_ancestor_sessions(child))
        owners.discard("")
        record = {**prev, **spec, **_owner_identity(), "subagent_id": sid,
                  "delegation_id": getattr(child, "_delegation_id", None), "owners": sorted(owners),
                  "start_cwd": prev.get("start_cwd") or cwd, "cwd": cwd, "worktree": prev.get("worktree")
                  or run.worktree_info, "outcome": "running", "resume_mode": None,
                  # Kept across continues: a tree the child made in an earlier run is still its own.
                  "worktrees_at_start": prev.get("worktrees_at_start") or _worktrees_of(cwd),
                  "auto_attempts": prev.get("auto_attempts", 0)}
        # A running record pins too (its paths are resolved at sweep time): a crash before boot recovery ran must
        # not hand its tree to the sweep. Bounded by the same TTL.
        _upsert(conn, sid, record, state="running", pinned=True, expires_at=time.time() + _settings()[0] * 3600)

    _write(path, _do)
    child._delegate_durable = {"db": path, "cwd": cwd}


def _ancestor_sessions(child: Any) -> set[str]:
    owners, cur = set(), child
    for _ in range(8):
        ref = getattr(cur, "_delegate_parent_ref", None)
        cur = ref() if callable(ref) else None
        if cur is None:
            break
        sid = getattr(cur, "session_id", None)
        if isinstance(sid, str) and sid:
            owners.add(sid)
    return owners


def _child_cwd(run: Any) -> Optional[str]:
    from tools.delegate_tool_routes import child_profile_scope
    from tools.terminal_tool import get_session_cwd
    with child_profile_scope(run.child):
        cwd = get_session_cwd(run.child_task_id)
    return os.path.realpath(cwd) if isinstance(cwd, str) and cwd else None


def checkpoint(run: Any) -> None:
    """Heartbeat: persist the child's terminal cwd when it moved, so a hard kill still knows where it was."""
    durable = getattr(run.child, "_delegate_durable", None)
    if not isinstance(durable, dict) or not run.subagent_id:
        return
    cwd = _child_cwd(run)
    if not cwd or cwd == durable.get("cwd"):
        return
    durable["cwd"] = cwd

    def _do(conn):
        conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.cwd', ?), updated_at=? "
                     "WHERE subagent_id=? AND state='running'", (cwd, time.time(), run.subagent_id))
    _write(durable.get("db"), _do)


def stop_cause_live(run: Any) -> Optional[str]:
    """``gateway_shutdown`` when the gateway's shutdown stopped this child (``note_gateway_shutdown`` marked it, or
    its own turn recorded the shutdown interrupt), else None."""
    if getattr(run.child, "_delegate_stop_cause", None) == GATEWAY_SHUTDOWN or GATEWAY_SHUTDOWN in str(
            getattr(run, "turn_exit_reason", "") or ""):
        return GATEWAY_SHUTDOWN
    return None


def stamp_entry(run: Any, entry: dict[str, Any]) -> None:
    """The result entry says how this stopped child may resume (absent when it completed or a person stopped it)."""
    durable = getattr(run.child, "_delegate_durable", None)
    if not run.subagent_id or not isinstance(durable, dict):
        return
    durable["cwd"] = _child_cwd(run) or durable.get("cwd")  # before cleanup drops the child's cwd record
    state, mode, _ = classify(entry, stop_cause_live(run))
    if state == "resumable":
        entry["resume_mode"] = mode


def record_stop(run: Any, entry: Optional[dict[str, Any]]) -> None:
    """The child's run ended and its resources are closed: its outcome, resume mode, last cwd and pins."""
    durable = getattr(run.child, "_delegate_durable", None)
    sid = run.subagent_id
    if not isinstance(durable, dict) or not sid:
        return
    entry = entry or {"status": "unknown"}
    cwd = durable.get("cwd") or _child_cwd(run)
    live_cause = stop_cause_live(run)

    def _do(conn):
        record = _row(conn, sid) or {}
        _finish(conn, sid, record, entry, live_cause or record.get("stop_cause"), cwd=cwd)
    _write(durable.get("db"), _do)


def _finish(conn, sid: str, record: dict[str, Any], entry: dict[str, Any], cause: Optional[str], *,
            cwd: Optional[str]) -> None:
    ttl, _cap = _settings()
    state, mode, next_at = classify(entry, cause)
    record = {**record, "cwd": cwd or record.get("cwd"), "outcome": entry.get("status"), "resume_mode": mode,
              "next_attempt_at": next_at, "failure_reason": entry.get("failure_reason"),
              "failure_resets_at": entry.get("failure_resets_at"), "usage_limit": entry.get("usage_limit"),
              "stopped_at": time.time()}
    if isinstance(entry.get("worktree"), dict):
        record["worktree"] = {**(record.get("worktree") or {}), **entry["worktree"]}
    record["pin_paths"] = _pin_paths(record) if state == "resumable" else []
    _upsert(conn, sid, record, state=state, pinned=state == "resumable" and bool(record["pin_paths"]),
            expires_at=time.time() + ttl * 3600, stop_cause=cause)
    _apply_limits(conn)


def note_gateway_shutdown() -> int:
    """Gateway shutdown is about to stop this process's children: mark them first (in memory and on every profile's
    running records), so each reads as a restart (resumable) rather than a user stop however its run ends, a
    SIGKILL mid-unwind included. Returns how many records were marked."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    with _active_subagents_lock:
        live = [r.get("agent") for r in _active_subagents.values()]
    for agent in live:
        if agent is not None:
            agent._delegate_stop_cause = GATEWAY_SHUTDOWN
    marked = 0
    for db in _profile_dbs():
        marked += _write(db, lambda conn: conn.execute(
            "UPDATE resumable_subagents SET stop_cause=? WHERE state='running' AND owner_pid=?",
            (GATEWAY_SHUTDOWN, os.getpid())).rowcount) or 0
    return marked


def recover_abandoned(conn, alive) -> dict[str, list[str]]:
    """Boot: a ``running`` record whose owner process is gone becomes resumable (outcome ``unknown``, or a gateway
    shutdown when that was marked), pinned on where it last was. ``{delegation_id: [subagent_id, ...]}``."""
    recovered: dict[str, list[str]] = {}
    rows = conn.execute("SELECT subagent_id, owner_pid, owner_started_at FROM resumable_subagents "
                        "WHERE state IN ('running','claimed')").fetchall()
    for sid, pid, started in rows:
        if alive(pid, started):
            continue
        record = _row(conn, sid) or {}
        cause = record.get("stop_cause")
        status = "interrupted" if cause == GATEWAY_SHUTDOWN else "unknown"
        _finish(conn, sid, record, {"status": status}, cause, cwd=record.get("cwd"))
        recovered.setdefault(str(record.get("delegation_id") or ""), []).append(sid)
    return recovered


# ── continue ───────────────────────────────────────────────────────────────
def claim(parent_agent: Any, sid: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Take a finished child's durable record for ``continue`` (``(record, None)``, or ``(None, error)``). The
    state flip to ``claimed`` is one guarded UPDATE, so two continues never run one conversation twice."""
    from tools.delegate_tool_registry import _owns_subagent_record
    path = _db_path_for(parent_agent)
    record = _write(path, lambda conn: (_apply_limits(conn), _row(conn, sid))[1])
    if record is None:
        return None, None
    owned = {"owner_agent_session_id": None, "ancestor_owner_session_ids": record.get("owners") or ()}
    if not _owns_subagent_record(owned, parent_agent):
        return None, None
    if record.get("state") not in _FINISHED:
        return None, (f"Subagent '{sid}' is still running or being continued. Try again shortly; if this persists, "
                      "spawn a new task.") if record.get("state") in ("running", "claimed") else None
    owner = _owner_identity()  # boot recovery must not reclaim a continue this live process is starting
    claimed = _write(path, lambda conn: conn.execute(
        "UPDATE resumable_subagents SET state='claimed', owner_pid=?, owner_started_at=?, updated_at=? "
        "WHERE subagent_id=? AND state=?",
        (owner["owner_pid"], owner["owner_started_at"], time.time(), sid, record["state"])).rowcount)
    return (record, None) if claimed == 1 else (None, f"Subagent '{sid}' is already being continued.")


def release(parent_agent: Any, sid: str, record: dict[str, Any]) -> None:
    """Hand a claimed record back (the continue was refused before anything ran)."""
    _write(_db_path_for(parent_agent), lambda conn: conn.execute(
        "UPDATE resumable_subagents SET state=? WHERE subagent_id=? AND state='claimed'", (record["state"], sid)))


def last_workspace(parent_agent: Any, sid: str) -> Optional[dict[str, Any]]:
    """``{cwd, worktree}`` the child stopped in, for a continue to start there instead of the parent's cwd."""
    record = _write(_db_path_for(parent_agent), lambda conn: _row(conn, sid)) or {}
    cwd = record.get("cwd")
    if not (isinstance(cwd, str) and os.path.isdir(cwd)):
        return None
    worktree = record.get("worktree") if isinstance(record.get("worktree"), dict) else None
    return {"cwd": cwd, "worktree": worktree if worktree and os.path.isdir(worktree.get("path") or "") else None}


# ── worktree sweep pin ─────────────────────────────────────────────────────
def _profile_dbs() -> list[Path]:
    from hermes_cli.profiles import profiles_to_serve
    homes = [home for _name, home in profiles_to_serve(multiplex=True, include_standalone=True, include_parked=True)]
    return [Path(home) / "state.db" for home in homes if (Path(home) / "state.db").is_file()]


def _read_pins(db: Path, now: float) -> Iterable[tuple[str, dict[str, Any]]]:
    import sqlite3
    from hermes_cli.sqlite_safe_read import connect_tracked
    from hermes_state_holders import read_only_db_uri
    _ttl, cap = _settings()
    try:
        conn = connect_tracked(read_only_db_uri(db), tracking_path=db, uri=True, timeout=2.0)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute("SELECT subagent_id, record_json, expires_at, state FROM resumable_subagents "
                            "WHERE pinned=1 AND state IN ('resumable','running','claimed') "
                            "AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC LIMIT ?",
                            (now, cap)).fetchall()
    except sqlite3.Error:
        rows = []  # no table yet: nothing pinned
    finally:
        conn.close()
    out = []
    for sid, raw, exp, state in rows:
        record = {**json.loads(raw or "{}"), "expires_at": exp}
        if state != "resumable":
            record["pin_paths"] = record.get("pin_paths") or _pin_paths(record)
        out.append((sid, record))
    return out


def pinned_worktrees(now: Optional[float] = None) -> dict[str, str]:
    """``{realpath: reason}`` for every worktree a live resumable record pins, across this install's profiles.
    The startup sweep's only view of the records."""
    now = now or time.time()
    pins: dict[str, str] = {}
    for db in _profile_dbs():
        for sid, record in _read_pins(db, now):
            until = time.strftime("%Y-%m-%d %H:%M", time.localtime(record["expires_at"])) if record.get(
                "expires_at") else "?"
            for path in record.get("pin_paths") or ():
                pins.setdefault(os.path.realpath(path), f"resumable subagent {sid} (pinned until {until})")
    return pins


def pin_reason(path: str, pins: dict[str, str]) -> Optional[str]:
    """Why the tree at *path* is pinned (a pin on the tree or any directory inside it), else None."""
    root = os.path.realpath(path)
    for pinned, reason in pins.items():
        if pinned == root or pinned.startswith(root + os.sep):
            return reason
    return None
