"""``delegation.auto_resume``: continue stopped subagents without a person, and only while the parent still wants them.

The gateway runs :func:`tick` every few seconds for each profile it serves, inside that profile's scope. A tick reads the
profile's ``resumable_subagents`` records (``delegate_tool_resumable``) and continues the ones that are due through the
same claim + ``continue`` path as ``delegate_task(action='continue')``:

- ``AUTO_AT`` once ``next_attempt_at`` (the limit's reset) has passed;
- ``AUTO_ON_BOOT`` (gateway shutdown, crash) once the process that ran it is gone and this one is up.

Never: ``ON_FIX`` / ``HANDOFF`` / ``REPAIR_THEN_CONTINUE`` records, user stops, expired records, records whose pin the
cap dropped, nested children (their parent is a subagent that joined them inside its own turn).

Explicit cancel only. A due record is ``superseded`` (kept, pinned and still continuable by hand until its TTL) only
when its parent session is gone or ended deliberately (``/new``, deleted, closed). ``/stop`` and session ends mark it the
same way at once (:func:`supersede_for_session`), ``delegate_task(action='stop')`` on a resumable child dismisses it
(:func:`dismiss`), and the parent's own continue takes it over. Other work the parent starts never cancels it.

Silent while pending: a stop that promised a resume delivers no completion (``async_delegation``); the parent sees it in
``delegate_task(action='list')`` (:func:`waiting_entries`). It reports once: the resumed run's own result, or ONE
terminal failure when it gives up (attempts used, ``delegation.auto_resume_deadline_minutes`` passed since its first
stop, or a permanent refusal) (:func:`_give_up`).

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
DEFAULT_DEADLINE_MINUTES = 360
SUPERSEDED = "superseded"
CANCELLED = "Not continued: its parent cancelled the automatic resume"
STAND_IN_FAILED = "Not continued: the scheduler could not build the parent's stand-in"
# A continue refused for one of these reasons ran nothing and can succeed on a later tick: the attempt is refunded.
# A depth-limit refusal is not one: the child's depth does not change between ticks.
_TRANSIENT_REFUSALS = ("still running", "being continued", "still stopping", "paused", "capacity", CANCELLED,
                       STAND_IN_FAILED)


def enabled() -> bool:
    """``delegation.auto_resume`` of the bound profile (opt-in, default off)."""
    from tools.delegate_tool_config import _load_config
    from utils import is_truthy_value
    return is_truthy_value(_load_config().get("auto_resume", False))


def _deadline_s() -> float:
    from tools.delegate_tool_config import _load_config
    try:
        return max(0.0, float(_load_config().get("auto_resume_deadline_minutes", DEFAULT_DEADLINE_MINUTES))) * 60
    except (TypeError, ValueError):
        return DEFAULT_DEADLINE_MINUTES * 60.0


def _first_stopped_at(record: dict[str, Any]) -> float:
    return float(record.get("auto_first_stopped_at") or record.get("stopped_at") or time.time())


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def give_up_reason(record: dict[str, Any], now: Optional[float] = None, *, at: Optional[float] = None) -> Optional[str]:
    """Why the scheduler stops trying *record* (its attempts are used, or its deadline passed, or the next attempt
    *at* would land past it), else None."""
    if int(record.get("auto_attempts") or 0) >= MAX_AUTO_ATTEMPTS:
        return (f"gave up after {MAX_AUTO_ATTEMPTS} automatic attempts (last stop: "
                f"{record.get('failure_reason') or record.get('outcome') or 'unknown'})")
    deadline = _first_stopped_at(record) + _deadline_s()
    if (now or time.time()) > deadline or (at is not None and at > deadline):
        return (f"gave up: the automatic-resume deadline ({_deadline_s() / 60:g} minutes after its first stop, "
                f"{_hhmm(deadline)}) " + ("passed" if at is None or (now or time.time()) > deadline
                                           else "falls before the limit resets"))
    return None


def promise(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """What the stop may promise for a just-stopped record: ``{"at": epoch, "attempt": n}`` or ``{"on_boot": True,
    "attempt": n}``, or None when this record will not be resumed automatically."""
    if not enabled() or not _scheduler_runs():
        return None
    if record.get("resume_mode") == AUTO_AT and isinstance(record.get("next_attempt_at"), (int, float)):
        auto: dict[str, Any] = {"at": float(record["next_attempt_at"])}
    elif record.get("resume_mode") == AUTO_ON_BOOT:
        auto = {"on_boot": True}
    else:
        return None
    if give_up_reason(record, at=auto.get("at")):
        return None
    return {**auto, "attempt": int(record.get("auto_attempts") or 0) + 1,
            "gives_up_at": _first_stopped_at(record) + _deadline_s()}


def gave_up(record: dict[str, Any]) -> Optional[str]:
    """For a just-stopped record the scheduler would resume but no longer will: why (the stop's own completion is
    then the one terminal report), else None."""
    if not enabled() or not _scheduler_runs() or record.get("resume_mode") not in (AUTO_AT, AUTO_ON_BOOT):
        return None
    return give_up_reason(record, at=record.get("next_attempt_at") if record.get("resume_mode") == AUTO_AT else None)


def _scheduler_runs() -> bool:
    """Whether this process is the gateway serving the bound profile, so its watcher will run the resume (a
    background child's completion route is this process's). A classic CLI parent has no scheduler: no promise."""
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home
    return process_ownership.owns(get_hermes_home())


def waiting_text(auto: dict[str, Any]) -> str:
    """``waiting to resume at HH:MM (attempt n of 2)``."""
    when = f"at {_hhmm(auto['at'])}" if auto.get("at") else "once the gateway is back up"
    out = f"waiting to resume {when} (attempt {auto.get('attempt') or 1} of {MAX_AUTO_ATTEMPTS})"
    return out + (f"; gives up at {_hhmm(auto['gives_up_at'])}" if auto.get("gives_up_at") else "")


def notice_line(sid: str, auto: dict[str, Any]) -> str:
    """The sentence a mixed unit's notice carries for a child the scheduler will continue."""
    return (f"Subagent {sid} is {waiting_text(auto)}, on its own conversation and worktree. Do not re-dispatch it: "
            "its result, or one failure if it gives up, will arrive here. To cancel it, call "
            f"delegate_task(action='stop', subagent_id='{sid}').")


def gave_up_line(sid: str, reason: str) -> str:
    return (f"Automatic resume of subagent {sid} {reason}. It is still continuable by hand: "
            f"delegate_task(action='continue', subagent_id='{sid}', message=...), or re-dispatch the task.")


def boot_notice(conn, sids: list[str], records: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Boot recovery's sentence for crashed children the scheduler will continue ('' when none will) and their ids;
    each promised record is stamped ``auto_promised`` on boot recovery's own connection."""
    promised = {sid: auto for sid in sids if (record := records.get(sid)) and record.get("unit_id")
                and not record.get("auto_off") and (auto := promise({**record, "resume_mode": AUTO_ON_BOOT}))}
    for sid in promised:
        conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_promised', json('true'))"
                     " WHERE subagent_id=?", (sid,))
    lines = [notice_line(sid, auto) for sid, auto in promised.items()]
    return ((" " + " ".join(lines)) if lines else ""), list(promised)


# ── storage ────────────────────────────────────────────────────────────────
def _db() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _write(fn) -> Any:
    from tools.delegate_tool_resumable import _write as write
    return write(_db(), fn)


def _note(conn, sid: str, *, state: Optional[str], note: str, from_states=("resumable",), unpin=False,
          unstarted_claim=False) -> bool:
    """Record why the scheduler will not (or no longer) run this record; *state* moves it when given.
    *unstarted_claim*: also a record an automatic continue claimed whose run has not started (``$.auto_claim``)."""
    marks = ",".join("?" * len(from_states))
    sets = "record_json=json_remove(json_set(record_json, '$.auto_note', ?), '$.auto_claim'), updated_at=?"
    if state:
        sets += ", state=?" + (", pinned=0" if unpin else "")
    claimed = (" OR (state='claimed' AND json_extract(record_json, '$.auto_claim') IS NOT NULL)"
               if unstarted_claim else "")
    params = [note, time.time(), *([state] if state else []), sid, *from_states]
    return conn.execute(f"UPDATE resumable_subagents SET {sets} WHERE subagent_id=? AND (state IN ({marks}){claimed})",
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
    cols = ("origin_session", "origin_ui_session_id", "parent_session_id", "origin_session_id", "task_json",
            "result_json", "dispatched_at")
    got = _write(lambda conn: conn.execute(f"SELECT {', '.join(cols)} FROM async_delegations WHERE delegation_id=?",
                                           (str(unit_id),)).fetchone())
    return dict(zip(cols, got)) if got else None


# ── verdicts ───────────────────────────────────────────────────────────────
def _lineage(conn, parent_sid: str) -> tuple[set[str], Optional[dict[str, Any]]]:
    """``(ids of the parent's compression lineage, the live tip's session row or None)``, read on *conn*."""
    from hermes_state_compression import _CHAIN_STEP_SQL, _walk_compression_chain

    def step(current):
        row = conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone()
        return {"id": row[0]} if row else None
    chain = _walk_compression_chain(parent_sid, step) or [parent_sid]
    tip = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id=?", (chain[-1],)).fetchone()
    return set(chain), ({"ended_at": tip[0], "end_reason": tip[1]} if tip else None)


def _parent_ended(row: Optional[dict[str, Any]]) -> Optional[str]:
    if row is None:
        return "its parent session no longer exists"
    reason = row.get("end_reason")
    if row.get("ended_at") is None or not reason:
        return None
    from hermes_state_common import is_automatic_end_reason
    return None if is_automatic_end_reason(reason) else f"its parent session ended ({reason})"


def superseded_reason(conn, record: dict[str, Any], parent_sid: str) -> Optional[str]:
    """Why the parent no longer wants *record* resumed, or None while it still does (read on *conn*, so inside a
    write transaction nothing the parent does can land between this answer and the write that acts on it)."""
    _lineage_ids, tip_row = _lineage(conn, parent_sid)
    return _parent_ended(tip_row)


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
    if not record.get("auto_promised"):  # its stop notice told the parent to continue it by hand (or re-dispatch)
        return "its stop never promised an automatic resume; continue it by hand"
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
            "of your last tool call before re-running anything with side effects. If your last tool call has no "
            "recorded result and you cannot establish whether it ran, do not repeat it: finish with a result that "
            "starts with \"BLOCKED:\" and names that tool call.")


def _claim(record: dict[str, Any], parent_sid: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The automatic claim: ONE write transaction re-reads the row, re-runs supersession on that connection and,
    only while it is still ``resumable``, unsuperseded, uncancelled and under the cap, moves it to ``claimed`` and
    spends the attempt (BEFORE it runs: a resume that kills the process still counts). A parent decision that
    committed first wins; none can land in between. ``(claimed record, None)``, or ``(None, supersede note)``,
    or ``(None, None)`` when the row moved."""
    from tools.delegate_tool_resumable import _begin_immediate, _owner_identity, _row
    sid, attempts = record["subagent_id"], int(record.get("auto_attempts") or 0)
    owner = _owner_identity()

    def do(conn):
        _begin_immediate(conn)
        row = _row(conn, sid)
        if (not row or row.get("state") != "resumable" or row.get("auto_off") or not row.get("auto_promised")
                or int(row.get("auto_attempts") or 0) != attempts or attempts >= MAX_AUTO_ATTEMPTS):
            return None, None
        if reason := superseded_reason(conn, row, parent_sid):
            _note(conn, sid, state=SUPERSEDED, note=reason)
            return None, reason
        now = time.time()
        conn.execute(
            "UPDATE resumable_subagents SET state='claimed', owner_pid=?, owner_started_at=?, updated_at=?, "
            "record_json=json_set(record_json, '$.auto_attempts', ?, '$.auto_last_at', ?, '$.auto_claim', ?) "
            "WHERE subagent_id=? AND state='resumable'",
            (owner["owner_pid"], owner["owner_started_at"], now, attempts + 1, now, parent_sid, sid))
        return {**row, "auto_attempts": attempts + 1}, None
    return _write(do) or (None, None)


def take_claim(path: Any, sid: str) -> Optional[dict[str, Any]]:
    """The continue path's view of this process's automatic claim of *sid* (``claimed``, ``$.auto_claim`` set, by
    this process), as the record a release hands back to ``resumable``; None once the parent cancelled it."""
    from tools.delegate_tool_resumable import _row, _write as write
    row = write(path, lambda conn: _row(conn, sid))
    if not row or row.get("state") != "claimed" or not row.get("auto_claim") or row.get("owner_pid") != os.getpid():
        return None
    return {**row, "state": "resumable"}


def confirm_start(path: Any, sid: str) -> Optional[str]:
    """Right before the continued child is dispatched: the eligibility check, in one write transaction. None while
    the parent has not cancelled it (stop, /stop, /new), else the reason, with
    the row left as the parent's decision put it (or moved to ``superseded``). The claim stays cancellable as
    unstarted until the run records itself: :func:`admit_start` gives the parent the last word there."""
    from tools.delegate_tool_resumable import _begin_immediate, _row, _write as write

    def do(conn):
        _begin_immediate(conn)
        row = _row(conn, sid)
        if not row or row.get("state") != "claimed" or not row.get("auto_claim"):
            return CANCELLED
        if reason := superseded_reason(conn, row, str(row["auto_claim"])):
            _note(conn, sid, state=SUPERSEDED, note=reason, from_states=(), unstarted_claim=True)
            return f"{CANCELLED}: {reason}"
        return None
    return write(path, do) if path is not None else CANCELLED


def admit_start(conn, sid: str) -> Optional[str]:
    """Inside ``record_start``'s write transaction for a run the scheduler started: None while this process's claim
    still stands and nothing superseded it, else why the parent cancelled it. The parent's decision stays on the row
    and the attempt is refunded; a stop that lands after this point meets a registered, running child."""
    from tools.delegate_tool_resumable import _begin_immediate, _row
    _begin_immediate(conn)
    row = _row(conn, sid) or {}
    if row.get("state") == "claimed" and row.get("auto_claim") and row.get("owner_pid") == os.getpid():
        reason = superseded_reason(conn, row, str(row["auto_claim"]))
        if not reason:
            return None
        _note(conn, sid, state=SUPERSEDED, note=reason, from_states=(), unstarted_claim=True)
    elif row.get("state") in ("stopped", SUPERSEDED):
        reason = row.get("auto_note") or "its parent stopped it"
    else:
        return f"{CANCELLED}: the record moved ({row.get('state') or 'gone'})"
    conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_attempts', "
                 "MAX(COALESCE(json_extract(record_json, '$.auto_attempts'), 1) - 1, 0)) WHERE subagent_id=?", (sid,))
    return f"{CANCELLED}: {reason}"


def _settle_refusal(sid: str, attempts: int, error: str) -> None:
    """A continue refused before anything ran: the claim goes back (unless the parent already moved the row), a
    transient refusal refunds the attempt, any other one (a changed route, a missing transcript) ends automatic
    resumes of this record."""
    note = f"automatic continue refused: {error}"[:500]
    if error.startswith(CANCELLED):  # the parent's decision already wrote the row: only the attempt is refunded
        sql, params = "json_set(record_json, '$.auto_attempts', ?)", (attempts,)
    elif any(word in error for word in _TRANSIENT_REFUSALS):
        sql, params = "json_set(record_json, '$.auto_attempts', ?, '$.auto_note', ?)", (attempts, note)
    else:
        sql, params = "json_set(record_json, '$.auto_off', ?, '$.auto_note', ?)", (note, note)
    _write(lambda conn: conn.execute(
        f"UPDATE resumable_subagents SET record_json=json_remove({sql}, '$.auto_claim'), "
        "state=CASE WHEN state='claimed' AND owner_pid=? THEN 'resumable' ELSE state END WHERE subagent_id=?",
        (*params, os.getpid(), sid)))


def _resume(record: dict[str, Any], unit: dict[str, Any], start: Callable[..., Optional[str]]) -> Optional[str]:
    """Claim, then continue; returns ``"resumed"``, ``"superseded"`` or None (nothing ran)."""
    sid, attempts = record["subagent_id"], int(record.get("auto_attempts") or 0)
    claimed, superseded = _claim(record, str(unit["parent_session_id"]))
    if claimed is None:
        if superseded:
            logger.info("Subagent %s will not resume automatically: %s", sid, superseded)
        return "superseded" if superseded else None
    try:
        error = start(claimed, unit, resume_message(claimed))
    except Exception as exc:  # health: allow BLE001 -- one record's failure must not stop the tick
        logger.warning("automatic resume of subagent %s failed", sid, exc_info=True)
        error = f"{type(exc).__name__}: {exc}"
    if error:
        _settle_refusal(sid, attempts, error)
        logger.info("automatic resume of subagent %s refused: %s", sid, error)
        return None
    # Dispatched: the run-start write may still find the parent cancelled it (the child then never runs).
    logger.info("Started automatic resume of subagent %s (attempt %d/%d)", sid, attempts + 1, MAX_AUTO_ATTEMPTS)
    return "resumed"


def _verdict(record: dict[str, Any], now: float) -> tuple[str, Any]:
    """``(kind, payload)``: ``wait``, ``never`` (note), ``give_up`` (reason: a promised record's one terminal
    failure), ``supersede`` (note) or ``run`` (its async ledger row)."""
    if record.get("auto_gave_up"):
        return "wait", None  # its terminal failure was delivered; nothing more to do
    promised = bool(record.get("auto_promised"))
    if promised and (reason := give_up_reason(record, now)):
        return "give_up", reason
    if not _due(record, now):
        return "wait", None
    if blocked := _blocked(record):
        return ("give_up" if promised else "never"), blocked
    unit = _unit_row(record.get("unit_id"))
    if unit is None or not unit.get("parent_session_id"):
        return "never", "no background completion route (a nested or joined child)"
    if reason := _write(lambda conn: superseded_reason(conn, record, str(unit["parent_session_id"]))):
        return "supersede", reason
    return "run", unit


def tick(start: Callable[..., Optional[str]], *, capacity: int, now: Optional[float] = None) -> list[str]:
    """One pass over the bound profile's records; returns the subagent ids it resumed. ``start(record, unit_row,
    message)`` performs the continue and returns an error string, or None once the child was dispatched."""
    from tools.delegate_tool_registry import is_spawn_paused
    if not enabled() or capacity <= 0 or is_spawn_paused():  # paused: no claim, no stand-in parent built
        return []
    now = now or time.time()
    resumed, endpoints = [], set()
    for record in _candidates(now):
        kind, payload = _verdict(record, now)
        sid = record["subagent_id"]
        if kind == "give_up":
            _give_up(record, payload)
            continue
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
        if _resume(record, payload, start) == "resumed":
            resumed.append(sid)
    return resumed


def _give_up(record: dict[str, Any], reason: str) -> bool:
    """A promised record will not be resumed: mark it (once) and deliver ONE failed completion to the parent, on the
    route of the unit that reported its last stop, with its last summary and transcript tail."""
    sid = record["subagent_id"]
    claimed = _write(lambda conn: conn.execute(
        "UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_gave_up', ?, '$.auto_off', ?, "
        "'$.auto_note', ?), updated_at=? WHERE subagent_id=? AND state='resumable' "
        "AND json_extract(record_json, '$.auto_gave_up') IS NULL", (reason, reason, reason, time.time(), sid)).rowcount)
    if claimed != 1:
        return False
    row = _unit_row(record.get("unit_id"))
    if row is None:
        logger.info("Subagent %s gave up on automatic resume (%s) but has no completion route", sid, reason)
        return False
    from tools.async_delegation import _ROUTING_KEYS, _push_completion_event
    from tools.async_delegation_recovery_hints import transcript_tails
    task = json.loads(row.get("task_json") or "{}")
    prior = json.loads(row.get("result_json") or "{}") or {}
    last = next((r for r in (prior.get("results") or [prior]) if r.get("subagent_id") == sid), {})
    index = last.get("task_index", (task.get("task_indexes") or [0])[0])
    tails = transcript_tails(task.get("task_transcripts") or {})
    tail = tails.get(str(index)) or tails.get(index) or next(iter(tails.values()), None)
    entry = {"task_index": index, "subagent_id": sid, "status": "failed", "summary": last.get("summary"),
             "error": f"automatic resume {reason}", "auto_gave_up": reason, "failure_reason": record.get("failure_reason"),
             **({"transcript_tail": tail} if tail else {})}
    unit = {"delegation_id": record["unit_id"], "session_key": row.get("origin_session") or "",
            "origin_ui_session_id": row.get("origin_ui_session_id") or "", "parent_session_id": row["parent_session_id"],
            "origin_session_id": row.get("origin_session_id") or "", "is_batch": True,
            **{k: task.get(k) for k in ("goal", "goals", "context", "toolsets", "role", "model")},
            "dispatched_at": row.get("dispatched_at"), "completed_at": time.time(),
            **{k: task[k] for k in _ROUTING_KEYS if task.get(k)}}
    _push_completion_event(unit, {"results": [entry], "error": None}, "failed")
    logger.info("Subagent %s: automatic resume %s; delivered one failure to its parent", sid, reason)
    return True


def waiting_entries(parent_agent: Any) -> list[dict[str, Any]]:
    """``delegate_task(action='list')`` rows for this parent's children waiting on a promised automatic resume."""
    if not enabled():
        return []
    from tools.delegate_tool_registry import _owns_subagent_record
    from tools.delegate_tool_resumable import _db_path_for, _write as write
    path = _db_path_for(parent_agent)
    rows = (write(path, lambda conn: conn.execute(
        "SELECT subagent_id, record_json FROM resumable_subagents WHERE state='resumable'").fetchall())
            if path is not None else None) or []
    out = []
    for sid, raw in rows:
        r = json.loads(raw or "{}")
        if not r.get("auto_promised") or r.get("auto_off") or r.get("auto_gave_up"):
            continue
        if not _owns_subagent_record({"ancestor_owner_session_ids": r.get("owners") or ()}, parent_agent):
            continue
        auto = {"at": r.get("next_attempt_at") if r.get("resume_mode") == AUTO_AT else None,
                "attempt": int(r.get("auto_attempts") or 0) + 1, "gives_up_at": _first_stopped_at(r) + _deadline_s()}
        out.append({"subagent_id": sid, "goal": r.get("goal"), "model": r.get("model"), "status": "waiting_to_resume",
                    "next_attempt_at": auto["at"], "auto_attempts": int(r.get("auto_attempts") or 0),
                    "note": waiting_text(auto)})
    return out


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
                            "WHERE state IN ('resumable','claimed')").fetchall()
        note = f"its parent conversation was stopped or closed ({reason or 'session end'})"
        return sum(_note(conn, sid, state=SUPERSEDED, note=note, unstarted_claim=True)
                   for sid, unit in rows if unit in units)
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
    waiting = record and (record.get("state") in ("resumable", SUPERSEDED)
                          or (record.get("state") == "claimed" and record.get("auto_claim")))
    if not waiting:
        return False
    if not _owns_subagent_record({"ancestor_owner_session_ids": record.get("owners") or ()}, parent_agent):
        return False
    # An automatic continue that claimed it but has not started it yet is cancelled too (``confirm_start``).
    return bool(write(path, lambda conn: _note(conn, sid, state="stopped", note="dismissed by its parent",
                                               from_states=("resumable", SUPERSEDED), unpin=True,
                                               unstarted_claim=True)))
