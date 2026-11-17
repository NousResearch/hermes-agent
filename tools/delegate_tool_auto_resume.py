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
from typing import Any, Callable, Optional

from tools.delegate_tool_resumable import AUTO_AT, AUTO_ON_BOOT

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

MAX_AUTO_ATTEMPTS = 2
DEFAULT_DEADLINE_MINUTES = 360
SUPERSEDED = "superseded"
CANCELLED = "Not continued: its parent cancelled the automatic resume"
STAND_IN_FAILED = "Not continued: the scheduler could not build the parent's stand-in"
TURNED_OFF = "gave up: automatic resume was turned off (delegation.auto_resume: false)"
EXPIRED = "gave up: its record expired (delegation.resumable_ttl_hours) before it could resume"
# A continue refused for one of these reasons ran nothing and can succeed on a later tick: the attempt is refunded.
# A depth-limit refusal is not one: the child's depth does not change between ticks.
_TRANSIENT_REFUSALS = ("still running", "being continued", "still stopping", "paused", "capacity", CANCELLED,
                       STAND_IN_FAILED)
# A give-up's completion is built from its unit row and lands only while that row is unchanged; a row that moved is
# read again.
_ROUTE_TRIES = 8


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


def boot_notice(sids: list[str], records: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Boot recovery's sentence for crashed children the scheduler will continue ('' when none will) and the ids
    the parent hears about only from the scheduler: the promised ones, and the ones whose crashed run was their last
    chance (attempts used or deadline passed). Boot recovery stamps each ``auto_promised`` in its own write, so the
    scheduler's next tick resumes it or, for the second kind, delivers its ONE give-up failure (:func:`_give_up`)
    instead of boot recovery's generic "outcome unknown"."""
    promised, spent = {}, []
    for sid in sids:
        record = records.get(sid)
        if not record or not record.get("unit_id") or record.get("auto_off"):
            continue
        candidate = {**record, "resume_mode": AUTO_ON_BOOT}
        if auto := promise(candidate):
            promised[sid] = auto
        elif gave_up(candidate):
            spent.append(sid)
    lines = [notice_line(sid, auto) for sid, auto in promised.items()]
    return ((" " + " ".join(lines)) if lines else ""), [*promised, *spent]


# ── storage ────────────────────────────────────────────────────────────────
def _db() -> Any:
    from persistence_assembly import profile_scope
    return profile_scope()


def _write(fn) -> Any:
    from tools.delegate_tool_resumable import _write as write
    return write(_db(), fn)


def _state(state: str) -> Any:
    from persistence.authority.subagents import SubagentState
    return SubagentState(state)


def _note(ledger, sid: str, *, state: Optional[str], note: str, from_states=("resumable",), unpin=False,
          unstarted_claim=False) -> bool:
    """Record why the scheduler will not (or no longer) run this record; *state* moves it when given (a cancel
    state also clears its debt). *unstarted_claim*: also a record an automatic continue claimed whose run has not
    started (``$.auto_claim``)."""
    return ledger.note_subagent(subagent_id=sid, note=note, state=_state(state) if state else None,
                                from_states=tuple(_state(s) for s in from_states), unpin=unpin,
                                unstarted_claim=unstarted_claim)


def _owes_any() -> bool:
    """With auto_resume off: does this profile's store hold a record still owed its one report? One read per tick,
    so an idle disabled tick never writes (or creates) the store."""
    from persistence import PersistenceError
    from persistence_assembly import open_persistence_reader, store_exists
    scope = _db()
    if not store_exists(scope):
        return False
    try:
        with open_persistence_reader(scope.profile_dir) as store:
            return store.coordination.has_owed_subagents()
    except PersistenceError:  # unreadable here: the settle pass's writer decides (and logs)
        return True


def _settle_disabled(now: float) -> None:
    if not _owes_any():
        return
    for record in _candidates(now, owed_only=True):
        _give_up(record, EXPIRED if record["expired"] else TURNED_OFF)


def _candidates(now: float, *, owed_only: bool = False) -> list[dict[str, Any]]:
    """Records the scheduler acts on, each with ``expired`` set once its TTL passed (``_apply_limits`` may already
    have moved it to ``expired``): only an owed one is read then, so its give-up still reports."""
    def read(ledger):
        # The owner fields come from the record (the process that RAN it); a claim moves only the columns.
        return [{**row.document, "subagent_id": row.subagent_id, "pinned": int(row.pinned),
                 "stop_cause": row.stop_cause,
                 "expired": row.state.value == "expired" or (row.expires_at is not None and row.expires_at <= now)}
                for row in ledger.list_auto_resume_candidates(owed_only=owed_only)]
    return [r for r in (_write(read) or []) if r.get("resume_mode") in (AUTO_AT, AUTO_ON_BOOT)
            and (not r["expired"] or (r.get("auto_promised") and not r.get("auto_gave_up")))]


def _unit_row(unit_id: Any) -> Optional[dict[str, Any]]:
    """The async ledger row the child's background unit reported on (its completion route), or None."""
    if not unit_id:
        return None
    return _write(lambda ledger: _route_row(ledger.get_delegation_route(delegation_id=str(unit_id))))


def _route_row(route: Any) -> Optional[dict[str, Any]]:
    """A unit's :class:`DelegationRoute` as the scheduler's row: its route columns, ``task`` and ``result`` decoded
    (an object, else empty), its dispatch time and ``updated_at``."""
    if route is None:
        return None
    return {"origin_session": route.origin_session, "origin_ui_session_id": route.origin_ui_session_id,
            "parent_session_id": route.parent_session_id, "origin_session_id": route.origin_session_id,
            "task": route.task if isinstance(route.task, dict) else {},
            "result": route.result if isinstance(route.result, dict) else {},
            "dispatched_at": route.dispatched_at, "updated_at": route.updated_at}


# ── verdicts ───────────────────────────────────────────────────────────────
def superseded_reason(ledger, parent_sid: str) -> Optional[str]:
    """Why the parent no longer wants its child resumed, or None while it still does: the live tip of its
    compression chain is gone, or ended for a deliberate (not automatic) reason."""
    return ledger.subagent_parent_superseded(parent_session_id=parent_sid)


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
    from tools.delegate_tool_resumable import _as_record, _owner_identity
    sid, attempts = record["subagent_id"], int(record.get("auto_attempts") or 0)
    owner = _owner_identity()

    def do(ledger):
        claim = ledger.claim_auto_resume(subagent_id=sid, expected_attempts=attempts, parent_session_id=parent_sid,
                                         owner_pid=owner["owner_pid"], owner_started_at=owner["owner_started_at"])
        if claim.record is None:
            return None, claim.superseded
        return {**(_as_record(claim.record) or {}), "auto_attempts": attempts + 1}, None
    return _write(do) or (None, None)


def take_claim(path: Any, sid: str) -> Optional[dict[str, Any]]:
    """The continue path's view of this process's automatic claim of *sid* (``claimed``, ``$.auto_claim`` set, by
    this process), as the record a release hands back to ``resumable``; None once the parent cancelled it."""
    from tools.delegate_tool_resumable import _row, _write as write
    row = write(path, lambda ledger: _row(ledger, sid))
    if not row or row.get("state") != "claimed" or not row.get("auto_claim") or row.get("owner_pid") != os.getpid():
        return None
    return {**row, "state": "resumable"}


def confirm_start(path: Any, sid: str) -> Optional[str]:
    """Right before the continued child is dispatched: the eligibility check, in one write. None while the parent
    has not cancelled it (stop, /stop, /new), else the reason, with the row left as the parent's decision put it (or
    moved to ``superseded``). The claim stays cancellable as unstarted until the run records itself: its run-start
    write gives the parent the last word there."""
    from tools.delegate_tool_resumable import _write as write
    return write(path, lambda ledger: ledger.confirm_auto_start(subagent_id=sid)) if path is not None else CANCELLED


def _settle_refusal(sid: str, attempts: int, error: str) -> None:
    """A continue refused before anything ran: the claim goes back (unless the parent already moved the row), a
    transient refusal refunds the attempt, any other one (a changed route, a missing transcript) ends automatic
    resumes of this record."""
    from persistence.authority.subagents import AutoRefusal
    note = f"automatic continue refused: {error}"[:500]
    if error.startswith(CANCELLED):  # the parent's decision already wrote the row: only the attempt is refunded
        kind = AutoRefusal.CANCELLED
    elif any(word in error for word in _TRANSIENT_REFUSALS):
        kind = AutoRefusal.TRANSIENT
    else:
        kind = AutoRefusal.PERMANENT
    _write(lambda ledger: ledger.settle_auto_refusal(subagent_id=sid, kind=kind, attempts=attempts, note=note,
                                                     owner_pid=os.getpid()))


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
    if record.get("expired"):
        return "give_up", EXPIRED  # _candidates reads an expired record only while it is owed its report
    if promised and (reason := give_up_reason(record, now)):
        return "give_up", reason
    if not _due(record, now):
        return "wait", None
    if blocked := _blocked(record):
        return ("give_up" if promised else "never"), blocked
    unit = _unit_row(record.get("unit_id"))
    if unit is None or not unit.get("parent_session_id"):
        return "never", "no background completion route (a nested or joined child)"
    if reason := _write(lambda ledger: superseded_reason(ledger, str(unit["parent_session_id"]))):
        return "supersede", reason
    return "run", unit


def tick(start: Callable[..., Optional[str]], *, capacity: int, now: Optional[float] = None) -> list[str]:
    """One pass over the bound profile's records; returns the subagent ids it resumed. ``start(record, unit_row,
    message)`` performs the continue and returns an error string, or None once the child was dispatched."""
    from tools.delegate_tool_registry import is_spawn_paused
    from tools.delegate_tool_resumable import retry_hand_backs
    retry_hand_backs()  # a claim a failed start could not hand back is owed again before candidates are read
    if not enabled():  # nothing starts; a child the stop promised a resume still gets its one report
        _settle_disabled(now or time.time())
        return []
    # A full pool or a spawn pause holds only new starts (no claim, no stand-in parent built): give-ups, notes and
    # supersessions still settle, so a deadline reports on time.
    starts = 0 if is_spawn_paused() else max(capacity, 0)
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
                _write(lambda ledger, n=payload: _note(ledger, sid, state=None, note=n))
            continue
        if kind == "supersede":
            _write(lambda ledger, n=payload: _note(ledger, sid, state=SUPERSEDED, note=n))
            logger.info("Subagent %s will not resume automatically: %s", sid, payload)
            continue
        endpoint = json.dumps(record.get("endpoint"))
        if kind != "run" or endpoint in endpoints or len(resumed) >= starts:
            continue  # one resume per endpoint per tick: children sharing one reset are staggered
        endpoints.add(endpoint)
        if _resume(record, payload, start) == "resumed":
            resumed.append(sid)
    return resumed


def gave_up_delegation_id(unit_id: Any, sid: str) -> str:
    """The give-up failure's own ledger row: one per child, so siblings of one unit each report (and ack) once."""
    return f"{unit_id}:auto:{sid}"


def _gave_up_completion(record: dict[str, Any], reason: str, row: dict[str, Any]) -> tuple[dict, dict]:
    """``(event, result)`` of the ONE failed completion, on the route of the unit that reported the last stop, with
    the child's last summary and transcript tail."""
    from tools.async_delegation import _ROUTING_KEYS, completion_event
    from tools.async_delegation_recovery_hints import transcript_tails
    sid = record["subagent_id"]
    task, prior = row.get("task") or {}, row.get("result") or {}
    last = next((r for r in (prior.get("results") or [prior]) if r.get("subagent_id") == sid), {})
    index = last.get("task_index", (task.get("task_indexes") or [0])[0])
    tails = transcript_tails(task.get("task_transcripts") or {})
    tail = tails.get(str(index)) or tails.get(index) or next(iter(tails.values()), None)
    entry = {"task_index": index, "subagent_id": sid, "status": "failed", "summary": last.get("summary"),
             "error": f"automatic resume {reason}", "auto_gave_up": reason, "failure_reason": record.get("failure_reason"),
             **({"transcript_tail": tail} if tail else {})}
    unit = {"delegation_id": gave_up_delegation_id(record["unit_id"], sid), "session_key": row.get("origin_session") or "",
            "origin_ui_session_id": row.get("origin_ui_session_id") or "", "parent_session_id": row["parent_session_id"],
            "origin_session_id": row.get("origin_session_id") or "", "is_batch": True,
            **{k: task.get(k) for k in ("goal", "goals", "context", "toolsets", "role", "model")},
            "dispatched_at": row.get("dispatched_at"), "completed_at": time.time(),
            **{k: task[k] for k in _ROUTING_KEYS if task.get(k)}}
    evt, result, _silent = completion_event(unit, {"results": [entry], "error": None}, "failed")
    return evt, result


def _give_up(record: dict[str, Any], reason: str) -> bool:
    """A promised record will not be resumed: ONE write transaction marks it and records its one failed completion
    as its own pending ledger row (:func:`gave_up_delegation_id`), so a crash on either side of the commit leaves
    the report replayable (restart replay, orphan sweep) and never doubled; then the in-memory copy is queued."""
    from gateway.status import get_process_start_time
    from persistence import DelegationId
    from persistence.authority.subagents import GaveUpCompletion, GiveUpOutcome
    from tools.async_delegation import redact_child_result, redact_completion_event
    sid, unit_id = record["subagent_id"], record.get("unit_id")

    def do(ledger):
        for _ in range(_ROUTE_TRIES):  # the completion is built from the unit row the write finds unchanged
            row = _route_row(ledger.get_delegation_route(delegation_id=str(unit_id))) if unit_id else None
            evt, completion = None, None
            if row is not None:
                evt, result = _gave_up_completion(record, reason, row)
                completion = GaveUpCompletion(
                    unit_id=DelegationId(str(unit_id)), unit_updated_at=row["updated_at"],
                    event=redact_completion_event(evt), result=redact_child_result(result), owner_pid=os.getpid(),
                    owner_started_at=get_process_start_time(os.getpid()))
            # A cancel since the tick read it cleared the debt: nothing to report.
            outcome = ledger.give_up_auto_resume(subagent_id=sid, reason=reason, completion=completion)
            if outcome == GiveUpOutcome.RECORDED:
                return evt
            if outcome == GiveUpOutcome.NO_ROUTE:
                return False
            if outcome == GiveUpOutcome.NOT_OWED:
                return None
        return None
    evt = _write(do)
    if evt is False:  # pruning keeps a waiting child's unit row: only a nested or joined child has none
        logger.info("Subagent %s gave up on automatic resume (%s) but has no completion route", sid, reason)
    if not evt:
        return False
    from tools.process_registry import process_registry
    process_registry.completion_queue.put(evt)
    logger.info("Subagent %s: automatic resume %s; delivered one failure to its parent", sid, reason)
    return True


def waiting_entries(parent_agent: Any) -> list[dict[str, Any]]:
    """``delegate_task(action='list')`` rows for this parent's children waiting on a promised automatic resume: not
    one that will give up instead (deadline, attempts, expiry, auto_resume turned off), even before a tick says so."""
    if not enabled():  # the next tick delivers each promised child's give-up: none is waiting
        return []
    from tools.delegate_tool_registry import _owns_subagent_record
    from tools.delegate_tool_resumable import _db_path_for, _write as write
    path = _db_path_for(parent_agent)
    rows = (write(path, lambda ledger: ledger.list_waiting_subagents()) if path is not None else None) or ()
    out = []
    for row in rows:
        sid, r = row.subagent_id, dict(row.document)
        if r.get("auto_off") or give_up_reason(r):
            continue
        if not _owns_subagent_record({"ancestor_owner_session_ids": r.get("owners") or ()}, parent_agent):
            continue
        auto = {"at": r.get("next_attempt_at") if r.get("resume_mode") == AUTO_AT else None,
                "attempt": int(r.get("auto_attempts") or 0) + 1, "gives_up_at": _first_stopped_at(r) + _deadline_s()}
        out.append({"subagent_id": sid, "goal": r.get("goal"), "model": r.get("model"), "status": "waiting_to_resume",
                    "next_attempt_at": auto["at"], "auto_attempts": int(r.get("auto_attempts") or 0),
                    "note": waiting_text(auto)})
    return out


def refuse_waiting_duplicates(task_list: list[dict[str, Any]], routes: list[Any], parent_agent: Any) -> list[Any]:
    """*routes* with each task whose goal is exactly (``strip()``) the goal of one of this parent's children waiting
    on a promised automatic resume refused alone: a second copy would race the first on the same work."""
    waiting = {str(w.get("goal") or "").strip(): w for w in waiting_entries(parent_agent)}
    waiting.pop("", None)
    if not waiting:
        return routes
    from dataclasses import replace
    out = list(routes)
    for i, task in enumerate(task_list):
        hit = waiting.get(str(task.get("goal") or "").strip())
        if hit and not out[i].error:
            sid = hit["subagent_id"]
            out[i] = replace(out[i], error=(
                f"Not dispatched: subagent {sid} already has this exact goal and is {hit['note']}. Either wait for "
                f"it (its result, or one failure if it gives up, arrives on its own), or call "
                f"delegate_task(action='stop', subagent_id='{sid}') first and then re-dispatch."))
    return out


def waiting_on_resume(entry: dict[str, Any]) -> bool:
    """A ledger result entry (or boot result) for a child still waiting on its promised automatic resume, or an
    automatic attempt that never started (``unstarted_auto``: not an outcome the parent is told about)."""
    return (isinstance(entry.get("auto_resume"), dict) or bool(entry.get("waiting_to_resume"))
            or bool(entry.get("unstarted_auto")))


# ── parent actions ─────────────────────────────────────────────────────────
def supersede_for_session(*, session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "",
                          reason: str = "") -> int:
    """A person stopped, reset or closed a conversation: its resumable children must not wake up later. Automatic
    cleanup (a gateway restart, an orphan reap) is not a decision about them and changes nothing. With
    ``auto_resume`` off it still cancels the promised ones, so turning it back on cannot wake them."""
    from persistence.conversation.sessions import is_automatic_end_reason
    if is_automatic_end_reason(reason.rsplit(":", 1)[-1]):
        return 0
    if not (session_key or origin_ui_session_id or parent_session_id):
        return 0
    note = f"its parent conversation was stopped or closed ({reason or 'session end'})"
    return int(_write(lambda ledger: ledger.supersede_subagents_for_session(
        note=note, session_key=session_key, origin_ui_session_id=origin_ui_session_id,
        parent_session_id=parent_session_id, promised_only=not enabled())) or 0)


def dismiss(parent_agent: Any, sid: str) -> Optional[str]:
    """``delegate_task(action='stop')`` on a child that is not running but waits to resume: it never resumes on its
    own and its worktree pin is released (its transcript stays): ``"dismissed"``. None when there is no such record
    of this parent. With ``auto_resume`` off only a promised one is dismissed (its parent was told to stop it this way).
    A stop on a row the parent's own continue holds is recorded on it (``$.auto_cancel``): ``"recorded"``.
    The decision is taken on the row it is written to: a refused continue handing the row back in between would
    otherwise make the cancel match nothing and lose the stop, so a row that moved is read and judged again."""
    from tools.delegate_tool_resumable import _CAS_TRIES, _db_path_for, _write as write

    def do(ledger):
        for _ in range(_CAS_TRIES):
            stored = ledger.get_subagent(subagent_id=sid)
            if not _dismissable(stored, parent_agent):
                return None
            # The parent's own continue holding it records the stop for its hand-back (``recorded``); an automatic
            # continue that claimed it but has not started it yet is cancelled too (``confirm_start``).
            if done := ledger.dismiss_subagent(expected=stored, note="dismissed by its parent"):
                return done
            if ledger.get_subagent(subagent_id=sid) == stored:
                return None  # nothing to change on an unmoved row
        return None
    return write(_db_path_for(parent_agent), do)


def _dismissable(stored: Any, parent_agent: Any) -> bool:
    """Whether the parent may stop *stored* here: a child of its own waiting to resume, or one its own continue
    holds and has not run yet (a refused continue must not hand its promise back)."""
    from tools.delegate_tool_registry import _owns_subagent_record
    from tools.delegate_tool_resumable import _as_record
    record = _as_record(stored)
    if not record:
        return False
    state = record.get("state")
    owed_expired = state == "expired" and not record.get("auto_gave_up")
    manual_claim = state == "claimed" and not record.get("auto_claim")
    waiting = state in ("resumable", SUPERSEDED) or owed_expired or (state == "claimed" and record.get("auto_claim"))
    if not (manual_claim or (waiting and (record.get("auto_promised") or enabled()))):
        return False
    return bool(_owns_subagent_record({"ancestor_owner_session_ids": record.get("owners") or ()}, parent_agent))
