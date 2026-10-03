"""Live-subagent registry + model-facing control plane (list/steer/stop) for delegate_task."""

from __future__ import annotations

import logging
import json
import threading
import time
from typing import Any, Dict, List, Optional
from agent.interrupt_compat import request_hard_interrupt
from tools.registry import tool_error

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

_spawn_pause_lock = threading.Lock()
_spawn_paused: bool = False
_active_subagents_lock = threading.Lock()
# subagent_id -> mutable record tracking the live child agent.  Stays only
# for the lifetime of the run; _run_single_child is the owner.
_active_subagents: Dict[str, Dict[str, Any]] = {}
# subagent_id -> {goal, delegation_id, owner_agent_session_id} retained AFTER the child finishes (bounded FIFO).
# Child-started background processes routinely outlive the child (its npm ci with notify_on_complete=true finishes
# after the summary was delivered); their completion notifications reach the parent via the shared completion_queue
# and need delegation attribution even though the live registry entry is gone.
_RECENT_SUBAGENTS_CAP = 200
_recent_subagents: Dict[str, Dict[str, Any]] = {}

# Per-parent-session circuit breaker for delegate_task control actions whose
# target subagent no longer exists or has closed steering. The original bug
# (#94858) was the model repeatedly calling delegate_task(action='steer',
# subagent_id='sa-0-X') against an already-finished child; the tool kept
# returning a recoverable-looking error ("No live subagent 'sa-0-X' ...")
# and the model kept retrying, burning tokens and CPU until the operator
# killed the gateway. The fix has two halves:
#   1. The control-path error is always marked non-retryable (recoverable=False
#      + an explicit "do not retry" hint) so a well-behaved model stops on
#      the first failure.
#   2. This counter tracks how many times the SAME parent-session has failed
#      against the SAME subagent_id; once the count exceeds
#      _STALE_SUBAGENT_RETRY_LIMIT, every further attempt short-circuits with
#      an obvious "you are in a retry loop" message that names the loop and
#      points at action='list' as the alternative. This catches the actual
#      misbehaving model that ignored the recoverable=False marker.
#
# Keyed by (parent_session_id, subagent_id) so siblings and grandchildren
# don't share state, and the live agent object identity isn't required (a
# parent-agent rebuild in the CLI mid-session would orphan anything keyed on
# the AIAgent instance). Cleared automatically when the child re-registers
# (a recycled public id maps to a new run) because we look it up at every
# call rather than caching success counts.
_dead_subagent_hits: Dict[tuple, int] = {}
_dead_subagent_hits_lock = threading.Lock()
# Bound is intentionally small: one transient miss is plausible, two is
# already suspicious, three is a loop. The model is given no slack past three.
_STALE_SUBAGENT_RETRY_LIMIT = 3
# Ceiling on live breaker keys. Only reached by a session that keeps meeting
# brand-new dead ids; eviction then prefers non-tripped keys (see
# _record_dead_subagent_hit) so an already-tripped breaker is never laundered.
_DEAD_SUBAGENT_HITS_CAP = 1024


def get_subagent_attribution(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """``{subagent_id, goal, delegation_id}`` for a process task_id that belongs to a live or recently-finished child
    (children run their terminal sessions under ``task_id == subagent_id``), else None."""
    if not task_id or not isinstance(task_id, str):
        return None
    with _active_subagents_lock:
        record = _active_subagents.get(task_id) or _recent_subagents.get(task_id)
    if record is None:
        return None
    return {"subagent_id": task_id, "goal": record.get("goal"), "delegation_id": record.get("delegation_id")}

def set_spawn_paused(paused: bool) -> bool:
    """Globally block/unblock NEW delegate_task spawns (active children keep running). Returns the new state."""
    global _spawn_paused
    with _spawn_pause_lock:
        _spawn_paused = bool(paused)
        return _spawn_paused

def is_spawn_paused() -> bool:
    with _spawn_pause_lock:
        return _spawn_paused

def _register_subagent(record: Dict[str, Any]) -> None:
    sid = record.get("subagent_id")
    if not sid:
        return
    record.setdefault("accepting_steer", True)
    with _active_subagents_lock:
        _active_subagents[sid] = record
    # A recycled public id means a brand-new run with a clean miss
    # counter. Without this, a parent that retried a dead id earlier in
    # the same session would inherit the per-(parent, target) count
    # and trip the loop detector on the very first call against the
    # new, perfectly healthy child. See #94858.
    #
    # The record may not carry owner_agent_session_id (test doubles, and any
    # future spawn path that forgets it), so fall back to the "" bucket the
    # miss counter itself uses when the parent has no session_id. Resetting
    # only when the field is present left a recycled id inheriting its
    # predecessor's counter -- the exact false positive the guard exists for.
    owner_sid = str(record.get("owner_agent_session_id") or "")
    _reset_dead_subagent_hits(owner_sid, sid)

def _unregister_subagent(subagent_id: str, *, agent: Any = None) -> None:
    """Drop the live record (exact agent identity when given) and keep a bounded attribution stub."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None or not (agent is None or record.get("agent") is agent):
            return
        _active_subagents.pop(subagent_id, None)
        sid = record.get("subagent_id")
        if not sid:
            return
        _recent_subagents[sid] = {k: record.get(k) for k in ("goal", "delegation_id", "owner_agent_session_id")}
        while len(_recent_subagents) > _RECENT_SUBAGENTS_CAP:
            _recent_subagents.pop(next(iter(_recent_subagents)), None)

def _record_dead_subagent_hit(parent_session_id: Optional[str], subagent_id: str) -> int:
    """Increment the per-(parent, target) miss counter and return the new total.

    Bounded eviction keeps the dict small even under spam: a runaway loop hits
    the same key forever, so eviction is defensive against fresh ids arriving
    while a breaker is already tripped. Two properties the plain FIFO lacked:
    eviction only ever drops a key that has NOT tripped (a live breaker is never
    silently un-tripped by unrelated traffic and handed full slack again), and it
    only runs when a NEW key is inserted (one session hammering one dead id never
    evicts anything, so the loop it is meant to catch keeps its own counter).
    """
    if not subagent_id:
        return 0
    key = (str(parent_session_id or ""), subagent_id)
    with _dead_subagent_hits_lock:
        is_new_key = key not in _dead_subagent_hits
        count = _dead_subagent_hits.get(key, 0) + 1
        _dead_subagent_hits[key] = count
        if is_new_key and len(_dead_subagent_hits) > _DEAD_SUBAGENT_HITS_CAP:
            for candidate, seen in _dead_subagent_hits.items():
                if candidate != key and seen <= _STALE_SUBAGENT_RETRY_LIMIT:
                    del _dead_subagent_hits[candidate]
                    break
        return count


def _reset_dead_subagent_hits(parent_session_id: Optional[str], subagent_id: str) -> None:
    """Drop the (parent, target) miss counter when the child re-registers.

    A recycled public id means a brand-new run, so any previous "stale"
    misses against the same string are no longer evidence of a loop.
    """
    if not subagent_id:
        return
    key = (str(parent_session_id or ""), subagent_id)
    with _dead_subagent_hits_lock:
        _dead_subagent_hits.pop(key, None)


def _non_retryable_subagent_error(
    *,
    subagent_id: str,
    reason: str,
    hint: str,
    loop_count: Optional[int] = None,
) -> str:
    """Build the structured JSON error for an unreachable / closed subagent.

    The model treats any ``{"error": "..."}`` result as "I should fix this
    and try again" unless the payload makes the terminal nature explicit.
    This helper always tags the error with ``recoverable=False`` and a
    hard-line "do not retry" message; when the per-session circuit breaker
    has tripped (loop_count > _STALE_SUBAGENT_RETRY_LIMIT), it additionally
    flags ``loop_detected=True`` and names the offending call so the model
    can see exactly what went wrong.

    The shape is stable JSON so a future prompt-builder or guardrail can
    pattern-match on the keys without re-parsing the prose.
    """
    payload: Dict[str, Any] = {
        "error": reason,
        "recoverable": False,
        "subagent_id": subagent_id,
        "do_not_retry": True,
        "hint": hint,
    }
    if loop_count is not None and loop_count > _STALE_SUBAGENT_RETRY_LIMIT:
        payload["loop_detected"] = True
        payload["attempts"] = loop_count
    return json.dumps(payload, ensure_ascii=False)


def _close_subagent_steering(subagent_id: str, agent: Any) -> Optional[str]:
    """Atomically close steer acceptance and drain its final durable artifact. ``steer_subagent`` holds the same
    registry lock through ``agent.steer``, so either acceptance wins and this drain sees its exact text, or closure
    wins and the caller is rejected. Exact agent identity prevents a finishing child with a recycled public id from
    closing its replacement."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None or record.get("agent") is not agent:
            return None
        record["accepting_steer"] = False
        drain = getattr(agent, "_drain_pending_steer", None)
        if not callable(drain):
            return None
        try:
            pending = drain()
        except Exception as exc:
            logger.debug("final steer drain for %s failed: %s", subagent_id, exc)
            return None
        return pending if isinstance(pending, str) and pending.strip() else None

def interrupt_subagent(subagent_id: str) -> bool:
    """Request that one running subagent stop at its next iteration boundary
    (cooperative: the flag propagates to in-flight tools and recurses into
    grandchildren via AIAgent.interrupt()). True iff a matching subagent was found."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    agent = record.get("agent") if record else None
    if agent is None:
        return False
    try:
        return bool(request_hard_interrupt(agent, f"Interrupted via TUI ({subagent_id})"))
    except Exception as exc:
        logger.debug("interrupt_subagent(%s) failed: %s", subagent_id, exc)
        return False

def _subagent_transport_matches(record, transport) -> bool:
    """Authority follows the owning session's LIVE transport slot, read at check time.

    ``owner_transport`` on the record is only the capture-time marker that a gateway session
    commissioned the child (``None`` = no RPC authority ever). The slot is authoritative because
    every reattach path (prompt.submit, queued drain, resume, activate, viewer failover) already
    mutates it; a per-record copy needed a matching registry sync at each of those sites and two
    were missed (#106663). Records whose owner is not a session dict keep the exact-object rule."""
    from tui_gateway.transport import FanoutTransport

    if record.get("owner_transport") is None:
        return False
    owner = record.get("owner_session_record")
    bound = owner.get("transport") if isinstance(owner, dict) else record.get("owner_transport")
    return bound is transport or (isinstance(bound, FanoutTransport) and bound.contains(transport))


def steer_subagent(
    subagent_id: str, text: str, *, owner_session_id: Optional[str] = None, owner_transport: Any = None,
    owner_session_record: Any = None,
) -> bool:
    """Queue steering text into a running subagent without stopping it.

    AIAgent.steer() appends the text to the child's last tool result at its next iteration boundary — the current tool
    call is never cut. True iff the text was QUEUED while the child still accepted work; False for unknown/closed id,
    ownership mismatch, no live agent, or empty text. ``owner_session_id=None`` keeps the in-process helper contract;
    gateway callers must pass exact authority. Acceptance and completion are linearized by the registry lock: if
    acceptance wins but no delivery boundary remains, the text lands in the entry as ``missed_steer``.
    """
    if not text or not text.strip():
        return False
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if not record or not record.get("accepting_steer", False):
            return False
        if owner_session_id is not None and (
            record.get("owner_session_id") != owner_session_id
            or owner_transport is None
            or not _subagent_transport_matches(record, owner_transport)
            or owner_session_record is None
            or record.get("owner_session_record") is not owner_session_record
        ):
            return False
        agent = record.get("agent")
        if agent is None:
            return False
        try:
            return bool(agent.steer(text))
        except Exception as exc:
            logger.debug("steer_subagent(%s) failed: %s", subagent_id, exc)
            return False

def _capture_gateway_steer_authority(owner_session_id: Optional[str]) -> tuple[Any, Any]:
    """Exact request transport + live session generation, if any — an in-process
    bridge, not a serializable capability. Non-gateway hosts get ``(None, None)``."""
    if not owner_session_id:
        return None, None
    try:
        from tui_gateway.server import _current_session_steer_authority
        return _current_session_steer_authority(owner_session_id)
    except Exception:
        return None, None

# Registry record fields never exposed to the TUI/RPC snapshot.
_PRIVATE_RECORD_KEYS = frozenset({"agent", "owner_session_id", "owner_transport", "owner_session_record", "accepting_steer"})

def list_active_subagents() -> List[Dict[str, Any]]:
    """Copy of the running subagent tree ({subagent_id, parent_id, depth, goal, model,
    started_at, tool_count, status, ...}); safe from any thread."""
    with _active_subagents_lock:
        return [{k: v for k, v in r.items() if k not in _PRIVATE_RECORD_KEYS} for r in _active_subagents.values()]

def _is_descendant_of(child_agent: Any, parent_agent: Any, max_hops: int = 8) -> bool:
    """True when *child_agent* sits below *parent_agent* in the spawn tree (walks the ``_delegate_parent_ref`` weakref
    chain stamped at build time). Identity only — a parent may steer/stop its own children and grandchildren, never
    a sibling tree owned by another conversation."""
    if child_agent is None or parent_agent is None:
        return False
    cur = child_agent
    for _ in range(max_hops):
        ref = getattr(cur, "_delegate_parent_ref", None)
        ancestor = ref() if callable(ref) else None
        if ancestor is None:
            return False
        if ancestor is parent_agent:
            return True
        cur = ancestor
    return False

# Model-facing control actions accepted by delegate_task(action=...).
# "spawn" (or omitted) keeps the historical spawn semantics.
_CONTROL_ACTIONS = frozenset({"list", "steer", "stop"})

def _resolve_session_lineage(session_id: Optional[str], parent_agent: Any) -> str:
    """Tip of a session id's compression lineage via the parent's live SessionDB (best-effort; input unchanged when
    unavailable) so a delegation dispatched before a compression rotation still matches the rotated parent."""
    sid = str(session_id or "")
    db = getattr(parent_agent, "_session_db", None)
    if not sid or db is None:
        return sid
    try:
        resolved = db.resolve_resume_session_id(sid)
        return str(resolved) if resolved else sid
    except Exception:
        return sid

def _owns_subagent_record(record: Dict[str, Any], parent_agent: Any) -> bool:
    """True when *parent_agent*'s conversation owns this live-child record.

    Tier 1: identity — the ``_delegate_parent_ref`` weakref chain reaches
    *parent_agent* (fast path while the parent AIAgent survives the run). Tier 2:
    durable lineage — the record's ``owner_agent_session_id`` matches the caller's
    ``session_id`` after resolving compression-rotation lineage on both sides.
    Tier 2 exists because the identity chain is BRITTLE across parent rebuilds:
    the CLI sets ``self.agent = None`` mid-session (route change, credential
    refresh, /model, MoA one-shots) and builds a NEW AIAgent while the child keeps
    a weakref to the old one. Delivery routes by durable session id; control must
    use the same spine or running children go invisible/unsteerable.
    """
    if _is_descendant_of(record.get("agent"), parent_agent):
        return True
    owner_sid = str(record.get("owner_agent_session_id") or "")
    parent_sid = str(getattr(parent_agent, "session_id", "") or "")
    if not owner_sid or not parent_sid:
        return False
    if owner_sid == parent_sid:
        return True
    # Compression rotation on either side: compare lineage tips.
    return _resolve_session_lineage(owner_sid, parent_agent) in {parent_sid, _resolve_session_lineage(parent_sid, parent_agent)}

def _list_payload(parent_agent: Any) -> Dict[str, Any]:
    with _active_subagents_lock:
        records = list(_active_subagents.values())
    entries = []
    for r in records:
        if not _owns_subagent_record(r, parent_agent):
            continue
        started = r.get("started_at")
        entries.append({
            "subagent_id": r.get("subagent_id"),
            "parent_id": r.get("parent_id"),
            "goal": r.get("goal"),
            "model": r.get("model"),
            "status": r.get("status"),
            "running_seconds": round(time.time() - started, 1) if isinstance(started, (int, float)) else None,
            "accepting_steer": bool(r.get("accepting_steer", False)),
            "live_transcript": getattr(r.get("agent"), "_live_transcript_path", None),
        })
    payload: Dict[str, Any] = {"action": "list", "count": len(entries), "subagents": entries}
    if not entries:
        payload["note"] = (
            "No live subagents right now. Children that already finished "
            "have delivered (or will deliver) their results as normal "
            "completion messages — there is nothing to steer or stop."
        )
    return payload

def _handle_control_action(action: str, subagent_id: Optional[str], message: Optional[str], parent_agent: Any) -> str:
    """Synchronous control plane for delegate_task: list/steer/stop. Runs in-turn (never backgrounded) over the same
    registry the TUI overlay drives, scoped so a conversation can only control its own spawn tree."""
    if action == "list":
        return json.dumps(_list_payload(parent_agent), ensure_ascii=False)

    # steer / stop need a resolvable, owned target.
    sid = (subagent_id or "").strip()
    if not sid:
        return tool_error(f"action='{action}' requires subagent_id (from the spawn dispatch response or action='list').")
    # The parent session id is the durable spine the rest of delegate_tool
    # uses (a CLI rebuild swaps the AIAgent instance mid-session but keeps
    # session_id). Using the live agent object directly would orphan the
    # counter on the very first rebuild and let a fresh instance re-enter
    # the loop with a clean slate.
    parent_sid = str(getattr(parent_agent, "session_id", "") or "")
    with _active_subagents_lock:
        record = _active_subagents.get(sid)
    if record is None or not _owns_subagent_record(record, parent_agent):
        return _stale_subagent_error(parent_sid, sid, action, target_missing=True)
    if action == "steer" and not (message or "").strip():
        return tool_error("action='steer' requires a non-empty 'message' describing the course correction.")
    outcome = _CONTROL_OUTCOMES.get(action)
    if outcome is None:
        return tool_error(f"Unknown action '{action}'. Use spawn, list, steer, or stop.")
    status, note, _failure = outcome
    ok = interrupt_subagent(sid) if action == "stop" else steer_subagent(sid, message.strip())
    if ok:
        return json.dumps({"action": action, "subagent_id": sid, "status": status, "note": note}, ensure_ascii=False)
    # Record still exists and we own it, but the child vanished / closed its steer
    # window between the ownership check and the call. Same terminal class as a
    # missing target from the model's point of view: retrying will never flip the
    # answer, and #94858 showed models WILL retry unless the error is unambiguous.
    return _stale_subagent_error(parent_sid, sid, action, target_missing=False)

# action -> (success status, success note, failure error template)
_CONTROL_OUTCOMES = {
    "stop": (
        "interrupt_requested",
        "The subagent stops at its next iteration boundary (in-flight tool calls are asked to cancel). Its "
        "partial result still re-enters the conversation as a completion message — do not wait or poll.",
        "Could not interrupt '{sid}' — it likely finished in the last "
        "moment. Its result arrives as a normal completion message.",
    ),
    "steer": (
        "queued",
        "Steering text queued. The subagent sees it appended to its next tool result — the current tool call is "
        "never cut. If the child finishes before a delivery boundary remains, the text is reported back as "
        "missed_steer in its completion entry.", "Subagent '{sid}' is no longer accepting steering (finishing or "
        "already finished). Its result arrives as a normal completion "
        "message; re-delegate a follow-up task if more work is needed.",
    ),
}


def _stale_subagent_error(
    parent_session_id: Optional[str],
    subagent_id: str,
    action: str,
    *,
    target_missing: bool,
) -> str:
    """Build the structured, non-retryable error for a dead/closed subagent.

    One entry point so the per-(parent, target) circuit breaker and the
    ``recoverable=False`` JSON shape stay in lockstep — any future change
    to the loop-detection wording has exactly one site to update.

    ``target_missing=True`` is the original #94858 case (no record at all,
    or the parent doesn't own it); ``target_missing=False`` is the
    "record exists but is no longer accepting" case (closed steering or
    a child that vanished between checks). Both are terminal: retrying
    will never flip the answer, so both go through the same bounded
    counter and structured payload.
    """
    miss_count = _record_dead_subagent_hit(parent_session_id, subagent_id)
    if target_missing:
        reason = (
            f"No live subagent '{subagent_id}' in this conversation's "
            "spawn tree. It may have already finished (its result "
            "arrives as a normal completion message)."
        )
    else:
        reason = (
            f"Subagent '{subagent_id}' is no longer accepting "
            f"{action} (finishing or already finished). Its result "
            "arrives as a normal completion message."
        )
    if miss_count > _STALE_SUBAGENT_RETRY_LIMIT:
        # Loop detected: same parent keeps hitting the same dead id. The
        # regular "do not retry" hint isn't enough — the model is ignoring
        # it, so name the loop and point at the actual alternative.
        reason = (
            f"RETRY LOOP DETECTED: you have called "
            f"delegate_task(action='{action}', subagent_id='{subagent_id}') "
            f"{miss_count} times against a subagent that no longer exists "
            f"or has already finished. The previous {miss_count - 1} "
            f"attempt(s) already returned this error and you ignored the "
            f"'do_not_retry' flag. Do not call delegate_task with this "
            f"subagent_id again."
        )
        hint = (
            "If you still need work done, call delegate_task(action='list') "
            "to see any live children, or spawn a fresh subagent with "
            "delegate_task(action='spawn', goal=...). The dead subagent's "
            "result, if any, is already in the conversation as a normal "
            "completion message — re-reading it does not require a tool call."
        )
    else:
        hint = (
            f"Do not retry delegate_task(action='{action}', "
            f"subagent_id='{subagent_id}'). The subagent is gone. If you "
            "need to know its current state, call "
            "delegate_task(action='list') for live children, or wait for "
            "its completion message to arrive."
        )
    return _non_retryable_subagent_error(
        subagent_id=subagent_id,
        reason=reason,
        hint=hint,
        loop_count=miss_count,
    )
