"""Live-subagent registry + model-facing control plane (list/steer/stop) for delegate_task."""

from __future__ import annotations

import contextlib
import logging
import json
import threading
import time
from functools import partial
from typing import Any, Dict, List, Optional
from agent.interrupt_compat import request_hard_interrupt
from agent.interrupt_control import _ic_codex_method
from tools.registry import tool_error

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

_spawn_pause_lock = threading.Lock()
_spawn_paused: bool = False
_active_subagents_lock = threading.Lock()
# subagent_id -> mutable record tracking the live child agent.  Stays only
# for the lifetime of the run; _run_single_child is the owner.
_active_subagents: dict[str, dict[str, Any]] = {}
# subagent_id -> {goal, delegation_id, owner_agent_session_id} retained AFTER the child finishes (bounded FIFO).
# Child-started background processes routinely outlive the child (its npm ci with notify_on_complete=true finishes
# after the summary was delivered); their completion notifications reach the parent via the shared completion_queue
# and need delegation attribution even though the live registry entry is gone.
_RECENT_SUBAGENTS_CAP = 200
_recent_subagents: dict[str, dict[str, Any]] = {}

def get_subagent_attribution(task_id: Optional[str]) -> Optional[dict[str, Any]]:
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

def _register_subagent(record: dict[str, Any]) -> None:
    sid = record.get("subagent_id")
    if not sid:
        return
    record.setdefault("accepting_steer", True)
    with _active_subagents_lock:
        _active_subagents[sid] = record

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
        pending = None
        drain = getattr(agent, "_drain_pending_steer", None)
        if callable(drain):
            try:
                pending = drain()
            except Exception as exc:
                logger.debug("final steer drain for %s failed: %s", subagent_id, exc)
        # Whatever no drain delivered by now (this one, or the turn finalizer's leftover handoff) was missed.
        for receipt in record.get("steers") or ():
            if receipt["status"] == "queued":
                receipt["status"] = "missed"
                logger.info("steer missed by subagent %s (it finished first)", subagent_id)
        if record.get("steers"):
            with contextlib.suppress(Exception):
                agent._steer_drain_hook = None  # drop the agent -> hook -> record cycle
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
    owner_session_record: Any = None, relayed_from: Optional[str] = None, relayed_to: Optional[list[str]] = None,
) -> bool:
    """Queue steering text into a running subagent without stopping it (and, labelled, into its live sub-helpers).

    AIAgent.steer() appends the text to the child's last tool result at its next iteration boundary — the current tool
    call is never cut. True iff the text was QUEUED while the child still accepted work; False for unknown/closed id,
    ownership mismatch, no live agent, or empty text. ``owner_session_id=None`` keeps the in-process helper contract;
    gateway callers must pass exact authority. Acceptance and completion are linearized by the registry lock: if
    acceptance wins but no delivery boundary remains, the text lands in the entry as ``missed_steer``. A Codex
    app-server child gets it in its active turn via native ``turn/steer`` (queued as above when no turn takes it).
    Sub-helpers that accepted a relayed copy are appended to ``relayed_to`` when given.
    """
    if not text or not text.strip():
        return False
    queued = (_RELAY_LABEL.format(via=relayed_from) if relayed_from else "") + text.strip()
    authority = (subagent_id, owner_session_id, owner_transport, owner_session_record)
    with _active_subagents_lock:
        target = _steer_target(*authority)
    native = _ic_codex_method(target[1], "request_steer") if target else None
    native_at, delivered = time.time(), False
    if native is not None and not getattr(target[1], "_interrupt_requested", False):
        # A Codex app-server child runs its own turn loop and never drains the Hermes queue: hand the text to its
        # active turn (`turn/steer`, as redirect() does). A JSON-RPC round trip, so outside the registry lock.
        try:
            delivered = bool(native(queued))
        except Exception:  # the native runtime is a subprocess boundary; fall back to the queue below
            logger.debug("steer_subagent(%s) turn/steer failed", subagent_id, exc_info=True)
    with _active_subagents_lock:
        if delivered:
            record, agent = target
            now = time.time()
            receipt = {"queued_at": native_at, "delivered_at": now, "latency_s": round(now - native_at, 3),
                       "status": "delivered", "relayed_from": relayed_from, "_text": queued}
            receipts = record.setdefault("steers", [])
            receipts.append(receipt)
        else:  # no native turn took it: queue it (re-checked, closing may have won meanwhile)
            target = _steer_target(*authority)
            if target is None:
                return False
            record, agent = target
            # Publish the receipt (and the drain hook) before queueing, so a drain racing this call always finds it.
            receipt = {"queued_at": time.time(), "delivered_at": None, "latency_s": None, "status": "queued",
                       "relayed_from": relayed_from, "_text": queued}
            receipts = record.setdefault("steers", [])
            receipts.append(receipt)
            if getattr(agent, "_steer_drain_hook", None) is None:
                with contextlib.suppress(Exception):
                    agent._steer_drain_hook = partial(_settle_steer_receipts, record, subagent_id)
            try:
                accepted = bool(agent.steer(queued))
            except Exception:  # an agent without a working steer(): refuse the steer, keep the registry intact
                logger.debug("steer_subagent(%s) failed", subagent_id, exc_info=True)
                accepted = False
            if not accepted:
                receipts.remove(receipt)
                return False
        del receipts[:-_RECEIPTS_CAP]
        # A helper below the top level joins its own sub-helpers synchronously, so its copy waits for the whole
        # sub-batch. Relay to them now; authority was checked on this record and passes down the spawn tree.
        sub_helpers = [sid for sid, r in _active_subagents.items()
                       if r.get("accepting_steer") and _record_parent_agent(r) is agent]
    logger.info("steer %s subagent %s (%d chars%s)", "delivered via Codex turn/steer to" if delivered else
                "queued for", subagent_id, len(queued), f", relayed from {relayed_from}" if relayed_from else "")
    for sid in sub_helpers:
        if steer_subagent(sid, text, relayed_from=subagent_id, relayed_to=relayed_to) and relayed_to is not None:
            relayed_to.append(sid)
    if not delivered and not sub_helpers and getattr(agent, "_executing_tools", False) is True:
        # Same yield a user steer gets (redirect): a long foreground command moves to the background registry.
        with contextlib.suppress(Exception):
            agent.request_tool_yield()
    return True


def _steer_target(subagent_id: str, owner_session_id: Optional[str], owner_transport: Any,
                  owner_session_record: Any) -> Optional[tuple[dict[str, Any], Any]]:
    """The live (record, agent) this caller may steer, else None. Registry lock held."""
    record = _active_subagents.get(subagent_id)
    if not record or not record.get("accepting_steer", False):
        return None
    if owner_session_id is not None and (
        record.get("owner_session_id") != owner_session_id
        or owner_transport is None
        or not _subagent_transport_matches(record, owner_transport)
        or owner_session_record is None
        or record.get("owner_session_record") is not owner_session_record
    ):
        return None
    agent = record.get("agent")
    return None if agent is None else (record, agent)


_RECEIPTS_CAP = 10
_RELAY_LABEL = ("[Steer relayed from above: your parent helper {via} was steered while it waits on you. "
                "Act on it if it concerns your task; {via} also gets it when you finish.]\n")


def _record_parent_agent(record: dict[str, Any]) -> Any:
    ref = getattr(record.get("agent"), "_delegate_parent_ref", None)
    return ref() if callable(ref) else None


def _settle_steer_receipts(record: dict[str, Any], subagent_id: str, drained: list[str], requeued: bool = False) -> None:
    """The child's steer-drain hook (its own thread): each drained part settles the oldest queued receipt with exactly
    its text; a requeue (the drain found no tool result to attach to) reopens the newest delivered one. Never takes
    the registry lock (``_close_subagent_steering`` holds it while draining); a drain after steering closed is that
    final one, and closing marks what is left as missed."""
    if not record.get("accepting_steer", False):
        return
    now, parts = time.time(), list(drained)
    receipts = list(record.get("steers") or ())
    for r in reversed(receipts) if requeued else receipts:
        if r["status"] == ("delivered" if requeued else "queued") and r["_text"] in parts:
            parts.remove(r["_text"])
            if requeued:
                r.update(status="queued", delivered_at=None, latency_s=None)
                logger.info("steer for subagent %s requeued (no tool result to attach to yet)", subagent_id)
            else:
                r.update(status="delivered", delivered_at=now, latency_s=round(now - r["queued_at"], 3))
                logger.info("steer delivered to subagent %s after %.2fs", subagent_id, now - r["queued_at"])

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
_PRIVATE_RECORD_KEYS = frozenset({
    "agent", "owner_session_id", "owner_transport", "owner_session_record", "accepting_steer", "steers"})

def list_active_subagents() -> list[dict[str, Any]]:
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

def _owns_subagent_record(record: dict[str, Any], parent_agent: Any) -> bool:
    """True when *parent_agent*'s conversation owns this live-child record.

    Tier 1: identity — the ``_delegate_parent_ref`` weakref chain reaches
    *parent_agent* (fast path while the parent AIAgent survives the run). Tier 2:
    durable lineage — the record's ``owner_agent_session_id`` matches the caller's
    ``session_id`` after resolving compression-rotation lineage on both sides.
    Tier 2 exists because the identity chain is BRITTLE across parent rebuilds:
    the CLI sets ``self.agent = None`` mid-session (route change, credential
    refresh, /model, MoA one-shots) and builds a NEW AIAgent while the child keeps
    a weakref to the old one. Delivery routes by durable session id; control must
    use the same spine or running children go invisible/unsteerable. Tier 3: a
    sub-helper whose own parent helper is owned (its owner id names that helper).
    """
    parent_sid = str(getattr(parent_agent, "session_id", "") or "")
    current: Optional[dict[str, Any]] = record
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_descendant_of(current.get("agent"), parent_agent):
            return True
        owner_sid = str(current.get("owner_agent_session_id") or "")
        if owner_sid and parent_sid and (owner_sid == parent_sid or _resolve_session_lineage(owner_sid, parent_agent) in {
                parent_sid, _resolve_session_lineage(parent_sid, parent_agent)}):  # compression rotation: lineage tips
            return True
        # Tier 3: climb to the live helper that spawned this one (its durable owner id names that helper).
        above_agent = _record_parent_agent(current)
        with _active_subagents_lock:
            current = next((r for r in _active_subagents.values()
                           if above_agent is not None and r.get("agent") is above_agent), None)
    return False

def _list_payload(parent_agent: Any) -> dict[str, Any]:
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
            "depth": r.get("depth"),
            "goal": r.get("goal"),
            "model": r.get("model"),
            "status": r.get("status"),
            "running_seconds": round(time.time() - started, 1) if isinstance(started, (int, float)) else None,
            "accepting_steer": bool(r.get("accepting_steer", False)),
            "live_transcript": getattr(r.get("agent"), "_live_transcript_path", None),
            "steers": [{k: v for k, v in s.items() if not k.startswith("_")} for s in (r.get("steers") or [])[-5:]],
        })
    payload: dict[str, Any] = {"action": "list", "count": len(entries), "subagents": entries}
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
    with _active_subagents_lock:
        record = _active_subagents.get(sid)
    if record is None or not _owns_subagent_record(record, parent_agent):
        return tool_error(
            f"No live subagent '{sid}' in this conversation's spawn tree. It "
            "may have already finished (its result arrives as a normal "
            "completion message). Use action='list' to see live children."
        )
    if action == "steer" and not (message or "").strip():
        return tool_error("action='steer' requires a non-empty 'message' describing the course correction.")
    outcome = _CONTROL_OUTCOMES.get(action)
    if outcome is None:
        return tool_error(f"Unknown action '{action}'. Use spawn, list, steer, or stop.")
    status, note, failure = outcome
    relayed: list[str] = []
    ok = interrupt_subagent(sid) if action == "stop" else steer_subagent(sid, message.strip(), relayed_to=relayed)
    if ok:
        out: dict[str, Any] = {"action": action, "subagent_id": sid, "status": status, "note": note}
        if action == "steer":
            out["relayed_to"] = relayed
        return json.dumps(out, ensure_ascii=False)
    return tool_error(failure.format(sid=sid))

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
        "Steering text queued. The subagent sees it appended to its next tool result (a Codex app-server child: "
        "in its active turn) — the current tool call is never cut; a busy foreground command is moved to the background so it lands sooner. A child waiting on "
        "its own sub-helpers also relays it to them (relayed_to). action='list' shows each steer's delivered_at. If "
        "the child finishes before a delivery boundary remains, the text is reported back as missed_steer in its "
        "completion entry.", "Subagent '{sid}' is no longer accepting steering (finishing or "
        "already finished). Its result arrives as a normal completion "
        "message; re-delegate a follow-up task if more work is needed.",
    ),
}
