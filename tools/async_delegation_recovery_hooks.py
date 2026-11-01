"""``subagent_stop`` for children whose owning process died before their unit joined.

A background unit fires ``subagent_stop`` at its join; an owner killed before then never fires it, so a plugin
pairing ``subagent_start``/``subagent_stop`` kept the child "running" forever. Restart recovery
(``recover_abandoned_delegations``) replays the stop from the identity persisted at dispatch: a child the unit had
already recorded reports its real result, the rest ``child_status="unknown"``. Every recovered stop carries
``recovered=True``.

Delivery is tracked per child, not per unit: the live join and recovery both mark a child's ``task_children`` entry
``stop_fired`` right after its hook returns, and recovery only replays unmarked children. That makes delivery
at-least-once: a process that dies between a hook and its mark repeats that one stop later, so consumers dedupe on
``child_subagent_id``. Nothing is lost when a recoverer dies mid-delivery: the next recovery finishes the row.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
# Two recoveries in one process must not both deliver the rows this process owns.
_DELIVERY_LOCK = threading.Lock()

# The subagent_start identity keys a recovered stop repeats (``task_children`` also carries each child's model).
_IDENTITY_KEYS = ("child_session_id", "child_subagent_id", "child_role", "parent_tool_call_id")


def recovered_stop_payloads(
    task: dict[str, Any], parent_session_id: Optional[str], results: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """One ``subagent_stop`` kwargs dict per child the unit started whose stop was not delivered yet (rows
    dispatched without ``task_children`` started none that recovery can name)."""
    from tools.delegate_tool_results import subagent_stop_outcome_fields

    recorded = {r.get("task_index"): r for r in results or () if isinstance(r, dict)}
    payloads = []
    for key, identity in sorted((task.get("task_children") or {}).items(), key=lambda kv: int(kv[0])):
        if identity.get("stop_fired"):
            continue
        index = int(key)
        # The child's own model, never the unit's: None when dispatch did not record it.
        entry = {"model": identity.get("model"), "cost_status": "unknown",
                 **(recorded.get(index) or {"task_index": index, "status": "unknown"})}
        payloads.append({
            "parent_session_id": parent_session_id, "parent_turn_id": "", "task_index": index,
            **{k: identity.get(k) for k in _IDENTITY_KEYS}, **subagent_stop_outcome_fields(entry), "recovered": True,
        })
    return payloads


def mark_stop_fired(delegation_id: str, task_index: Any) -> None:
    """Durably record that one child's ``subagent_stop`` went out, so recovery never replays it. Best-effort: a
    failed write only costs a duplicate stop after a crash."""
    from agent.runtime_session_store import is_worker_process
    if is_worker_process():
        return  # worker-dispatched rows persist no task_children: recovery has nothing to replay
    from tools.async_delegation import _DB_LOCK, _transaction
    try:
        with _DB_LOCK, _transaction() as conn:
            row = conn.execute("SELECT task_json FROM async_delegations WHERE delegation_id=?",
                               (delegation_id,)).fetchone()
            task = json.loads(row[0] or "{}") if row else {}
            child = (task.get("task_children") or {}).get(str(task_index))
            if not isinstance(child, dict) or child.get("stop_fired"):
                return
            child["stop_fired"] = True
            conn.execute("UPDATE async_delegations SET task_json=? WHERE delegation_id=?",
                         (json.dumps(task), delegation_id))
    except Exception:
        logger.warning("Async delegation %s: could not record subagent_stop for task %s",
                       delegation_id, task_index, exc_info=True)


def claim_recovered_stops(task: dict[str, Any]) -> dict[str, Any]:
    """``task`` stamped with this process as the one delivering its children's recovered stops; written by the same
    conditional UPDATE that classifies the row, so concurrent recoverers never both deliver."""
    if not task.get("task_children"):
        return task
    from gateway.status import get_process_start_time
    return {**task, "stop_owner": [os.getpid(), get_process_start_time(os.getpid())]}


def deliver_recovered_stops(alive: Callable[[Any, Any], bool]) -> None:
    """Fire every recovered child's undelivered ``subagent_stop``, marking each as it goes. Rows this process
    classified, and rows whose recoverer died mid-delivery, are delivered here; a live recoverer keeps its own.
    Runs after the recovery transaction commits, under the ledger owner's profile scope (recovery callers bind it),
    so the profile's own plugins receive them."""
    from hermes_cli.plugins import invoke_hook
    from tools.async_delegation import _DB_LOCK, _transaction

    with _DELIVERY_LOCK:
        with _DB_LOCK, _transaction() as conn:
            rows = conn.execute("""SELECT delegation_id, parent_session_id, task_json, result_json
                   FROM async_delegations WHERE state='unknown' AND task_json LIKE '%"stop_owner"%'""").fetchall()
        _deliver_rows(rows, alive, invoke_hook)


def _deliver_rows(rows, alive: Callable[[Any, Any], bool], invoke_hook) -> None:
    """Hook, then mark, each unmarked child: at-least-once (a death between the two repeats that one stop)."""
    for delegation_id, parent_id, task_json, result_json in rows:
        task = json.loads(task_json or "{}")
        owner_pid, owner_started = task.get("stop_owner") or (None, None)
        if owner_pid != os.getpid() and alive(owner_pid, owner_started):
            continue
        results = (json.loads(result_json or "{}") or {}).get("results")
        for payload in recovered_stop_payloads(task, parent_id, results):
            try:
                invoke_hook("subagent_stop", **payload)
            except Exception:
                logger.debug("recovered subagent_stop hook invocation failed", exc_info=True)
            mark_stop_fired(delegation_id, payload["task_index"])
