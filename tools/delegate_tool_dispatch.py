"""Batch execution + background dispatch for delegate_task: ``delegate_task`` builds a ``_Batch``
(children + origin identity) and hands it to ``_run_batch``, which runs it synchronously or as ONE
detached async unit."""

from __future__ import annotations

import contextvars
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, wait as _cf_wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.delegate_tool_child_run import _detach_child, _fabricated_entry, _signal_child_stop
from tools.delegate_tool_progress import (
    SUBAGENT_FAILURE_STATUSES, _clean_error_text, _print_completion_line, _quiet, format_batch_tag,
)
from tools.delegate_tool_registry import _capture_gateway_steer_authority
from tools.delegate_tool_results import _finalize_child_results

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module


@dataclass
class _Batch:
    """One delegate_task call's built children plus everything needed to run them
    and assemble the combined result (shared by the sync path and the background runner)."""

    task_list: List[Dict[str, Any]]
    children: List[tuple]
    parent_agent: Any
    creds: Dict[str, Any]
    context: Optional[str]
    top_role: str
    max_children: int
    live_deleg_id: Optional[str]
    live_writers: list
    live_paths: list
    origin_wake_sid: str
    origin_ui_session_id: str
    origin_owner_transport: Any
    origin_owner_session_record: Any
    overall_start: float

    def owner_kwargs(self) -> Dict[str, Any]:
        """Steer/stop authority of the originating session, passed to every child run."""
        return {
            "owner_session_id": self.origin_ui_session_id or None, "owner_transport": self.origin_owner_transport,
            "owner_session_record": self.origin_owner_session_record,
        }

    def run_child(self, i: int, task: Dict[str, Any], child: Any) -> Dict[str, Any]:
        from tools.delegate_tool import _run_single_child
        return _run_single_child(task_index=i, goal=task["goal"], child=child, parent_agent=self.parent_agent, **self.owner_kwargs())


def _announce_batch(parent_agent, n_tasks: int, live_deleg_id: Optional[str]) -> None:
    """Announce the batch tag once so interleaved ``[tag n/N]`` lines are attributable."""
    if n_tasks > 1 and live_deleg_id:
        _hdr = f"  🔀 [{format_batch_tag(live_deleg_id)}] delegating {n_tasks} tasks"
        _print_completion_line(parent_agent, getattr(parent_agent, "_delegate_spinner", None), _hdr, console_line=_hdr)

def _capture_origin() -> tuple[str, str, Any, Any]:
    """``(wake_sid, ui_session_id, owner_transport, owner_session_record)`` of the
    ORIGINATING session, captured BEFORE building any child: AIAgent construction
    clobbers the HERMES_SESSION_ID ContextVar/os.environ with the subagent's id."""
    from tools.async_delegation import _current_origin_session_id
    _origin_wake_sid = _current_origin_session_id()
    _origin_ui_session_id = ""
    with _quiet(None):
        from gateway.session_context import get_session_env
        _origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
    return (_origin_wake_sid, _origin_ui_session_id, *_capture_gateway_steer_authority(_origin_ui_session_id))

def _report_child_done(parent_agent, spinner_ref, entry, tag, task_labels, n_tasks, remaining) -> None:
    """Print one completion line for a finished child and refresh the spinner text. Failed/errored/timed-out children
    say WHY on the same line — a bare ✗ reads as "silently dropped"."""
    idx = entry["task_index"]
    label = task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
    status = entry.get("status", "?")
    _slot = f"{tag} · {idx+1}/{n_tasks}" if tag else f"{idx+1}/{n_tasks}"
    completion_line = f"{'✓' if status == 'completed' else '✗'} [{_slot}] {label}  ({entry.get('duration_seconds', 0)}s)"
    _err_line = _clean_error_text(entry.get("error"), max_chars=120) if status in SUBAGENT_FAILURE_STATUSES else ""
    if _err_line:
        completion_line += f" — {_err_line}"
    _print_completion_line(parent_agent, spinner_ref, completion_line)
    if spinner_ref and remaining > 0:
        with _quiet("Spinner update_text failed: %s"):
            spinner_ref.update_text(f"🔀 {'[' + tag + '] ' if tag else ''}{remaining} task{'s' if remaining != 1 else ''} remaining")

def _run_children_parallel(batch: _Batch, results: list, *, honor_parent_interrupt: bool, detached: bool = False) -> None:
    """Run the batch's children in parallel, appending entries to ``results`` (sorted by task_index on return, one
    completion line printed per child). Polls futures with a short ``wait()`` timeout instead of ``as_completed()``
    so a wedged child cannot block the parent forever after an interrupt; on parent interrupt the still-pending
    children are reported ``interrupted`` and abandoned (they already got the interrupt signal)."""
    # Daemon workers (tools.daemon_pool): the `with` block still joins normally, but if the parent is interrupted
    # while a child is wedged, the abandoned worker must not block interpreter exit.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    parent_agent, n_tasks = batch.parent_agent, len(batch.task_list)
    task_labels = [t["goal"][:40] for t in batch.task_list]
    spinner_ref = getattr(parent_agent, "_delegate_spinner", None)
    _tag = format_batch_tag(batch.live_deleg_id)
    # Fabricated entries for still-pending / raised futures carry the correct _delegate_role.
    _child_by_index = {i: child for (i, _, child) in batch.children}

    def _entry_of(future, idx):
        if not future.done():
            return _fabricated_entry(
                idx, "interrupted", "Parent agent interrupted — child did not finish in time", _child_by_index.get(idx),
            )
        try:
            return future.result()
        except Exception as exc:
            return _fabricated_entry(idx, "error", str(exc), _child_by_index.get(idx))

    with DaemonThreadPoolExecutor(max_workers=batch.max_children) as executor:
        futures = {executor.submit(contextvars.copy_context().run, batch.run_child, i, t, child): i for i, t, child in batch.children}
        pending = set(futures)
        while pending:
            if honor_parent_interrupt and getattr(parent_agent, "_interrupt_requested", False) is True:
                results.extend(_entry_of(f, futures[f]) for f in pending)
                break
            done, pending = _cf_wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                entry = _entry_of(future, futures[future])
                results.append(entry)
                _report_child_done(parent_agent, spinner_ref, entry, _tag, task_labels, n_tasks, n_tasks - len(results))
                if detached and entry.get("status") in SUBAGENT_FAILURE_STATUSES and len(results) < n_tasks:
                    # The parent is not watching a spinner: tell it NOW, not when the last sibling finishes.
                    with _quiet("task failure notice failed", exc_info=True):
                        from tools.async_delegation import push_task_failure_notice
                        _i = entry.get("task_index", -1)
                        _live = batch.live_paths[_i] if isinstance(_i, int) and 0 <= _i < len(batch.live_paths) else None
                        push_task_failure_notice(
                            batch.live_deleg_id, {**entry, **({"live_transcript": _live} if _live else {})}, n_tasks=n_tasks)
    results.sort(key=lambda r: r["task_index"])  # match input order

def _execute_and_aggregate(batch: _Batch, *, honor_parent_interrupt: bool = True, detached: bool = False) -> dict:
    """Run all built children, join, finalize (hooks + cost rollup), return the combined dict. Shared by the sync path
    and the background runner: even in the background the batch JOINS on itself here so ONE consolidated results
    block re-enters the conversation. Live transcripts are finalized but retained as the full-fidelity record
    (retention pruning happens on future dispatches)."""
    from tools.delegation_live_log import update_manifest_statuses
    results: list = []
    if len(batch.task_list) == 1:
        results.append(batch.run_child(*batch.children[0]))
    else:
        _run_children_parallel(batch, results, honor_parent_interrupt=honor_parent_interrupt, detached=detached)

    _finalize_child_results(results, batch.task_list, batch.children, batch.parent_agent)
    total_duration = round(time.monotonic() - batch.overall_start, 2)
    for entry in results:
        _idx = entry.get("task_index", -1)
        if isinstance(_idx, int) and 0 <= _idx < len(batch.live_writers) and batch.live_writers[_idx] is not None:
            with _quiet("Live transcript finalize failed", exc_info=True):
                batch.live_writers[_idx].finalize(entry)
            if _idx < len(batch.live_paths):
                entry["live_transcript"] = batch.live_paths[_idx]
    update_manifest_statuses(batch.live_deleg_id, results)

    combined: Dict[str, Any] = {"results": results, "total_duration_seconds": total_duration}
    if batch.live_paths:
        combined["live_transcripts"] = list(batch.live_paths)
    return combined

_SYNC_FALLBACK_NOTES = {
    "no_async": (
        "background=true is not available in this session — it cannot "
        "receive a detached subagent result after the turn ends (a "
        "one-shot runner such as `hermes -z`, a cron job, a Kanban "
        "worker, or a stateless HTTP endpoint). The subagent(s) ran SYNCHRONOUSLY and the result is included above."
    ),
    "at_capacity": (
        "The background delegation pool was at capacity (delegation.max_concurrent_children), so the subagent(s) ran "
        "SYNCHRONOUSLY and the result is included above. Raise "
        "delegation.max_concurrent_children in config.yaml to allow more concurrent background delegations."
    ),
}

def _run_sync_with_note(batch: _Batch, reason: str) -> str:
    """Inline fallback: run the batch now and explain why it was not detached."""
    result = _execute_and_aggregate(batch)
    if isinstance(result, dict):
        result["note"] = _SYNC_FALLBACK_NOTES[reason]
    return json.dumps(result, ensure_ascii=False)

def _resolve_async_wake_sid(origin_wake_sid: str) -> Optional[str]:
    """Wake target for a detached batch, or None to force synchronous execution.

    Finite sessions (stateless HTTP requests, one-shot Kanban workers) cannot route a detached result back after their
    turn/process ends — but if a raw session id is bound (the API server always binds one), gateway.wake can still
    reach it by self-POSTing /v1/chat/completions, so only fall back to sync when there is truly no session id to
    wake. Uses the origin captured BEFORE child construction — HERMES_SESSION_ID here would be the subagent's internal
    id.
    """
    try:
        # Finite sessions cannot route a detached subagent result back to the agent after their turn/process
        # ends. This includes stateless HTTP requests (#10760) and one-shot Kanban workers (#63169). Fall
        # back to SYNCHRONOUS execution so the result returns in this same turn instead of handing out a
        # handle with no durable consumer. Mirrors the pool-at-capacity inline fallback below.
        from gateway.session_context import async_delivery_supported
        if async_delivery_supported():
            return ""
    except Exception:
        return ""
    if origin_wake_sid:
        logger.info(
            "delegate_task: async delivery unsupported on this session, but a session id is bound (%s) — dispatching "
            "in the background and waking the session via self-post when it completes instead of forcing synchronous "
            "execution.", origin_wake_sid,
        )
        return origin_wake_sid
    return None

def _resolve_async_session_key(parent_agent: Any, origin_ui_session_id: str) -> tuple[str, str]:
    """``(session_key, origin_ui_session_id)`` the async registry routes completions by.

    Desktop/TUI: the routable key is the durable AIAgent.session_id — compression can rotate it mid-turn before the
    TUI-side dict is re-anchored, and a stale approval-context key would orphan the completion. Gateway chats keep the
    platform conversation key (agent:main:...). The CLI has no bound approval contextvar and no HERMES_SESSION_KEY, so
    the key resolves empty; its drain is a positive-ownership filter on the durable session_id (empty would fail
    closed), so stamp the parent's durable id.
    """
    from tools.approval_context import get_current_session_key
    session_key = get_current_session_key(default="")
    agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
    with _quiet(None):
        from gateway.session_context import get_session_env
        source = get_session_env("HERMES_SESSION_SOURCE", "")
        # Refresh from the task-local source when available, else retain the
        # immutable value captured before child construction.
        origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or origin_ui_session_id
        if source == "tui" and agent_session_id:
            session_key = agent_session_id
    return session_key or agent_session_id, origin_ui_session_id

def _batch_progress_token(child_agents: List[Any]) -> tuple:
    """Progress token for the async registry's stale monitor: every child's (api_call_count, current_tool,
    last_activity_ts). last_activity_ts ticks on streamed chunks, tool transitions and API-call start/completion,
    so a child streaming a long response counts as alive; a fully frozen token past the threshold means the batch
    is wedged. ``in_tool`` is True while ANY child is inside a tool so slow tools get the higher ceiling (mirrors
    the sync heartbeat)."""
    # Progress token for the async registry's stale monitor: the combined (api_call_count, current_tool,
    # last_activity_ts) of every child. last_activity_ts is ticked by _touch_activity on every streamed
    # chunk ("receiving stream response"), every tool transition, and every API-call start/completion — so a
    # child streaming a long response is alive even though api_call_count only advances when the call
    # completes (same liveness signal as the compaction inactivity budget, PR #71508).
    parts = []
    in_tool = False
    for c in child_agents:
        try:
            summary = c.get_activity_summary()
            tool = summary.get("current_tool")
            parts.append((summary.get("api_call_count", 0), tool, summary.get("last_activity_ts")))
            in_tool = in_tool or bool(tool)
        except Exception:
            parts.append(None)
    return tuple(parts), in_tool

_BACKGROUND_NOTES = {
    "one": (
        "Subagent is running in the background. You and the user can keep working; its full result re-enters the "
        "conversation as a new message when it finishes. Do not wait or poll — just continue."
    ),
    "many": (
        "{n} subagents are running in parallel in the background. You and the user can keep working; they wait on "
        "each other and their consolidated results re-enter the conversation as a single message once ALL of them "
        "finish. Do not wait or poll — just continue."
    ),
    "control_hint": (
        "While a child runs you can orchestrate it live with this same tool: delegate_task(action='list') to see live "
        "children, action='steer' with subagent_id + message to redirect one, action='stop' with subagent_id to end "
        "one early."
    ),
    "live_transcripts_hint": (
        "Each subagent streams a human-readable transcript of its operations to the file listed above (append-only, "
        "one per task). Read or `tail -f` these paths at any time to watch a child work while it runs."
    ),
}

def _dispatched_payload(dispatch: dict, goals: List[str], child_agents: List[Any], live_paths: List[str]) -> dict:
    """Model-facing handle for an accepted background batch."""
    n = len(goals)
    payload = {
        "status": "dispatched", "mode": "background", "count": n, "delegation_id": dispatch["delegation_id"],
        "goals": goals, "note": _BACKGROUND_NOTES["one"] if n == 1 else _BACKGROUND_NOTES["many"].format(n=n),
    }
    sids = [getattr(c, "_subagent_id", None) for c in child_agents]
    if any(isinstance(s, str) and s for s in sids):
        payload["subagent_ids"] = sids
        payload["control_hint"] = _BACKGROUND_NOTES["control_hint"]
    if live_paths:
        payload["live_transcripts"] = list(live_paths)
        payload["live_transcripts_hint"] = _BACKGROUND_NOTES["live_transcripts_hint"]
    return payload

def _dispatch_background(batch: _Batch) -> str:
    """Dispatch the WHOLE batch as one async unit and return the tool result JSON. The runner joins on every child and
    yields ONE consolidated results block that re-enters the conversation as a single message when ALL children
    finish. Falls back to running synchronously (with an explanatory ``note``) when the session cannot receive
    detached completions or the async pool is at capacity."""
    from tools.delegate_tool import _get_max_async_children
    from tools.async_delegation import dispatch_async_delegation_batch
    wake_sid = _resolve_async_wake_sid(batch.origin_wake_sid)
    if wake_sid is None:
        logger.info("delegate_task: async delivery unsupported on this session runtime; running the batch synchronously instead.")
        return _run_sync_with_note(batch, "no_async")

    parent_agent = batch.parent_agent
    session_key, origin_ui_session_id = _resolve_async_session_key(parent_agent, batch.origin_ui_session_id)
    child_agents = [c for (_, _, c) in batch.children]
    # The batch's lifecycle is owned by the async registry now: drop the children from the parent's
    # interrupt-propagation list (_build_child_agent attached them, which is correct for sync runs).
    for c in child_agents:
        _detach_child(parent_agent, c)

    def _batch_interrupt():
        # Cancellation path for the detached batch (owned by the async registry).
        for c in child_agents:
            _signal_child_stop(c, "Async delegation cancelled")

    goals = [t["goal"] for t in batch.task_list]
    dispatch = dispatch_async_delegation_batch(
        goals=goals, context=batch.context,
        toolsets=None,  # metadata for the completion block only; subagents inherit the parent's toolsets
        role=batch.top_role, model=batch.creds["model"], session_key=session_key,
        origin_ui_session_id=origin_ui_session_id, origin_session_id=wake_sid,
        parent_session_id=getattr(parent_agent, "session_id", None),
        runner=lambda: _execute_and_aggregate(batch, honor_parent_interrupt=False, detached=True),
        interrupt_fn=_batch_interrupt, max_async_children=_get_max_async_children(),
        # Reuse the live-transcript directory's id (when created) so the returned delegation_id matches
        # cache/delegation/live/<id>/.
        delegation_id=batch.live_deleg_id,
        progress_fn=lambda: _batch_progress_token(child_agents),
    )
    if dispatch.get("status") == "dispatched":
        return json.dumps(_dispatched_payload(dispatch, goals, child_agents, batch.live_paths), ensure_ascii=False)
    # Pool at capacity / schedule failure: the async unit was never accepted, so just run inline (re-attaching to the
    # parent list is not needed).
    logger.info(
        "delegate_task: async pool at capacity (%s); running the whole batch synchronously instead.",
        dispatch.get("error", "rejected"),
    )
    return _run_sync_with_note(batch, "at_capacity")

def _run_batch(batch: _Batch, background: bool) -> str:
    """Tool result JSON: a dispatch handle (background) or the joined combined results."""
    if background:
        return _dispatch_background(batch)
    return json.dumps(_execute_and_aggregate(batch), ensure_ascii=False)
