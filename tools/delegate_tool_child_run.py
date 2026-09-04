"""Running ONE pre-built child agent: heartbeat, registry entry, workspace seeding,
timeout/failure handling, result-entry assembly and cleanup (``_ChildRun``)."""

from __future__ import annotations

import logging
import contextvars
import json
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional
from agent.interrupt_compat import request_hard_interrupt
from dataclasses import dataclass, field
from tools import file_state
from tools.delegate_tool_progress import _quiet, _safe_progress
from tools.delegate_tool_registry import (
    _capture_gateway_steer_authority, _close_subagent_steering, _register_subagent, _unregister_subagent,
)
from tools.delegate_tool_results import (
    _extract_output_tail, _looks_like_error_output, _stringify_tool_content, _summarize_tool_arguments,
)

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

def _num(value: Any, default: int = 0) -> int:
    """int() for counters that may be mocks/None on test doubles."""
    return int(value) if isinstance(value, (int, float)) else default

def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None

def _looks_like_child_error_summary(summary: str) -> bool:
    """Detect legacy child result shapes that put the provider/client error in
    ``final_response`` but omitted the newer ``failed``/``error`` fields.

    Keep this intentionally narrower than a generic substring search so genuine
    max-iteration summaries remain completed+truncated.
    """
    if _looks_like_error_output(summary):
        return True
    lines = (summary or "").strip().splitlines()
    first = lines[0].strip().lower() if lines else ""
    if first.startswith("http "):
        parts = first.split(maxsplit=2)
        status_token = parts[1].rstrip(":,") if len(parts) > 1 else ""
        if status_token.isdigit() and 400 <= int(status_token) <= 599:
            return True
    return any(
        marker in first
        for marker in (
            "authenticationerror",
            "authorizationerror",
            "badrequesterror",
            "permissiondeniederror",
            "rate limit",
            "token expired",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "invalid x-api-key",
        )
    )

def _fabricated_entry(idx: int, status: str, error: str, child: Any, duration: float = 0) -> Dict[str, Any]:
    """Result entry for a child that raised, never finished, or was abandoned."""
    return {
        "task_index": idx, "status": status, "summary": None, "error": error, "api_calls": 0,
        "duration_seconds": duration, "_child_role": getattr(child, "_delegate_role", None),
    }

def _append_missed_steer(entry: Dict[str, Any], late_steer: Optional[str]) -> None:
    """Record steer text that won the race with the child's failure/timeout."""
    if late_steer:
        entry["missed_steer"] = late_steer
        entry["error"] += (" [steer did not land before the subagent stopped: " f"{late_steer}]")

def _close_child(child: Any, log_message: str) -> None:
    """Best-effort ``child.close()`` (tool sandboxes, browser daemons, httpx clients)."""
    with _quiet(log_message, exc_info=True):
        close = getattr(child, "close", None)
        if callable(close):
            close()

def _with_children_lock(parent_agent: Any, op: str, child: Any) -> None:
    """``parent_agent._active_children.<op>(child)`` under the parent's lock when it has one."""
    lock = getattr(parent_agent, "_active_children_lock", None)
    if lock:
        with lock:
            getattr(parent_agent._active_children, op)(child)
    else:
        getattr(parent_agent._active_children, op)(child)

def _attach_child(parent_agent: Any, child: Any) -> None:
    """Register the child for parent interrupt propagation."""
    if hasattr(parent_agent, "_active_children"):
        _with_children_lock(parent_agent, "append", child)

def _detach_child(parent_agent: Any, child: Any) -> None:
    """Remove the child from parent interrupt propagation (no-op if absent)."""
    if not hasattr(parent_agent, "_active_children"):
        return
    try:
        _with_children_lock(parent_agent, "remove", child)
    except (ValueError, UnboundLocalError) as e:
        logger.debug("Could not remove child from active_children: %s", e)

def _signal_child_stop(child: Any, *reason: str) -> None:
    """Cooperative interrupt so the child's worker thread can exit cleanly."""
    with _quiet(None):
        if child is not None and not request_hard_interrupt(child, *reason) and hasattr(child, "_interrupt_requested"):
            child._interrupt_requested = True

# ── 0-API-call timeout diagnostic ────────────────────────────────────────────

def _format_thread_stack(frame: Any, indent: str) -> List[str]:
    import traceback as _traceback
    return [f"{indent}{sub}" for frame_line in _traceback.format_stack(frame) for sub in frame_line.rstrip().split("\n")]

_DIAG_CHILD_ATTRS = (
    "model", "provider", "api_mode", "base_url", "max_iterations", "quiet_mode", "skip_memory", "skip_context_files",
    "platform", "_delegate_role", "_delegate_depth",
)

def _diag_section(label: str, produce) -> List[str]:
    """Lines from ``produce()``, or one ``<label: ...exc>`` line so a broken attribute never aborts the dump."""
    try:
        return list(produce())
    except Exception as exc:
        return [f"  {label}{exc}>"]

def _diag_sizes(child: Any) -> List[str]:
    def _prompt():
        sys_prompt = getattr(child, "ephemeral_system_prompt", None) or getattr(child, "system_prompt", None) or ""
        is_str = isinstance(sys_prompt, str)
        return [
            f"  system_prompt_bytes: {len(sys_prompt.encode('utf-8')) if is_str else 'n/a'}",
            f"  system_prompt_chars: {len(sys_prompt) if is_str else 'n/a'}",
        ]

    def _tools():
        tools_schema = getattr(child, "tools", None)
        if tools_schema is None:
            return []
        return [
            f"  tool_schema_count: {len(tools_schema)}",
            f"  tool_schema_bytes: {len(json.dumps(tools_schema, default=str).encode('utf-8'))}",
        ]

    return ["## Prompt / schema sizes"] + _diag_section("system_prompt: <error: ", _prompt) + _diag_section("tool_schema: <error: ", _tools)

def _diag_threads(worker_thread: Optional[threading.Thread]) -> List[str]:
    """Worker stack plus all other live threads (bounded to 40): the worker is often parked on a helper thread, so a
    pre-HTTP wedge is indistinguishable from a slow provider without the full picture."""
    import sys as _sys
    lines = ["## Worker thread stack at timeout"]
    frames = _sys._current_frames()
    if worker_thread is not None and worker_thread.is_alive():
        worker_frame = frames.get(worker_thread.ident)
        lines.extend(_format_thread_stack(worker_frame, "  ") if worker_frame is not None else ["  <worker frame not available>"])
    else:
        lines.append("  <no worker thread handle>" if worker_thread is None else "  <worker thread already exited>")

    def _all_threads():
        frames = _sys._current_frames()
        by_ident = {th.ident: th for th in threading.enumerate() if th.ident}
        worker_ident = worker_thread.ident if worker_thread else None
        out: List[str] = []
        for dumped, (ident, frame) in enumerate(f for f in frames.items() if f[0] != worker_ident):  # worker dumped above
            if dumped >= 40:
                out.append(f"  <{len(frames) - dumped - 1} more threads omitted>")
                break
            th = by_ident.get(ident)
            out.append(f"  --- {th.name if th else f'ident={ident}'}{' daemon' if (th and th.daemon) else ''} ---")
            out.extend(_format_thread_stack(frame, "    "))
        return out

    return lines + ["", "## All thread stacks at timeout"] + _diag_section("<all-thread dump failed: ", _all_threads)

def _dump_subagent_timeout_diagnostic(
    *, child: Any, task_index: int, timeout_seconds: float, duration_seconds: float,
    worker_thread: Optional[threading.Thread], goal: str,
) -> Optional[str]:
    """Structured diagnostic for a subagent that timed out before any API call (otherwise "timed out with no
    response", 0 API calls, nothing to inspect): ``~/.hermes/logs/subagent-timeout-<sid>-<ts>.log`` with the
    child's config, prompt/schema sizes, activity snapshot and worker stack. Path, or None on failure."""
    try:
        from hermes_constants import get_hermes_home
        import datetime as _dt
        logs_dir = get_hermes_home() / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        subagent_id = getattr(child, "_subagent_id", None) or f"idx{task_index}"
        dump_path = logs_dir / f"subagent-timeout-{subagent_id}-{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _goal_preview = (goal or "").strip()
        if len(_goal_preview) > 1000:
            _goal_preview = _goal_preview[:1000] + " ...[truncated]"
        def _attr_line(attr):
            try:
                return f"  {attr}: {getattr(child, attr, None)!r}"
            except Exception:
                return f"  {attr}: <unreadable>"

        tool_names = getattr(child, "valid_tool_names", None)
        lines: List[str] = [
            "# Subagent timeout diagnostic — issue #14726", f"# Generated: {_dt.datetime.now().isoformat()}", "",
            "## Timeout", f"  task_index:        {task_index}", f"  subagent_id:       {subagent_id}",
            f"  configured_timeout: {timeout_seconds}s", f"  actual_duration:   {duration_seconds:.2f}s", "", "## Goal",
            _goal_preview or "(empty)", "", "## Child config", *map(_attr_line, _DIAG_CHILD_ATTRS),
            "", "## Toolsets", f"  enabled_toolsets:  {getattr(child, 'enabled_toolsets', None)!r}",
        ]
        if tool_names:
            lines.append(f"  loaded tool count: {len(tool_names)}")
            with _quiet(None):
                lines.append(f"  loaded tools:      {sorted(tool_names)}")
        lines += ["", *_diag_sizes(child), "", "## Activity summary"]
        lines += _diag_section(
            "<get_activity_summary failed: ", lambda: [f"  {k}: {v!r}" for k, v in child.get_activity_summary().items()],
        )
        lines += ["", *_diag_threads(worker_thread), "", "## Notes",
            "  This file is written ONLY when a subagent times out with 0 API calls.",
            "  0-API-call timeouts mean the child never reached its first LLM request.",
            "  Common causes: oversized prompt rejected by provider, transport hang,",
            "  credential resolution stuck. See issue #14726 for context.",
        ]
        dump_path.write_text("\n".join(lines), encoding="utf-8")
        return str(dump_path)
    except Exception as exc:
        logger.warning("Subagent timeout diagnostic dump failed: %s", exc)
        return None

# ── Per-run helpers ──────────────────────────────────────────────────────────

class _Heartbeat:
    """One child's parent-activity heartbeat on the shared periodic scheduler thread
    (``agent.periodic_scheduler``) — not one daemon thread per child. NOT started at construction:
    the caller calls ``start()`` inside its ``try`` so a failed schedule (OS thread exhaustion on
    first use) leaves ``handle`` None and ``stop()`` is a no-op."""

    def __init__(self, child: Any, parent_agent: Any, task_index: int):
        self.child, self.parent_agent, self.task_index = child, parent_agent, task_index
        # Stale detection: a cycle counts as stale when (tool, iteration,
        # activity_ts) all froze; thresholds differ idle vs in-tool.
        self.last_seen = {"iter": 0, "tool": None, "ts": None, "stale": 0}
        self.handle = None

    def start(self) -> None:
        from agent.periodic_scheduler import schedule
        from tools.delegate_tool import _HEARTBEAT_INTERVAL
        self.handle = schedule(self.tick, _HEARTBEAT_INTERVAL)

    def stop(self) -> None:
        """wait=5 mirrors the old thread join: an in-flight tick finishes."""
        if self.handle is not None:
            self.handle.cancel(wait=5)

    def tick(self):
        """Returning False stops the periodic callback."""
        from tools.delegate_tool import _HEARTBEAT_STALE_CYCLES_IDLE, _HEARTBEAT_STALE_CYCLES_IN_TOOL
        child, parent_agent, task_index, last_seen = self.child, self.parent_agent, self.task_index, self.last_seen
        touch = getattr(parent_agent, "_touch_activity", None) if parent_agent is not None else None
        if not touch:
            return None
        desc = f"delegate_task: subagent {task_index} working"
        try:
            child_summary = child.get_activity_summary()
            child_tool = child_summary.get("current_tool")
            child_iter = child_summary.get("api_call_count", 0)
            child_max = child_summary.get("max_iterations", 0)
            child_activity_ts = child_summary.get("last_activity_ts")
            # A slow model wait refreshes last_activity_ts (direct_api_call
            # heartbeat), so it never looks stale at the idle threshold.
            activity_advanced = child_activity_ts is not None and (
                last_seen["ts"] is None or child_activity_ts > last_seen["ts"]
            )
            if child_iter > last_seen["iter"] or child_tool != last_seen["tool"] or activity_advanced:
                last_seen.update(iter=child_iter, tool=child_tool, stale=0)
                if child_activity_ts is not None:
                    last_seen["ts"] = child_activity_ts
            else:
                last_seen["stale"] += 1
            if last_seen["stale"] >= (_HEARTBEAT_STALE_CYCLES_IN_TOOL if child_tool else _HEARTBEAT_STALE_CYCLES_IDLE):
                logger.warning(
                    "Subagent %d appears stale (no progress for %d heartbeat cycles, tool=%s) — stopping heartbeat",
                    task_index, last_seen["stale"], child_tool or "<none>",
                )
                return False  # stop touching parent, let gateway timeout fire
            if child_tool:
                desc = f"delegate_task: subagent running {child_tool} (iteration {child_iter}/{child_max})"
            elif child_summary.get("last_activity_desc", ""):
                desc = f"delegate_task: subagent {child_summary.get('last_activity_desc', '')} (iteration {child_iter}/{child_max})"
        except Exception:
            pass
        with _quiet(None):
            touch(desc)
        return None


def _start_heartbeat(child: Any, parent_agent: Any, task_index: int) -> _Heartbeat:
    """Build (not start) one child's heartbeat; see ``_Heartbeat``."""
    return _Heartbeat(child, parent_agent, task_index)

def _register_child(
    child: Any, parent_agent: Any, goal: str, *, owner_session_id: Optional[str], owner_transport: Any,
    owner_session_record: Any,
) -> Optional[str]:
    """Register the live child in the module registry; return its subagent_id. Test doubles without a stable string
    ``_subagent_id`` are not registered (None) and the caller skips every registry interaction for them."""
    _subagent_id = getattr(child, "_subagent_id", None)
    if not isinstance(_subagent_id, str) or not _subagent_id:
        return None
    if owner_session_id is None:
        with _quiet(None):
            from gateway.session_context import get_session_env
            owner_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or None
    if owner_session_id and (owner_transport is None or owner_session_record is None):
        owner_transport, owner_session_record = _capture_gateway_steer_authority(owner_session_id)
    _raw_depth = getattr(child, "_delegate_depth", 1)
    _register_subagent({
        "subagent_id": _subagent_id,
        "parent_id": _str_or_none(getattr(child, "_parent_subagent_id", None)),
        "depth": max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0,
        "goal": goal,
        "delegation_id": _str_or_none(getattr(child, "_delegation_id", None)),
        "model": _str_or_none(getattr(child, "model", None)),
        "started_at": time.time(), "status": "running", "tool_count": 0, "agent": child,
        # Owning conversation's durable session id (same lineage completion delivery routes by), sourced from the
        # child's stamp so it survives a parent_agent rebuild between dispatch and run; used for list/steer/stop
        # ownership when the weakref chain breaks.
        "owner_agent_session_id": (
            str(getattr(child, "_parent_session_id", "") or "") or str(getattr(parent_agent, "session_id", "") or "") or None
        ),
        # Immutable live gateway/TUI session that commissioned this child.
        # Empty outside those hosts; RPC authority fails closed.
        "owner_session_id": owner_session_id,
        "owner_transport": owner_transport,
        "owner_session_record": owner_session_record,
    })
    return _subagent_id

def _create_isolated_worktree(parent_agent: Any, parent_task_id: Any, subagent_id: Optional[str]):
    """Opt-in worktree isolation: own git worktree off the parent's HEAD (the
    child's terminal starts there). Git-only, local-backend-only; failures
    degrade silently to the shared workspace. Returns the worktree info or None."""
    from tools.delegate_tool import _get_worktree_isolation, _resolve_workspace_hint
    if not _get_worktree_isolation():
        return None
    with _quiet("worktree isolation setup failed: %s"):
        from tools import subagent_worktree
        if not subagent_worktree.local_backend_active():
            logger.debug("worktree isolation skipped: non-local terminal backend")
            return None
        _parent_cwd = None
        with _quiet(None):
            from tools.terminal_tool import get_session_cwd as _gsc
            _parent_cwd = _gsc(parent_task_id)
        return subagent_worktree.create_subagent_worktree(
            _parent_cwd or _resolve_workspace_hint(parent_agent), subagent_id=subagent_id,
        )
    return None

def _defer_close_after_timeout(child: Any, child_future: Any) -> None:
    """Hand ``child.close()`` to a Future done-callback and drain its transports.

    The interrupt is cooperative: the worker still runs its finally path, so closing now could close SQLite under its
    final write — the done-callback is the first safe boundary. The abandoned worker is usually parked in an OpenSSL
    read; NEVER hard-close that transport from this thread (cross-thread FD release under a live SSL read corrupts
    native state) — shutdown() the pooled sockets so the read settles with EOF and the worker unwinds. One immediate
    sweep + one delayed re-sweep for a connection opened in between; a worker that still won't settle keeps its
    resources until process exit.
    """
    child_future.add_done_callback(lambda _done: _close_child(child, "Failed to close timed-out child after worker exit"))
    # Bounded drain (#94248 native half): the deferred close above only fires once the abandoned worker
    # unwinds, but that worker is typically parked inside an in-flight OpenSSL read (Codex / httpx). Never
    # hard-close that transport from this thread — releasing FDs under a live SSL read is the #29507/#70773
    # native-corruption family. Instead shutdown() the child's pooled sockets, which is FD-safe from any
    # thread and settles the blocked read with EOF/EPIPE so the worker can unwind and trigger the deferred
    # close.
    _drain = getattr(child, "_drain_transports_after_abandonment", None)
    if not callable(_drain):
        return

    def _drain_once(phase: str) -> None:
        with _quiet("Timed-out child transport drain (%s) failed", phase, exc_info=True):
            _drain(reason=f"delegate_timeout_{phase}")

    _drain_once("immediate")
    _resweep_timer = threading.Timer(5.0, lambda: None if child_future.done() else _drain_once("resweep"))
    _resweep_timer.daemon = True
    _resweep_timer.start()

def _lease_child_credential(child: Any) -> tuple[Any, Optional[str]]:
    """Lease a credential from the child's pool (if any) and bind it; ``(pool, lease_id)``."""
    child_pool = getattr(child, "_credential_pool", None)
    if child_pool is None:
        return None, None
    leased_cred_id = child_pool.acquire_lease()
    if leased_cred_id is not None:
        with _quiet("Failed to bind child to leased credential: %s"):
            leased_entry = child_pool.current()
            if leased_entry is not None and hasattr(child, "_swap_credential"):
                child._swap_credential(leased_entry)
    return child_pool, leased_cred_id

def _merge_late_steer(result: Dict[str, Any], subagent_id: Optional[str], child: Any) -> None:
    """Linearization boundary for registry steering: from here the child cannot consume another steer. Closing under
    the registry lock either rejects a concurrent caller or drains every accepted exact text into the result before
    callbacks/result assembly run."""
    late = _close_subagent_steering(subagent_id, child) if subagent_id else None
    if late:
        existing = result.get("pending_steer")
        result["pending_steer"] = f"{existing}\n{late}" if isinstance(existing, str) and existing else late


@dataclass
class _SchemaOutcome:
    schema: Optional[Dict[str, Any]]
    valid: Optional[bool]
    errors: List[str]
    retries: int

def _validate_child_output_schema(
    child: Any, result: Dict[str, Any], task_index: int, child_task_id: str, relay_child_text: Any
) -> _SchemaOutcome:
    """Validate the final answer against the attached output_schema with ONE bounded retry. Schema-less children (no
    dict on ``child._delegate_output_schema``) take no branch here so their result entry stays byte-identical."""
    _output_schema = getattr(child, "_delegate_output_schema", None)
    if not isinstance(_output_schema, dict):
        return _SchemaOutcome(_output_schema, None, [], 0)
    from tools.delegation_output_schema import build_retry_message, validate_output
    _first_text = result.get("final_response") or ""
    _schema_valid, _schema_errors = validate_output(_first_text, _output_schema)
    if _schema_valid or not _first_text.strip() or result.get("interrupted", False):
        return _SchemaOutcome(_output_schema, _schema_valid, _schema_errors, 0)

    # Exactly one retry turn, carrying the validation errors verbatim (no
    # schema re-paste — the child already holds the contract in its context).
    _retry_result = None
    try:
        _retry_result = child.run_conversation(
            user_message=build_retry_message(_schema_errors), task_id=child_task_id, stream_callback=relay_child_text,
        )
    except Exception as _retry_exc:
        logger.warning("Subagent %d schema-retry turn failed: %s", task_index, _retry_exc)
    if isinstance(_retry_result, dict):
        _retry_text = _retry_result.get("final_response") or ""
        if _retry_text.strip():
            result["final_response"] = _retry_text
        try:
            result["api_calls"] = int(result.get("api_calls", 0) or 0) + int(_retry_result.get("api_calls", 0) or 0)
        except (TypeError, ValueError):
            pass
        _retry_messages = _retry_result.get("messages")
        if isinstance(_retry_messages, list) and isinstance(result.get("messages"), list):
            result["messages"] = result["messages"] + _retry_messages
        _schema_valid, _schema_errors = validate_output(_retry_text, _output_schema)
    return _SchemaOutcome(_output_schema, _schema_valid, _schema_errors, 1)

def _build_tool_trace(messages: Any) -> list[Dict[str, Any]]:
    """Tool trace from the child's conversation messages, pairing parallel
    tool calls with their results by tool_call_id."""
    tool_trace: list[Dict[str, Any]] = []
    trace_by_id: Dict[str, Dict[str, Any]] = {}
    if not isinstance(messages, list):
        return tool_trace
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                arguments = fn.get("arguments", "")
                entry_t = {
                    "tool": fn.get("name", "unknown"), "args_bytes": len(arguments),
                    "input_summary": _summarize_tool_arguments(arguments),
                }
                tool_trace.append(entry_t)
                if tc.get("id"):
                    trace_by_id[tc["id"]] = entry_t
        elif msg.get("role") == "tool":
            content = _stringify_tool_content(msg.get("content", ""))
            result_meta = {"result_bytes": len(content), "status": "error" if _looks_like_error_output(content) else "ok"}
            tc_id = msg.get("tool_call_id")
            target = trace_by_id.get(tc_id) if tc_id else None
            if target is not None:
                target.update(result_meta)
            elif tool_trace:
                tool_trace[-1].update(result_meta)  # no tool_call_id: pair with the latest call
    return tool_trace

def _build_result_entry(
    child: Any, result: Dict[str, Any], task_index: int, duration: float, schema: _SchemaOutcome,
) -> Dict[str, Any]:
    """Parent-visible result entry (status, exit_reason, tool trace, tokens, cost).
    ``status``/``exit_reason``/``truncated`` follow the ``_run_single_child`` contract; a structured failure always
    wins over the summary-presence heuristic (a fallback for legacy/mock results only)."""
    summary = result.get("final_response") or ""
    # "(empty)" is run_agent's give-up sentinel after repeated empty LLM
    # responses (usually a transport bug) — a failure, not a success.
    usable_summary = bool(summary) and summary.strip() != "(empty)"
    if result.get("interrupted", False):
        status, exit_reason = "interrupted", "interrupted"
    elif result.get("failed") or result.get("error") or (
        not result.get("completed", False) and _looks_like_child_error_summary(summary)
    ):
        # The loop returns the error text as final_response, which would otherwise read as "completed". Never report a
        # provider rejection as "max_iterations" — that is only truthful for real budget exhaustion.
        status, exit_reason = "failed", "error"
    else:
        # exit_reason ("completed" vs "max_iterations") tells the parent HOW the task ended; completed=False with no
        # failure = budget exhaustion. A declared schema still violated after the bounded retry makes the summary
        # unusable under the contract, so status must not say completed (orchestrators reading only status/icon would
        # accept an empty verdict).
        exit_reason = "completed" if result.get("completed", False) else "max_iterations"
        status = "completed" if schema.valid is not False and usable_summary else "failed"

    _cost = getattr(child, "session_estimated_cost_usd", 0.0)
    _cost_status = getattr(child, "session_cost_status", None)
    # Result entry contract: see the _run_single_child docstring.
    entry: Dict[str, Any] = {
        "task_index": task_index,
        "status": status,
        "summary": summary,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": duration,
        "model": _str_or_none(getattr(child, "model", None)),
        "exit_reason": exit_reason,
        # A budget-exhausted child still returns a summary (status stays
        # "completed"), so the parent needs this explicit flag.
        "truncated": exit_reason == "max_iterations",
        "tokens": {
            "input": _num(getattr(child, "session_prompt_tokens", 0)),
            "output": _num(getattr(child, "session_completion_tokens", 0)),
        },
        "tool_trace": _build_tool_trace(result.get("messages") or []),
        # Captured before the finally block calls child.close() so the parent thread can fire subagent_stop with the
        # correct role; stripped before the dict is serialised back to the model (as is _child_cost_usd, folded into
        # the parent's session cost by the aggregator).
        "_child_role": getattr(child, "_delegate_role", None),
        "_child_cost_usd": float(_cost or 0.0) if isinstance(_cost, (int, float)) else 0.0,
    }
    # Model-visible per-delegation spend (unlike _child_cost_usd above).
    entry["cost_usd"] = round(entry["_child_cost_usd"], 6)
    entry["cost_status"] = _cost_status if isinstance(_cost_status, str) and _cost_status else "unknown"
    if status == "failed":
        if schema.valid is False and usable_summary:
            # The child DID respond; name the contract violation instead of the generic "no response" error.
            entry["error"] = (
                "Final answer does not satisfy the declared output_schema" + (" (after 1 retry)." if schema.retries else ".")
            )
        else:
            entry["error"] = result.get("error") or summary or "Subagent did not produce a response."
        # Classified reason from the child loop (e.g. "rate_limit", "billing")
        # lets the parent tell a quota wall from a task error without parsing prose.
        _failure_reason = result.get("failure_reason")
        if isinstance(_failure_reason, str) and _failure_reason:
            entry["failure_reason"] = _failure_reason

    # Schema-validation outcome — emitted ONLY when a schema was requested, so
    # legacy (schema-less) payloads keep their exact shape.
    if isinstance(schema.schema, dict):
        entry["schema_valid"] = bool(schema.valid)
        if schema.retries:
            entry["schema_retries"] = schema.retries
        if not schema.valid and schema.errors:
            entry["schema_errors"] = schema.errors

    # A steer queued after the final assistant turn had no tool batch to land
    # in; name it so the parent sees it was MISSED rather than silently absorbed.
    _missed_steer = result.get("pending_steer")
    if isinstance(_missed_steer, str) and _missed_steer.strip():
        entry["missed_steer"] = _missed_steer
        _miss_note = ("[steer did not land — the subagent finished before it could " f"be delivered: {_missed_steer}]")
        entry["summary"] = f"{summary}\n\n{_miss_note}" if summary else _miss_note
    return entry


@dataclass
class _ChildRun:
    """State of one child run, shared by every phase of ``_run_single_child``.
    ``worktree_info`` stays None until isolation engages (``attach_worktree`` is
    then a no-op on every early error path); ``seed_workspace`` sets the rest."""

    child: Any
    parent_agent: Any
    task_index: int
    goal: str
    subagent_id: Optional[str]
    child_progress_cb: Any
    child_start: float = field(default_factory=time.monotonic)
    worktree_info: Optional[Dict[str, str]] = None
    child_task_id: str = ""
    parent_task_id: Optional[str] = None
    wall_start: float = 0.0
    parent_reads_snapshot: list = field(default_factory=list)

    def elapsed(self) -> float:
        return round(time.monotonic() - self.child_start, 2)

    def relay_text(self, delta: str) -> None:
        """Stream callback forwarding the child's reply text up the progress relay so gateway watch windows mirror it
        live (subagent.text → message.delta). Inert under CLI/TUI: their progress handlers ignore non-tool events."""
        if delta:
            _safe_progress(self.child_progress_cb, "subagent.text", preview=delta)

    def attach_worktree(self, entry_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect + prune the child worktree, reporting into the entry (no-op without isolation)."""
        info = self.worktree_info
        if info is None:
            return entry_dict
        from tools import subagent_worktree
        try:
            entry_dict["worktree"] = subagent_worktree.finalize_subagent_worktree(info)
        except Exception as e:
            # State is unknown: emit the SAME flagged schema the parent expects,
            # via the shared factory so the two producers never drift.
            logger.warning("worktree finalize failed: %s", e)
            entry_dict["worktree"] = subagent_worktree.unproven_worktree_payload(info, f"finalize raised: {e}")
        return entry_dict

    def seed_workspace(self) -> None:
        """Seed cwd/container aliases and optional worktree isolation for the child;
        ``goal`` is extended with the worktree contract note when isolation engaged."""
        import uuid as _uuid
        self.child_task_id = self.subagent_id or f"subagent-{self.task_index}-{_uuid.uuid4().hex[:8]}"
        self.parent_task_id = getattr(self.parent_agent, "_current_task_id", None)
        # Seed the child's cwd record from the parent's: same starting directory,
        # but the child's later `cd`s stay in its own record. Per-session container
        # isolation keys containers by task_id; the child must share the PARENT's.
        with _quiet("Child cwd seed failed: %s"):
            from tools.terminal_tool import get_session_cwd, record_session_cwd, register_container_alias
            record_session_cwd(self.child_task_id, get_session_cwd(self.parent_task_id))
            register_container_alias(self.child_task_id, self.parent_task_id)

        self.worktree_info = _create_isolated_worktree(self.parent_agent, self.parent_task_id, self.subagent_id)
        if self.worktree_info is not None:
            with _quiet("worktree cwd seed failed: %s"):
                from tools.terminal_tool import record_session_cwd as _rsc
                _rsc(self.child_task_id, self.worktree_info["path"])
            # The child's context is already built; carry the isolation contract on
            # the goal message instead (same turn, no system-prompt mutation).
            from tools.subagent_worktree import build_worktree_context_note
            self.goal = self.goal + build_worktree_context_note(self.worktree_info)
        self.wall_start = time.time()
        self.parent_reads_snapshot = list(file_state.known_reads(self.parent_task_id)) if self.parent_task_id else []

    def finish_failed(
        self, entry: Dict[str, Any], late_steer: Optional[str], *, preview: str, summary: str = "", status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shared tail of every failure path: emit ``subagent.complete`` (``status`` defaults to the entry's), note
        the steer text that won the race with the failure, report the worktree."""
        _safe_progress(
            self.child_progress_cb, "subagent.complete", preview=preview, status=status or entry["status"],
            duration_seconds=entry["duration_seconds"], summary=summary,
        )
        _append_missed_steer(entry, late_steer)
        return self.attach_worktree(entry)

    def close_steering(self) -> Optional[str]:
        """Close steer acceptance (see ``_merge_late_steer``); returns late steer text, if any."""
        return _close_subagent_steering(self.subagent_id, self.child) if self.subagent_id else None

    def await_child(self) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
        """Run the child's conversation on a daemon worker: ``(result, None, False)`` or ``(None, error_entry,
        close_deferred)`` on timeout/exception.

        Hard timeout is off by default (``result(timeout=None)``; stuck children are the heartbeat's job). Daemon
        worker: an abandoned timed-out child on a non-daemon thread would block interpreter exit at atexit join. The
        worker installs a non-interactive approval callback (deny/approve per delegation.subagent_auto_approve) so
        dangerous-command prompts never fall back to ``input()`` and deadlock the parent TUI. On failure: steer
        acceptance closes BEFORE the stop signal (a concurrent steer is drained into the entry or rejected, never
        lost); a 0-API-call timeout gets a diagnostic dump; a worker that still owns the child gets ``child.close()``
        via a Future done-callback (``close_deferred=True``) — closing here would race its still-unwinding finally
        path.
        """
        from tools.delegate_tool import (_get_child_timeout, _get_subagent_approval_callback, _set_subagent_approval_cb)
        from tools.daemon_pool import DaemonThreadPoolExecutor
        child, task_index = self.child, self.task_index
        child_timeout = _get_child_timeout()
        executor = DaemonThreadPoolExecutor(
            max_workers=1, initializer=_set_subagent_approval_cb, initargs=(_get_subagent_approval_callback(),),
        )
        # Worker thread handle so the timeout diagnostic can dump its stack.
        worker_thread_holder: Dict[str, Optional[threading.Thread]] = {"t": None}

        def _run_with_thread_capture():
            worker_thread_holder["t"] = threading.current_thread()
            from agent.delegation_context import delegated_child_context
            with delegated_child_context(str(getattr(child, "session_id", "") or "")):
                return child.run_conversation(
                    user_message=self.goal, task_id=self.child_task_id, stream_callback=self.relay_text,
                )

        future = executor.submit(contextvars.copy_context().run, _run_with_thread_capture)
        try:
            return future.result(timeout=child_timeout), None, False
        except Exception as wait_exc:
            exc: BaseException = wait_exc  # ``as`` targets are unbound after the except block
        finally:
            # Shut down without waiting — a child stuck on blocking I/O would hang wait=True forever.
            executor.shutdown(wait=False)

        _late_pending_steer = self.close_steering()
        _signal_child_stop(child)
        is_timeout = isinstance(exc, (FuturesTimeoutError, TimeoutError))
        duration = self.elapsed()
        logger.warning("Subagent %d %s after %.1fs", task_index, "timed out" if is_timeout else f"raised {type(exc).__name__}", duration)
        child_api_calls = 0
        with _quiet(None):
            child_api_calls = int(child.get_activity_summary().get("api_call_count", 0) or 0)
        # A timeout BEFORE any API call is a black box without a diagnostic dump.
        before_first_call = is_timeout and child_api_calls == 0
        diagnostic_path: Optional[str] = None
        if before_first_call:
            diagnostic_path = _dump_subagent_timeout_diagnostic(
                child=child, task_index=task_index,
                # is_timeout implies a cap was configured (result(timeout=None)
                # never raises FuturesTimeoutError); guard for the type checker.
                timeout_seconds=float(child_timeout or 0.0), duration_seconds=float(duration),
                worker_thread=worker_thread_holder.get("t"), goal=self.goal,
            )
            if diagnostic_path:
                logger.warning("Subagent %d 0-API-call timeout — diagnostic written to %s", task_index, diagnostic_path)
        if not is_timeout:
            _err = str(exc)
        elif before_first_call:
            _err = (
                f"Subagent timed out after {child_timeout}s without making any API call — the child never reached its "
                f"first LLM request (prompt construction, credential resolution, or transport may be stuck)."
            )
        else:
            _err = (
                f"Subagent timed out after {child_timeout}s with {child_api_calls} API call(s) completed — likely "
                f"stuck on a slow API call, tool call, or unresponsive network request."
            )
        if diagnostic_path:
            _err += f" Diagnostic: {diagnostic_path}"
        status = "timeout" if is_timeout else "error"
        _error_entry = {
            "task_index": task_index, "status": status, "summary": None, "error": _err, "exit_reason": status,
            "api_calls": child_api_calls, "duration_seconds": duration,
            "timeout_seconds": child_timeout if is_timeout else None,
            "timed_out_after_seconds": duration if is_timeout else None,
            "timeout_phase": "before_first_llm_call" if before_first_call else "after_llm_calls" if is_timeout else None,
            "_child_role": getattr(child, "_delegate_role", None),
            "diagnostic_path": diagnostic_path,
        }
        self.finish_failed(_error_entry, _late_pending_steer, preview=f"Timed out after {duration}s" if is_timeout else str(exc))
        close_deferred = is_timeout and not future.done()
        if close_deferred:
            _defer_close_after_timeout(child, future)
        return None, _error_entry, close_deferred

    def append_sibling_write_reminder(self, entry: Dict[str, Any]) -> None:
        """Warn the parent when this child wrote files the parent had already read. Checks writes by ANY non-parent
        task_id (not just this child's) so nested orchestrator→worker chains are covered too."""
        if not (self.parent_task_id and self.parent_reads_snapshot):
            return
        with _quiet("file_state sibling-write check failed", exc_info=True):
            sibling_writes = file_state.writes_since(self.parent_task_id, self.wall_start, self.parent_reads_snapshot)
            mod_paths = sorted({p for paths in sibling_writes.values() for p in paths}) if sibling_writes else []
            if not mod_paths:
                return
            reminder = (
                "\n\n[NOTE: subagent modified files the parent previously read — re-read before editing: "
                + ", ".join(mod_paths[:8])
                + (f" (+{len(mod_paths) - 8} more)" if len(mod_paths) > 8 else "")
                + "]"
            )
            if entry.get("summary"):
                entry["summary"] = entry["summary"] + reminder
            else:
                entry["stale_paths"] = mod_paths

    def emit_complete(self, result: Dict[str, Any], entry: Dict[str, Any], duration: float) -> None:
        """Fire ``subagent.complete`` with the per-branch observability payload (tokens, cost, files touched,
        tool-output tail); every field is optional and degrades gracefully on the client."""
        if not self.child_progress_cb:
            return
        child = self.child
        summary = entry["summary"]
        _files_read: list = []
        with _quiet(None):
            _files_read = list(file_state.known_reads(self.child_task_id))[:40]
        _files_written_map: dict = {}
        with _quiet(None):
            _files_written_map = file_state.writes_since("", self.wall_start, [])  # all writes since wall_start
        complete_kwargs: Dict[str, Any] = {
            "preview": summary[:160] if summary else entry.get("error", ""),
            "status": entry["status"],
            "duration_seconds": duration,
            "summary": summary[:500] if summary else entry.get("error", ""),
            "input_tokens": _num(getattr(child, "session_prompt_tokens", 0)),
            "output_tokens": _num(getattr(child, "session_completion_tokens", 0)),
            "reasoning_tokens": _num(getattr(child, "session_reasoning_tokens", 0)),
            "api_calls": _num(entry["api_calls"]),
            "files_read": _files_read,
            "files_written": sorted({p for tid, paths in _files_written_map.items() if tid == self.child_task_id for p in paths})[:40],
            "output_tail": _extract_output_tail(result, max_entries=8, max_chars=600),
        }
        _cost_usd = getattr(child, "session_estimated_cost_usd", None)
        if _cost_usd is not None:
            with _quiet(None):
                complete_kwargs["cost_usd"] = float(_cost_usd)
        _safe_progress(self.child_progress_cb, "subagent.complete", **complete_kwargs)

    def cleanup(self, *, heartbeat: _Heartbeat, child_pool: Any, leased_cred_id: Any, close_deferred: bool) -> None:
        """Finally-path teardown (idempotent, never raises). Order matters: stop heartbeat → drop registry entry →
        release credential lease → restore the parent's process-global tool names → detach from the parent's
        interrupt list → close the child (unless a timed-out worker still owns it) → pop the child's Relay scope if
        no turn is active."""
        child = self.child
        heartbeat.stop()

        # Safe even if the child was never registered (ID missing on test doubles).
        if self.subagent_id:
            _unregister_subagent(self.subagent_id, agent=child)

        if child_pool is not None and leased_cred_id is not None:
            with _quiet("Failed to release credential lease: %s"):
                child_pool.release_lease(leased_cred_id)

        # Restore the parent's tool names so the process-global is correct for
        # any subsequent execute_code calls or other consumers.
        import model_tools
        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        _detach_child(self.parent_agent, child)

        # Close tool resources (terminal sandboxes, browser daemons, background
        # processes, httpx clients) so subagent subprocesses don't outlive the delegation.
        if not close_deferred:
            _close_child(child, "Failed to close child agent after delegation")

        # The AIAgent turn boundary normally closes the child scope itself. This fallback covers failures before that
        # boundary starts, but must not pop a scope while a timed-out child worker is still unwinding.
        with _quiet("Failed to close child Relay session after delegation"):
            from agent import relay_runtime
            runtime = relay_runtime.get_runtime(create=False)
            child_session_id = str(getattr(child, "session_id", "") or "")
            child_turn_is_active = relay_runtime.SESSION_COORDINATOR.has_active_turn(
                profile_key=relay_runtime.current_profile_key(), session_id=child_session_id,
            )
            if runtime is not None and child_session_id and not child_turn_is_active:
                runtime.unregister_subagent({"child_session_id": child_session_id})
