"""Running ONE pre-built child agent: heartbeat, registry entry, workspace seeding,
timeout/failure handling, result-entry assembly and cleanup (``_ChildRun``)."""

from __future__ import annotations

import logging
import contextvars
import json
import os
import threading
import time
from tools.delegation_redaction import redact_child_result, redact_child_text
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional
from agent import reasoning_effort as _reasoning_effort
from agent.compression_marker import elide
from agent.interrupt_compat import request_hard_interrupt
from agent.reasoning_effort import EFFORT_LADDER, clamp_effort, route_supported_efforts
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

def _tier_in_effect(child: Any) -> Any:
    """The child's speed tier: its own (a route's tier), else the one its request settings carry (an inherited
    ``service_tier: priority`` or ``speed: fast``, top level or in ``extra_body``)."""
    if getattr(child, "service_tier", None):
        return child.service_tier
    overrides = getattr(child, "request_overrides", None)
    overrides = overrides if isinstance(overrides, dict) else {}
    extra = overrides.get("extra_body") if isinstance(overrides.get("extra_body"), dict) else {}
    for scope in (overrides, extra):
        if scope.get("service_tier"):
            return scope["service_tier"]
        if scope.get("speed") == "fast":
            return "priority"
    return None


def _route_report(child: Any) -> Optional[dict[str, Any]]:
    """What a backend-routed child actually ran on: its backend and label plus the provider, model, reasoning effort
    (None = the provider's default) and speed tier read off the child when its result is assembled, so a fallback
    provider that took over shows (``fallback_activated``). None for the default backend's unlabelled route."""
    route = getattr(child, "_delegate_route", None)
    if not isinstance(route, dict):
        return None
    from agent.fast_mode import service_tier_word
    reasoning = getattr(child, "reasoning_config", None)
    effort = None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort") if reasoning.get("enabled", True) else "none"
    return {**route, "provider": _str_or_none(getattr(child, "provider", None)),
            "model": _str_or_none(getattr(child, "model", None)), "reasoning_effort": _str_or_none(effort),
            "service_tier": service_tier_word(_tier_in_effect(child)),
            **({"fallback_activated": True} if getattr(child, "_fallback_activated", False) is True else {})}

def _fabricated_entry(idx: int, status: str, error: str, child: Any, duration: float = 0) -> dict[str, Any]:
    """Result entry for a child that raised, never finished, or was abandoned."""
    entry = {
        "task_index": idx, "status": status, "summary": None, "error": error, "api_calls": 0,
        "duration_seconds": duration, "_child_role": getattr(child, "_delegate_role", None),
    }
    route = _route_report(child)
    if route is not None:
        entry["route"] = route
    return entry

def _append_missed_steer(entry: dict[str, Any], late_steer: Optional[str]) -> None:
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
    """Register the child for parent interrupt propagation.

    ``interrupt()`` fans out to a SNAPSHOT of ``_active_children``; a child attached after the stop
    landed (a fan-out still building its siblings, a turn that has not reached its iteration check yet)
    would otherwise start with no signal and run to completion as an orphan. Mirror a pending stop here
    so the whole spawn tree dies with its parent."""
    if hasattr(parent_agent, "_active_children") and child not in parent_agent._active_children:
        _with_children_lock(parent_agent, "append", child)
    if getattr(parent_agent, "_interrupt_requested", False) is not True:
        return
    # Same soft/hard split as ``interrupt()``'s own fan-out: a hard stop cancels, a soft one redirects.
    message = getattr(parent_agent, "_interrupt_message", None)
    hard = getattr(parent_agent, "_hard_interrupt_requested", None)
    if hard is None or hard.is_set():
        _signal_child_stop(child, message or "parent agent interrupted",
                           tool_reason=getattr(parent_agent, "_tool_interrupt_reason", None) or "parent agent interrupted")
    else:
        with _quiet("Failed to propagate interrupt to late child: %s"):
            child.interrupt(message)

def _detach_child(parent_agent: Any, child: Any) -> None:
    """Remove the child from parent interrupt propagation (no-op if absent)."""
    if not hasattr(parent_agent, "_active_children"):
        return
    try:
        _with_children_lock(parent_agent, "remove", child)
    except (ValueError, UnboundLocalError) as e:
        logger.debug("Could not remove child from active_children: %s", e)

def _signal_child_stop(child: Any, *reason: str, tool_reason: str = "parent delegation ended") -> None:
    """Cooperative interrupt so the child's worker thread can exit cleanly. ``tool_reason`` is the
    fixed cause the child's tools see (a pending approval wait reports it instead of a user deny)."""
    with _quiet(None):
        if (child is not None and not request_hard_interrupt(child, *reason, tool_reason=tool_reason)
                and hasattr(child, "_interrupt_requested")):
            child._interrupt_requested = True

# ── 0-API-call timeout diagnostic ────────────────────────────────────────────

def _format_thread_stack(frame: Any, indent: str) -> list[str]:
    import traceback as _traceback
    return [f"{indent}{sub}" for frame_line in _traceback.format_stack(frame) for sub in frame_line.rstrip().split("\n")]

_DIAG_CHILD_ATTRS = (
    "model", "provider", "api_mode", "base_url", "max_iterations", "quiet_mode", "skip_memory", "skip_context_files",
    "platform", "_delegate_role", "_delegate_depth",
)

def _diag_section(label: str, produce) -> list[str]:
    """Lines from ``produce()``, or one ``<label: ...exc>`` line so a broken attribute never aborts the dump."""
    try:
        return list(produce())
    except Exception as exc:
        return [f"  {label}{exc}>"]

def _diag_sizes(child: Any) -> list[str]:
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

def _diag_threads(worker_thread: Optional[threading.Thread]) -> list[str]:
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
        out: list[str] = []
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
        _goal_preview = elide((goal or "").strip(), 1000)
        def _attr_line(attr):
            try:
                return f"  {attr}: {getattr(child, attr, None)!r}"
            except Exception:
                return f"  {attr}: <unreadable>"

        tool_names = getattr(child, "valid_tool_names", None)
        lines: list[str] = [
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

# Granularity for re-checking a child's progress while a configured ``child_timeout_seconds`` budget runs.
# Five seconds is finer than the 30s heartbeat and cheap (one activity-summary read per slice).
_LIVENESS_POLL_SECONDS = 5.0
# Fraction of the inactivity budget after which the child is warned once (via its steer channel) that the window is
# closing, so a child that is merely slow can wrap up and return instead of losing its whole context (#116001).
_BUDGET_WARNING_FRACTION = 0.8

def _effective_child_timeout(child: Any) -> Optional[float]:
    """The route's pinned idle cap (``child._delegate_child_timeout``) through the same parse as the global knob;
    otherwise (unpinned or not a valid route timeout) the global ``delegation.child_timeout_seconds``."""
    from agent.delegation_provider import valid_route_child_timeout
    pinned = getattr(child, "_delegate_child_timeout", None)
    if valid_route_child_timeout(pinned):
        from tools.delegate_tool_config import _parse_timeout
        return _parse_timeout(pinned)
    from tools.delegate_tool import _get_child_timeout
    return _get_child_timeout()


_BUDGET_WARNING_PREFIX = "[delegation budget warning]"


def _budget_warning_text(idle_seconds: float, child_timeout: float) -> str:
    return (
        f"{_BUDGET_WARNING_PREFIX} No progress signal for {idle_seconds:.0f}s of your {child_timeout:.0f}s "
        "inactivity window. Finish the current step and return your summary now — the work is discarded if the "
        "window elapses."
    )

def _warn_child_budget(child: Any, idle_seconds: float, child_timeout: float) -> None:
    """Queue the one-line warning through the child's steer path (delivered at its next iteration boundary)."""
    steer = getattr(child, "steer", None)
    if not callable(steer):
        return
    try:
        steer(_budget_warning_text(idle_seconds, child_timeout))
    except Exception as exc:
        logger.debug("budget warning steer failed: %s", exc)

def _parent_steer_text(late: Optional[str]) -> Optional[str]:
    """Drained steer text minus the runtime's own budget warnings: they ride the steer channel, but an undelivered
    one is not a parent steer to report as missed or hand to a retry."""
    if not late:
        return None
    kept = "\n".join(line for line in late.split("\n") if not line.startswith(_BUDGET_WARNING_PREFIX))
    return kept if kept.strip() else None


def _child_activity_fingerprint(child: Any) -> tuple:
    """``(completed calls, current tool, activity clock)`` — the progress signals the heartbeat's stale verdict
    already watches. A child waiting on an in-flight LLM completion is ALIVE: its activity clock ticks while the
    provider works (``direct_api_call``'s 15s heartbeat) and the call itself is bounded by the per-call stale
    watchdog, so at the delegation layer it must not read as stuck (#116001)."""
    try:
        summary = child.get_activity_summary() or {}
    except Exception:
        return (None, None, None)
    return (summary.get("api_call_count"), summary.get("current_tool"), summary.get("last_activity_ts"))

def _child_last_event_age(child: Any) -> Optional[float]:
    """Seconds since the child's activity clock last ticked, or None when the child exposes no clock. Reported on a
    timeout so an operator can tell a slow-but-live provider from a runaway without transcript forensics."""
    try:
        ts = (child.get_activity_summary() or {}).get("last_activity_ts")
        return round(max(0.0, time.time() - float(ts)), 2) if ts is not None else None
    except Exception:
        return None

class _Heartbeat:
    """One child's parent-activity heartbeat via the shared periodic scheduler
    (``agent.periodic_scheduler``) — not one daemon thread per child. NOT started at construction:
    the caller calls ``start()`` inside its ``try`` so a failed schedule (OS thread exhaustion on
    first use) leaves ``handle`` None and ``stop()`` is a no-op."""

    def __init__(self, child: Any, parent_agent: Any, task_index: int):
        self.child, self.parent_agent, self.task_index = child, parent_agent, task_index
        # Stale detection: a cycle counts as stale when (tool, iteration,
        # activity_ts) all froze; thresholds differ idle vs in-tool.
        self.last_seen = {"iter": 0, "tool": None, "ts": None, "stale": 0}
        self.handle = None
        # Set on the stale verdict; ``await_child`` waits on it (its worker's done-callback
        # sets it too) so a wedged child ends the wait instead of only ending the heartbeat.
        self.settled = threading.Event()
        # Threshold (seconds of frozen activity) the stale verdict fired at; None until it does.
        # ``await_child`` reads it to name the real cause when a configured cap was still pending.
        self.stale_threshold_seconds: Optional[float] = None

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
        from tools.delegate_tool import _HEARTBEAT_INTERVAL, _HEARTBEAT_STALE_CYCLES_IDLE, _HEARTBEAT_STALE_CYCLES_IN_TOOL
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
            stale_cycles = _HEARTBEAT_STALE_CYCLES_IN_TOOL if child_tool else _HEARTBEAT_STALE_CYCLES_IDLE
            if last_seen["stale"] >= stale_cycles:
                logger.warning(
                    "Subagent %d appears stale (no progress for %d heartbeat cycles, tool=%s) — abandoning its wait",
                    task_index, last_seen["stale"], child_tool or "<none>",
                )
                # A finite/-Q turn has no gateway watchdog behind this; the wait itself must end (#109749).
                self.stale_threshold_seconds = stale_cycles * _HEARTBEAT_INTERVAL
                self.settled.set()
                return False
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

#: How long a timed-out attempt's worker gets to exit after the stop signal before a reasoning-escalation
#: retry may take over its workspace.
_ESCALATION_TIMEOUT_SETTLE_SECONDS = 2.0


def _await_timed_out_worker_exit(child: Any, future: Any, grace: Optional[float] = None) -> bool:
    """Wait (bounded) for a timed-out child's worker to exit after the stop signal; True once it has. Drains the
    child's pooled transports first so a worker parked in an SSL read wakes with EOF instead of riding out ``grace``."""
    if future.done():
        return True
    settled_ev = threading.Event()
    future.add_done_callback(lambda _f: settled_ev.set())
    _drain = getattr(child, "_drain_transports_after_abandonment", None)
    if callable(_drain):
        with _quiet("Timed-out child transport drain (escalate) failed", exc_info=True):
            _drain(reason="delegate_timeout_escalate")
    settled_ev.wait(timeout=_ESCALATION_TIMEOUT_SETTLE_SECONDS if grace is None else grace)
    return bool(future.done())


def _lease_child_credential(child: Any) -> tuple[Any, Optional[str]]:
    """Lease a credential from the child's pool (if any) and bind it; ``(pool, lease_id)``. The bound entry must
    serve the child's endpoint: on a mixed same-provider pool the least-leased pick may target another host, so it is
    released and an endpoint-matching entry is leased by id instead (#68237). Child work, so it runs in the child's
    profile scope: a lease may refresh that profile's OAuth entries, and ``_swap_credential`` re-reads TLS/header
    config for the bound endpoint."""
    child_pool = getattr(child, "_credential_pool", None)
    if child_pool is None:
        return None, None
    from agent.credential_pool import credential_pool_entry_serves_endpoint as _entry_serves_endpoint
    from tools.delegate_tool_routes import child_profile_scope
    base_url = getattr(child, "base_url", None)
    with child_profile_scope(child):
        leased_cred_id = child_pool.acquire_lease()
        if leased_cred_id is not None:
            with _quiet("Failed to bind child to leased credential: %s"):
                # Resolve the leased entry by id: the pool is shared with the parent/siblings, so current() is a
                # mutable cursor that may already point at someone else's pick.
                leased_entry = next((e for e in child_pool.entries() if e.id == leased_cred_id), None)
                if not _entry_serves_endpoint(leased_entry, base_url):
                    child_pool.release_lease(leased_cred_id)
                    leased_entry = next(
                        (e for e in child_pool.entries() if e.last_status != "dead" and _entry_serves_endpoint(e, base_url)),
                        None,
                    )
                    leased_cred_id = child_pool.acquire_lease(leased_entry.id) if leased_entry is not None else None
                if leased_entry is not None and hasattr(child, "_swap_credential"):
                    child._swap_credential(leased_entry)
    return child_pool, leased_cred_id

def _merge_late_steer(result: dict[str, Any], subagent_id: Optional[str], child: Any) -> None:
    """Linearization boundary for registry steering: from here the child cannot consume another steer. Closing under
    the registry lock either rejects a concurrent caller or drains every accepted exact text into the result before
    callbacks/result assembly run. Both channels drop the runtime's budget warnings: the turn finalizer already put
    whatever was queued at the turn's end into ``pending_steer``, and this drain adds what arrived after it."""
    late = _close_subagent_steering(subagent_id, child) if subagent_id else None
    existing = result.get("pending_steer")
    steer = _parent_steer_text("\n".join(t for t in (existing, late) if isinstance(t, str) and t))
    if steer:
        result["pending_steer"] = steer
    else:
        result.pop("pending_steer", None)


@dataclass
class _SchemaOutcome:
    schema: Optional[dict[str, Any]]
    valid: Optional[bool]
    errors: list[str]
    retries: int

def _validate_child_output_schema(
    child: Any, result: dict[str, Any], task_index: int, child_task_id: str, relay_child_text: Any
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
        # Same identity as the main child turn: this runs on the parent worker's thread, and an
        # unmarked turn is misread as the dispatcher-owned worker by every HERMES_KANBAN_* gate.
        from agent.delegation_context import delegated_child_context
        from tools.delegate_tool_routes import child_profile_scope
        with child_profile_scope(child), delegated_child_context(str(getattr(child, "session_id", "") or "")):
            _retry_result = child.run_conversation(
                user_message=build_retry_message(_schema_errors), task_id=child_task_id,
                stream_callback=relay_child_text,
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

def _build_tool_trace(messages: Any) -> list[dict[str, Any]]:
    """Tool trace from the child's conversation messages, pairing parallel
    tool calls with their results by tool_call_id."""
    tool_trace: list[dict[str, Any]] = []
    trace_by_id: dict[str, dict[str, Any]] = {}
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

def _result_exit_reason(result: dict[str, Any]) -> str:
    """How a finished conversation ended (``_build_result_entry``'s ``exit_reason``), readable before the entry is
    built so the escalation decision can skip work an exhausted attempt would waste (the schema retry turn)."""
    if result.get("interrupted", False):
        return "interrupted"
    if result.get("failed") or result.get("error"):
        return "error"
    return "completed" if result.get("completed", False) else "max_iterations"


# ── Reasoning escalation (route ``escalate_reasoning`` + ``escalate_ceiling``) ─────────────────────────────────

# Budget stops only: a child that ran out of iterations or went quiet inside its inactivity window may finish at a
# stronger level. Errors, interrupts and successes never retry.
_ESCALATING_EXIT_REASONS = frozenset({"max_iterations", "timeout"})
_ORIGINAL_CLAMP_EFFORT = clamp_effort
_ORIGINAL_ROUTE_SUPPORTED_EFFORTS = route_supported_efforts
_HANDOFF_SUMMARY_CHARS = 8000


def _effort_ran(child: Any) -> str:
    """The reasoning level the child actually ran at (not the route field, which may be blank and inherited):
    ``none`` when thinking was disabled, ``medium`` (Hermes' default) when no level was set."""
    for attr in ("reasoning_config", "_wire_reasoning_config"):
        config = getattr(child, attr, None)
        if isinstance(config, dict):
            if config.get("enabled", True) is False:
                return "none"
            return str(config.get("effort") or "").strip().lower() or "medium"
    return "medium"


def _effort_clamp_fns() -> tuple[Any, Any]:
    """``(clamp_effort, clamp_target)``: this module's names when patched here, else the owner's. ``clamp_target``
    maps ``(provider, model, api_mode)`` to ``(supported, overrides)`` — the provider's final wire vocabulary and
    declared remap (TokenHub tops out at ``high``, Kimi K3 sends ``medium`` as ``high``), not just the entry clamp."""
    clamp = clamp_effort if clamp_effort is not _ORIGINAL_CLAMP_EFFORT else _reasoning_effort.clamp_effort
    if route_supported_efforts is not _ORIGINAL_ROUTE_SUPPORTED_EFFORTS:
        patched = route_supported_efforts
        return clamp, lambda provider, model, api_mode: (patched(provider, model, api_mode), None)
    return clamp, _reasoning_effort.route_effort_clamp_target


def _escalation_ceiling(child: Any) -> Optional[str]:
    """The pinned ceiling when escalation is on for this child; enabled without a ceiling is off."""
    from agent.delegation_provider import valid_route_escalate_ceiling
    if getattr(child, "_delegate_escalate_reasoning", None) is not True:
        return None
    ceiling = getattr(child, "_delegate_escalate_ceiling", None)
    return ceiling.strip().lower() if isinstance(ceiling, str) and valid_route_escalate_ceiling(ceiling) else None


def _entry_api_calls(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("api_calls", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _parent_stop_reaches(child: Any, parent_agent: Any) -> bool:
    """A parent turn's stop ends the chain of a child attached for its interrupt fan-out (a foreground task); a
    detached background unit outlives the turn that spawned it on purpose, so only an explicit stop of the unit or
    subagent (``_delegate_unit_cancelled``) ends its chain."""
    return (getattr(parent_agent, "_interrupt_requested", False) is True
            and child in getattr(parent_agent, "_active_children", ()))


def _next_escalation_effort(child: Any, entry: dict[str, Any], parent_agent: Any) -> Optional[str]:
    """The level the retry runs at, or None when this attempt ends the chain. Triggers on ``exit_reason`` only (a
    budget stop with no summary is ``status: failed`` and still retries); a 0-call timeout never reached the model.
    Steps one rung up from what the child ran, never past the ceiling, and only while the provider's clamp sends a
    strictly stronger level than the last attempt's."""
    ceiling = _escalation_ceiling(child)
    if (ceiling is None or _parent_stop_reaches(child, parent_agent)
            or getattr(child, "_delegate_unit_cancelled", False) is True):
        return None
    exit_reason = entry.get("exit_reason")
    if exit_reason not in _ESCALATING_EXIT_REASONS:
        return None
    if exit_reason == "timeout" and _entry_api_calls(entry) == 0:
        return None
    ladder = EFFORT_LADDER
    current = _effort_ran(child)
    if current not in ladder or ceiling not in ladder or ladder.index(current) >= ladder.index(ceiling):
        return None
    next_effort = ladder[ladder.index(current) + 1]
    clamp, clamp_target = _effort_clamp_fns()
    supported, overrides = clamp_target(
        _str_or_none(getattr(child, "provider", None)), _str_or_none(getattr(child, "model", None)),
        _str_or_none(getattr(child, "api_mode", None)),
    )
    if overrides:
        last_clamped, next_clamped = clamp(current, supported, overrides), clamp(next_effort, supported, overrides)
    else:
        last_clamped, next_clamped = clamp(current, supported), clamp(next_effort, supported)
    if (last_clamped not in ladder or next_clamped not in ladder
            or ladder.index(next_clamped) <= ladder.index(last_clamped)):
        return None
    return next_effort


def _escalation_handoff(
    original_goal: str, *, effort: str, next_effort: str, entry: dict[str, Any], transcript_path: Optional[str],
    working_dir: Optional[str], processes: dict[str, list[dict[str, Any]]],
) -> str:
    """The retry's goal: the original task, what the stopped attempt left (its summary, or a pointer to the
    transcript when there is none), the transcript path, the working directory to continue in and the runtime's
    facts about its background processes (a watcher it left running dies with it; a result it never read)."""
    exit_reason = entry.get("exit_reason")
    stopped = ("hit its iteration limit" if exit_reason == "max_iterations"
               else "timed out (no progress inside its inactivity window)")
    lines = [
        original_goal, "",
        f"[REASONING ESCALATION] A previous attempt at this task ran at reasoning effort '{effort}' and {stopped} "
        f"before finishing. You are a fresh attempt at reasoning effort '{next_effort}' with the same limits. "
        "Continue from where it stopped instead of starting over.",
    ]
    summary, missed = entry.get("summary"), entry.get("missed_steer")
    if isinstance(summary, str) and isinstance(missed, str) and missed.strip():
        summary = summary.removesuffix(_missed_steer_note(missed))  # the steer is handed off on its own below
    if isinstance(summary, str) and summary.strip() and summary.strip() != "(empty)":
        lines += ["", "Previous attempt's summary:", elide(summary.strip(), _HANDOFF_SUMMARY_CHARS)]
    else:
        lines += ["", "The previous attempt left no summary"
                  + ("; read its transcript (below) to see what it already did." if transcript_path else ".")]
    lines.append(f"Transcript of the earlier attempt(s): {transcript_path}" if transcript_path
                 else "No transcript of the earlier attempt is available.")
    if working_dir:
        lines.append(f"Working directory: {working_dir} — keep working there; the previous attempt's changes are "
                     "already in it.")
    if isinstance(missed, str) and missed.strip():
        lines.append(f"A steer from the parent arrived after the previous attempt stopped and was never delivered: "
                     f"{missed.strip()}")
    if processes:
        from tools.process_registry_notifications import _process_accounting_lines
        lines += ["", "The previous attempt's background processes (runtime facts, not its own claims):",
                  *_process_accounting_lines(processes)]
    return "\n".join(lines)


def _cost_of(child: Any) -> tuple[float, Optional[str]]:
    cost = getattr(child, "session_estimated_cost_usd", 0.0)
    status = getattr(child, "session_cost_status", None)
    value = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0
    return value, status if isinstance(status, str) and status else None


@dataclass
class _EscalationChain:
    """One task's attempts inside a single ``delegate_task`` unit. The first attempt seeds the workspace (and the
    only worktree); later attempts reuse it. ``pending`` holds the next attempt's ``(child, goal)`` once an
    attempt decides to retry; that attempt then skips its completion event and worktree finalize."""

    original_goal: str
    first_child: Any = None
    worktree_info: Optional[dict[str, str]] = None
    seeded: bool = False
    working_dir: Optional[str] = None
    pending: Optional[tuple[Any, str]] = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    costs: list[tuple[float, Optional[str]]] = field(default_factory=list)
    reused_child: Any = None
    # The sibling-write reminder spans the chain: the first attempt's start and parent-read snapshot, plus the
    # parent-read paths each escalated attempt modified (recorded before its close releases its writer claims).
    wall_start: float = 0.0
    parent_reads_snapshot: list = field(default_factory=list)
    modified_parent_paths: set[str] = field(default_factory=set)
    # ``orphaned_processes`` / ``unread_completions`` of escalated attempts, snapshot before their teardown kills the
    # processes (and with them the only record that a watcher was still running or a result was never read).
    process_report: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, child: Any, exit_reason: Any) -> None:
        self.attempts.append({"effort": _effort_ran(child), "exit_reason": exit_reason})
        self.costs.append(_cost_of(child))

    def finish(self, entry: dict[str, Any]) -> dict[str, Any]:
        """The parent's single result: unchanged when only one attempt ran; else the last attempt's entry plus
        ``attempts`` and the cost summed across every attempt."""
        if len(self.attempts) <= 1:
            return entry
        entry["attempts"] = [dict(a) for a in self.attempts]
        total = sum(cost for cost, _ in self.costs)
        if self.reused_child is not None:
            self.reused_child.session_estimated_cost_usd = total
        entry["_child_cost_usd"] = total
        entry["cost_usd"] = round(total, 6)
        if entry.get("cost_status") in (None, "", "unknown"):
            known = next((s for _, s in reversed(self.costs) if s and s != "unknown"), None)
            entry["cost_status"] = known or "unknown"
        return entry


def _build_retry_child(child: Any, next_effort: str) -> Any:
    """The next attempt's child. A child built by ``_build_children`` carries ``_delegate_build_retry``, which
    rebuilds it on the exact same route with only the reasoning level changed (the handoff goal is the attempt's
    first user message, not build input). A child without one (a direct ``_run_single_child`` caller) is rerun
    itself at the next level."""
    builder = getattr(child, "_delegate_build_retry", None) if "_delegate_build_retry" in dir(child) else None
    if builder is not None:
        return builder(next_effort) if callable(builder) else None
    from hermes_constants import parse_reasoning_effort
    child.reasoning_config = parse_reasoning_effort(next_effort)
    child.session_estimated_cost_usd = 0.0  # the chain already recorded this attempt
    return child


def _missed_steer_note(steer: str) -> str:
    return f"[steer did not land — the subagent finished before it could be delivered: {steer}]"


def _build_result_entry(
    child: Any, result: dict[str, Any], task_index: int, duration: float, schema: _SchemaOutcome,
) -> dict[str, Any]:
    """Parent-visible result entry (status, exit_reason, tool trace, tokens, cost).
    ``status``/``exit_reason``/``truncated`` follow the ``_run_single_child`` contract; a structured failure always
    wins over the summary-presence heuristic (a fallback for legacy/mock results only)."""
    summary = result.get("final_response") or ""
    # "(empty)" is run_agent's give-up sentinel after repeated empty LLM
    # responses (usually a transport bug) — a failure, not a success.
    usable_summary = bool(summary) and summary.strip() != "(empty)"
    interrupt_note = ""
    if result.get("interrupted", False):
        status, exit_reason = "interrupted", "interrupted"
        # The loop's final_response is a placeholder here ("Operation interrupted…", also appended as the closing
        # assistant row); the completion must carry what the child actually had so far — its last real assistant
        # text — and keep the placeholder as the error.
        from agent.message_content import flatten_message_text
        placeholders = {"", summary.strip(), "Operation interrupted."}
        partial = next((t for m in reversed(result.get("messages") or []) if m.get("role") == "assistant"
                        and (t := flatten_message_text(m.get("content")).strip()) not in placeholders), "")
        if partial:
            interrupt_note, summary = summary.strip(), partial
    elif result.get("failed") or result.get("error"):
        # The loop returns the error text as final_response, which would otherwise read as "completed". Never report a
        # provider rejection as "max_iterations" — that is only truthful for real budget exhaustion.
        status, exit_reason = "failed", "error"
    else:
        # exit_reason ("completed" vs "max_iterations") tells the parent HOW the task ended; completed=False with no
        # failure = budget exhaustion. A declared schema still violated after the bounded retry does NOT fail the
        # run: the child's raw final text is the deliverable (audits of up to 68 min were written off as "failed"
        # over a stray code fence or one missing field); ``schema_valid: false`` + ``schema_errors`` carry the
        # contract verdict, and the summary is prefixed with a notice so a status-only reader cannot mistake it
        # for validated output.
        exit_reason = "completed" if result.get("completed", False) else "max_iterations"
        status = "completed" if usable_summary else "failed"

    _cost = getattr(child, "session_estimated_cost_usd", 0.0)
    _cost_status = getattr(child, "session_cost_status", None)
    # Result entry contract: see the _run_single_child docstring.
    entry: dict[str, Any] = {
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
    _route = _route_report(child)
    if _route is not None:
        entry["route"] = _route
    if status == "failed":
        entry["error"] = result.get("error", "Subagent did not produce a response.")
        # Classified reason from the child loop (e.g. "rate_limit", "billing")
        # lets the parent tell a quota wall from a task error without parsing prose.
        _failure_reason = result.get("failure_reason")
        if isinstance(_failure_reason, str) and _failure_reason:
            entry["failure_reason"] = _failure_reason
    elif interrupt_note:
        entry["error"] = interrupt_note

    # Schema-validation outcome — emitted ONLY when a schema was requested, so
    # legacy (schema-less) payloads keep their exact shape.
    if isinstance(schema.schema, dict):
        entry["schema_valid"] = bool(schema.valid)
        if schema.retries:
            entry["schema_retries"] = schema.retries
        if not schema.valid and schema.errors:
            entry["schema_errors"] = schema.errors
        if schema.valid is False and usable_summary:
            entry["schema_note"] = (
                "Final answer does not satisfy the declared output_schema"
                + (" (after 1 retry)" if schema.retries else "")
                + "; `summary` is the child's raw, UNVALIDATED final text — extract what you need from it "
                "yourself (see schema_errors) rather than re-running the task."
            )

    # A steer queued after the final assistant turn had no tool batch to land
    # in; name it so the parent sees it was MISSED rather than silently absorbed.
    _missed_steer = result.get("pending_steer")
    if isinstance(_missed_steer, str) and _missed_steer.strip():
        entry["missed_steer"] = _missed_steer
        _miss_note = _missed_steer_note(_missed_steer)
        entry["summary"] = f"{summary}\n\n{_miss_note}" if summary else _miss_note
    return entry


def _is_image_url(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "data:image/"))


def _build_child_goal_message(goal: str, images: list[str], child) -> Any:
    """The child's first user message when a task forwards ``images``.

    Routing reuses the inbound-image policy (``agent.image_routing``, honouring ``agent.image_input_mode``): a
    vision-capable child gets an OpenAI-style content list (text part + one ``image_url`` part per image; local files
    as data URLs behind the read guard, http(s)/data URLs verbatim); otherwise the goal gains ``[Image attached …]``
    hint lines for ``vision_analyze``. Any failure degrades to the text-only goal so image plumbing never breaks a
    spawn — logged at warning since the caller asked for the images.
    """
    try:
        # data: URLs ride as image parts only — their base64 never goes into the text hint or a text-mode goal.
        data_urls = [s for s in images if s.startswith("data:image/")]
        urls = [s for s in images if _is_image_url(s) and s not in data_urls]
        paths = [s for s in images if not _is_image_url(s)]
        from agent.image_routing import build_native_content_parts, decide_image_input_mode
        cfg = None
        with _quiet(None):
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
        mode = decide_image_input_mode(
            str(getattr(child, "provider", "") or ""), str(getattr(child, "model", "") or ""), cfg,
            requested_provider=str(getattr(child, "requested_provider", "") or ""),
        )
        if mode == "native":
            parts, skipped = build_native_content_parts(goal, paths, urls)
            if skipped:
                logger.warning("delegate_task: skipped %d unreadable image(s) for subagent: %s", len(skipped), ", ".join(skipped[:3]))
            if data_urls:
                parts = (parts or [{"type": "text", "text": goal}]) + [{"type": "image_url", "image_url": {"url": u}} for u in data_urls]
            return parts if any(p.get("type") == "image_url" for p in parts) else goal
        if data_urls:
            logger.warning("delegate_task: %d inline data-URL image(s) dropped for a non-vision subagent", len(data_urls))
        hints: list[str] = []
        for p in paths:
            if os.path.isfile(p):
                hints.append(f"[Image attached at: {p}]")
            else:
                logger.warning("delegate_task: image path not found, not forwarded: %s", p)
        hints.extend(f"[Image attached: {u}]" for u in urls)
        if not hints:
            return goal
        return goal + "\n\n" + "\n".join(hints) + "\nUse vision_analyze to inspect these images."
    except Exception:
        logger.warning("delegate_task: image forwarding failed; sending text-only goal", exc_info=True)
        return goal


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
    heartbeat: Any = None
    child_start: float = field(default_factory=time.monotonic)
    worktree_info: Optional[dict[str, str]] = None
    child_task_id: str = ""
    parent_task_id: Optional[str] = None
    wall_start: float = 0.0
    parent_reads_snapshot: list = field(default_factory=list)
    # Set by ``_run_single_child``: the reasoning-escalation chain this attempt belongs to (None = a lone run).
    chain: Optional[_EscalationChain] = None

    def elapsed(self) -> float:
        return round(time.monotonic() - self.child_start, 2)

    def working_dir(self) -> Optional[str]:
        """Where this attempt was working: its own terminal cwd, else the worktree, the parent's cwd, the hint."""
        from tools.delegate_tool_routes import child_profile_scope
        cwd: Optional[str] = None
        with _quiet("Could not read the child's cwd: %s"):
            from tools.terminal_tool import get_session_cwd
            with child_profile_scope(self.child):
                cwd = get_session_cwd(self.child_task_id) if self.child_task_id else None
        if isinstance(cwd, str) and cwd:
            return cwd
        if self.worktree_info is not None and self.worktree_info.get("path"):
            return self.worktree_info["path"]
        with _quiet(None):
            from tools.terminal_tool import get_session_cwd as _gsc
            cwd = _gsc(self.parent_task_id) if self.parent_task_id else None
        if isinstance(cwd, str) and cwd:
            return cwd
        with _quiet(None):
            from tools.delegate_tool import _resolve_workspace_hint
            return _resolve_workspace_hint(self.parent_agent)
        return None

    def cancelled(self) -> bool:
        """The unit was cancelled (marked on the chain's first child) or this attempt was stopped by subagent id."""
        first = self.chain.first_child if self.chain is not None else None
        return any(getattr(c, "_delegate_unit_cancelled", False) is True for c in (first, self.child))

    def escalate(self, entry: dict[str, Any]) -> bool:
        """Decide whether this attempt hands off to a retry at the next reasoning level and, if so, build the
        retry child on the same route (same limits, fresh budget) and queue it on the chain. True = this attempt is
        intermediate: its caller skips the completion event and the worktree finalize (the chain is still live)."""
        chain = self.chain
        if chain is None or self.cancelled():
            return False
        next_effort = _next_escalation_effort(self.child, entry, self.parent_agent)
        if next_effort is None:
            return False
        effort = _effort_ran(self.child)
        working_dir = self.working_dir()
        transcript = _str_or_none(getattr(self.child, "_live_transcript_path", None))
        processes = self.background_process_report()  # before teardown kills them
        retry_goal = _escalation_handoff(
            chain.original_goal, effort=effort, next_effort=next_effort, entry=entry, transcript_path=transcript,
            working_dir=working_dir, processes=processes,
        )
        chain.record(self.child, entry.get("exit_reason"))  # before the build: a reused child's level changes
        try:
            next_child = _build_retry_child(self.child, next_effort)
        except Exception as exc:  # boundary: a failed retry build degrades to "no escalation", never crashes the parent
            logger.warning("Subagent %d reasoning escalation to %s not started: %s", self.task_index, next_effort, exc,
                           exc_info=True)
            next_child = None
        if next_child is not None and self.cancelled():
            _signal_child_stop(next_child, "Async delegation cancelled")
            if next_child is not self.child:
                _detach_child(self.parent_agent, next_child)
                _close_child(next_child, "Failed to close cancelled retry child")
            else:
                next_child.session_estimated_cost_usd = chain.costs[-1][0]
            next_child = None
        if next_child is None:
            chain.attempts.pop()
            chain.costs.pop()
            return False
        logger.info(
            "Subagent %d stopped on %s at reasoning effort %s; retrying at %s", self.task_index,
            entry.get("exit_reason"), effort, next_effort,
        )
        if next_child is self.child:
            chain.reused_child = next_child
        elif getattr(self.child, "_handed_off_processes", None):
            # The parent already owns these; the final attempt's result names them (and the per-subagent cap holds).
            next_child._handed_off_processes = list(self.child._handed_off_processes)
        chain.working_dir = working_dir
        chain.modified_parent_paths |= self.modified_parent_paths()
        for key, items in processes.items():
            chain.process_report.setdefault(key, []).extend(items)
        chain.pending = (next_child, retry_goal)
        return True

    def relay_text(self, delta: str) -> None:
        """Stream callback forwarding the child's reply text up the progress relay so gateway watch windows mirror it
        live (subagent.text → message.delta). Inert under CLI/TUI: their progress handlers ignore non-tool events."""
        if delta:
            _safe_progress(self.child_progress_cb, "subagent.text", preview=delta)

    def attach_worktree(self, entry_dict: dict[str, Any]) -> dict[str, Any]:
        """Inspect + prune the child worktree, reporting into the entry (no-op without isolation)."""
        info = self.worktree_info
        if info is None:
            return redact_child_result(entry_dict, getattr(self.child, "_delegate_output_schema", None), force=False)
        from tools import subagent_worktree
        try:
            entry_dict["worktree"] = subagent_worktree.finalize_subagent_worktree(info)
        except Exception as e:
            # State is unknown: emit the SAME flagged schema the parent expects,
            # via the shared factory so the two producers never drift.
            logger.warning("worktree finalize failed: %s", e)
            entry_dict["worktree"] = subagent_worktree.unproven_worktree_payload(info, f"finalize raised: {e}")
        return redact_child_result(entry_dict, getattr(self.child, "_delegate_output_schema", None), force=False)

    def seed_workspace(self) -> None:
        """Seed cwd/container aliases and optional worktree isolation for the child;
        ``goal`` is extended with the worktree contract note when isolation engaged."""
        import uuid as _uuid
        chain = self.chain
        # A retry continues the chain's workspace: the previous attempt's directory, the one worktree.
        retry = chain is not None and chain.seeded
        base_id = self.subagent_id or f"subagent-{self.task_index}-{_uuid.uuid4().hex[:8]}"
        # Every attempt keeps the stable subagent_id (steer/stop/list) but runs its own terminal session and process
        # ownership; get_subagent_attribution strips the ``-attempt-N`` suffix back to the subagent.
        self.child_task_id = f"{base_id}-attempt-{len(chain.attempts) + 1}" if chain is not None and retry else base_id
        self.parent_task_id = getattr(self.parent_agent, "_current_task_id", None)
        # Seed the child's cwd record from the parent's: same starting directory,
        # but the child's later `cd`s stay in its own record. Per-session container
        # isolation keys containers by task_id; the child must share the PARENT's.
        # The parent's record is read in the parent's scope; the child's is written in the child's (cwd records
        # are keyed by the active profile, and a profile-routed child looks its own up as that profile).
        from tools.delegate_tool_routes import child_profile_scope
        retry_dir = chain.working_dir if chain is not None and retry else None
        with _quiet("Child cwd seed failed: %s"):
            from tools.terminal_tool import get_session_cwd, record_session_cwd, register_container_alias
            start_cwd = retry_dir or get_session_cwd(self.parent_task_id)
            with child_profile_scope(self.child):
                record_session_cwd(self.child_task_id, start_cwd)
            register_container_alias(self.child_task_id, self.parent_task_id)

        if chain is not None and retry:
            self.worktree_info = chain.worktree_info
        else:
            self.worktree_info = _create_isolated_worktree(self.parent_agent, self.parent_task_id, self.subagent_id)
            if chain is not None:
                chain.worktree_info, chain.seeded = self.worktree_info, True
        if self.worktree_info is not None:
            if not retry:
                with _quiet("worktree cwd seed failed: %s"), child_profile_scope(self.child):
                    from tools.terminal_tool import record_session_cwd as _rsc
                    _rsc(self.child_task_id, self.worktree_info["path"])
            # The child's context is already built; carry the isolation contract on
            # the goal message instead (same turn, no system-prompt mutation).
            from tools.subagent_worktree import build_worktree_context_note
            self.goal = self.goal + build_worktree_context_note(self.worktree_info)
        if chain is not None and retry:
            self.wall_start, self.parent_reads_snapshot = chain.wall_start, chain.parent_reads_snapshot
            return
        self.wall_start = time.time()
        self.parent_reads_snapshot = list(file_state.known_reads(self.parent_task_id)) if self.parent_task_id else []
        if chain is not None:
            chain.wall_start, chain.parent_reads_snapshot = self.wall_start, self.parent_reads_snapshot

    def finish_failed(
        self, entry: dict[str, Any], late_steer: Optional[str], *, preview: str, summary: str = "", status: Optional[str] = None,
    ) -> dict[str, Any]:
        """Shared tail of every failure path: emit ``subagent.complete`` (``status`` defaults to the entry's), note
        the steer text that won the race with the failure, add the chain's final-result facts, report the
        worktree."""
        _safe_progress(
            self.child_progress_cb, "subagent.complete", preview=redact_child_text(preview, force=False), status=status or entry["status"],
            duration_seconds=entry["duration_seconds"],
            summary=redact_child_text(summary, force=False) if summary is not None else None,
        )
        _append_missed_steer(entry, late_steer)
        self.add_final_facts(entry)
        return self.attach_worktree(entry)

    def add_final_facts(self, entry: dict[str, Any]) -> None:
        """What the parent must hear from the attempt that ends the chain, however it ended (completed, raised, timed
        out, cancelled): parent-read files the chain modified and the chain's background-process accounting."""
        self.append_sibling_write_reminder(entry)
        self.account_background_processes(entry)

    def close_steering(self) -> Optional[str]:
        """Close steer acceptance (see ``_merge_late_steer``); returns late steer text, if any."""
        return _parent_steer_text(_close_subagent_steering(self.subagent_id, self.child) if self.subagent_id else None)

    def wait_liveness_aware(self, settled: threading.Event, child_future: Any, child_timeout: Optional[float]) -> None:
        """Wait for the worker or the stale verdict, where ``child_timeout`` bounds INACTIVITY, not total runtime.

        A configured cap used to be a dispatch-to-death stopwatch, so it only ever killed children the runtime had
        already judged healthy: all 75 reported deaths happened while waiting on an in-flight LLM completion, none
        mid-tool (#116001). A provider serving multi-minute completions is progress — the child's activity clock
        ticks during the wait and the request is bounded by the per-call stale watchdog — so the budget restarts on
        every sign of progress, exactly the signals the heartbeat's stale verdict reads. Genuinely frozen children
        keep two authorities: this budget (when configured) and the heartbeat's stale threshold.
        """
        if child_timeout is None:
            settled.wait()
            return
        deadline = time.monotonic() + child_timeout
        warn_at = deadline - child_timeout * (1.0 - _BUDGET_WARNING_FRACTION)
        warned = False
        fingerprint = _child_activity_fingerprint(self.child)
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return  # no progress for the whole budget
            wait = min(_LIVENESS_POLL_SECONDS, remaining)
            if not warned:
                wait = min(wait, max(warn_at - now, 0.0))
            settled.wait(timeout=wait)
            if settled.is_set():
                return  # the worker finished, or the heartbeat declared the child stale
            current = _child_activity_fingerprint(self.child)
            if current != fingerprint:
                fingerprint = current
                deadline = time.monotonic() + child_timeout
                warn_at = deadline - child_timeout * (1.0 - _BUDGET_WARNING_FRACTION)
                warned = False  # a fresh window gets its own warning
            elif not warned and time.monotonic() >= warn_at:
                warned = True
                _warn_child_budget(self.child, child_timeout - (deadline - time.monotonic()), child_timeout)

    def await_child(self) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], bool]:
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
        from tools.delegate_tool import (
            _STALE_RESULT_GRACE_SECONDS, _get_subagent_approval_callback, _set_subagent_approval_cb,
        )
        from tools.daemon_pool import DaemonThreadPoolExecutor
        child, task_index = self.child, self.task_index
        child_timeout = _effective_child_timeout(child)
        executor = DaemonThreadPoolExecutor(
            max_workers=1, initializer=_set_subagent_approval_cb, initargs=(_get_subagent_approval_callback(),),
        )
        # Worker thread handle so the timeout diagnostic can dump its stack.
        worker_thread_holder: dict[str, Optional[threading.Thread]] = {"t": None}
        # Resolved after seed_workspace so a multimodal goal's text part carries the worktree note too.
        _images = list(getattr(child, "_delegate_images", None) or [])
        user_message: Any = _build_child_goal_message(self.goal, _images, child) if _images else self.goal

        def _run_with_thread_capture():
            worker_thread_holder["t"] = threading.current_thread()
            from agent.delegation_context import delegated_child_context
            from tools.delegate_tool_routes import child_profile_scope
            # A profile-routed child converses as that profile (config, secrets, skills, terminal policy).
            with child_profile_scope(child), delegated_child_context(str(getattr(child, "session_id", "") or "")):
                return child.run_conversation(
                    user_message=user_message, task_id=self.child_task_id, stream_callback=self.relay_text,
                )

        future = executor.submit(contextvars.copy_context().run, _run_with_thread_capture)
        # One wait covers both ways out: the worker finishing, or the heartbeat's stale verdict.
        # Without the second, a worker wedged after its final answer holds a finite (-Q / Bot Chat
        # one-shot) turn — and its session lease — forever, since that runtime has no gateway
        # inactivity watchdog (#109749). The configured cap bounds INACTIVITY (see wait_liveness_aware).
        settled = self.heartbeat.settled if self.heartbeat is not None else threading.Event()
        future.add_done_callback(lambda _f: settled.set())
        # Set when the stale verdict — not the configured cap — ended the wait; the entry must name that cause.
        stale_after: Optional[float] = None
        try:
            self.wait_liveness_aware(settled, future, child_timeout)
            if not future.done():
                stale_after = getattr(self.heartbeat, "stale_threshold_seconds", None)
                if stale_after is None:
                    # A configured inactivity cap elapsing keeps its teeth: the user armed a
                    # budget, the fingerprint never moved, abandon now (#116001 contract).
                    raise FuturesTimeoutError()
                # Grace collection (#113222): the heartbeat's stale verdict ended the wait
                # while the worker was still pending — a child that already wrote its final
                # answer often lands its result moments later, and that real result must be
                # collected instead of a synthesized timeout that strands a finished batch
                # as `running`. Signal the cooperative stop (a stoppable worker can then
                # unwind inside the window), poll the future for the grace period, and only
                # fall through to the timeout path if nothing lands. Steering stays OPEN
                # here on purpose: the success path's `_merge_late_steer` (or the failure
                # path below) is the linearization boundary, so a concurrent steer is never
                # lost either way.
                _signal_child_stop(child)
                grace = _STALE_RESULT_GRACE_SECONDS
                grace_done = threading.Event()
                future.add_done_callback(lambda _f: grace_done.set())
                if not grace_done.wait(timeout=grace) or not future.done():
                    raise FuturesTimeoutError()
                logger.info(
                    "Subagent %d settled inside the %.1fs stale-result grace window — collecting its real result",
                    task_index, grace,
                )
                return future.result(), None, False
            return future.result(), None, False
        except Exception as wait_exc:
            exc: BaseException = wait_exc  # ``as`` targets are unbound after the except block
        finally:
            # Shut down without waiting — a child stuck on blocking I/O would hang wait=True forever.
            executor.shutdown(wait=False)

        _late_pending_steer = self.close_steering()
        _signal_child_stop(child)
        is_timeout = isinstance(exc, (FuturesTimeoutError, TimeoutError))
        # What actually ended the wait: the stale threshold pre-empts a longer configured cap.
        timeout_cause = stale_after if stale_after is not None else child_timeout
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
                # A stale verdict or a configured cap; ``or 0.0`` guards the type checker.
                timeout_seconds=float(timeout_cause or 0.0), duration_seconds=float(duration),
                worker_thread=worker_thread_holder.get("t"), goal=self.goal,
            )
            if diagnostic_path:
                logger.warning("Subagent %d 0-API-call timeout — diagnostic written to %s", task_index, diagnostic_path)
        if not is_timeout:
            _err = str(exc)
        elif stale_after is not None:
            _err = (
                f"Subagent stopped making progress after {child_api_calls} API call(s) — no activity for "
                f"{stale_after:g}s (heartbeat stale threshold); the pending worker was abandoned."
            )
        elif before_first_call:
            _err = (
                f"Subagent timed out after {child_timeout}s without making any API call — the child never reached its "
                f"first LLM request (prompt construction, credential resolution, or transport may be stuck)."
            )
        else:
            _err = (
                f"Subagent timed out after {child_timeout}s with {child_api_calls} API call(s) completed — no "
                f"progress in that window (no completed call, tool change, or activity-clock tick): a stalled "
                f"provider request or an unresponsive network request."
            )
        if diagnostic_path:
            _err += f" Diagnostic: {diagnostic_path}"
        status = "timeout" if is_timeout else "error"
        _error_entry = {
            "task_index": task_index, "status": status, "summary": None, "error": _err, "exit_reason": status,
            "api_calls": child_api_calls, "duration_seconds": duration,
            "timeout_seconds": timeout_cause if is_timeout else None,
            "timed_out_after_seconds": duration if is_timeout else None,
            "timeout_phase": "before_first_llm_call" if before_first_call else "after_llm_calls" if is_timeout else None,
            # How long the child had been silent when the wait ended: distinguishes a slow provider from a runaway
            # without transcript forensics (#116001).
            "last_event_age": _child_last_event_age(child) if is_timeout else None,
            "_child_role": getattr(child, "_delegate_role", None),
            "diagnostic_path": diagnostic_path,
        }
        _error_entry["model"] = _str_or_none(getattr(child, "model", None))
        if (_route := _route_report(child)) is not None:
            _error_entry["route"] = _route
        _handoff_view = {**_error_entry, **({"missed_steer": _late_pending_steer} if _late_pending_steer else {})}
        may_escalate = True
        if (is_timeout and not future.done()
                and _next_escalation_effort(child, _handoff_view, self.parent_agent) is not None):
            # The stop is cooperative: a retry started now would share the workspace (and worktree) with a
            # worker that is still writing. Hand off only once the predecessor has actually exited.
            if not _await_timed_out_worker_exit(child, future):
                may_escalate = False
                logger.warning(
                    "Subagent %d timed-out worker did not exit after interrupt; skipping reasoning escalation so "
                    "the workspace is not shared with a live worker", task_index,
                )
        close_deferred = is_timeout and not future.done()
        if may_escalate and self.escalate(_handoff_view):
            # Intermediate attempt: the chain goes on in the same workspace, so no completion event and no
            # worktree finalize yet — the retry's own completion is the one the parent sees.
            _append_missed_steer(_error_entry, _late_pending_steer)
        else:
            _error_entry = self.finish_failed(_error_entry, _late_pending_steer, preview=f"Timed out after {duration}s" if is_timeout else str(exc))
        if close_deferred:
            _defer_close_after_timeout(child, future)
        return None, _error_entry, close_deferred

    def modified_parent_paths(self) -> set[str]:
        """Files the parent had read that were written since this run (any non-parent writer) — so nested
        orchestrator→worker chains are covered too."""
        if not (self.parent_task_id and self.parent_reads_snapshot):
            return set()
        with _quiet("file_state sibling-write check failed", exc_info=True):
            sibling_writes = file_state.writes_since(self.parent_task_id, self.wall_start, self.parent_reads_snapshot)
            return {p for paths in sibling_writes.values() for p in paths}
        return set()

    def append_sibling_write_reminder(self, entry: dict[str, Any]) -> None:
        """Warn the parent when this child (or an earlier attempt of its chain) wrote files the parent had read."""
        earlier = self.chain.modified_parent_paths if self.chain is not None else set()
        mod_paths = sorted(self.modified_parent_paths() | earlier)
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

    def background_process_report(self) -> dict[str, list[dict[str, Any]]]:
        """``orphaned_processes`` (still running, about to be terminated with the child) and ``unread_completions``
        (exited notify processes nobody read) for this attempt's own background processes; empty keys omitted."""
        report: dict[str, list[dict[str, Any]]] = {}
        with _quiet(None):
            from tools.process_registry import process_registry, _output_tail
            leftover = process_registry.running_owned_by(self.child_task_id)
            if leftover:
                report["orphaned_processes"] = [
                    {"session_id": s.id, "command": redact_child_text(s.command, force=False)[:200], "runtime_seconds": round(time.time() - s.started_at)}
                    for s in leftover]
            unread = process_registry.unread_completions_owned_by(self.child_task_id)
            if unread:
                report["unread_completions"] = [
                    {"session_id": s.id, "command": redact_child_text(s.command, force=False)[:200], "exit_code": s.exit_code,
                     "output_tail": redact_child_text(_output_tail(s, len(s.output_buffer)), force=False)[-600:]} for s in unread]
        return report

    def account_background_processes(self, entry: dict[str, Any]) -> None:
        """Name the child's background processes on the result BEFORE ``cleanup`` kills them: handed-off ones now
        belong to the parent (their completion lands in the parent's chat); anything else still running is about to be
        terminated, and the parent must hear that from the runtime rather than trust a child's "watcher running".
        Earlier attempts of the chain were accounted before their own teardown (``escalate``); they come first."""
        handed = list(getattr(self.child, "_handed_off_processes", None) or [])
        if handed:
            entry["handed_off_processes"] = handed
        earlier = self.chain.process_report if self.chain is not None else {}
        current = self.background_process_report()
        for key in ("orphaned_processes", "unread_completions"):
            combined = earlier.get(key, []) + current.get(key, [])
            if combined:
                entry[key] = combined

    def emit_complete(self, result: dict[str, Any], entry: dict[str, Any], duration: float) -> None:
        """Fire ``subagent.complete`` with the per-branch observability payload (tokens, cost, files touched,
        tool-output tail); every field is optional and degrades gracefully on the client."""
        if not self.child_progress_cb:
            return
        child = self.child
        # Parent copy + live progress: honour security.redact_secrets=false (persistence boundaries force).
        entry.update(redact_child_result(entry, getattr(child, "_delegate_output_schema", None), force=False))
        summary = entry["summary"]
        _files_read: list = []
        with _quiet(None):
            _files_read = list(file_state.known_reads(self.child_task_id))[:40]
        _files_written_map: dict = {}
        with _quiet(None):
            _files_written_map = file_state.writes_since("", self.wall_start, [])  # all writes since wall_start
        complete_kwargs: dict[str, Any] = {
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
        if entry.get("failure_reason"):
            # Classified verdict rides the event so every surface glosses the failure the same way.
            complete_kwargs["failure_reason"] = entry["failure_reason"]
        _cost_usd = getattr(child, "session_estimated_cost_usd", None)
        if _cost_usd is not None:
            with _quiet(None):
                complete_kwargs["cost_usd"] = float(_cost_usd)
        _safe_progress(self.child_progress_cb, "subagent.complete", **redact_child_result(complete_kwargs, force=False))

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
        # The retry builder closes over the parent and the route; drop it once this attempt is over (it was used,
        # if at all, before this point) so a lingering child record never pins the parent.
        with _quiet(None):
            vars(child).pop("_delegate_build_retry", None)

        # Close tool resources (terminal sandboxes, browser daemons, background
        # processes, httpx clients) so subagent subprocesses don't outlive the delegation.
        reused = self.chain is not None and self.chain.pending is not None and self.chain.pending[0] is child
        if not close_deferred and not reused:
            _close_child(child, "Failed to close child agent after delegation")
        # The child's execute_code kernels live exactly as long as the child (pinned against the LRU
        # cap while it runs); dispose them here so they never squat the cap after the child is gone.
        with _quiet("Failed to dispose child execute_code kernels: %s"):
            from tools.code_kernel import shutdown_kernels_for_delegated_child
            if not reused:
                shutdown_kernels_for_delegated_child(str(getattr(child, "session_id", "") or ""))

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
