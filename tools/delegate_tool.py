#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with isolated context, inherited toolsets,
and their own terminal sessions. Supports single-task and batch (parallel)
modes. Top-level model calls run in the background; orchestrator children
wait for their own workers so they can synthesize the results.

Each child gets:
  - A fresh conversation (no parent history)
  - Its own task_id (own terminal session, file ops cache)
  - The parent's toolsets, with child-only blocked tools stripped
  - A focused system prompt built from the delegated goal + context

The parent's context only sees the delegation call and the summary result,
never the child's intermediate tool calls or reasoning.
"""


import contextvars
import json
import logging
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from agent.interrupt_compat import request_hard_interrupt
from tools import file_state
from tools.delegation_outcome import (
    apply_schema_evidence,
    classify_delegation_result,
    delegation_batch_icon,
    failed_delegation_evidence,
    schema_evidence_payload,
    terminal_tool_error_count,
)
from tools.delegation_output_schema import validate_and_repair_output
from tools.registry import registry, tool_error
from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb
from utils import is_truthy_value

logger = logging.getLogger(__name__)

from tools.delegate_tool_control import (
    _RUNTIME_PROVIDER_CUSTOM,
    DELEGATE_BLOCKED_TOOLS,
    _subagent_auto_deny,
    _subagent_auto_approve,
    _get_subagent_approval_callback,
    _DEFAULT_MAX_CONCURRENT_CHILDREN,
    _HIGH_CONCURRENCY_WARNED,
    MAX_DEPTH,
    _MIN_SPAWN_DEPTH,
    _spawn_pause_lock,
    _spawn_paused,
    _active_subagents_lock,
    _active_subagents,
    _RECENT_SUBAGENTS_CAP,
    _recent_subagents,
    SUBAGENT_FAILURE_STATUSES,
    _clean_error_text,
    format_subagent_failure_line,
    get_subagent_attribution,
    set_spawn_paused,
    is_spawn_paused,
    _register_subagent,
    _retain_recent_subagent,
    _unregister_subagent,
    _close_subagent_steering,
    interrupt_subagent,
    steer_subagent,
    _capture_gateway_steer_authority,
    list_active_subagents,
    _is_descendant_of,
    _CONTROL_ACTIONS,
    _resolve_session_lineage,
    _owns_subagent_record,
    _handle_control_action,
    _extract_output_tail,
    _stringify_tool_content,
    _TOOL_INPUT_TARGET_KEYS,
    _TOOL_INPUT_URL_KEYS,
    _sanitize_tool_target,
    _summarize_tool_arguments,
    _sanitize_tool_input_summary,
    _subagent_stop_tool_call_history,
    _looks_like_error_output,
    _normalize_role,
    _get_max_concurrent_children,
    _get_worktree_isolation,
    _LEGACY_MAX_ASYNC_WARNED,
    _get_max_async_children,
    _get_child_timeout,
    _get_max_spawn_depth,
    _get_orchestrator_enabled,
    _get_inherit_mcp_toolsets,
    _is_mcp_toolset_name,
    _expand_parent_toolsets,
    _preserve_parent_mcp_toolsets,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SUMMARY_CHARS,
    _SUMMARY_HEADROOM_FRACTION,
    _MIN_SUMMARY_CHARS,
    DEFAULT_CHILD_TIMEOUT,
    _HEARTBEAT_INTERVAL,
    _HEARTBEAT_STALE_CYCLES_IDLE,
    _HEARTBEAT_STALE_CYCLES_IN_TOOL,
    DEFAULT_TOOLSETS,
    DelegateEvent,
    _LEGACY_EVENT_MAP,
    check_delegate_requirements,
    _load_config,
)
from tools.delegate_tool_setup import (
    _build_child_system_prompt,
    _resolve_workspace_hint,
    _strip_blocked_tools,
    _blocked_toolsets_for_role,
    _BATCH_ORDINALS,
    format_batch_tag,
    _batch_prefix as _batch_prefix,
    _emit_parent_console,
    _build_child_progress_callback,
    _normalized_runtime_url,
    _inherit_parent_capabilities,
    _inherit_parent_base_url,
    _build_child_agent,
    _dump_subagent_timeout_diagnostic,
    _spill_summary_to_file,
    _trim_summary_with_footer,
    _parent_summary_char_budget,
    _apply_summary_budget,
)
from tools.delegate_tool_runtime import (
    _PARENT_FINALIZATION_LOCK_GUARD,
    _PARENT_FINALIZATION_FALLBACK_LOCK,
    _CHILD_CONSTRUCTION_LOCK,
    _build_child_preserving_parent_tools,
    _parent_finalization_lock,
    _finalize_child_results,
    _run_child_lifecycle,
    _recover_tasks_from_json_string,
)
from tools.delegate_tool_dispatch import (
    _PLACEHOLDER_GOAL_RE,
    _TEMPLATE_MARKER_RE,
    _MIN_BATCH_GOAL_LEN,
    _validate_batch_tasks,
    _resolve_child_credential_pool,
    _merge_request_overrides,
    _resolve_delegation_credentials,
    _load_config,
    _build_top_level_description,
    _build_tasks_param_description,
    _build_role_param_description,
    _build_dynamic_schema_overrides,
    DELEGATE_TASK_SCHEMA,
    _model_background_value,
    _MODEL_HIDDEN_TASK_FIELDS,
    _strip_model_hidden_task_fields,
)


def _run_single_child(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    *,
    owner_session_id: Optional[str] = None,
    owner_transport: Any = None,
    owner_session_record: Any = None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Run a pre-built child agent. Called from within a thread.
    Returns a structured result dict with a ``status`` and ``exit_reason``
    that are derived honestly from the child's structured completion fields.

    ``status`` ∈ {``"completed"``, ``"interrupted"``, ``"failed"``}:
        * ``"completed"``  — the child reached a normal finish (may still have
          hit its iteration budget; see ``exit_reason``).
        * ``"interrupted"`` — the child was interrupted (``interrupted=True``).
        * ``"failed"``    — a structured failure (``failed=True`` or a non-empty
          ``error``) or a summary-less/invalid terminal state.

    ``exit_reason`` ∈ {``"completed"``, ``"max_iterations"``, ``"interrupted"``,
    ``"error"``}:
        * ``"completed"``       — normal finish.
        * ``"max_iterations"``  — genuine per-child iteration-budget exhaustion
          (``completed=False`` with no failure fields).
        * ``"interrupted"``     — interrupted by the parent.
        * ``"error"``           — provider rejection / terminal failure; NOT
          budget exhaustion (this is the case #97655 fixed).

    ``truncated`` is derived as ``exit_reason == "max_iterations"`` only, so the
    parent-visible truncation flag stays truthful for all of the above.
    """
    child_start = time.monotonic()
    # A timed-out Future may still be unwinding on its daemon worker. Closing
    # the child from this owner thread before that Future settles races every
    # resource the conversation's finally path still touches (notably its
    # owned SessionDB). The timeout branch flips this when close ownership is
    # handed to a Future done-callback instead.
    _child_close_deferred = False

    # Get the progress callback from the child agent
    child_progress_cb = getattr(child, "tool_progress_callback", None)

    # Restore parent tool names using the value saved before child construction
    # mutated the global. This is the correct parent toolset, not the child's.
    import model_tools

    _saved_tool_names = getattr(
        child, "_delegate_saved_tool_names", list(model_tools._last_resolved_tool_names)
    )

    child_pool = getattr(child, "_credential_pool", None)
    leased_cred_id = None
    if child_pool is not None:
        leased_cred_id = child_pool.acquire_lease()
        if leased_cred_id is not None:
            try:
                leased_entry = child_pool.current()
                if leased_entry is not None and hasattr(child, "_swap_credential"):
                    child._swap_credential(leased_entry)
            except Exception as exc:
                logger.debug("Failed to bind child to leased credential: %s", exc)

    # Heartbeat: periodically propagate child activity to the parent so the
    # gateway inactivity timeout doesn't fire while the subagent is working.
    # Without this, the parent's _last_activity_ts freezes when delegate_task
    # starts and the gateway eventually kills the agent for "no activity".
    _heartbeat_stop = threading.Event()
    # Stale detection: track the child's (tool, iteration, activity_ts) across
    # heartbeat cycles. If none advances, count the cycle as stale.
    # Different thresholds for idle vs in-tool (see _HEARTBEAT_STALE_CYCLES_*).
    # last_activity_ts is the same liveness signal the async stall monitor
    # already uses (streamed chunks + direct_api_call mid-wait heartbeats).
    _last_seen_iter = [0]
    _last_seen_tool = [None]  # type: list
    _last_seen_activity_ts = [None]  # type: list
    _stale_count = [0]

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            if parent_agent is None:
                continue
            touch = getattr(parent_agent, "_touch_activity", None)
            if not touch:
                continue
            # Pull detail from the child's own activity tracker
            desc = f"delegate_task: subagent {task_index} working"
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)
                child_activity_ts = child_summary.get("last_activity_ts")

                # Stale detection: count cycles where iteration, current_tool,
                # AND last_activity_ts are all frozen. A child running a
                # legitimately long-running tool keeps current_tool set; a
                # child waiting on a slow model refreshes last_activity_ts
                # via direct_api_call's activity heartbeat — neither should
                # look stale at the idle threshold.
                iter_advanced = child_iter > _last_seen_iter[0]
                tool_changed = child_tool != _last_seen_tool[0]
                activity_advanced = (
                    child_activity_ts is not None
                    and (
                        _last_seen_activity_ts[0] is None
                        or child_activity_ts > _last_seen_activity_ts[0]
                    )
                )
                if iter_advanced or tool_changed or activity_advanced:
                    _last_seen_iter[0] = child_iter
                    _last_seen_tool[0] = child_tool
                    if child_activity_ts is not None:
                        _last_seen_activity_ts[0] = child_activity_ts
                    _stale_count[0] = 0
                else:
                    _stale_count[0] += 1

                # Pick threshold based on whether the child is currently
                # inside a tool call. In-tool threshold is high enough to
                # cover legitimately slow tools; idle threshold stays
                # tight so the gateway timeout can fire on a truly wedged
                # child.
                stale_limit = (
                    _HEARTBEAT_STALE_CYCLES_IN_TOOL
                    if child_tool
                    else _HEARTBEAT_STALE_CYCLES_IDLE
                )
                if _stale_count[0] >= stale_limit:
                    logger.warning(
                        "Subagent %d appears stale (no progress for %d "
                        "heartbeat cycles, tool=%s) — stopping heartbeat",
                        task_index,
                        _stale_count[0],
                        child_tool or "<none>",
                    )
                    break  # stop touching parent, let gateway timeout fire

                if child_tool:
                    desc = (
                        f"delegate_task: subagent running {child_tool} "
                        f"(iteration {child_iter}/{child_max})"
                    )
                else:
                    child_desc = child_summary.get("last_activity_desc", "")
                    if child_desc:
                        desc = (
                            f"delegate_task: subagent {child_desc} "
                            f"(iteration {child_iter}/{child_max})"
                        )
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)

    # Register the live agent in the module-level registry so the TUI can
    # target it by subagent_id (kill, pause, status queries).  Unregistered
    # in the finally block, even when the child raises.  Test doubles that
    # hand us a MagicMock don't carry stable ids; skip registration then.
    _raw_sid = getattr(child, "_subagent_id", None)
    _subagent_id = _raw_sid if isinstance(_raw_sid, str) else None
    if _subagent_id:
        if owner_session_id is None:
            try:
                from gateway.session_context import get_session_env

                owner_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or None
            except Exception:
                owner_session_id = None
        if owner_session_id and (
            owner_transport is None or owner_session_record is None
        ):
            owner_transport, owner_session_record = (
                _capture_gateway_steer_authority(owner_session_id)
            )
        _raw_depth = getattr(child, "_delegate_depth", 1)
        _tui_depth = max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0
        _parent_sid = getattr(child, "_parent_subagent_id", None)
        # Durable ownership spine: the OWNING CONVERSATION's session id (the
        # same lineage the delivery path routes completions by). Sourced from
        # the child's _parent_session_id stamp so it stays correct even when
        # parent_agent has been rebuilt between dispatch and this run.
        _owner_agent_session_id = (
            str(getattr(child, "_parent_session_id", "") or "")
            or str(getattr(parent_agent, "session_id", "") or "")
        )
        _delegation_id = getattr(child, "_delegation_id", None)
        _register_subagent(
            {
                "subagent_id": _subagent_id,
                "parent_id": _parent_sid if isinstance(_parent_sid, str) else None,
                "depth": _tui_depth,
                "goal": goal,
                "delegation_id": (
                    _delegation_id if isinstance(_delegation_id, str) else None
                ),
                "model": (
                    getattr(child, "model", None)
                    if isinstance(getattr(child, "model", None), str)
                    else None
                ),
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": child,
                # Durable conversation lineage for the model-facing control
                # plane (list/steer/stop). The weakref identity chain breaks
                # when the CLI rebuilds its AIAgent mid-session; this id is
                # the same spine completion delivery routes by.
                "owner_agent_session_id": _owner_agent_session_id or None,
                # Immutable live gateway/TUI session that commissioned this
                # child. Empty outside those hosts; RPC authority fails closed.
                "owner_session_id": owner_session_id,
                "owner_transport": owner_transport,
                "owner_session_record": owner_session_record,
            }
        )

    # Worktree-isolation state: populated inside the try once the child's
    # task id is known; the default no-op keeps every early error path safe.
    _worktree_info: Optional[Dict[str, str]] = None

    def _attach_worktree(entry_dict: Dict[str, Any]) -> None:
        """Inspect + prune the child worktree, reporting into the entry."""
        if _worktree_info is None:
            return
        try:
            from tools import subagent_worktree

            entry_dict["worktree"] = (
                subagent_worktree.finalize_subagent_worktree(_worktree_info)
            )
        except Exception as e:
            # finalize is written hard not to raise, but if it ever does the
            # state is unknown — emit the SAME schema the parent expects,
            # flagged, via the shared factory so the two producers of this
            # payload can never drift.
            logger.warning("worktree finalize failed: %s", e)
            try:
                from tools import subagent_worktree as _sw

                entry_dict["worktree"] = _sw.unproven_worktree_payload(
                    _worktree_info, f"finalize raised: {e}"
                )
            except Exception:
                # Import itself failed — inline the same shape rather than
                # dropping the flag (the parent must still see the warning).
                entry_dict["worktree"] = {
                    "path": _worktree_info.get("path", ""),
                    "branch": _worktree_info.get("branch", ""),
                    "commits": 0,
                    "dirty": False,
                    "pruned": False,
                    "inspection_failed": True,
                    "note": (
                        f"worktree finalize raised ({e}) and the reporting "
                        "helper was unavailable: 'commits' and 'dirty' are "
                        "UNKNOWN, not zero/clean. Inspect "
                        f"{_worktree_info.get('path', '')} before assuming "
                        "no work."
                    ),
                }

    try:
        _heartbeat_thread.start()
        if child_progress_cb:
            try:
                child_progress_cb("subagent.start", preview=goal)
            except Exception as e:
                logger.debug("Progress callback start failed: %s", e)

        # File-state coordination: reuse the stable subagent_id as the child's
        # task_id so file_state writes, active-subagents registry, and TUI
        # events all share one key.  Falls back to a fresh uuid only if the
        # pre-built id is somehow missing.
        import uuid as _uuid

        child_task_id = _subagent_id or f"subagent-{task_index}-{_uuid.uuid4().hex[:8]}"
        parent_task_id = getattr(parent_agent, "_current_task_id", None)
        # Seed the child's session-cwd record from the parent's (cwd rearch):
        # children share the parent's container, and today they inherit the
        # parent's live env.cwd implicitly. Seeding at spawn preserves that
        # starting directory while keeping the child's subsequent `cd`s
        # isolated in its own record (a child's cd no longer bleeds back into
        # the parent once readers flip to the record store).
        try:
            from tools.terminal_tool import (
                get_session_cwd,
                record_session_cwd,
                register_container_alias,
            )

            record_session_cwd(child_task_id, get_session_cwd(parent_task_id))
            # Per-session container isolation (docker + container_persistent:
            # false) keys containers by session task_id. The child must share
            # the PARENT's container — register the alias so the child's
            # task_id resolves to the parent's container key.
            register_container_alias(child_task_id, parent_task_id)
        except Exception as e:
            logger.debug("Child cwd seed failed: %s", e)

        # Opt-in worktree isolation (delegation.worktree_isolation, inspired
        # by Muse Code's --subagent-worktree-isolation): give this child its
        # own git worktree branched from the parent repo's HEAD, and start its
        # terminal there. Git-only and local-backend-only; any failure
        # degrades silently to the shared-workspace behavior above.
        if _get_worktree_isolation():
            try:
                from tools import subagent_worktree

                if subagent_worktree.local_backend_active():
                    _parent_cwd = None
                    try:
                        from tools.terminal_tool import get_session_cwd as _gsc

                        _parent_cwd = _gsc(parent_task_id)
                    except Exception:
                        pass
                    _worktree_info = subagent_worktree.create_subagent_worktree(
                        _parent_cwd or _resolve_workspace_hint(parent_agent),
                        subagent_id=_subagent_id,
                    )
                else:
                    logger.debug(
                        "worktree isolation skipped: non-local terminal backend"
                    )
            except Exception as e:
                logger.debug("worktree isolation setup failed: %s", e)
            if _worktree_info is not None:
                try:
                    from tools.terminal_tool import record_session_cwd as _rsc

                    _rsc(child_task_id, _worktree_info["path"])
                except Exception as e:
                    logger.debug("worktree cwd seed failed: %s", e)
                # The child's context is already built; carry the isolation
                # contract on the goal message instead (same turn, no
                # system-prompt mutation).
                from tools.subagent_worktree import build_worktree_context_note

                goal = goal + build_worktree_context_note(_worktree_info)

        wall_start = time.time()
        parent_reads_snapshot = (
            list(file_state.known_reads(parent_task_id)) if parent_task_id else []
        )

        # Run child with an optional hard timeout (off by default —
        # result(timeout=None) blocks until the child finishes). Stuck-child
        # protection comes from the heartbeat staleness monitor instead.
        child_timeout = _get_child_timeout()
        # Daemon worker (tools.daemon_pool): a timed-out child is abandoned
        # below; a stdlib non-daemon worker would then block interpreter
        # exit at atexit-join time if the child never unwinds.
        from tools.daemon_pool import DaemonThreadPoolExecutor
        _timeout_executor = DaemonThreadPoolExecutor(
            max_workers=1,
            # Install a non-interactive approval callback in the worker thread
            # so dangerous-command prompts from the subagent don't fall back to
            # input() and deadlock the parent's prompt_toolkit TUI.
            # Callback (deny vs approve) is governed by delegation.subagent_auto_approve.
            initializer=_set_subagent_approval_cb,
            initargs=(_get_subagent_approval_callback(),),
        )
        # Capture the worker thread so the timeout diagnostic can dump its
        # Python stack (see #14726 — 0-API-call hangs are opaque without it).
        _worker_thread_holder: Dict[str, Optional[threading.Thread]] = {"t": None}

        def _relay_child_text(delta: str) -> None:
            # Forward the child's streamed reply text up the progress relay so
            # gateway watch windows mirror it live (subagent.text → message.delta).
            # Inert under CLI/TUI: their progress handlers ignore non-tool events.
            if not delta or not child_progress_cb:
                return
            try:
                child_progress_cb("subagent.text", preview=delta)
            except Exception as e:
                logger.debug("Child text relay failed: %s", e)

        def _run_with_thread_capture():
            _worker_thread_holder["t"] = threading.current_thread()
            from agent.delegation_context import delegated_child_context

            with delegated_child_context(str(getattr(child, "session_id", "") or "")):
                return child.run_conversation(
                    user_message=goal,
                    task_id=child_task_id,
                    stream_callback=_relay_child_text,
                )

        _child_context = contextvars.copy_context()
        _child_future = _timeout_executor.submit(
            _child_context.run,
            _run_with_thread_capture,
        )
        try:
            result = _child_future.result(timeout=child_timeout)
        except Exception as _timeout_exc:
            # No consumer boundary remains once this owner stops waiting for
            # the child. Close acceptance before any completion callback and
            # retain steer text that won the race with this failure/timeout.
            _late_pending_steer = (
                _close_subagent_steering(_subagent_id, child) if _subagent_id else None
            )
            # Signal the child to stop so its thread can exit cleanly.
            try:
                interrupted = child is not None and request_hard_interrupt(child)
                if not interrupted and child is not None and hasattr(child, "_interrupt_requested"):
                    child._interrupt_requested = True
            except Exception:
                pass

            is_timeout = isinstance(_timeout_exc, (FuturesTimeoutError, TimeoutError))
            duration = round(time.monotonic() - child_start, 2)
            logger.warning(
                "Subagent %d %s after %.1fs",
                task_index,
                "timed out" if is_timeout else f"raised {type(_timeout_exc).__name__}",
                duration,
            )

            # When a subagent times out BEFORE making any API call, dump a
            # diagnostic to help users (and us) see what the child was doing.
            # See #14726 — without this, 0-API-call hangs are black boxes.
            diagnostic_path: Optional[str] = None
            child_api_calls = 0
            try:
                _summary = child.get_activity_summary()
                child_api_calls = int(_summary.get("api_call_count", 0) or 0)
            except Exception:
                pass
            if is_timeout and child_api_calls == 0:
                diagnostic_path = _dump_subagent_timeout_diagnostic(
                    child=child,
                    task_index=task_index,
                    # is_timeout implies a cap was configured (result(timeout=None)
                    # never raises FuturesTimeoutError); guard for the type checker.
                    timeout_seconds=float(child_timeout or 0.0),
                    duration_seconds=float(duration),
                    worker_thread=_worker_thread_holder.get("t"),
                    goal=goal,
                )
                if diagnostic_path:
                    logger.warning(
                        "Subagent %d 0-API-call timeout — diagnostic written to %s",
                        task_index,
                        diagnostic_path,
                    )

            if child_progress_cb:
                try:
                    child_progress_cb(
                        "subagent.complete",
                        preview=(
                            f"Timed out after {duration}s"
                            if is_timeout
                            else str(_timeout_exc)
                        ),
                        status="timeout" if is_timeout else "error",
                        outcome="failed",
                        exit_reason="timeout" if is_timeout else "error",
                        interrupted=False,
                        tool_error_count=0,
                        duration_seconds=duration,
                        summary="",
                    )
                except Exception:
                    pass

            if is_timeout:
                if child_api_calls == 0:
                    _err = (
                        f"Subagent timed out after {child_timeout}s without "
                        f"making any API call — the child never reached its "
                        f"first LLM request (prompt construction, credential "
                        f"resolution, or transport may be stuck)."
                    )
                    if diagnostic_path:
                        _err += f" Diagnostic: {diagnostic_path}"
                else:
                    _err = (
                        f"Subagent timed out after {child_timeout}s with "
                        f"{child_api_calls} API call(s) completed — likely "
                        f"stuck on a slow API call, tool call, or unresponsive "
                        f"network request."
                    )
                    if diagnostic_path:
                        _err += f" Diagnostic: {diagnostic_path}"
            else:
                _err = str(_timeout_exc)

            _error_entry = {
                "task_index": task_index,
                **failed_delegation_evidence(
                    status="timeout" if is_timeout else "error",
                    exit_reason="timeout" if is_timeout else "error",
                ),
                "summary": None,
                "error": _err,
                "api_calls": child_api_calls,
                "duration_seconds": duration,
                "timeout_seconds": child_timeout if is_timeout else None,
                "timed_out_after_seconds": duration if is_timeout else None,
                "timeout_phase": (
                    "before_first_llm_call" if is_timeout and child_api_calls == 0
                    else "after_llm_calls" if is_timeout
                    else None
                ),
                "_child_role": getattr(child, "_delegate_role", None),
                "diagnostic_path": diagnostic_path,
            }
            if _late_pending_steer:
                _error_entry["missed_steer"] = _late_pending_steer
                _error_entry["error"] += (
                    " [steer did not land before the subagent stopped: "
                    f"{_late_pending_steer}]"
                )
            _attach_worktree(_error_entry)
            if is_timeout and not _child_future.done():
                # request_hard_interrupt() is cooperative: the worker still
                # executes run_conversation's finally path before its Future
                # becomes done. child.close() tears down that same agent's
                # clients, messages, and owned SQLite handle, so calling it in
                # our outer finally while the worker is alive can close SQLite
                # underneath its final activity write. Future callbacks run
                # only after the worker has fully returned (or raised), which
                # is the first safe close boundary.
                def _close_after_timed_out_worker(_done_future) -> None:
                    try:
                        close = getattr(child, "close", None)
                        if callable(close):
                            close()
                    except Exception:
                        logger.debug(
                            "Failed to close timed-out child after worker exit",
                            exc_info=True,
                        )

                _child_future.add_done_callback(_close_after_timed_out_worker)
                _child_close_deferred = True

                # Bounded drain (#94248 native half): the deferred close above
                # only fires once the abandoned worker unwinds, but that worker
                # is typically parked inside an in-flight OpenSSL read (Codex /
                # httpx). Never hard-close that transport from this thread —
                # releasing FDs under a live SSL read is the #29507/#70773
                # native-corruption family. Instead shutdown() the child's
                # pooled sockets, which is FD-safe from any thread and settles
                # the blocked read with EOF/EPIPE so the worker can unwind and
                # trigger the deferred close. One immediate sweep plus one
                # delayed re-sweep (covers a fresh connection opened between
                # the interrupt and the first sweep); a worker that still
                # doesn't settle keeps its resources until process exit rather
                # than risking a cross-thread FD release.
                _drain = getattr(child, "_drain_transports_after_abandonment", None)
                if callable(_drain):
                    def _drain_once(phase: str) -> None:
                        try:
                            _drain(reason=f"delegate_timeout_{phase}")
                        except Exception:
                            logger.debug(
                                "Timed-out child transport drain (%s) failed",
                                phase,
                                exc_info=True,
                            )

                    _drain_once("immediate")

                    def _drain_resweep() -> None:
                        if not _child_future.done():
                            _drain_once("resweep")

                    _resweep_timer = threading.Timer(5.0, _drain_resweep)
                    _resweep_timer.daemon = True
                    _resweep_timer.start()
            return _error_entry
        finally:
            # Shut down executor without waiting — if the child thread
            # is stuck on blocking I/O, wait=True would hang forever.
            _timeout_executor.shutdown(wait=False)

        _output_schema = getattr(child, "_delegate_output_schema", None)
        _schema_valid: Optional[bool] = None
        _schema_errors: List[str] = []
        _schema_retries = 0
        _terminal_result = result
        if isinstance(_output_schema, dict):
            assert child is not None
            _schema_repair = validate_and_repair_output(
                result,
                _output_schema,
                retry=lambda message: child.run_conversation(
                    user_message=message,
                    task_id=child_task_id,
                    stream_callback=_relay_child_text,
                ),
            )
            result = _schema_repair.aggregate_result
            _terminal_result = _schema_repair.terminal_result
            _schema_valid = _schema_repair.schema_valid
            _schema_errors = _schema_repair.schema_errors
            _schema_retries = _schema_repair.schema_retries

        # Linearization boundary for registry steering. From this point on the
        # child cannot consume another steer. Closing under the registry lock
        # either rejects a concurrent caller or drains every previously accepted
        # exact text into the result before callbacks/result assembly can run.
        _late_pending_steer = (
            _close_subagent_steering(_subagent_id, child) if _subagent_id else None
        )
        if _late_pending_steer:
            _existing_pending = result.get("pending_steer")
            result["pending_steer"] = (
                f"{_existing_pending}\n{_late_pending_steer}"
                if isinstance(_existing_pending, str) and _existing_pending
                else _late_pending_steer
            )

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, "_flush"):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - child_start, 2)

        summary = result.get("final_response") or ""
        api_calls = result.get("api_calls", 0)
        classification = classify_delegation_result(
            _terminal_result,
            aggregate_result=result,
            summary=summary,
            schema_requested=isinstance(_output_schema, dict),
            schema_valid=_schema_valid,
        )
        completed = classification.completed
        interrupted = classification.interrupted
        runtime_error = classification.runtime_error
        status = classification.status
        outcome = classification.outcome
        exit_reason = classification.exit_reason

        # Build tool trace from conversation messages (already in memory).
        # Uses tool_call_id to correctly pair parallel tool calls with results.
        tool_trace: list[Dict[str, Any]] = []
        trace_by_id: Dict[str, Dict[str, Any]] = {}
        messages = result.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        arguments = fn.get("arguments", "")
                        entry_t = {
                            "tool": fn.get("name", "unknown"),
                            "args_bytes": len(arguments),
                            "input_summary": _summarize_tool_arguments(arguments),
                        }
                        tool_trace.append(entry_t)
                        tc_id = tc.get("id")
                        if tc_id:
                            trace_by_id[tc_id] = entry_t
                elif msg.get("role") == "tool":
                    content = _stringify_tool_content(msg.get("content", ""))
                    is_error = _looks_like_error_output(content)
                    result_meta = {
                        "result_bytes": len(content),
                        "status": "error" if is_error else "ok",
                    }
                    # Match by tool_call_id for parallel calls
                    tc_id = msg.get("tool_call_id")
                    target = trace_by_id.get(tc_id) if tc_id else None
                    if target is not None:
                        target.update(result_meta)
                    elif tool_trace:
                        # Fallback for messages without tool_call_id
                        tool_trace[-1].update(result_meta)

        # Deterministic tool-error evidence derived from the actual tool
        # messages (never from child prose): how many tool results looked like
        # errors. Lets the parent gauge child reliability without trusting the
        # summary's self-report.
        tool_error_count = sum(
            1 for t in tool_trace if t.get("status") == "error"
        )
        terminal_attempt_tool_errors: Optional[int] = None
        if _terminal_result is not result:
            terminal_attempt_tool_errors = terminal_tool_error_count(
                _terminal_result,
                stringify_tool_content=_stringify_tool_content,
                looks_like_error_output=_looks_like_error_output,
            )

        # Extract token counts (safe for mock objects)
        _input_tokens = getattr(child, "session_prompt_tokens", 0)
        _output_tokens = getattr(child, "session_completion_tokens", 0)
        _model = getattr(child, "model", None)

        # --- result entry contract (see _run_single_child docstring) ---
        # status ∈ {completed, interrupted, failed}
        # exit_reason ∈ {completed, max_iterations, interrupted, error}
        # truncated is exactly (exit_reason == "max_iterations").
        entry: Dict[str, Any] = {
            "task_index": task_index,
            "status": status,
            # Logical task outcome, decoupled from lifecycle status. Parents
            # must verify returned evidence before treating unverified/partial
            # as success.
            "outcome": outcome,
            "summary": summary,
            "api_calls": api_calls,
            "duration_seconds": duration,
            "model": _model if isinstance(_model, str) else None,
            "exit_reason": exit_reason,
            # Explicit, parent-visible truncation flag. A subagent that
            # exhausts its per-child iteration budget still returns a summary,
            # so `status` stays "completed" (see above) — without this the
            # parent can't tell truncated-but-summarized from cleanly-finished
            # work except by parsing the summary prose. exit_reason is computed
            # authoritatively from the child's `completed` flag.
            "truncated": exit_reason == "max_iterations",
            # Runtime-derived evidence for parent verification (not child prose).
            "interrupted": bool(interrupted),
            "tool_error_count": tool_error_count,
            "tokens": {
                "input": (
                    _input_tokens if isinstance(_input_tokens, (int, float)) else 0
                ),
                "output": (
                    _output_tokens if isinstance(_output_tokens, (int, float)) else 0
                ),
            },
            "tool_trace": tool_trace,
            # Captured before the finally block calls child.close() so the
            # parent thread can fire subagent_stop with the correct role.
            # Stripped before the dict is serialised back to the model.
            "_child_role": getattr(child, "_delegate_role", None),
            # Captured before child.close() so the parent aggregator can fold
            # the child's total spend into the parent's session cost.  Port of
            # Kilo-Org/kilocode#9448 — previously the footer only reflected the
            # parent's direct API calls and under-counted subagent-heavy runs.
            # Stripped before the dict is serialised back to the model.
            "_child_cost_usd": (
                float(getattr(child, "session_estimated_cost_usd", 0.0) or 0.0)
                if isinstance(
                    getattr(child, "session_estimated_cost_usd", 0.0),
                    (int, float),
                )
                else 0.0
            ),
        }
        # Per-delegation spend, serialized back to the model alongside
        # tokens/api_calls so the parent can see what each delegation cost.
        # Mirrors _child_cost_usd (which is stripped pre-serialization and
        # only feeds the parent session rollup).
        # Inspired by: Perplexity Agent API result shape (idea-level).
        entry["cost_usd"] = round(entry["_child_cost_usd"], 6)
        _cost_status = getattr(child, "session_cost_status", None)
        entry["cost_status"] = (
            _cost_status if isinstance(_cost_status, str) and _cost_status
            else "unknown"
        )
        if terminal_attempt_tool_errors is not None:
            entry["terminal_tool_error_count"] = terminal_attempt_tool_errors
        if status == "failed":
            entry["error"] = (
                runtime_error
                or (summary if classification.usable_summary else None)
                or "Subagent did not produce a response."
            )
            # Preserve the child's machine-readable provider/runtime class.
            _failure_reason = result.get("failure_reason")
            if isinstance(_failure_reason, str) and _failure_reason:
                entry["failure_reason"] = _failure_reason

        apply_schema_evidence(
            entry,
            classification=classification,
            schema_requested=isinstance(_output_schema, dict),
            schema_valid=_schema_valid,
            schema_retries=_schema_retries,
            schema_errors=_schema_errors,
        )

        # A steer that queued after the child's final assistant turn had no
        # tool batch left to drain into.  The finalizer hands the undelivered
        # text back (turn_finalizer.py "pending_steer"); retain it here so the
        # parent sees the steer was MISSED rather than silently absorbed —
        # steer_subagent() returning True means "queued", and this is where a
        # queued-but-never-delivered steer gets named.
        _missed_steer = result.get("pending_steer")
        if isinstance(_missed_steer, str) and _missed_steer.strip():
            entry["missed_steer"] = _missed_steer
            _miss_note = (
                "[steer did not land — the subagent finished before it could "
                f"be delivered: {_missed_steer}]"
            )
            entry["summary"] = f"{summary}\n\n{_miss_note}" if summary else _miss_note

        # Cross-agent file-state reminder.  If this subagent wrote any
        # files the parent had already read, surface it so the parent
        # knows to re-read before editing — the scenario that motivated
        # the registry.  We check writes by ANY non-parent task_id (not
        # just this child's), which also covers transitive writes from
        # nested orchestrator→worker chains.
        try:
            if parent_task_id and parent_reads_snapshot:
                sibling_writes = file_state.writes_since(
                    parent_task_id, wall_start, parent_reads_snapshot
                )
                if sibling_writes:
                    mod_paths = sorted(
                        {p for paths in sibling_writes.values() for p in paths}
                    )
                    if mod_paths:
                        reminder = (
                            "\n\n[NOTE: subagent modified files the parent "
                            "previously read — re-read before editing: "
                            + ", ".join(mod_paths[:8])
                            + (
                                f" (+{len(mod_paths) - 8} more)"
                                if len(mod_paths) > 8
                                else ""
                            )
                            + "]"
                        )
                        if entry.get("summary"):
                            entry["summary"] = entry["summary"] + reminder
                        else:
                            entry["stale_paths"] = mod_paths
        except Exception:
            logger.debug("file_state sibling-write check failed", exc_info=True)

        # Per-branch observability payload: tokens, cost, files touched, and
        # a tail of tool-call results.  Fed into the TUI's overlay detail
        # pane + accordion rollups (features 1, 2, 4).  All fields are
        # optional — missing data degrades gracefully on the client.
        _cost_usd = getattr(child, "session_estimated_cost_usd", None)
        _reasoning_tokens = getattr(child, "session_reasoning_tokens", 0)
        try:
            _files_read = list(file_state.known_reads(child_task_id))[:40]
        except Exception:
            _files_read = []
        try:
            _files_written_map = file_state.writes_since(
                "", wall_start, []
            )  # all writes since wall_start
        except Exception:
            _files_written_map = {}
        _files_written = sorted(
            {
                p
                for tid, paths in _files_written_map.items()
                if tid == child_task_id
                for p in paths
            }
        )[:40]

        _output_tail = _extract_output_tail(result, max_entries=8, max_chars=600)

        complete_kwargs: Dict[str, Any] = {
            "preview": summary[:160] if summary else entry.get("error", ""),
            "status": status,
            "outcome": outcome,
            "exit_reason": exit_reason,
            "interrupted": bool(interrupted),
            "tool_error_count": tool_error_count,
            "duration_seconds": duration,
            "summary": summary[:500] if summary else entry.get("error", ""),
            "input_tokens": (
                int(_input_tokens) if isinstance(_input_tokens, (int, float)) else 0
            ),
            "output_tokens": (
                int(_output_tokens) if isinstance(_output_tokens, (int, float)) else 0
            ),
            "reasoning_tokens": (
                int(_reasoning_tokens)
                if isinstance(_reasoning_tokens, (int, float))
                else 0
            ),
            "api_calls": int(api_calls) if isinstance(api_calls, (int, float)) else 0,
            "files_read": _files_read,
            "files_written": _files_written,
            "output_tail": _output_tail,
        }
        complete_kwargs.update(schema_evidence_payload(entry))
        if _cost_usd is not None:
            try:
                complete_kwargs["cost_usd"] = float(_cost_usd)
            except (TypeError, ValueError):
                pass

        if child_progress_cb:
            try:
                child_progress_cb("subagent.complete", **complete_kwargs)
            except Exception as e:
                logger.debug("Progress callback completion failed: %s", e)

        _attach_worktree(entry)
        return entry

    except Exception as exc:
        _late_pending_steer = (
            _close_subagent_steering(_subagent_id, child) if _subagent_id else None
        )
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        if child_progress_cb:
            try:
                child_progress_cb(
                    "subagent.complete",
                    preview=str(exc),
                    status="failed",
                    outcome="failed",
                    exit_reason="error",
                    interrupted=False,
                    tool_error_count=0,
                    duration_seconds=duration,
                    summary=str(exc),
                )
            except Exception as e:
                logger.debug("Progress callback failure relay failed: %s", e)
        _error_entry = {
            "task_index": task_index,
            **failed_delegation_evidence(status="error"),
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
            "_child_role": getattr(child, "_delegate_role", None),
        }
        if _late_pending_steer:
            _error_entry["missed_steer"] = _late_pending_steer
            _error_entry["error"] += (
                " [steer did not land before the subagent stopped: "
                f"{_late_pending_steer}]"
            )
        # _attach_worktree defaults to a no-op when isolation never engaged.
        _attach_worktree(_error_entry)
        return _error_entry

    finally:
        # Stop the heartbeat thread so it doesn't keep touching parent activity
        # after the child has finished (or failed).  Guard the join: .start()
        # now lives inside the try block, so if it raised (OS thread
        # exhaustion) the thread was never started and Thread.join() would
        # raise RuntimeError.  ident is None until start() succeeds.
        _heartbeat_stop.set()
        if _heartbeat_thread.ident is not None:
            _heartbeat_thread.join(timeout=5)

        # Drop the TUI-facing registry entry.  Safe to call even if the
        # child was never registered (e.g. ID missing on test doubles).
        if _subagent_id:
            _unregister_subagent(_subagent_id, agent=child)

        if child_pool is not None and leased_cred_id is not None:
            try:
                child_pool.release_lease(leased_cred_id)
            except Exception as exc:
                logger.debug("Failed to release credential lease: %s", exc)

        # Restore the parent's tool names so the process-global is correct
        # for any subsequent execute_code calls or other consumers.
        import model_tools

        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        # Remove child from active tracking

        # Unregister child from interrupt propagation
        if hasattr(parent_agent, "_active_children"):
            try:
                lock = getattr(parent_agent, "_active_children_lock", None)
                if lock:
                    with lock:
                        parent_agent._active_children.remove(child)
                else:
                    parent_agent._active_children.remove(child)
            except (ValueError, UnboundLocalError) as e:
                logger.debug("Could not remove child from active_children: %s", e)

        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) so subagent subprocesses
        # don't outlive the delegation.
        if not _child_close_deferred:
            try:
                close = getattr(child, "close", None)
                if callable(close):
                    close()
            except Exception:
                logger.debug("Failed to close child agent after delegation")

        # The AIAgent turn boundary normally closes the child scope itself. This
        # fallback covers failures before that boundary starts, but must not pop
        # a scope while a timed-out child worker is still unwinding.
        try:
            from agent import relay_runtime

            runtime = relay_runtime.get_runtime(create=False)
            child_session_id = str(getattr(child, "session_id", "") or "")
            child_turn_is_active = relay_runtime.SESSION_COORDINATOR.has_active_turn(
                profile_key=relay_runtime.current_profile_key(),
                session_id=child_session_id,
            )
            if runtime is not None and child_session_id and not child_turn_is_active:
                runtime.unregister_subagent({"child_session_id": child_session_id})
        except Exception:
            logger.debug("Failed to close child Relay session after delegation")


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    role: Optional[str] = None,
    background: Optional[bool] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    action: Optional[str] = None,
    subagent_id: Optional[str] = None,
    message: Optional[str] = None,
    parent_agent=None,
    credentials_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks, or control
    already-running ones.

    Spawn modes (action='spawn' or omitted):
      - Single: provide goal (+ optional context and role)
      - Batch:  provide tasks array [{goal, context, role}, ...]

    Control modes (synchronous, never backgrounded):
      - action='list'  -> live children of this conversation's spawn tree
      - action='steer' -> queue course-correction text into a running child
                          (subagent_id + message)
      - action='stop'  -> interrupt a running child early (subagent_id)

    The 'role' parameter controls whether a child can further delegate:
    'leaf' (default) cannot; 'orchestrator' retains the delegation
    toolset and can spawn its own workers, bounded by
    delegation.max_spawn_depth.  Per-task role beats the top-level one.

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    # ── Control plane: list/steer/stop run synchronously and return here.
    # They never spawn, so they bypass the pause gate, depth limit, and the
    # async dispatch machinery entirely.
    normalized_action = (action or "").strip().lower()
    if normalized_action in _CONTROL_ACTIONS:
        return _handle_control_action(
            normalized_action, subagent_id, message, parent_agent
        )
    if normalized_action and normalized_action != "spawn":
        return tool_error(
            f"Unknown action '{action}'. Use spawn (default), list, steer, or stop."
        )

    # Operator-controlled kill switch — lets the TUI freeze new fan-out
    # when a runaway tree is detected, without interrupting already-running
    # children.  Cleared via the matching `delegation.pause` RPC.
    if is_spawn_paused():
        return tool_error(
            "Delegation spawning is paused. Clear the pause via the TUI "
            "(`p` in /agents) or the `delegation.pause` RPC before retrying."
        )

    # Normalise the top-level role once; per-task overrides re-normalise.
    top_role = _normalize_role(role)

    # Background (async) delegation now applies to BOTH single tasks and
    # batches. A batch is dispatched as ONE async unit: the whole fan-out runs
    # on the daemon executor, joins on every child (see _execute_and_aggregate
    # / dispatch_async_delegation_batch), and pushes a SINGLE completion event
    # carrying the consolidated per-task results. It re-enters the conversation
    # as one message once ALL children finish — the chat is not blocked while
    # they run.
    background = is_truthy_value(background, default=False) if background is not None else False

    # Depth limit — configurable via delegation.max_spawn_depth,
    # default 2 for parity with the original MAX_DEPTH constant.
    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return tool_error(
            f"Delegation depth limit reached (depth={depth}, "
            f"max_spawn_depth={max_spawn}). Raise "
            f"delegation.max_spawn_depth in config.yaml if deeper "
            f"nesting is required (no hard ceiling, but each level "
            f"multiplies API cost)."
        )

    # Load config
    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    # Model-supplied max_iterations is ignored — the config value is authoritative
    # so users get predictable budgets. The kwarg is retained for internal callers
    # and tests; a model-emitted value here would only shrink the budget and
    # surprise the user mid-run. Log and drop it if one slips through from a
    # cached tool schema or a stale provider.
    if max_iterations is not None and max_iterations != default_max_iter:
        logger.debug(
            "delegate_task: ignoring caller-supplied max_iterations=%s; "
            "using delegation.max_iterations=%s from config",
            max_iterations, default_max_iter,
        )
    effective_max_iter = default_max_iter

    # Resolve delegation credentials (provider:model pair).
    # When delegation.provider is configured, this resolves the full credential
    # bundle (base_url, api_key, api_mode) via the same runtime provider system
    # used by CLI/gateway startup.  When unconfigured, returns None values so
    # children inherit from the parent.
    #
    # ``credentials_cfg`` (internal callers only — never model-facing) is a
    # per-call override shaped like the delegation config section
    # ({provider, model, base_url, api_key, api_mode}); the /review engine
    # uses it to route its reviewer subagent onto ``auxiliary.review``
    # without touching the global delegation pin.
    try:
        creds = _resolve_delegation_credentials(
            credentials_cfg if credentials_cfg else cfg, parent_agent
        )
    except ValueError as exc:
        return tool_error(str(exc))

    # Normalize to task list
    max_children = _get_max_concurrent_children()
    recovered_tasks, tasks_error = _recover_tasks_from_json_string(tasks)
    if tasks_error:
        return tool_error(tasks_error)
    if recovered_tasks is not None:
        tasks = recovered_tasks

    # Small models frequently emit an empty tasks array ([]) alongside a
    # single goal. Treat that as "no batch" instead of letting the batch
    # quality gate below reject the goal-derived single task ("Batch mode
    # requires at least 2 tasks") — the intent is unambiguous.
    if isinstance(tasks, list) and not tasks:
        tasks = None

    if tasks and isinstance(tasks, list):
        if len(tasks) > max_children:
            return tool_error(
                f"Too many tasks: {len(tasks)} provided, but "
                f"max_concurrent_children is {max_children}. "
                f"Either reduce the task count, split into multiple "
                f"delegate_task calls, or increase "
                f"delegation.max_concurrent_children in config.yaml."
            )
        task_list = tasks
    elif goal and isinstance(goal, str) and goal.strip():
        single_task: Dict[str, Any] = {"goal": goal, "context": context, "role": top_role}
        if output_schema is not None:
            single_task["output_schema"] = output_schema
        task_list = [single_task]
    else:
        return tool_error(
            "No tasks provided. Pass tasks=[{goal: '...', context: '...'}, "
            "...] — one entry per subagent (a single task is a one-entry "
            "array)."
        )

    if not task_list:
        return tool_error("No tasks provided.")

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not isinstance(task, dict):
            return tool_error(
                f"Task {i} must be an object, got {type(task).__name__}."
            )
        if not task.get("goal", "").strip():
            return tool_error(f"Task {i} is missing a 'goal'.")

    # Batch-only quality gate: catch malformed fan-outs (placeholder goals,
    # unexpanded multi-word template markers, 1-task batches) before any
    # child is spawned.  The single-`goal` form is deliberately exempt —
    # short goals are valid there.  Duplicate goals are allowed (best-of-N).
    # Inspired by: MoonshotAI/kimi-code agent-swarm.md validation rules (MIT).
    if tasks is not None and isinstance(tasks, list):
        batch_error = _validate_batch_tasks(task_list)
        if batch_error:
            return tool_error(batch_error)

    # T1-24: coerce/validate optional per-task output_schema up front so a
    # malformed schema fails the whole call loudly instead of spawning
    # children that can never satisfy their contract. Runs AFTER the
    # existing goal checks; schema-less tasks resolve to None and take no
    # new code paths downstream.
    from tools.delegation_output_schema import coerce_output_schema

    task_schemas: List[Optional[Dict[str, Any]]] = []
    for i, task in enumerate(task_list):
        raw_schema = task.get("output_schema")
        if raw_schema is None and len(task_list) == 1 and output_schema is not None:
            raw_schema = output_schema
        coerced_schema, schema_err = coerce_output_schema(raw_schema)
        if schema_err:
            return tool_error(f"Task {i} output_schema invalid: {schema_err}")
        task_schemas.append(coerced_schema)

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    # Live transcripts: one pre-headered append-only log per task under
    # cache/delegation/live/<delegation_id>/task-<n>.log so the caller can
    # tail each child's operations while it runs (side-channel only — zero
    # effect on message content or prompt caching). Best-effort: on failure
    # live_paths is empty and delegation proceeds exactly as before.
    from tools.delegation_live_log import (
        create_live_transcripts,
        update_manifest_statuses,
        wrap_progress_callback,
    )

    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        task_list, context, model=creds.get("model"), provider=creds.get("provider")
    )
    # Announce the batch tag once so nested or concurrent completion lines
    # remain attributable to the delegation id returned by this dispatch.
    if n_tasks > 1 and live_deleg_id:
        header = f"🔀 [{format_batch_tag(live_deleg_id)}] delegating {n_tasks} tasks"
        header_spinner = getattr(parent_agent, "_delegate_spinner", None)
        if header_spinner:
            try:
                header_spinner.print_above(f"  {header}")
            except Exception:
                _emit_parent_console(parent_agent, f"  {header}")
        else:
            _emit_parent_console(parent_agent, f"  {header}")

    # Capture the ORIGINATING session's wake target BEFORE any child agent is
    # constructed: _build_child_agent() -> AIAgent() -> agent_init calls
    # set_current_session_id(child.session_id), which clobbers the
    # HERMES_SESSION_ID ContextVar and os.environ with the subagent's internal
    # id before the background-dispatch code below would read it. The
    # request-scoped chat_id binding (the raw X-Hermes-Session-Id on
    # api_server) is untouched by child construction, so read it here and
    # thread it through the dispatch.
    from tools.async_delegation import _current_origin_session_id

    _origin_wake_sid = _current_origin_session_id()
    try:
        from gateway.session_context import get_session_env

        _origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
    except Exception:
        _origin_ui_session_id = ""
    _origin_owner_transport, _origin_owner_session_record = (
        _capture_gateway_steer_authority(_origin_ui_session_id)
    )

    # Build all child agents on the main thread (thread-safe construction).
    # _build_child_preserving_parent_tools saves/restores the parent's
    # resolved tool names around each construction under a lock, so child
    # toolset resolution never leaks into the parent (shared with the plugin
    # subagent-lifecycle API).
    children = []
    for i, t in enumerate(task_list):
        # Per-task role beats top-level; normalise again so unknown
        # per-task values warn and degrade to leaf uniformly.
        effective_role = _normalize_role(t.get("role") or top_role)
        # T1-24: schema'd tasks get the contract appended to their context
        # so the child knows the expected output shape before it starts.
        _task_schema = task_schemas[i] if i < len(task_schemas) else None
        _child_context = t.get("context")
        if _task_schema is not None:
            from tools.delegation_output_schema import append_output_contract

            _child_context = append_output_contract(_child_context, _task_schema)
        try:
            child = _build_child_preserving_parent_tools(
                task_index=i,
                goal=t["goal"],
                context=_child_context,
                # Subagents always inherit the parent's toolsets; the model
                # cannot choose or narrow them (no model-facing toolsets arg).
                toolsets=None,
                model=creds["model"],
                max_iterations=effective_max_iter,
                task_count=n_tasks,
                parent_agent=parent_agent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_request_overrides=creds.get("request_overrides"),
                override_max_tokens=creds.get("max_output_tokens"),
                override_acp_command=creds.get("command"),
                override_acp_args=creds.get("args"),
                role=effective_role,
            )
        except ValueError as exc:
            # Explicit-pin preflight failures (e.g. pinned delegation.command
            # missing from PATH) refuse the spawn loudly (#80450).
            return tool_error(str(exc))
        # Attach the validated schema for the completion-side validation
        # hook in _run_single_child. Absent (None) on schema-less tasks.
        if _task_schema is not None:
            try:
                child._delegate_output_schema = _task_schema
            except Exception:
                logger.debug("Could not attach output schema to child %d", i)
        # Tee the child's progress events into its live transcript log.
        # wrap_progress_callback preserves the inner callback contract
        # (including the _flush attribute) and never lets writer failures
        # reach the agent loop. When no parent display exists the inner
        # callback is None and the wrapper still records events.
        _writer = live_writers[i] if i < len(live_writers) else None
        if _writer is not None:
            child.tool_progress_callback = wrap_progress_callback(
                getattr(child, "tool_progress_callback", None), _writer
            )
            child._live_transcript_path = str(_writer.path)
        # Delegation identity for the live registry + process-notification
        # attribution (child-started background processes report under it).
        if live_deleg_id:
            setattr(child, "_delegation_id", live_deleg_id)
            identity_ref = getattr(child, "_progress_identity_ref", None)
            if isinstance(identity_ref, dict):
                identity_ref["delegation_id"] = live_deleg_id
        children.append((i, t, child))

    def _execute_and_aggregate(*, honor_parent_interrupt: bool = True) -> dict:
        """Run all built children (1 or N), join on them, aggregate results,
        fire subagent_stop hooks + cost rollup, and return the combined result
        dict. Used by BOTH the synchronous path and the background runner. In
        the background case this whole function runs on the daemon executor, so
        the parent turn isn't blocked — but the batch still JOINS on itself
        here (all children must finish) before producing ONE consolidated
        results block. That is the contract: fan-out runs in the background,
        waits on each other, and returns together.
        """
        if n_tasks == 1:
            # Single task -- run directly (no thread pool overhead)
            _i, _t, child = children[0]
            result = _run_single_child(
                _i,
                _t["goal"],
                child,
                parent_agent,
                owner_session_id=_origin_ui_session_id or None,
                owner_transport=_origin_owner_transport,
                owner_session_record=_origin_owner_session_record,
            )
            results.append(result)
        else:
            # Batch -- run in parallel with per-task progress lines
            completed_count = 0
            spinner_ref = getattr(parent_agent, "_delegate_spinner", None)

            # Daemon workers (tools.daemon_pool): the `with` block still joins
            # normally, but if the parent is interrupted while a child is
            # wedged, the abandoned worker must not block interpreter exit.
            from tools.daemon_pool import DaemonThreadPoolExecutor
            with DaemonThreadPoolExecutor(max_workers=max_children) as executor:
                futures = {}
                for i, t, child in children:
                    child_context = contextvars.copy_context()
                    future = executor.submit(
                        child_context.run,
                        _run_single_child,
                        task_index=i,
                        goal=t["goal"],
                        child=child,
                        parent_agent=parent_agent,
                        owner_session_id=_origin_ui_session_id or None,
                        owner_transport=_origin_owner_transport,
                        owner_session_record=_origin_owner_session_record,
                    )
                    futures[future] = i

                # Poll futures with interrupt checking.  as_completed() blocks
                # until ALL futures finish — if a child agent gets stuck,
                # the parent blocks forever even after interrupt propagation.
                # Instead, use wait() with a short timeout so we can bail
                # when the parent is interrupted.
                # Map task_index -> child agent, so fabricated entries for
                # still-pending futures can carry the correct _delegate_role.
                _child_by_index = {i: child for (i, _, child) in children}

                pending = set(futures.keys())
                while pending:
                    if (
                        honor_parent_interrupt
                        and getattr(parent_agent, "_interrupt_requested", False) is True
                    ):
                        # Parent interrupted — collect whatever finished and
                        # abandon the rest.  Children already received the
                        # interrupt signal; we just can't wait forever.
                        for f in pending:
                            idx = futures[f]
                            if f.done():
                                try:
                                    entry = f.result()
                                except Exception as exc:
                                    entry = {
                                        "task_index": idx,
                                        **failed_delegation_evidence(status="error"),
                                        "summary": None,
                                        "error": str(exc),
                                        "api_calls": 0,
                                        "duration_seconds": 0,
                                        "_child_role": getattr(
                                            _child_by_index.get(idx), "_delegate_role", None
                                        ),
                                    }
                            else:
                                entry = {
                                    "task_index": idx,
                                    **failed_delegation_evidence(
                                        status="interrupted",
                                        exit_reason="interrupted",
                                        interrupted=True,
                                    ),
                                    "summary": None,
                                    "error": "Parent agent interrupted — child did not finish in time",
                                    "api_calls": 0,
                                    "duration_seconds": 0,
                                    "_child_role": getattr(
                                        _child_by_index.get(idx), "_delegate_role", None
                                    ),
                                }
                            results.append(entry)
                            completed_count += 1
                        break

                    from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED

                    done, pending = _cf_wait(
                        pending, timeout=0.5, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        try:
                            entry = future.result()
                        except Exception as exc:
                            idx = futures[future]
                            entry = {
                                "task_index": idx,
                                **failed_delegation_evidence(status="error"),
                                "summary": None,
                                "error": str(exc),
                                "api_calls": 0,
                                "duration_seconds": 0,
                                "_child_role": getattr(
                                    _child_by_index.get(idx), "_delegate_role", None
                                ),
                            }
                        results.append(entry)
                        completed_count += 1

                        # Print per-task completion line above the spinner
                        idx = entry["task_index"]
                        label = (
                            task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                        )
                        dur = entry.get("duration_seconds", 0)
                        status = entry.get("status", "?")
                        icon = delegation_batch_icon(entry)
                        remaining = n_tasks - completed_count
                        tag = format_batch_tag(live_deleg_id)
                        slot = (
                            f"{tag} · {idx + 1}/{n_tasks}"
                            if tag
                            else f"{idx + 1}/{n_tasks}"
                        )
                        completion_line = f"{icon} [{slot}] {label}  ({dur}s)"
                        # Failed/errored/timed-out children: say WHY on the
                        # same line, cleaned to one short human-readable
                        # fragment — a bare ✗ reads as "silently dropped".
                        if status in SUBAGENT_FAILURE_STATUSES:
                            _err_line = _clean_error_text(
                                entry.get("error"), max_chars=120
                            )
                            if _err_line:
                                completion_line += f" — {_err_line}"
                        if spinner_ref:
                            try:
                                spinner_ref.print_above(completion_line)
                            except Exception:
                                _emit_parent_console(parent_agent, f"  {completion_line}")
                        else:
                            _emit_parent_console(parent_agent, f"  {completion_line}")

                        # Update spinner text to show remaining count
                        if spinner_ref and remaining > 0:
                            try:
                                tag_prefix = f"[{tag}] " if tag else ""
                                spinner_ref.update_text(
                                    f"🔀 {tag_prefix}{remaining} "
                                    f"task{'s' if remaining != 1 else ''} remaining"
                                )
                            except Exception as e:
                                logger.debug("Spinner update_text failed: %s", e)

            # Sort by task_index so results match input order
            results.sort(key=lambda r: r["task_index"])

        # Cap subagent summaries against the parent's remaining context
        # headroom (split across the batch) before they enter the parent's
        # conversation. Full text is spilled to disk so nothing is lost.
        # Covers both the single-task and batch paths. See PR #9126.
        _finalize_child_results(results, task_list, children, parent_agent)

        total_duration = round(time.monotonic() - overall_start, 2)

        # Close out the live transcripts: terminal marker per task + manifest
        # status update. The files are retained (retention pruning happens on
        # future dispatches) — they double as the full-fidelity operational
        # record alongside the summary spill files.
        for entry in results:
            _idx = entry.get("task_index", -1)
            _w = (
                live_writers[_idx]
                if isinstance(_idx, int) and 0 <= _idx < len(live_writers)
                else None
            )
            if _w is not None:
                try:
                    _w.finalize(entry)
                except Exception:
                    logger.debug("Live transcript finalize failed", exc_info=True)
                if _idx < len(live_paths):
                    entry["live_transcript"] = live_paths[_idx]
        update_manifest_statuses(live_deleg_id, results)

        combined: Dict[str, Any] = {
            "results": results,
            "total_duration_seconds": total_duration,
        }
        if live_paths:
            combined["live_transcripts"] = list(live_paths)
        return combined

    # ----- Background dispatch: run the WHOLE batch as one async unit -----
    # When background is true, the entire fan-out runs on the daemon executor
    # via a single async delegation. _execute_and_aggregate() joins on every
    # child and produces ONE consolidated results block, which re-enters the
    # conversation as a single message when ALL children finish. The chat is
    # not blocked in the meantime. This is the contract: dispatch N subagents,
    # keep chatting, get the combined summaries back together at the end.
    if background:
        from tools.async_delegation import dispatch_async_delegation_batch
        from tools.approval import get_current_session_key

        # Finite sessions cannot route a detached subagent result back to the
        # agent after their turn/process ends. This includes stateless HTTP
        # requests (#10760) and one-shot Kanban workers (#63169). Fall back to
        # SYNCHRONOUS execution so the result returns in this same turn instead
        # of handing out a handle with no durable consumer. Mirrors the
        # pool-at-capacity inline fallback below.
        try:
            from gateway.session_context import async_delivery_supported
            _async_ok = async_delivery_supported()
        except Exception:
            _async_ok = True

        _wake_sid = ""
        if not _async_ok:
            # The adapter itself cannot push, but if a raw session id is
            # bound (the API server always binds one — see
            # ApiServerAdapter._bind_api_server_session), gateway.wake can
            # still reach the session by self-POSTing /v1/chat/completions
            # with that id in X-Hermes-Session-Id once the batch completes.
            # Only fall back to forced-sync execution when there is truly no
            # session id to wake. Uses the origin captured before child
            # construction (see _origin_wake_sid above) — reading
            # HERMES_SESSION_ID here would return the subagent's internal id.
            _wake_sid = _origin_wake_sid
            if _wake_sid:
                logger.info(
                    "delegate_task: async delivery unsupported on this "
                    "session, but a session id is bound (%s) — dispatching "
                    "in the background and waking the session via self-post "
                    "when it completes instead of forcing synchronous "
                    "execution.",
                    _wake_sid,
                )
                _async_ok = True

        if not _async_ok:
            logger.info(
                "delegate_task: async delivery unsupported on this session "
                "runtime; running the batch synchronously instead."
            )
            _sync_result = _execute_and_aggregate()
            if isinstance(_sync_result, dict):
                _sync_result["note"] = (
                    "background=true is not available in this session — it cannot "
                    "receive a detached subagent result after the turn ends (a "
                    "one-shot runner such as `hermes -z`, a cron job, a Kanban "
                    "worker, or a stateless HTTP endpoint). The subagent(s) ran "
                    "SYNCHRONOUSLY and the result is included above."
                )
            return json.dumps(_sync_result, ensure_ascii=False)

        _session_key = get_current_session_key(default="")
        try:
            from gateway.session_context import get_session_env

            _source = get_session_env("HERMES_SESSION_SOURCE", "")
            # Refresh from the same task-local source when available, but retain
            # the immutable value captured before child construction otherwise.
            _origin_ui_session_id = (
                get_session_env("HERMES_UI_SESSION_ID", "") or _origin_ui_session_id
            )
            # In desktop/TUI, the routable session key is the durable
            # AIAgent.session_id. Context compression can rotate that id during
            # the same turn before the TUI-side session dict is re-anchored;
            # if we capture the stale approval/session context key here, the
            # async completion becomes an orphan and any desktop poller may
            # consume it. Gateway chats are different: their session_key is the
            # platform conversation key (agent:main:...), so keep it there.
            if _source == "tui":
                _agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
                if _agent_session_id:
                    _session_key = _agent_session_id
        except Exception:
            _source = ""
        if not _session_key:
            # CLI (single-process) path: the approval contextvar is only bound
            # during gateway/TUI turns and HERMES_SESSION_KEY is not in the CLI
            # environment, so the key resolves empty here. Since #64240 the CLI
            # drains completions through a positive-ownership filter keyed on
            # the durable AIAgent.session_id — an empty session_key would fail
            # closed and the CLI could never claim its own completions, while
            # a restored foreign event with an empty key could leak into any
            # unfiltered consumer (#64484). Stamp the parent's durable session
            # id instead; compression rotations are handled on the drain side
            # via resolve_resume_session_id lineage resolution.
            _agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
            if _agent_session_id:
                _session_key = _agent_session_id
        _parent_session_id = getattr(parent_agent, "session_id", None)
        _child_agents = [c for (_, _, c) in children]

        # Detach every child from the parent's interrupt-propagation list — the
        # batch's lifecycle is owned by the async registry now, not the parent
        # turn. _build_child_agent attached them (correct for sync runs).
        if hasattr(parent_agent, "_active_children"):
            _ac_lock = getattr(parent_agent, "_active_children_lock", None)
            for _c in _child_agents:
                try:
                    if _ac_lock:
                        with _ac_lock:
                            parent_agent._active_children.remove(_c)
                    else:
                        parent_agent._active_children.remove(_c)
                except ValueError:
                    pass

        def _batch_runner():
            # This batch is detached from the foreground turn. Its lifecycle is
            # owned by the async registry and cancelled only via _batch_interrupt.
            return _execute_and_aggregate(honor_parent_interrupt=False)

        def _batch_interrupt():
            for _c in _child_agents:
                try:
                    interrupted = request_hard_interrupt(_c, "Async delegation cancelled")
                    if not interrupted and hasattr(_c, "_interrupt_requested"):
                        _c._interrupt_requested = True
                except Exception:
                    pass

        def _batch_progress():
            # Progress token for the async registry's stale monitor: the
            # combined (api_call_count, current_tool, last_activity_ts) of
            # every child. last_activity_ts is ticked by _touch_activity on
            # every streamed chunk ("receiving stream response"), every tool
            # transition, and every API-call start/completion — so a child
            # streaming a long response is alive even though api_call_count
            # only advances when the call completes (same liveness signal as
            # the compaction inactivity budget, PR #71508). A fully frozen
            # token past the stale threshold means the detached batch is
            # wedged (e.g. stuck inside the first model API call — #60203).
            # in_tool=True while ANY child is inside a tool so legitimately
            # slow tools get the higher staleness ceiling, mirroring the
            # sync-path heartbeat monitor.
            parts = []
            in_tool = False
            for _c in _child_agents:
                try:
                    _summary = _c.get_activity_summary()
                    _tool = _summary.get("current_tool")
                    parts.append(
                        (
                            _summary.get("api_call_count", 0),
                            _tool,
                            _summary.get("last_activity_ts"),
                        )
                    )
                    in_tool = in_tool or bool(_tool)
                except Exception:
                    parts.append(None)
            return tuple(parts), in_tool

        _goals = [t["goal"] for t in task_list]
        dispatch = dispatch_async_delegation_batch(
            goals=_goals,
            context=context,
            # Metadata for the completion block only; subagents inherit the
            # parent's toolsets (no model-facing toolsets arg).
            toolsets=None,
            role=top_role,
            model=creds["model"],
            session_key=_session_key,
            origin_ui_session_id=_origin_ui_session_id,
            origin_session_id=_wake_sid,
            parent_session_id=_parent_session_id,
            runner=_batch_runner,
            interrupt_fn=_batch_interrupt,
            max_async_children=_get_max_async_children(),
            # Reuse the live-transcript directory's id (when created) so the
            # returned delegation_id matches cache/delegation/live/<id>/.
            delegation_id=live_deleg_id,
            progress_fn=_batch_progress,
        )

        if dispatch.get("status") == "dispatched":
            n = len(_goals)
            note = (
                "Subagent is running in the background. You and the user can "
                "keep working; its full result re-enters the conversation as a "
                "new message when it finishes. Do not wait or poll — just "
                "continue."
                if n == 1 else
                f"{n} subagents are running in parallel in the background. You "
                f"and the user can keep working; they wait on each other and "
                f"their consolidated results re-enter the conversation as a "
                f"single message once ALL of them finish. Do not wait or poll "
                f"— just continue."
            )
            payload = {
                "status": "dispatched",
                "mode": "background",
                "count": n,
                "delegation_id": dispatch["delegation_id"],
                "goals": _goals,
                "note": note,
            }
            _sids = [
                getattr(_c, "_subagent_id", None) for _c in _child_agents
            ]
            if any(isinstance(s, str) and s for s in _sids):
                payload["subagent_ids"] = _sids
                payload["control_hint"] = (
                    "While a child runs you can orchestrate it live with this "
                    "same tool: delegate_task(action='list') to see live "
                    "children, action='steer' with subagent_id + message to "
                    "redirect one, action='stop' with subagent_id to end one "
                    "early."
                )
            if live_paths:
                payload["live_transcripts"] = list(live_paths)
                payload["live_transcripts_hint"] = (
                    "Each subagent streams a human-readable transcript of its "
                    "operations to the file listed above (append-only, one per "
                    "task). Read or `tail -f` these paths at any time to watch "
                    "a child work while it runs."
                )
            return json.dumps(payload, ensure_ascii=False)

        # Pool at capacity / schedule failure — children are still attached
        # (we detach above only on the parent list, but the async unit was
        # never accepted, so re-attaching isn't needed: we just run inline).
        logger.info(
            "delegate_task: async pool at capacity (%s); running the whole "
            "batch synchronously instead.",
            dispatch.get("error", "rejected"),
        )
        _cap_result = _execute_and_aggregate()
        if isinstance(_cap_result, dict):
            _cap_result["note"] = (
                "The background delegation pool was at capacity "
                "(delegation.max_concurrent_children), so the subagent(s) ran "
                "SYNCHRONOUSLY and the result is included above. Raise "
                "delegation.max_concurrent_children in config.yaml to allow "
                "more concurrent background delegations."
            )
        return json.dumps(_cap_result, ensure_ascii=False)

    # ----- Synchronous path -----
    return json.dumps(_execute_and_aggregate(), ensure_ascii=False)


registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        tasks=_strip_model_hidden_task_fields(args.get("tasks")),
        max_iterations=args.get("max_iterations"),
        role=args.get("role"),
        background=_model_background_value(args, kw.get("parent_agent")),
        output_schema=args.get("output_schema"),
        action=args.get("action"),
        subagent_id=args.get("subagent_id"),
        message=args.get("message"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
    dynamic_schema_overrides=_build_dynamic_schema_overrides,
)
