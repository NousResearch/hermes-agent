#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with a fresh conversation, their own task_id
(terminal session, file-ops cache), the parent's toolsets minus child-blocked
tools, and a focused system prompt built from goal + context. Single-task and
batch (parallel) modes; top-level model calls run in the background while
orchestrator children wait for their own workers. The parent only ever sees
the delegation call and the summary result, never the child's intermediate
tool calls or reasoning.
"""

import json
import logging
import time
import weakref
from typing import Any, Dict, List, Optional

from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb  # noqa: F401  (used via _ChildRun.await_child)
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# The delegate_tool_* siblings hold the pieces split out of this module; every name callers or patching tests reach as
# ``tools.delegate_tool.<name>`` is re-imported here. Mutable flag globals live only in their owning module.
from tools.delegate_tool_child_run import (  # noqa: F401
    _ChildRun, _attach_child, _build_result_entry, _dump_subagent_timeout_diagnostic, _fabricated_entry,
    _lease_child_credential, _merge_late_steer, _register_child, _start_heartbeat, _validate_child_output_schema,
)
from tools.delegate_tool_config import (  # noqa: F401
    _DEFAULT_MAX_CONCURRENT_CHILDREN, _get_child_timeout, _get_max_async_children, _get_max_concurrent_children,
    _get_max_spawn_depth, _get_orchestrator_enabled, _get_subagent_approval_callback, _get_worktree_isolation,
    _inherit_parent_capabilities, _load_config, _merge_request_overrides, _resolve_child_credential_pool,
    _resolve_child_runtime, _resolve_delegation_credentials,
    _subagent_auto_approve, _subagent_auto_deny,
)
from tools.delegate_tool_dispatch import _Batch, _announce_batch, _capture_origin, _run_batch
from tools.delegate_tool_progress import (  # noqa: F401
    DelegateEvent, SUBAGENT_FAILURE_STATUSES, _batch_prefix, _build_child_progress_callback,
    _build_child_system_prompt, _clean_error_text, _emit_parent_console, _quiet, _resolve_workspace_hint,
    _safe_progress, format_batch_tag, format_subagent_failure_line,
)
from tools.delegate_tool_registry import (  # noqa: F401
    _CONTROL_ACTIONS, _active_subagents, _active_subagents_lock, _capture_gateway_steer_authority,
    _handle_control_action, _is_descendant_of, _owns_subagent_record, _register_subagent, _unregister_subagent,
    get_subagent_attribution, interrupt_subagent, is_spawn_paused, list_active_subagents, set_spawn_paused,
    steer_subagent,
)
from tools.delegate_tool_tasks import _coerce_task_schemas, _normalize_task_list
from tools.delegate_tool_toolsets import (  # noqa: F401
    DELEGATE_BLOCKED_TOOLS, _apply_exact_tool_policy, _expand_parent_toolsets, _resolve_child_toolsets,
    _strip_blocked_tools,
)
from tools.delegate_tool_results import (  # noqa: F401
    _apply_summary_budget, _build_child_preserving_parent_tools, _run_child_lifecycle, _summarize_tool_arguments,
)

_ROLES = frozenset({"leaf", "orchestrator"})

# Nested delegation is granted by depth/role in _build_child_agent, never by the
# model naming toolsets (there is no model-facing toolsets argument).
def _normalize_role(r: Optional[str]) -> str:
    """'leaf' | 'orchestrator'; None/empty/unknown -> 'leaf' (unknown warns)."""
    r_norm = str(r).strip().lower() if r else "leaf"
    if r_norm not in _ROLES:
        logger.warning("Unknown delegate_task role=%r, coercing to 'leaf'", r)
        return "leaf"
    return r_norm

DEFAULT_MAX_ITERATIONS = 250
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
# Stale-heartbeat thresholds (cycles of _HEARTBEAT_INTERVAL with no progress). Progress = iteration, current_tool OR
# last_activity_ts advancing; an in-flight model wait refreshes last_activity_ts, so slow models are not "idle". Idle
# stays tight so a truly wedged child doesn't mask the gateway timeout; in-tool is much higher so legitimately long
# tools can finish.
_HEARTBEAT_STALE_CYCLES_IDLE = 15  # 450s idle between turns → stale
_HEARTBEAT_STALE_CYCLES_IN_TOOL = 40  # 1200s stuck on same tool → stale

def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _validate_spawn_admission(parent_agent, requested_children: int = 0) -> int:
    """Shared parent/plugin tree admission; returns the effective child batch cap."""
    if is_spawn_paused():
        raise ValueError(
            "Delegation spawning is paused. Clear the pause via the TUI (`p` in /agents) "
            "or the delegation.pause RPC before retrying.")
    if getattr(parent_agent, "_delegate_spawn_allowed", True) is False:
        raise ValueError("This worker may use delegate_task controls but its profile/depth does not permit spawning.")
    depth = int(getattr(parent_agent, "_delegate_depth", 0) or 0)
    max_spawn = _get_max_spawn_depth()
    parent_profile_spawn = getattr(parent_agent, "_delegate_profile_max_spawn_depth", None)
    if isinstance(parent_profile_spawn, int):
        if parent_profile_spawn <= 0:
            raise ValueError("This worker profile does not permit spawning descendants.")
        max_spawn = min(max_spawn, depth + parent_profile_spawn)
    if depth >= max_spawn:
        raise ValueError(
            f"Delegation depth limit reached (depth={depth}, max_spawn_depth={max_spawn}). Raise "
            "delegation.max_spawn_depth in config.yaml if deeper nesting is required.")

    max_children = _get_max_concurrent_children()
    parent_profile_concurrency = getattr(parent_agent, "_delegate_profile_max_concurrent_children", None)
    if isinstance(parent_profile_concurrency, int):
        if parent_profile_concurrency <= 0:
            raise ValueError("This worker profile does not permit spawning children.")
        max_children = min(max_children, parent_profile_concurrency)
    if requested_children > max_children:
        raise ValueError(
            f"Requested {requested_children} children exceeds this worker's concurrency limit {max_children}.")
    if requested_children:
        from agent.subagent_lifecycle import _owner_session_id_of, _persistent_store
        store = _persistent_store(parent_agent)
        owner = _owner_session_id_of(parent_agent)
        if store is not None and owner:
            store.recover_expired_runs(owner)
            active = store.active_run_count(owner)
            global_cap = _get_max_concurrent_children()
            if active + requested_children > global_cap:
                raise ValueError(
                    f"Durable worker concurrency limit would be exceeded: {active} active + "
                    f"{requested_children} requested > {global_cap}. Wait for a worker to finish before retrying.")
    return max_children


def _open_child_session_db(parent_agent) -> Any:
    """DEDICATED SessionDB handle for the child, or None: the parent's handle can be closed by its own lifecycle while
    a background child still flushes (transcript silently dropped). It MUST open the same db FILE as the parent's
    handle (non-launch profiles), else lineage / session_search break; released by the child's close() via
    _owns_session_db."""
    # Each child gets a DEDICATED SessionDB connection instead of the parent's live object. The parent's
    # handle is owned by the parent's lifecycle (cron run_job's finally block, gateway session end, /new)
    # and can be closed while a fire-and-forget background child is still flushing on a daemon thread —
    # every subsequent flush then hits the closed handle and the child's transcript is silently dropped
    # (#81267). It MUST point at the same database FILE as the parent's handle: parents can hold non-default
    # per-profile handles (tui_gateway opens SessionDB(db_path=<profile>/ state.db) for non-launch
    # profiles), and a bare SessionDB() would write the child's transcript into the launch profile's db,
    # breaking parent_session_id lineage and session_search. AsyncSessionDB wrappers (gateway) forward
    # .db_path via __getattr__, so this works through them.
    parent_session_db = getattr(parent_agent, "_session_db", None)
    if parent_session_db is None:
        return None
    with _quiet("subagent: failed to open dedicated SessionDB; child persistence disabled", exc_info=True):
        from hermes_state_registry import acquire
        _parent_db_path = getattr(parent_session_db, "db_path", None)
        return acquire(_parent_db_path) if _parent_db_path is not None else acquire()
    return None

def _apply_child_cache_ttl(child) -> None:
    """A delegated child never uses the 1h cache tier. The tier is priced for a person who steps
    away between turns (2x write vs 1.25x for 5m, #14971); a subagent calls every few seconds for
    minutes and is gone, so it pays the 2x on every tool result and never collects the retention.
    Caching itself stays exactly as configured (disabled stays disabled)."""
    if getattr(child, "_cache_ttl", None) == "1h":
        child._cache_ttl = "5m"

_CHILD_CAP_MIN = 16_000  # below this a child compresses on every call; treat as a config error


def _child_compression_cap_tokens(raw) -> "int | None":
    """Validated ``delegation.compression_threshold_tokens``: an int >= 16000, or None for "no cap".

    Unset / ``0`` / ``false`` / ``null`` mean no subagent-specific cap: the child compacts at the
    same ratio trigger as everyone else (0.50 x window). A bool ``true`` (YAML) would coerce to 1
    and make every call compress; a string like ``"200k"`` would silently read as no cap. Both are
    config errors: warn and treat as unset so a typo never changes compaction behaviour."""
    if raw is None or raw is False or raw == 0:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) < _CHILD_CAP_MIN:
        logger.warning(
            "delegation.compression_threshold_tokens=%r is not a token count >= %d; ignoring it "
            "(children keep the ratio trigger).", raw, _CHILD_CAP_MIN,
        )
        return None
    return int(raw)


def _apply_child_compression_cap(child, delegation_cfg: dict) -> None:
    """Optional absolute cap on the child's compaction trigger, ``delegation.compression_threshold_tokens``
    (lower of it and any global ``compression.threshold_tokens``). Off by default: a 1M-window child
    compacts at 500K like its parent. The compressor applies the cap on first window resolution, which
    happens after construction, so setting it here is exactly equivalent to config."""
    from agent.context_compressor import ContextCompressor

    cc = getattr(child, "context_compressor", None)
    if not isinstance(cc, ContextCompressor):
        return
    cap = _child_compression_cap_tokens((delegation_cfg or {}).get("compression_threshold_tokens"))
    if cap is None:
        return
    existing = cc.threshold_tokens_cap
    cc.threshold_tokens_cap = min(cap, existing) if isinstance(existing, int) and existing > 0 else cap
    if cc._threshold_tokens is not None:  # already resolved: re-clamp now
        cc._apply_threshold_tokens_cap()


def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    task_count: int,
    parent_agent,
    # Credential overrides from delegation config
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    override_request_overrides: Optional[Dict[str, Any]] = None,

    # ACP transport overrides from trusted delegation config.
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    override_fallback_model: Optional[List[Dict[str, Any]]] = None,
    override_reasoning_config: Optional[Dict[str, Any]] = None,
    override_supports_tools: Optional[bool] = None,
    requested_profile: Optional[str] = None,
    profile_instructions: Optional[str] = None,
    profile_tool_policy: Any = None,
    profile_workspace_context: Any = None,
    profile_route_receipt: Optional[Dict[str, Any]] = None,
    profile_execution_limits: Any = None,
    request_blocked_tools: Optional[List[str]] = None,
    frozen_system_prompt: Optional[str] = None,
    retained_child_depth: Optional[int] = None,
    retained_parent_worker_id: Optional[str] = None,
    worker_interface_contract: Optional[Dict[str, Any]] = None,
    # Configuration block that owns the selected provider/model route. Internal
    # callers such as /review pass auxiliary.review here so fallback policy is
    # not accidentally read from the general delegation block.
    routing_cfg: Optional[Dict[str, Any]] = None,
    # Legacy; accepted for wire compat but ignored (capability is depth-derived).
    role: str = "leaf",
):
    """Build (don't run) a child AIAgent on the main thread. override_* (from delegation config) replace parent
    inheritance so children can run on a different provider:model pair."""
    import uuid as _uuid
    from run_agent import AIAgent
    from agent.delegation_context import delegated_child_context
    # Role is depth-derived: a child may delegate iff the kill switch is on and
    # depth budget remains below max_spawn_depth. The `role` arg is ignored.
    child_depth = (
        retained_child_depth
        if isinstance(retained_child_depth, int) and retained_child_depth > 0
        else getattr(parent_agent, "_delegate_depth", 0) + 1
    )
    max_spawn = _get_max_spawn_depth()
    profile_spawn_depth = getattr(profile_execution_limits, "max_spawn_depth", None)
    profile_can_spawn = profile_spawn_depth is None or profile_spawn_depth > 0
    effective_role = (
        "orchestrator"
        if _get_orchestrator_enabled() and child_depth < max_spawn and profile_can_spawn
        else "leaf"
    )

    # One subagent_id shared by the progress callback, spawn_requested event and
    # the live registry; parent_id is set when THIS parent is itself a subagent.
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)

    # General delegation behavior (reasoning, compression, capabilities) stays
    # global. Only fallback policy follows the owner of a per-call route such
    # as auxiliary.review.
    delegation_cfg = _load_config()
    profile_allowed_toolsets = getattr(profile_tool_policy, "allowed_toolsets", None)
    requested_toolsets = toolsets if toolsets is not None else (
        list(profile_allowed_toolsets) if profile_allowed_toolsets is not None else None)
    child_toolsets, child_disabled_toolsets = _resolve_child_toolsets(parent_agent, requested_toolsets, effective_role)
    if override_supports_tools is False and child_toolsets:
        raise ValueError(
            f"Delegation profile '{requested_profile}' pins a model that does not support tool calling, "
            f"but the child would run with toolsets {sorted(child_toolsets)}")
    if profile_instructions and frozen_system_prompt is None:
        context = f"{context}\n\nWorker profile instructions:\n{profile_instructions}" if context else (
            f"Worker profile instructions:\n{profile_instructions}")
    child_prompt = frozen_system_prompt
    if child_prompt is None:
        child_prompt = _build_child_system_prompt(
            goal, context, workspace_path=_resolve_workspace_hint(parent_agent), role=effective_role,
            max_spawn_depth=max_spawn, child_depth=child_depth,
        )
    context_mode = getattr(profile_workspace_context, "mode", None)
    inherit_context = profile_workspace_context is not None and context_mode == "inherit"
    requested_context = getattr(profile_workspace_context, "include_context_files", None)
    requested_memory = getattr(profile_workspace_context, "include_memory", None)
    parent_allows_context = not bool(getattr(parent_agent, "skip_context_files", False))
    parent_allows_memory = not bool(getattr(parent_agent, "skip_memory", False))
    include_context_files = context_mode != "none" and parent_allows_context and (
        requested_context is True or (requested_context is None and inherit_context)
    )
    include_memory = context_mode != "none" and parent_allows_memory and (
        requested_memory is True or (requested_memory is None and inherit_context)
    )
    if frozen_system_prompt is not None:
        include_context_files = False
        include_memory = False
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Shared ref: session_id once the child exists, delegation_id once
    # delegate_task stamps it — both ride on every relayed event.
    child_session_ref: Dict[str, Any] = {}
    child_progress_cb = _build_child_progress_callback(
        task_index, goal, parent_agent, task_count, subagent_id=subagent_id, parent_id=parent_subagent_id,
        depth=max(0, child_depth - 1),  # 0 = first-level child for the UI
        model=model or getattr(parent_agent, "model", None), toolsets=child_toolsets, session_ref=child_session_ref,
    )
    rt = _resolve_child_runtime(
        parent_agent, delegation_cfg, parent_api_key, model=model, override_provider=override_provider,
        override_base_url=override_base_url, override_api_key=override_api_key, override_api_mode=override_api_mode,
        override_acp_command=override_acp_command,
        override_acp_args=override_acp_args,
        routing_cfg=routing_cfg, override_fallback_model=override_fallback_model,
        override_reasoning_config=override_reasoning_config,
    )
    if override_request_overrides is not None:
        # honored whenever set, incl. the inherit branch where
        # _resolve_delegation_credentials already merged OVER the parent's
        request_overrides = dict(override_request_overrides)
    else:
        request_overrides = {} if override_provider else dict(getattr(parent_agent, "request_overrides", {}) or {})
    parent_sid = getattr(parent_agent, "session_id", None)
    child_session_db = _open_child_session_db(parent_agent)
    with delegated_child_context():
        try:
            child = AIAgent(
                **rt, max_iterations=max_iterations, prefill_messages=getattr(parent_agent, "prefill_messages", None),
                enabled_toolsets=child_toolsets, disabled_toolsets=child_disabled_toolsets, quiet_mode=True,
                ephemeral_system_prompt=child_prompt, log_prefix=f"[subagent-{task_index}]", platform="subagent",
                # Legacy delegates remain isolated. Profiles may inherit a parent-enabled startup source,
                # but can never elevate above the parent or reload it after this frozen prompt is built.
                skip_context_files=not include_context_files, skip_memory=not include_memory, clarify_callback=None,
                thinking_callback=(
                    (lambda text: _safe_progress(child_progress_cb, "_thinking", text) if text else None)
                    if child_progress_cb else None
                ),
                session_db=child_session_db, parent_session_id=parent_sid, request_overrides=request_overrides,
                worker_interface_contract=worker_interface_contract,
                tool_progress_callback=child_progress_cb,
                iteration_budget=None,  # fresh budget per subagent
            )
        except BaseException:
            # No child close() will ever run: release the dedicated handle here.
            if child_session_db is not None:
                with _quiet(None):
                    from hermes_state_registry import release_or_close
                    release_or_close(child_session_db)
            raise
    parent_exact_tools = getattr(
        parent_agent, "_worker_effective_tool_names",
        getattr(parent_agent, "_executable_tool_names", getattr(parent_agent, "valid_tool_names", None)),
    )
    _apply_exact_tool_policy(
        child,
        profile_tool_policy,
        request_toolsets=toolsets,
        request_blocked_tools=request_blocked_tools,
        ancestor_allowed_tools=parent_exact_tools,
    )
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    _apply_child_cache_ttl(child)
    if child_session_db is not None:
        child._owns_session_db = True  # released by the child's close(), never by the parent
    # Ownership transfer for the dedicated handle: the child's close() must release it (nothing else holds a
    # reference), and no parent teardown can close it out from under a background child (#81267).
    child_session_ref["session_id"] = getattr(child, "session_id", "") or ""
    child._progress_identity_ref = child_session_ref
    child._delegate_depth, child._delegate_role = child_depth, effective_role  # post-degrade role
    child._subagent_id, child._parent_subagent_id = subagent_id, parent_subagent_id
    child._worker_profile = requested_profile
    child._worker_route_receipt = dict(profile_route_receipt or {})
    child._delegate_profile_max_spawn_depth = profile_spawn_depth
    child._delegate_profile_max_concurrent_children = getattr(
        profile_execution_limits, "max_concurrent_children", None)
    child._worker_timeout_seconds = getattr(profile_execution_limits, "timeout_seconds", None)
    child._worker_max_followups = getattr(profile_execution_limits, "max_followups", None)
    child._worker_max_tool_calls = getattr(profile_execution_limits, "max_tool_calls", None)
    child._delegate_spawn_allowed = effective_role == "orchestrator"
    parent_worker_id = (
        retained_parent_worker_id
        if retained_parent_worker_id is not None
        else getattr(parent_agent, "_worker_id", None)
    )
    child._delegate_parent_worker_id = (
        parent_worker_id if isinstance(parent_worker_id, str) and parent_worker_id else None
    )
    child._delegate_outbound_messages = []

    def _parent_message_sink(content: str) -> Dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message must be nonempty text")
        from agent.subagent_lifecycle import queue_worker_parent_message
        durable = queue_worker_parent_message(child, content)
        item = ({
            "message_id": durable["message_id"],
            "status": durable["status"],
            "content": content,
        } if durable is not None else {
            "message_id": "message-" + _uuid.uuid4().hex,
            "status": "DELIVERED_ON_COMPLETION",
            "content": content,
        })
        child._delegate_outbound_messages.append(item)
        return item

    child._delegate_parent_message_sink = _parent_message_sink
    _apply_child_compression_cap(child, delegation_cfg)
    # Ownership chain for action=list/steer/stop; weakref so a finished parent
    # can be collected while a detached child record lingers in the registry.
    try:
        child._delegate_parent_ref = weakref.ref(parent_agent)
    except TypeError:
        child._delegate_parent_ref = None  # non-weakref-able test doubles
    # Sidebar marker: subagent sessions stay out of session pickers even when a
    # parent delete orphans them (mirrors /branch's ``_branched_from``).
    if parent_sid and getattr(child, "_session_init_model_config", None) is not None:
        child._session_init_model_config["_delegate_from"] = parent_sid
    # Shared pool lets children rotate credentials on rate limits.
    child_pool = _resolve_child_credential_pool(rt["provider"], parent_agent, rt["base_url"])
    if child_pool is not None:
        child._credential_pool = child_pool

    _attach_child(parent_agent, child)  # interrupt propagation
    # spawn_requested now — the child may queue for seconds when the pool is
    # saturated — then the subagent_start lifecycle hook.
    _safe_progress(child_progress_cb, "subagent.spawn_requested", preview=goal)
    with _quiet("subagent_start hook invocation failed", exc_info=True):
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "subagent_start", parent_session_id=parent_sid,
            parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "", parent_subagent_id=parent_subagent_id,
            child_session_id=getattr(child, "session_id", None), child_subagent_id=subagent_id,
            child_role=effective_role, child_goal=goal,
        )
    return child


def _stable_worker_identity(child: Any) -> Optional[tuple[str, str]]:
    """Return only durable scalar worker identity, never proxy metadata."""
    worker_id = getattr(child, "_worker_id", None)
    run_id = getattr(child, "_worker_run_id", None)
    if isinstance(worker_id, str) and worker_id and isinstance(run_id, str) and run_id:
        return worker_id, run_id
    return None

def _run_single_child(
    task_index: int, goal: str, child=None, parent_agent=None, *, owner_session_id: Optional[str] = None,
    owner_transport: Any = None, owner_session_record: Any = None, **_kwargs,
) -> Dict[str, Any]:
    """Run a pre-built child agent (called from a worker thread) and return its result entry.

    Contract, derived from the child's structured completion fields:
      status      ∈ {completed, interrupted, failed} — a structured failure
                    (failed=True / non-empty error) or an invalid terminal state
                    is "failed" even when a summary exists.
      exit_reason ∈ {completed, max_iterations, interrupted, error} —
                    "max_iterations" only for genuine budget exhaustion
                    (completed=False with no failure fields), never for errors.
      truncated   == (exit_reason == "max_iterations").

    * ``"completed"``       — normal finish. See #97655.
    """
    child_progress_cb = getattr(child, "tool_progress_callback", None)
    child_pool, leased_cred_id = _lease_child_credential(child)
    # Heartbeat keeps the parent's _last_activity_ts moving so the gateway inactivity timeout doesn't fire while the
    # child works; it stops itself once the child looks stale (see _HEARTBEAT_STALE_CYCLES_*).
    heartbeat = _start_heartbeat(child, parent_agent, task_index)
    # TUI/RPC registry entry (kill/pause/status by subagent_id); None for test
    # doubles without a stable id. Unregistered in the finally block.
    _subagent_id = _register_child(
        child, parent_agent, goal, owner_session_id=owner_session_id, owner_transport=owner_transport,
        owner_session_record=owner_session_record,
    )
    run = _ChildRun(child, parent_agent, task_index, goal, _subagent_id, child_progress_cb)
    # Set when a timed-out Future still owns the child: closing it from this
    # thread before the worker settles races the conversation's finally path.
    _child_close_deferred = False
    try:
        heartbeat.start()
        _safe_progress(child_progress_cb, "subagent.start", preview=goal)
        run.seed_workspace()
        result, failure_entry, _child_close_deferred = run.await_child()
        if failure_entry is not None:
            from agent.subagent_lifecycle import SubagentLifecycleService
            worker_identity = _stable_worker_identity(child)
            if worker_identity:
                failure_entry["worker_id"], failure_entry["run_id"] = worker_identity
            SubagentLifecycleService.complete_adopted_child(child, failure_entry)
            return failure_entry

        child._worker_last_history = list(result.get("messages") or [])

        schema = _validate_child_output_schema(child, result, task_index, run.child_task_id, run.relay_text)
        _merge_late_steer(result, _subagent_id, child)
        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, "_flush"):
            with _quiet("Progress callback flush failed: %s"):
                child_progress_cb._flush()

        duration = run.elapsed()
        entry = _build_result_entry(child, result, task_index, duration, schema)
        run.append_sibling_write_reminder(entry)
        run.account_background_processes(entry)
        run.emit_complete(result, entry, duration)
        entry = run.attach_worktree(entry)
        worker_identity = _stable_worker_identity(child)
        if worker_identity:
            entry["worker_id"], entry["run_id"] = worker_identity
        outbound = list(getattr(child, "_delegate_outbound_messages", None) or [])
        if outbound:
            entry["messages_to_parent"] = outbound
        from agent.subagent_lifecycle import SubagentLifecycleService
        SubagentLifecycleService.complete_adopted_child(child, entry)
        return entry
    except Exception as exc:
        # Close steer acceptance before any completion callback (see _merge_late_steer).
        _late_pending_steer = run.close_steering()
        logging.exception(f"[subagent-{task_index}] failed")
        # Entry status "error" (contract), progress event status "failed" (UI vocabulary).
        entry = run.finish_failed(
            _fabricated_entry(task_index, "error", str(exc), child, run.elapsed()), _late_pending_steer,
            preview=str(exc), summary=str(exc), status="failed",
        )
        worker_identity = _stable_worker_identity(child)
        if worker_identity:
            entry["worker_id"], entry["run_id"] = worker_identity
        outbound = list(getattr(child, "_delegate_outbound_messages", None) or [])
        if outbound:
            entry["messages_to_parent"] = outbound
        from agent.subagent_lifecycle import SubagentLifecycleService
        SubagentLifecycleService.complete_adopted_child(child, entry)
        return entry
    finally:
        run.cleanup(heartbeat=heartbeat, child_pool=child_pool, leased_cred_id=leased_cred_id, close_deferred=_child_close_deferred)


def _profile_task_overrides(creds: Dict[str, Any]) -> Dict[str, Any]:
    overrides = {
        "override_provider": creds["provider"], "override_base_url": creds["base_url"],
        "override_api_key": creds["api_key"], "override_api_mode": creds["api_mode"],
        "override_request_overrides": creds.get("request_overrides"),
        "override_acp_command": creds.get("command"), "override_acp_args": creds.get("args"),
    }
    if creds.get("requested_profile"):
        overrides.update({
            "override_fallback_model": creds.get("fallback_model"),
            "override_reasoning_config": creds.get("reasoning_config"),
            "override_supports_tools": creds.get("supports_tools"),
            "requested_profile": creds.get("requested_profile"),
            "profile_instructions": creds.get("profile_instructions"),
            "profile_tool_policy": creds.get("tool_policy"),
            "profile_workspace_context": creds.get("workspace_context"),
            "profile_route_receipt": {
                key: creds.get(key) for key in (
                    "requested_profile", "requested_provider", "requested_model",
                    "requested_reasoning_effort", "resolved_provider", "resolved_model",
                    "resolved_reasoning_effort", "route_provenance", "normalization_events",
                    "transmitted_model", "provider_reported_model",
                )
            },
            "profile_execution_limits": creds.get("execution_limits"),
        })
    return overrides


def _resolve_task_credentials(
    task_list: List[Dict[str, Any]], creds: Dict[str, Any], routing_cfg: Dict[str, Any], parent_agent,
    *, top_profile: Optional[str] = None, top_provider: Optional[str] = None,
    top_model: Optional[str] = None, top_reasoning_effort: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Resolve every task route before building any child, preventing partial batch launch."""
    from agent.delegation_model_routing import select_profile_name
    from tools.delegate_tool_config import _resolve_profile_credentials
    resolved: List[Dict[str, Any]] = []
    for task in task_list:
        name = select_profile_name(task.get("profile") or task.get("model_profile"), top_profile, routing_cfg)
        provider = task.get("provider", top_provider)
        model = task.get("model", top_model)
        effort = task.get("reasoning_effort", top_reasoning_effort)
        if not name:
            if provider or model or effort:
                return [], "Per-task provider/model/reasoning_effort overrides require a delegation profile."
            resolved.append(creds)
            continue
        try:
            task_creds = _resolve_profile_credentials(
                name, routing_cfg, parent_agent,
                requested_provider=provider, requested_model=model, requested_reasoning_effort=effort,
            )
            from agent.delegation_model_routing import parse_profiles
            task_creds["profile_instructions"] = parse_profiles(routing_cfg)[name].instructions
            resolved.append(task_creds)
        except ValueError as exc:
            return [], str(exc)
    return resolved, None


def _build_children(
    task_list: List[Dict[str, Any]], task_schemas: List[Optional[Dict[str, Any]]], creds: Dict[str, Any], *,
    top_role: str, max_iterations: int, parent_agent, routing_cfg: Dict[str, Any],
    live_deleg_id: Optional[str], live_writers: list, task_creds: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[tuple], Optional[str]]:
    """Build every child on the main thread (construction is not thread-safe);
    ``(children, None)`` or ``([], error)`` on an explicit-pin preflight failure."""
    from tools.delegation_live_log import wrap_progress_callback
    from tools.delegation_output_schema import append_output_contract
    children = []
    from agent.subagent_lifecycle import SubagentLifecycleService
    lifecycle = SubagentLifecycleService(lambda: parent_agent)
    for i, t in enumerate(task_list):
        task_route = task_creds[i] if task_creds and i < len(task_creds) else creds
        overrides = _profile_task_overrides(task_route)
        overrides["routing_cfg"] = routing_cfg
        profile_max = task_route.get("max_iterations") if task_route.get("requested_profile") else None
        task_max_iterations = min(max_iterations, profile_max) if isinstance(profile_max, int) else max_iterations
        _task_schema = task_schemas[i] if i < len(task_schemas) else None
        _child_context = t.get("context")
        if _task_schema is not None:
            _child_context = append_output_contract(_child_context, _task_schema)
        try:
            child = _build_child_preserving_parent_tools(
                task_index=i, goal=t["goal"], context=_child_context,
                toolsets=None,  # always inherit the parent's toolsets
                model=task_route["model"], max_iterations=task_max_iterations, task_count=len(task_list),
                parent_agent=parent_agent, role=_normalize_role(t.get("role") or top_role), **overrides,
            )
        except ValueError as exc:
            return [], str(exc)
        try:
            lifecycle.adopt_delegate_child(
                child,
                goal=t["goal"],
                context=_child_context,
                profile=task_route.get("requested_profile"),
                creds=task_route,
                cfg=routing_cfg,
            )
        except ValueError as exc:
            return [], str(exc)
        if _task_schema is not None:
            with _quiet("Could not attach output schema to child %d", i):
                child._delegate_output_schema = _task_schema
        # Tee progress events into the live transcript (wrapper keeps the
        # _flush contract and swallows writer failures).
        _writer = live_writers[i] if i < len(live_writers) else None
        if _writer is not None:
            child.tool_progress_callback = wrap_progress_callback(getattr(child, "tool_progress_callback", None), _writer)
            child._live_transcript_path = str(_writer.path)
        if live_deleg_id:
            setattr(child, "_delegation_id", live_deleg_id)
            _ident_ref = getattr(child, "_progress_identity_ref", None)
            if isinstance(_ident_ref, dict):
                _ident_ref["delegation_id"] = live_deleg_id
        children.append((i, t, child))
    return children, None


def delegate_task(
    goal: Optional[str] = None, context: Optional[str] = None, tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None, role: Optional[str] = None, background: Optional[bool] = None,
    output_schema: Optional[Dict[str, Any]] = None, action: Optional[str] = None, subagent_id: Optional[str] = None,
    message: Optional[str] = None, parent_agent=None, credentials_cfg: Optional[Dict[str, Any]] = None,
    profile: Optional[str] = None, model_profile: Optional[str] = None,
    provider: Optional[str] = None, model: Optional[str] = None, reasoning_effort: Optional[str] = None,
    worker_id: Optional[str] = None, run_id: Optional[str] = None, timeout_seconds: Optional[float] = None,
    reconciliation_disposition: Optional[str] = None, reference: Optional[str] = None,
) -> str:
    """Spawn child agents (single ``goal`` or ``tasks=[...]`` batch) or control running ones. ``action``
    list/steer/stop run synchronously and bypass the pause gate, depth limit and async dispatch. ``role`` is legacy
    (per-task beats top-level; capability is depth-derived). Returns JSON with one results entry per task, or a
    dispatch handle when running in the background."""
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    normalized_action = (action or "").strip().lower()
    if normalized_action == "discover":
        from agent.delegation_model_routing import discover_workers
        catalog = discover_workers(_load_config(), parent_agent)
        if profile:
            selected = next((item for item in catalog["profiles"] if item["name"] == profile), None)
            if selected is None:
                return tool_error(f"Unknown delegation profile '{profile}'.")
            catalog = {**catalog, "profiles": [selected]}
        else:
            catalog = {
                **catalog,
                "profiles": [
                    {
                        key: item.get(key) for key in (
                            "name", "description", "provider", "model", "reasoning_effort",
                            "supported_reasoning_efforts", "supports_tools", "availability",
                            "freshness", "provenance",
                        )
                    }
                    for item in catalog["profiles"]
                ],
            }
        from agent.shared_discovery import discover_shared_references
        try:
            shared = discover_shared_references(parent_agent, reference=reference)
        except PermissionError as exc:
            return tool_error(str(exc))
        return json.dumps({"success": True, **catalog, **shared}, ensure_ascii=False)
    if normalized_action in {
        "status", "inspect", "completions", "message", "wait", "interrupt", "cancel", "reconcile", "resume", "ack",
    }:
        from agent.subagent_lifecycle import SubagentLifecycleError, SubagentLifecycleService
        if normalized_action == "message" and not worker_id:
            worker_id = getattr(parent_agent, "_delegate_parent_worker_id", None)
            if not worker_id:
                sink = getattr(parent_agent, "_delegate_parent_message_sink", None)
                if callable(sink):
                    try:
                        queued = sink(message or "")
                    except ValueError as exc:
                        return tool_error(str(exc))
                    return json.dumps({
                        "success": True,
                        "message_id": queued["message_id"],
                        "status": queued["status"],
                    }, ensure_ascii=False)
                return tool_error("action='message' requires worker_id when no parent worker is available.")
        try:
            payload = SubagentLifecycleService(lambda: parent_agent).control(
                normalized_action,
                worker_id=worker_id,
                run_id=run_id,
                message=message,
                timeout_seconds=timeout_seconds,
                reconciliation_disposition=reconciliation_disposition,
            )
        except (SubagentLifecycleError, PermissionError, ValueError) as exc:
            return tool_error(str(exc))
        return json.dumps({"success": True, **payload}, ensure_ascii=False)
    if normalized_action in _CONTROL_ACTIONS:
        return _handle_control_action(normalized_action, subagent_id, message, parent_agent)
    if normalized_action and normalized_action != "spawn":
        return tool_error(
            f"Unknown action '{action}'. Use spawn, discover, status, inspect, completions, message, wait, interrupt, cancel, "
            "reconcile, resume, ack, list, steer, or stop.")

    top_role = _normalize_role(role)
    # background applies to single tasks AND batches: a batch is ONE async unit
    # that joins on every child and re-enters as a single consolidated message.
    background = is_truthy_value(background, default=False) if background is not None else False

    try:
        max_children = _validate_spawn_admission(parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    # Caller-supplied max_iterations is ignored: the config value is authoritative
    # so budgets stay predictable (kwarg kept for internal callers/tests).
    if max_iterations is not None and max_iterations != default_max_iter:
        logger.debug(
            "delegate_task: ignoring caller-supplied max_iterations=%s; using delegation.max_iterations=%s from config",
            max_iterations, default_max_iter,
        )
    # credentials_cfg (internal callers only, e.g. /review → auxiliary.review) is
    # a per-call routing owner shaped like the delegation config section. Keep
    # the route and its fallback policy together through child construction.
    routing_cfg = credentials_cfg if credentials_cfg is not None else cfg
    selected_profile = str(profile or model_profile or "").strip() or None
    if profile and model_profile and str(profile).strip() != str(model_profile).strip():
        return tool_error("profile and legacy model_profile disagree; provide only one worker profile.")
    task_list, err = _normalize_task_list(goal, context, tasks, output_schema, top_role, max_children)
    if not err:
        task_schemas, err = _coerce_task_schemas(task_list, output_schema)
    if err:
        return tool_error(err)
    # A batch may select its profile per item without a top-level/default
    # profile.  Resolve those item routes first instead of rejecting or
    # initializing an unrelated inherited runtime.
    item_profile_only = not selected_profile and bool(task_list) and all(
        str(task.get("profile") or task.get("model_profile") or "").strip()
        for task in task_list
    )
    creds = None
    if not item_profile_only:
        try:
            route_overrides = (provider, model, reasoning_effort)
            if selected_profile is None and all(value is None for value in route_overrides):
                creds = _resolve_delegation_credentials(routing_cfg, parent_agent)
            elif all(value is None for value in route_overrides):
                creds = _resolve_delegation_credentials(routing_cfg, parent_agent, selected_profile)
            else:
                creds = _resolve_delegation_credentials(
                    routing_cfg, parent_agent, selected_profile,
                    requested_provider=provider, requested_model=model,
                    requested_reasoning_effort=reasoning_effort,
                )
        except ValueError as exc:
            # Explicit-pin preflight failures refuse the entire batch before
            # any child is built.
            return tool_error(str(exc))
    task_creds, err = _resolve_task_credentials(
        task_list,
        creds,
        routing_cfg,
        parent_agent,
        top_profile=selected_profile,
        top_provider=provider,
        top_model=model,
        top_reasoning_effort=reasoning_effort,
    )
    if err:
        return tool_error(err)
    if creds is None:
        creds = task_creds[0]

    try:
        _validate_spawn_admission(parent_agent, len(task_list))
    except ValueError as exc:
        return tool_error(str(exc))

    overall_start = time.monotonic()
    # Live transcripts: cache/delegation/live/<id>/task-<n>.log per task, a side channel with zero effect on message
    # content or prompt caching. Best-effort: on failure live_paths is empty and delegation proceeds.
    from tools.delegation_live_log import create_live_transcripts
    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        task_list, context, model=creds.get("model"), provider=creds.get("provider")
    )
    _announce_batch(parent_agent, len(task_list), live_deleg_id)
    origin = _capture_origin()

    children, err = _build_children(
        task_list, task_schemas, creds, top_role=top_role, max_iterations=default_max_iter, parent_agent=parent_agent,
        routing_cfg=routing_cfg, live_deleg_id=live_deleg_id, live_writers=live_writers, task_creds=task_creds,
    )
    if err:
        return tool_error(err)
    batch = _Batch(
        task_list, children, parent_agent, creds, context, top_role, max_children,
        live_deleg_id, live_writers, live_paths, *origin, overall_start,
    )
    return _run_batch(batch, background)


# ── OpenAI function-calling schema ──────────────────────────────────────────

def _build_top_level_description(*, independent_completions=None) -> str:
    """delegate_task description: ONLY guidance stated nowhere else in the schema
    (limits live in the 'tasks' parameter description, rebuilt per get_definitions())."""
    try:
        orchestration_available = _get_max_spawn_depth() >= 2 and _get_orchestrator_enabled()
    except Exception:
        orchestration_available = False
    # Mention recursion only where it's actually available. send_message is deliberately not named (gateway-internal
    # vocabulary); model_tools session-filters the list to tools the session has.
    if orchestration_available:
        restrictions_rule = (
            "- Children cannot call clarify, memory, or cronjob.\n"
            f"- Children can themselves delegate while depth remains (max_spawn_depth={_get_max_spawn_depth()}); the "
            "runtime derives this from depth automatically.\n"
        )
    else:
        restrictions_rule = "- Children cannot call delegate_task, clarify, memory, or cronjob.\n"
    from tools.delegate_tool_config import _get_independent_completions

    if independent_completions is None:
        independent_completions = _get_independent_completions()
    delivery = (
        "each ungrouped task / `group` returns on its own"
        if independent_completions else "one message per call"
    )
    return _DESCRIPTION_HEAD.format(delivery=delivery) + restrictions_rule + _DESCRIPTION_TAIL

_DESCRIPTION_HEAD = (
    "Spawn subagents in isolated contexts; each gets its own conversation, terminal session, and toolset, and only its "
    "final summary returns to you. Pass every task in `tasks` — one entry spawns one subagent, several run in parallel "
    "(limit in the tasks description).\n\n"
    "Sessions without a later-result consumer (including one-shot CLI and cron) join parallel children "
    "and return results in this tool call. "
    "Otherwise runs in the background: dispatch returns live transcript paths and results re-enter "
    "as a new message when subagents finish ({delivery}). Background results are delivered only "
    "BETWEEN your turns: finish whatever does not depend on them, then give a one-line status and END YOUR TURN. Never "
    "wait or poll on transcripts, artifact files, or CI for a child. "
    "While children run, `action` (list/steer/stop) controls them live — steer when a transcript shows a "
    "child drifting.\n\n"
    "USE FOR: reasoning-heavy subtasks, work that would flood your context with intermediate data, or independent "
    "parallel workstreams.\n"
    "DO NOT USE FOR (use these instead):\n"
    "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
    "- A single tool call -> call the tool directly\n"
    "- Tasks needing user interaction -> subagents cannot ask questions\n"
    "- Long-running scheduled or shell work -> cronjob or terminal(background=True, notify=True). Worker profiles "
    "persist ids, messages, history, and terminal receipts when the parent session has durable state; restart "
    "interrupts an expired run and requires an explicit resume.\n\n"
    "RULES:\n"
    "- Children know nothing of this conversation: pass everything needed via 'context', including any required "
    "output language, tone, or style (e.g. \"respond in Chinese\").\n"
    "- Child summaries are SELF-REPORTS, not verified facts: a child claiming \"uploaded successfully\" or "
    "\"file written\" may be wrong. For external side effects (uploads, remote writes, publishing), require a "
    "verifiable handle (URL, ID, absolute path) and verify it yourself before telling the user the operation "
    "succeeded.\n"
)
_DESCRIPTION_TAIL = (
    "- Children inherit the parent model unless pinned via delegation.provider / delegation.model in config.yaml."
)

def _build_tasks_param_description() -> str:
    """Compose the 'tasks' parameter description with current concurrency limit."""
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    return (
        f"The task(s), up to {max_children} in parallel for this user (set "
        "via delegation.max_concurrent_children). Each entry spawns one "
        "subagent with isolated context and terminal session; a single task "
        "is a one-entry array. Required when spawning."
    )

def _build_dynamic_schema_overrides() -> dict:
    """Per-call schema overrides (ToolEntry.dynamic_schema_overrides): every
    get_definitions() pass rewrites the descriptions to the user's actual limits."""
    from tools.delegate_tool_config import _get_independent_completions

    independent_completions = _get_independent_completions()
    overrides_params = {**DELEGATE_TASK_SCHEMA["parameters"]}
    # Copy properties so the static schema dict is never mutated.
    overrides_params["properties"] = {k: dict(v) for k, v in DELEGATE_TASK_SCHEMA["parameters"]["properties"].items()}
    overrides_params["properties"]["tasks"]["description"] = _build_tasks_param_description()

    if not independent_completions:
        tasks = overrides_params["properties"]["tasks"]
        tasks["items"] = {**tasks["items"], "properties": {
            k: v for k, v in tasks["items"]["properties"].items() if k != "group"
        }}

    return {
        "description": _build_top_level_description(independent_completions=independent_completions),
        "parameters": overrides_params,
    }

def _p(type_: str, description: str, **extra) -> dict:
    return {"type": type_, **extra, "description": description}

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    # description / tasks.description are placeholders: the real text is built per get_definitions() call by
    # _build_dynamic_schema_overrides() so the model sees the user's actual max_concurrent_children / max_spawn_depth.
    # Lazy (not at import) so cli.CLI_CONFIG isn't forced to load before the test conftest redirects HERMES_HOME.
    "description": (
        "Spawn one or more subagents in isolated contexts. "
        "Description is rebuilt at every get_definitions() call to reflect the user's current delegation limits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            # The handler also accepts the legacy single-goal shape (top-level `goal`/`context`/`output_schema`),
            # wrapped into a one-entry batch at dispatch, and a per-task `role` (legacy, ignored: capability is
            # depth-derived). Both unadvertised on purpose (old transcripts only); do not re-add. No maxItems — the
            # runtime limit (delegation.max_concurrent_children) is enforced with a clear error in delegate_task().
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": _p(
                            "string",
                            "What this subagent should accomplish. Be specific and self-contained — it knows "
                            "nothing about your conversation history.",
                        ),
                        "context": _p(
                            "string",
                            "Background THIS child needs: file paths, error messages, constraints. Each child "
                            "sees only its own context — repeat shared background in every task that needs it.",
                        ),
                        "profile": _p(
                            "string",
                            "Configured worker profile name. Use action='discover' to inspect the stable catalog.",
                        ),
                        "provider": _p(
                            "string",
                            "Optional provider override; accepted only with routing_mode=dynamic and an enabled route.",
                        ),
                        "model": _p(
                            "string",
                            "Optional model override; accepted only with routing_mode=dynamic and an enabled route.",
                        ),
                        "reasoning_effort": _p(
                            "string",
                            "Optional effort override; accepted only with routing_mode=dynamic and an enabled route.",
                        ),
                        "output_schema": _p(
                            "object",
                            "Optional JSON Schema this child's final answer must validate against (told to the "
                            "child up front; parent validates with one bounded correction retry; result gains "
                            "schema_valid, plus schema_errors on failure). Keep it forgiving — require only "
                            "fields you will read.",
                        ),
                        "group": _p(
                            "string",
                            "Optional result-delivery bucket within this call (only when delegation.independent_completions "
                            "is enabled; otherwise the whole call returns as one message). Tasks sharing a group return "
                            "together in ONE message; ungrouped tasks return individually as each finishes. This does not "
                            "order execution; if B needs A's output, dispatch B after A returns.",
                        ),
                    },
                    "required": ["goal"],
                },
                "description": "(rebuilt at get_definitions() time)",
            },
            "profile": _p(
                "string",
                "Profile for every spawned task unless that task sets its own profile; with action='discover', "
                "returns that profile's full policy and instructions.",
            ),
            "reference": _p(
                "string",
                "Optional typed reference returned by action='discover'. Re-resolves one worker, run, Bot, "
                "room, or Kanban task under current authority without returning transcripts.",
            ),
            "provider": _p(
                "string",
                "Optional batch route override, accepted only by a dynamic profile and its enabled routes.",
            ),
            "model": _p(
                "string",
                "Optional batch model override, accepted only by a dynamic profile and its enabled routes.",
            ),
            "reasoning_effort": _p(
                "string",
                "Optional batch effort override, accepted only by a dynamic profile and its enabled routes.",
            ),
            # `background` (bool) is also accepted — DEPRECATED, ignored: top-level
            # delegations always run in the background. Unadvertised; do not re-add.
            "action": _p(
                "string",
                "Default 'spawn'. 'discover' returns configured worker profiles. Durable worker actions are "
                "'status', 'inspect' (explicit visible conversation plus receipt metadata), 'completions', "
                "'message', 'wait', 'interrupt' (one run), 'cancel' (worker tree), 'reconcile', 'resume', and 'ack'. "
                "Legacy live controls are "
                "'list' = ids/goals/status/transcripts; 'steer' = queue "
                "course-correction text into one child (subagent_id + "
                "message) without stopping it; 'stop' = end one child "
                "early (subagent_id; partial result still returns). "
                "Control actions return immediately; goal/tasks are ignored unless spawning.",
                enum=[
                    "spawn", "discover", "status", "inspect", "completions", "message", "wait",
                    "interrupt", "cancel", "reconcile", "resume", "ack", "list", "steer", "stop",
                ],
            ),
            "subagent_id": _p("string", "Target for action='steer'/'stop' (ids from the spawn response or action='list')."),
            "worker_id": _p(
                "string",
                "Stable worker target. action='inspect' returns its retained user/assistant/tool conversation "
                "without hidden reasoning or system prompts.",
            ),
            "run_id": _p("string", "Optional exact run target for inspect/wait/cancel/resume/ack."),
            "reconciliation_disposition": _p(
                "string",
                "For action='reconcile': explicit decision about the prior uncertain effect. This records evidence "
                "and never replays the tool.",
                enum=["confirmed_applied", "confirmed_not_applied", "accepted_unknown_no_replay"],
            ),
            "timeout_seconds": _p("number", "For action='wait', block for at most 60 seconds; omit for a snapshot."),
            "message": _p(
                "string",
                "For action='steer': the course correction, appended to "
                "the child's next tool result mid-run. Be directive and specific.",
            ),
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

def _model_background_value(args: dict, parent_agent=None) -> bool:
    """Background flag for the MODEL-facing dispatch path (registry fallback). Top-level delegations always run in the
    background — the model does not choose — for single tasks and fan-out batches alike (one async unit, one
    consolidated result); an orchestrator subagent (depth > 0) is the exception since it needs its workers' results
    within its own turn. The live path is ``run_agent._dispatch_delegate_task``; this mirrors it for the rare case
    the intercept is bypassed. Direct Python callers keep the synchronous default."""
    return not getattr(parent_agent, "_delegate_depth", 0) > 0

_MODEL_HIDDEN_TASK_FIELDS = {"acp_command", "acp_args"}

def _strip_model_hidden_task_fields(tasks: Any) -> Any:
    """Drop trusted-config-only task fields from model-supplied tasks (same list object back when nothing changed)."""
    if not isinstance(tasks, list) or not any(isinstance(t, dict) and _MODEL_HIDDEN_TASK_FIELDS & t.keys() for t in tasks):
        return tasks
    return [{k: v for k, v in t.items() if k not in _MODEL_HIDDEN_TASK_FIELDS} if isinstance(t, dict) else t for t in tasks]


registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"), context=args.get("context"), tasks=_strip_model_hidden_task_fields(args.get("tasks")),
        max_iterations=args.get("max_iterations"), role=args.get("role"),
        background=_model_background_value(args, kw.get("parent_agent")), output_schema=args.get("output_schema"),
        action=args.get("action"), subagent_id=args.get("subagent_id"), message=args.get("message"),
        worker_id=args.get("worker_id"), run_id=args.get("run_id"), timeout_seconds=args.get("timeout_seconds"),
        reconciliation_disposition=args.get("reconciliation_disposition"), reference=args.get("reference"),
        profile=args.get("profile"), provider=args.get("provider"), model=args.get("model"),
        reasoning_effort=args.get("reasoning_effort"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
    dynamic_schema_overrides=_build_dynamic_schema_overrides,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from concurrent.futures import TimeoutError as FuturesTimeoutError  # noqa: F401,E402
import contextvars  # noqa: F401,E402
import enum  # noqa: F401,E402
import json  # noqa: F401,E402
import os  # noqa: F401,E402
import re  # noqa: F401,E402
import threading  # noqa: F401,E402
from urllib.parse import urlsplit  # noqa: F401,E402
from urllib.parse import urlunsplit  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_CHILD_TIMEOUT': ('tools.delegate_tool_config', 'DEFAULT_CHILD_TIMEOUT'),
    'DEFAULT_MAX_SUMMARY_CHARS': ('tools.delegate_tool_results', 'DEFAULT_MAX_SUMMARY_CHARS'),
    'DEFAULT_TOOLSETS': ('tools.delegate_tool_toolsets', 'DEFAULT_TOOLSETS'),
    'MAX_DEPTH': ('tools.delegate_tool_config', 'MAX_DEPTH'),
    'TOOLSETS': ('toolsets', 'TOOLSETS'),
    'base_url_hostname': ('utils', 'base_url_hostname'),
    'file_state': ('tools', 'file_state'),
    'request_hard_interrupt': ('agent.interrupt_compat', 'request_hard_interrupt'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
