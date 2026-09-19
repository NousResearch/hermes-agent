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

import functools
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
    _ChildRun, _attach_child, _build_child_goal_message, _build_result_entry, _dump_subagent_timeout_diagnostic, _fabricated_entry,
    _lease_child_credential, _merge_late_steer, _register_child, _start_heartbeat, _validate_child_output_schema,
)
from tools.delegate_tool_config import (  # noqa: F401
    _DEFAULT_MAX_CONCURRENT_CHILDREN, _get_child_timeout, _get_max_async_children, _get_max_concurrent_children,
    _get_max_spawn_depth, _get_orchestrator_enabled, _get_subagent_approval_callback, _get_worktree_isolation,
    _inherit_parent_capabilities, _load_config, _merge_request_overrides, _resolve_child_credential_pool,
    _resolve_child_runtime, _resolve_delegation_credentials,
    _subagent_auto_approve, _subagent_auto_deny,
)
from tools.delegate_tool_dispatch import (  # noqa: F401
    _Batch, _Descent, _Rung, _announce_batch, _capture_origin, _run_batch,
)
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
from tools.delegate_tool_tasks import (  # noqa: F401
    _MAX_TASK_IMAGES, _coerce_task_images, _coerce_task_schemas, _normalize_task_images, _normalize_task_list,
)
from tools.delegate_tool_toolsets import (  # noqa: F401
    DELEGATE_BLOCKED_TOOLS, _expand_parent_toolsets, _parent_toolsets, _resolve_child_toolsets,
    _strip_blocked_tools,
    _unknown_toolset_names,
)
from tools.delegate_tool_config import _get_provider_toolsets
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
    compacts where its parent does. The compressor applies the cap on first window resolution, which
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
    # Per-task reasoning pin from tasks[].reasoning_effort; None = fall back to delegation.reasoning_effort.
    override_reasoning_effort: Any = None,
    # True when ``toolsets`` is an explicit per-task pin: the child gets exactly that set, with no MCP re-add.
    exact_toolsets: bool = False,
    # Role and ``(enabled, disabled)`` toolsets already resolved by the caller. Both derivations read mutable
    # global state — the orchestrator kill switch, the depth budget, the MCP alias registry — so a caller that
    # vetted a child against one answer must hand that answer over rather than let it be computed again here.
    resolved_role: Optional[str] = None,
    resolved_toolsets: Optional[tuple[List[str], List[str]]] = None,

    # ACP transport overrides from trusted delegation config.
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
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
    child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
    max_spawn = _get_max_spawn_depth()
    effective_role = resolved_role if resolved_role is not None else (
        "orchestrator" if _get_orchestrator_enabled() and child_depth < max_spawn else "leaf"
    )

    # One subagent_id shared by the progress callback, spawn_requested event and
    # the live registry; parent_id is set when THIS parent is itself a subagent.
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)

    # General delegation behavior (reasoning, compression, capabilities) stays
    # global. Only fallback policy follows the owner of a per-call route such
    # as auxiliary.review.
    delegation_cfg = _load_config()
    child_toolsets, child_disabled_toolsets = resolved_toolsets if resolved_toolsets is not None else (
        _resolve_child_toolsets(parent_agent, toolsets, effective_role, exact=exact_toolsets))
    child_prompt = _build_child_system_prompt(
        goal, context, workspace_path=_resolve_workspace_hint(parent_agent), role=effective_role,
        max_spawn_depth=max_spawn, child_depth=child_depth,
    )
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
        routing_cfg=routing_cfg, override_reasoning_effort=override_reasoning_effort,
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
                skip_context_files=True, skip_memory=True, clarify_callback=None,
                thinking_callback=(
                    (lambda text: _safe_progress(child_progress_cb, "_thinking", text) if text else None)
                    if child_progress_cb else None
                ),
                session_db=child_session_db, parent_session_id=parent_sid, request_overrides=request_overrides,
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
    # child works; once the child looks stale (see _HEARTBEAT_STALE_CYCLES_*) it also ends await_child's wait.
    heartbeat = _start_heartbeat(child, parent_agent, task_index)
    # TUI/RPC registry entry (kill/pause/status by subagent_id); None for test
    # doubles without a stable id. Unregistered in the finally block.
    _subagent_id = _register_child(
        child, parent_agent, goal, owner_session_id=owner_session_id, owner_transport=owner_transport,
        owner_session_record=owner_session_record,
    )
    run = _ChildRun(child, parent_agent, task_index, goal, _subagent_id, child_progress_cb, heartbeat=heartbeat)
    # Set when a timed-out Future still owns the child: closing it from this
    # thread before the worker settles races the conversation's finally path.
    _child_close_deferred = False
    try:
        heartbeat.start()
        _safe_progress(child_progress_cb, "subagent.start", preview=goal)
        run.seed_workspace()
        result, failure_entry, _child_close_deferred = run.await_child()
        if failure_entry is not None:
            return failure_entry

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
        return run.attach_worktree(entry)
    except Exception as exc:
        # Close steer acceptance before any completion callback (see _merge_late_steer).
        _late_pending_steer = run.close_steering()
        logging.exception(f"[subagent-{task_index}] failed")
        # Entry status "error" (contract), progress event status "failed" (UI vocabulary).
        return run.finish_failed(
            _fabricated_entry(task_index, "error", str(exc), child, run.elapsed()), _late_pending_steer,
            preview=str(exc), summary=str(exc), status="failed",
        )
    finally:
        run.cleanup(heartbeat=heartbeat, child_pool=child_pool, leased_cred_id=leased_cred_id, close_deferred=_child_close_deferred)


def _task_route_pin(value: Any) -> Optional[str]:
    """Non-empty string pin after strip. Non-strings and blanks inherit the batch route."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _child_credential_overrides(creds_i: Dict[str, Any], routing_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """_build_child_agent credential kwargs for one already-resolved route."""
    return {
        "override_provider": creds_i["provider"], "override_base_url": creds_i["base_url"],
        "override_api_key": creds_i["api_key"], "override_api_mode": creds_i["api_mode"],
        "override_request_overrides": creds_i.get("request_overrides"),
        "override_acp_command": creds_i.get("command"),
        "override_acp_args": creds_i.get("args"),
        "routing_cfg": routing_cfg,
    }


def _resolve_task_credentials(
    task: Dict[str, Any], creds: Dict[str, Any], routing_cfg: Dict[str, Any], parent_agent,
    cache: Dict[tuple, tuple],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Batch creds, or a per-task overlay when provider/model is a non-empty string.

    Cached on the (provider, model) pin pair, not on the provider alone: api_mode and base_url are
    model-derived for some providers (OpenCode routes gpt-/grok- to codex_responses and qwen to
    anthropic_messages), so a provider-only key would hand the second model the first model's wire.
    """
    pin_provider = _task_route_pin(task.get("provider"))
    pin_model = _task_route_pin(task.get("model"))
    if not pin_provider and not pin_model:
        return creds, routing_cfg
    key = (pin_provider, pin_model)
    if key not in cache:
        overlay = dict(routing_cfg)
        if pin_provider:
            # Drop the batch endpoint AND its request_overrides: a leftover base_url would land the child
            # on the parent's box, and _runtime_provider_credentials merges explicit overrides OVER the
            # pinned provider's own, so an extra_body tuned for one provider would ride onto another.
            overlay["provider"], overlay["base_url"] = pin_provider, ""
            overlay["request_overrides"] = None
        if pin_model:
            overlay["model"] = pin_model
        cache[key] = (_resolve_delegation_credentials(overlay, parent_agent), overlay)
    return cache[key]


def _task_toolsets(
    task: Dict[str, Any], index: int, parent_universe: set
) -> tuple[Optional[List[str]], Optional[str]]:
    """``(toolsets, None)`` for this task's explicit pin, or ``(None, error)`` on a bad one.

    ``None`` inherits the parent's toolsets (the default). An explicit ``[]`` means NO toolsets — a pure
    reasoning child — and must never fall back to inherit. Unknown names abort the batch rather than being
    dropped: a child pinned to a narrow set because it runs on a restricted provider must not quietly keep
    the parent's tools because a name was misspelled.
    """
    raw = task.get("toolsets")
    if raw is None:
        return None, None
    if not isinstance(raw, list) or any(not isinstance(name, str) for name in raw):
        return None, f"Task {index} 'toolsets' must be a list of toolset names."
    unknown = _unknown_toolset_names(raw)
    if unknown:
        return None, (
            f"Task {index} requested unknown toolset(s): {', '.join(sorted(unknown))}. "
            f"Use known toolset names, [] for a child with no tools, or omit 'toolsets' to inherit the parent's."
        )
    # Distinct from the above: the name exists, the parent just cannot delegate it. _resolve_child_toolsets
    # drops these silently (it is shared with other callers), which would hand back a child quietly wider or
    # narrower than the caller asked for.
    missing = [name for name in raw if name not in parent_universe]
    if missing:
        return None, (
            f"Task {index} requested toolset(s) the parent does not have: {', '.join(sorted(missing))}. "
            f"A child can only narrow the parent's toolsets, never add to them."
        )
    return raw, None


def _task_reasoning_effort_error(task: Dict[str, Any], index: int) -> Optional[str]:
    """Abort message for an unrecognized per-task reasoning_effort, or None.

    Only an explicit per-task pin aborts: it is a promise the caller made about this one task, so silently
    inheriting a different thinking level is a wrong answer delivered confidently. delegation.reasoning_effort
    keeps its warn-and-inherit — it is a batch default, not a per-task claim. Absent/blank is not a pin.
    """
    raw = task.get("reasoning_effort")
    # Absent or blank is the only non-pin: 0, [], {} are not levels, and inheriting on them would
    # contradict this function's whole contract. Bools are pins — ``False`` disables thinking.
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    from agent.reasoning_effort import EFFORT_LADDER
    from hermes_constants import parse_reasoning_effort
    if parse_reasoning_effort(raw) is not None:
        return None
    return (
        f"Task {index} has an unrecognized reasoning_effort {raw!r}. "
        f"Use one of: {', '.join(EFFORT_LADDER)}, or omit it to inherit."
    )


def _task_pin_error(index: int, task: Dict[str, Any], exc: Exception) -> str:
    """Name the offending task so an aborted batch is diagnosable; unpinned tasks keep the bare message."""
    pin = _task_route_pin(task.get("provider")) or _task_route_pin(task.get("model"))
    return f"Task {index} (pinned to '{pin}'): {exc}" if pin else str(exc)


def _route_label(creds: Dict[str, Any]) -> str:
    """``provider/model`` of a resolved route, for the switch stamp. A blank half inherited the parent's."""
    return f"{creds.get('provider') or 'inherit'}/{creds.get('model') or 'inherit'}"


def _same_route(a: tuple, b: tuple) -> bool:
    """Route pins equal, case-insensitively; ``None`` (inherit) only matches ``None``."""
    return [x.casefold() if isinstance(x, str) else x for x in a] == [
        y.casefold() if isinstance(y, str) else y for y in b]


def _route_pins(raw: Any, where: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``({provider, model}, None)`` for one declared route, or ``(None, error)``.

    One shape for both places a route is written down — a task's explicit ``fallback`` and a rung of
    ``delegation.descent_order`` — so the two cannot disagree about what a route is. A bare string is the
    common case (a model on the task's own provider); the dict form exists so a ladder can end somewhere
    else entirely. ``where`` is the caller-facing name of the thing being validated.
    """
    if isinstance(raw, str):
        model = _task_route_pin(raw)
        if not model:
            return None, f"{where} is blank. Name the model this route runs."
        return {"provider": None, "model": model}, None
    if not isinstance(raw, dict):
        return None, (
            f"{where} must be a model name or an object like "
            f"{{\"model\": \"...\", \"provider\": \"...\"}}, got {type(raw).__name__}."
        )
    unsupported = sorted(set(raw) - {"model", "provider"})
    if unsupported:
        return None, (
            f"{where} has unsupported key(s): {', '.join(unsupported)}. A route is {{model, provider}} and "
            f"nothing else — toolsets, reasoning_effort, goal and context carry over from the task, so a "
            f"descended child differs from the one it replaces in where it runs and nothing more."
        )
    model = _task_route_pin(raw.get("model"))
    if not model:
        return None, (
            f"{where} is missing a 'model'. Name the model this route runs — an inherited model would "
            f"re-run the walled route under a different name."
        )
    return {"provider": _task_route_pin(raw.get("provider")), "model": model}, None


# A task opting out of descent entirely, spelled as a value rather than a key so "no fallback" and
# "deliberately no fallback" stay distinguishable: the first inherits the ladder, the second refuses it.
_FALLBACK_OPT_OUT = "none"


def _task_fallback_mode(task: Dict[str, Any], index: int) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """``(mode, pins, error)`` for this task's ``fallback`` key.

    ``mode`` is ``"ladder"`` (absent — walk delegation.descent_order), ``"none"`` (explicit opt-out, for
    work that must not silently move provider) or ``"explicit"`` (this route and no other, one rung).

    Fail closed, at validation time: a fallback dropped for being malformed would leave the caller
    believing a quota wall is covered when it is not, and they would only find out at the wall.
    """
    raw = task.get("fallback")
    if raw is None:
        return "ladder", None, None
    if isinstance(raw, str) and raw.strip().casefold() == _FALLBACK_OPT_OUT:
        return "none", None, None
    pins, err = _route_pins(raw, f"Task {index} 'fallback'")
    if err:
        return "explicit", None, (
            f"{err} Use a model name, {{model, provider}}, or '{_FALLBACK_OPT_OUT}' to opt this task out "
            f"of descent entirely."
        )
    # Deliberately NOT compared against the task's pins here: a fallback that names no provider inherits
    # the task's, and a task that pins no provider inherits the batch's, so two routes that look different
    # as pins can be the same route once resolved. The guard lives where both sides are resolved.
    return "explicit", pins, None


def _normalize_descent_order(routing_cfg: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """``delegation.descent_order`` as normalized route pins, or ``([], error)``.

    The order is an OPERATOR statement of which model to try next when a quota wall closes the current
    one — it exists nowhere else in the runtime, so an unparseable one aborts rather than degrading to no
    ladder: silently losing the descent is indistinguishable, from the caller's side, from every rung
    being walled. Duplicate models abort for the same reason position is looked up by model name.
    """
    raw = routing_cfg.get("descent_order") if isinstance(routing_cfg, dict) else None
    if raw is None or raw == []:
        return [], None
    if not isinstance(raw, list):
        return [], (
            f"delegation.descent_order must be a list of models, ordered best-first, got "
            f"{type(raw).__name__}."
        )
    order: List[Dict[str, Any]] = []
    for position, entry in enumerate(raw):
        pins, err = _route_pins(entry, f"delegation.descent_order[{position}]")
        if err:
            return [], err
        if any(_same_route((p["provider"], p["model"]), (pins["provider"], pins["model"])) for p in order):
            return [], (
                f"delegation.descent_order[{position}] repeats {pins['model']!r}. A rung is found by model "
                f"name, so a repeat has no single position to descend from."
            )
        order.append(pins)
    return order, None


def _descent_rungs(
    creds_i: Dict[str, Any], mode: str, explicit: Optional[Dict[str, Any]], order: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """``(rung_pins_below_this_child, halt_reason)``.

    Every branch that yields NO rungs names why, because "this task failed on a wall and did not descend"
    is only actionable with the reason attached — an operator cannot tell a deliberate opt-out from a
    model nobody added to the order from a ladder already at its bottom.
    """
    if mode == "none":
        return [], f"the task set fallback: '{_FALLBACK_OPT_OUT}'"
    if mode == "explicit":
        return [explicit], None
    if not order:
        return [], "delegation.descent_order is not configured"
    model = str(creds_i.get("model") or "").strip()
    provider = str(creds_i.get("provider") or "").strip()
    for position, rung in enumerate(order):
        # A rung that named no provider matches on model alone: it is a step within whatever provider the
        # task is already on, which is what an all-one-provider ladder wants.
        if not _same_route((rung["model"],), (model,)):
            continue
        if rung["provider"] and not _same_route((rung["provider"],), (provider,)):
            continue
        below = order[position + 1:]
        return (below, None) if below else (
            [], f"{provider or 'inherit'}/{model} is the last rung of delegation.descent_order")
    return [], (
        f"{model or 'the inherited model'} is not in delegation.descent_order, so there is no rung below "
        f"it to descend to")


def _cross_provider_toolset_error(index: int, provider: str, child_toolsets: List[str]) -> Optional[str]:
    """Abort message when a child leaving the parent's provider carries toolsets that provider was never
    granted. Checked on the FINAL resolved list, so it holds for an explicit pin, for inheritance, and for
    a batch-level ``delegation.provider`` alike.

    Fail closed: a provider the operator never declared receives nothing, not everything. Matching is on
    literal names, so a toolset spelled differently in config than in the parent's set is over-denied
    rather than under-denied — wrong in the recoverable direction.
    """
    allowed = _get_provider_toolsets(provider)
    denied = [name for name in child_toolsets if name not in set(allowed or ())]
    if not denied:
        # Nothing crossed, so there is nothing to grant. A toolless child on an undeclared provider is
        # the one case where "granted nothing" and "allowed" are the same answer.
        return None
    if allowed is None:
        return (
            f"Task {index} routes to provider '{provider}', which is not the parent's, carrying "
            f"toolset(s) {', '.join(sorted(denied))} — but no delegation.provider_toolsets entry declares "
            f"what that provider may receive, and an undeclared provider is granted nothing. Declare one "
            f"before routing tool-bearing work there."
        )
    return (
        f"Task {index} routes to provider '{provider}' carrying toolset(s) it is not granted: "
        f"{', '.join(sorted(denied))}. delegation.provider_toolsets['{provider}'] grants: "
        f"{', '.join(sorted(allowed)) or '(nothing)'}. Narrow the task's 'toolsets', or grant them "
        f"in config if that provider is trusted with them."
    )


def _build_children(
    task_list: List[Dict[str, Any]], task_schemas: List[Optional[Dict[str, Any]]], creds: Dict[str, Any], *,
    top_role: str, max_iterations: int, parent_agent, routing_cfg: Dict[str, Any],
    live_deleg_id: Optional[str], live_writers: list, task_images: Optional[List[Optional[List[str]]]] = None,
) -> tuple[List[tuple], Optional[str]]:
    """Build every child on the main thread (construction is not thread-safe);
    ``(children, None)`` or ``([], error)`` on an explicit-pin preflight failure."""
    from tools.delegation_live_log import wrap_progress_callback
    from tools.delegation_output_schema import append_output_contract
    children = []
    task_creds_cache: Dict[tuple, tuple] = {}
    # The same set _resolve_child_toolsets will intersect against, so "the parent does not have it" here and
    # a silent drop there cannot disagree.
    parent_universe = _expand_parent_toolsets(_parent_toolsets(parent_agent))
    parent_provider = str(getattr(parent_agent, "provider", None) or "").strip()
    # Same expression _build_child_agent uses, so the gate below vets the list the child is really built
    # with rather than a lookalike that could drift from it.
    _child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
    effective_role = (
        "orchestrator" if _get_orchestrator_enabled() and _child_depth < _get_max_spawn_depth() else "leaf"
    )
    # Preflight every distinct route before constructing anything: a bad pin must abort the batch without
    # leaving half-built children (each already holding a child session-db handle) behind. Everything the
    # construction loop needs is resolved HERE and carried in ``prepared`` — re-deriving any of it below
    # would mean the gate vetted one answer while the child was built from another, and the widest of the
    # two answers is the one that wins by default.
    # Parsed once for the whole batch: it is operator config, identical for every task, and a broken one
    # must abort before the first child is constructed rather than at the first wall.
    descent_order, order_err = _normalize_descent_order(routing_cfg)
    if order_err:
        return [], order_err
    prepared: List[tuple] = []
    for i, t in enumerate(task_list):
        pinned_toolsets, toolsets_err = _task_toolsets(t, i, parent_universe)
        if toolsets_err:
            return [], toolsets_err
        effort_err = _task_reasoning_effort_error(t, i)
        if effort_err:
            return [], effort_err
        try:
            creds_i, routing_i = _resolve_task_credentials(t, creds, routing_cfg, parent_agent, task_creds_cache)
        except ValueError as exc:
            return [], _task_pin_error(i, t, exc)
        # None = inherit the parent's toolsets; a pinned list is exact, [] meaning no tools at all.
        resolved_toolsets = _resolve_child_toolsets(
            parent_agent, pinned_toolsets, effective_role, exact=pinned_toolsets is not None)
        # A falsy resolved provider means the child stays on the parent's own route, so nothing crossed a
        # trust boundary and today's behaviour is preserved exactly.
        child_provider = str(creds_i.get("provider") or "").strip()
        if child_provider and child_provider.casefold() != parent_provider.casefold():
            provider_err = _cross_provider_toolset_error(i, child_provider, resolved_toolsets[0])
            if provider_err:
                return [], provider_err
        # EVERY rung below this child is vetted HERE, on the same two gates the primary pin passed
        # (resolvable credentials, and the cross-provider toolset grant against the SAME resolved toolsets
        # the child really carries). Vetting a rung only when the walk reaches it would turn a quota wall
        # into a second, differently-shaped failure three rungs down — and would let a task reach an
        # ungranted provider by descending onto it. Resolution is cached per (provider, model), so a shared
        # ladder costs one resolution per distinct rung for the whole batch, not one per task.
        mode, explicit_pins, fb_err = _task_fallback_mode(t, i)
        if fb_err:
            return [], fb_err
        rung_pins, halted = _descent_rungs(creds_i, mode, explicit_pins, descent_order)
        rungs = []
        for rung in rung_pins:
            try:
                creds_r, routing_r = _resolve_task_credentials(
                    rung, creds, routing_cfg, parent_agent, task_creds_cache)
            except ValueError as exc:
                return [], f"Task {i} descent rung {rung['model']!r}: {exc}"
            # Compared resolved, not as pins: a rung naming no provider inherits the child's, so
            # ``fallback: "<the model I am already on>"`` only looks like a different route until both
            # sides resolve. Descending onto the walled route buys nothing but a second wall.
            if _same_route((creds_r.get("provider"), creds_r.get("model")),
                           (creds_i.get("provider"), creds_i.get("model"))):
                return [], (
                    f"Task {i} 'fallback' resolves to {_route_label(creds_r)}, the same route the task "
                    f"already runs on. A fallback onto the walled route buys nothing but a second wall — "
                    f"point it at a different provider or model."
                )
            rung_provider = str(creds_r.get("provider") or "").strip()
            if rung_provider and rung_provider.casefold() != parent_provider.casefold():
                rung_err = _cross_provider_toolset_error(i, rung_provider, resolved_toolsets[0])
                if rung_err:
                    return [], f"Task {i} descent rung {rung['model']!r}: {rung_err}"
            rungs.append(_Rung(_route_label(creds_r), creds_r, routing_r, resolved_toolsets))
        prepared.append((creds_i, routing_i, resolved_toolsets, tuple(rungs), halted))
    def _construct(i: int, t: Dict[str, Any], creds_i, routing_i, resolved_toolsets) -> tuple:
        """``(child, None)`` for one task on one already-vetted route, or ``(None, error)``.

        Shared by the primary build below and the fallback retry (via ``_FallbackRoute.build``) so a
        retried child differs from the one it replaces in its route and nothing else.
        """
        _task_schema = task_schemas[i] if i < len(task_schemas) else None
        _child_context = t.get("context")
        if _task_schema is not None:
            _child_context = append_output_contract(_child_context, _task_schema)
        try:
            child = _build_child_preserving_parent_tools(
                task_index=i, goal=t["goal"], context=_child_context,
                toolsets=None, resolved_role=effective_role, resolved_toolsets=resolved_toolsets,
                model=creds_i["model"], max_iterations=max_iterations, task_count=len(task_list),
                parent_agent=parent_agent, role=_normalize_role(t.get("role") or top_role),
                override_reasoning_effort=t.get("reasoning_effort"),
                **_child_credential_overrides(creds_i, routing_i),
            )
        except ValueError as exc:
            # Fail loud: one bad pin aborts the whole batch, no child runs, no silent parent fallback.
            return None, _task_pin_error(i, t, exc)
        if _task_schema is not None:
            with _quiet("Could not attach output schema to child %d", i):
                child._delegate_output_schema = _task_schema
        # Validated per-task images; absent on image-less tasks, which keep the text-only goal turn.
        _t_images = task_images[i] if task_images and i < len(task_images) else None
        if _t_images:
            with _quiet("Could not attach images to child %d", i):
                child._delegate_images = _t_images
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
        return child, None

    for i, t in enumerate(task_list):
        creds_i, routing_i, resolved_toolsets, rungs, halted = prepared[i]
        child, err = _construct(i, t, creds_i, routing_i, resolved_toolsets)
        if err:
            return [], err
        # Parked on the child (as _delegate_output_schema/_delegate_images are) so the dispatch layer needs
        # no new plumbing. Attached even with zero rungs: a walled task that did NOT descend still owes the
        # caller the reason, and ``halted`` is where it comes from.
        child._delegate_descent = _Descent(
            label=_route_label(creds_i), rungs=rungs, halted=halted,
            build=functools.partial(_construct, i, t),
        )
        children.append((i, t, child))
    return children, None


def delegate_task(
    goal: Optional[str] = None, context: Optional[str] = None, tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None, role: Optional[str] = None, background: Optional[bool] = None,
    output_schema: Optional[Dict[str, Any]] = None, images: Optional[List[str]] = None, action: Optional[str] = None,
    subagent_id: Optional[str] = None, message: Optional[str] = None, parent_agent=None,
    credentials_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Spawn child agents (single ``goal`` or ``tasks=[...]`` batch) or control running ones. ``action``
    list/steer/stop run synchronously and bypass the pause gate, depth limit and async dispatch. ``role`` is legacy
    (per-task beats top-level; capability is depth-derived). Returns JSON with one results entry per task, or a
    dispatch handle when running in the background."""
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    normalized_action = (action or "").strip().lower()
    if normalized_action in _CONTROL_ACTIONS:
        return _handle_control_action(normalized_action, subagent_id, message, parent_agent)
    if normalized_action and normalized_action != "spawn":
        return tool_error(f"Unknown action '{action}'. Use spawn (default), list, steer, or stop.")

    # Operator kill switch (TUI / delegation.pause RPC): blocks NEW spawns only.
    if is_spawn_paused():
        return tool_error(
            "Delegation spawning is paused. Clear the pause via the TUI "
            "(`p` in /agents) or the `delegation.pause` RPC before retrying."
        )

    top_role = _normalize_role(role)
    # background applies to single tasks AND batches: a batch is ONE async unit
    # that joins on every child and re-enters as a single consolidated message.
    background = is_truthy_value(background, default=False) if background is not None else False

    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return tool_error(
            f"Delegation depth limit reached (depth={depth}, max_spawn_depth={max_spawn}). Raise "
            f"delegation.max_spawn_depth in config.yaml if deeper nesting is required (no hard ceiling, but each level "
            f"multiplies API cost)."
        )

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
    try:
        creds = _resolve_delegation_credentials(routing_cfg, parent_agent)
    except ValueError as exc:
        # Explicit-pin preflight failures (e.g. pinned delegation.command missing from PATH) refuse the
        # spawn loudly (#80450).
        return tool_error(str(exc))
    max_children = _get_max_concurrent_children()
    task_list, err = _normalize_task_list(goal, context, tasks, output_schema, top_role, max_children)
    if not err:
        task_schemas, err = _coerce_task_schemas(task_list, output_schema)
    if not err:
        task_images, err = _coerce_task_images(task_list, images)
    if err:
        return tool_error(err)

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
        routing_cfg=routing_cfg, live_deleg_id=live_deleg_id, live_writers=live_writers, task_images=task_images,
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
    "While children run, `action` (list/steer/stop) controls them live.\n\n"
    "USE FOR: reasoning-heavy subtasks, work that would flood your context, or independent parallel workstreams.\n"
    "DO NOT USE FOR (use these instead):\n"
    "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
    "- A single tool call -> call the tool directly\n"
    "- Tasks needing user interaction -> subagents cannot ask questions\n"
    "- Durable work that must survive this session -> cronjob or terminal(background=True, notify=True); /stop, /new, "
    "or process exit halts running subagents (whole tree); each returns an 'interrupted' completion with partial output.\n\n"
    "RULES:\n"
    "- Children know nothing of this conversation: pass everything needed via 'context', including any required "
    "output language, tone, or style (e.g. \"respond in Chinese\").\n"
    "- Child summaries are SELF-REPORTS, not verified facts: a child claiming \"uploaded successfully\" or "
    "\"file written\" may be wrong. For external side effects (uploads, remote writes, publishing), require a "
    "verifiable handle (URL, ID, absolute path) and verify it yourself before telling the user the operation "
    "succeeded.\n"
    "- Children cannot close tracked work: a child asked to close it returns findings instead; "
    "the parent applies the transition.\n"
)
_DESCRIPTION_TAIL = (
    "- Children inherit the parent model and toolsets unless a task sets provider/model/reasoning_effort/"
    "toolsets, or via delegation.provider / delegation.model in config.yaml."
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
                        "output_schema": _p(
                            "object",
                            "Optional JSON Schema this child's final answer must validate against (told to the "
                            "child up front; parent validates with one bounded correction retry; result gains "
                            "schema_valid, plus schema_errors on failure — the child's raw text is still returned "
                            "as summary, never discarded). Keep it forgiving — require only fields you will read.",
                        ),
                        "images": _p(
                            "array",
                            "Optional images this child must SEE (max 8): local file paths or http(s) URLs — e.g. a "
                            "screenshot the user sent, a design mock, a chart. Vision-capable children receive the "
                            "pixels on their first turn; non-vision children get path hints for vision_analyze. Text "
                            "files do NOT belong here — put paths in 'context' instead.",
                            items={"type": "string"},
                        ),
                        "group": _p(
                            "string",
                            "Optional result-delivery bucket within this call (only when delegation.independent_completions "
                            "is enabled; otherwise the whole call returns as one message). Tasks sharing a group return "
                            "together in ONE message; ungrouped tasks return individually as each finishes. This does not "
                            "order execution; if B needs A's output, dispatch B after A returns.",
                        ),
                        "provider": _p(
                            "string",
                            "Optional provider pin for THIS child only (named provider or a custom providers: table "
                            "entry). Resolved like CLI/config. An unresolvable provider aborts the WHOLE call "
                            "naming this task; omit or leave blank to inherit the batch/parent route. A child "
                            "routed to a provider other than the parent's may only carry toolsets the operator "
                            "granted that provider in delegation.provider_toolsets — an inheriting child keeps the "
                            "parent's MCP servers, so narrow it with 'toolsets'. Carrying an ungranted toolset "
                            "aborts the WHOLE call naming this task.",
                        ),
                        "model": _p(
                            "string",
                            "Optional model pin for THIS child only — mix cheap and strong models in one batch. "
                            "Omit or leave blank to inherit the batch/parent model.",
                        ),
                        "reasoning_effort": _p(
                            "string",
                            "Optional thinking budget for THIS child only: low, medium, high, xhigh, or "
                            "'none'/'false' to disable thinking. Omit to use delegation.reasoning_effort, "
                            "else the parent's level. An unrecognized value aborts the WHOLE call naming this task.",
                        ),
                        "fallback": _p(
                            "string",
                            "Optional override of where THIS child goes when its route hits a quota wall "
                            "(billing / rate limit / upstream rate limit). OMIT IT for the normal case: the "
                            "child then descends delegation.descent_order, moving one model down per wall "
                            "until the order runs out. Name a model ('qwen3.8-max') to use that ONE route "
                            "instead of the ladder, or 'none' to opt this task out of descent entirely — do "
                            "that for work that must not move provider. "
                            "Only a quota wall ever moves a child: every other failure is returned as a "
                            "failure and never retried, and an exhausted ladder fails loud. The goal, context, "
                            "toolsets and reasoning_effort carry over unchanged. A malformed value aborts the "
                            "WHOLE call naming this task.",
                        ),
                        "toolsets": _p(
                            "array",
                            "Optional EXACT toolset list for THIS child (e.g. ['web','file']), replacing the "
                            "parent's — use it to keep a child off tools it should not have, especially when "
                            "pinning it to a restricted provider. An empty list [] means NO tools at all (a pure "
                            "reasoning child); OMIT the key to inherit the parent's toolsets. A child can never "
                            "gain a toolset the parent lacks, and an unknown name aborts the WHOLE call.",
                            items={"type": "string"},
                        ),
                    },
                    "required": ["goal"],
                },
                "description": "(rebuilt at get_definitions() time)",
            },
            # `background` (bool) is also accepted — DEPRECATED, ignored: top-level
            # delegations always run in the background. Unadvertised; do not re-add.
            "action": _p(
                "string",
                "Default 'spawn'. Live control of running children: "
                "'list' = ids/goals/status/transcripts; 'steer' = queue "
                "course-correction text into one child (subagent_id + "
                "message) without stopping it; 'stop' = end one child "
                "early (subagent_id; partial result still returns). "
                "Control actions return immediately; goal/tasks are ignored unless spawning.",
                enum=["spawn", "list", "steer", "stop"],
            ),
            "subagent_id": _p("string", "Target for action='steer'/'stop' (ids from the spawn response or action='list')."),
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
        images=args.get("images"), action=args.get("action"), subagent_id=args.get("subagent_id"), message=args.get("message"),
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
