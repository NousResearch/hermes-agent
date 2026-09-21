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

import logging
import time
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb  # noqa: F401  (used via _ChildRun.await_child)
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# The delegate_tool_* siblings hold the pieces split out of this module; every name callers or patching tests reach as
# ``tools.delegate_tool.<name>`` is re-imported here. Mutable flag globals live only in their owning module.
from tools.delegate_tool_child_run import (  # noqa: F401
    _ChildRun, _attach_child, _build_child_goal_message, _build_result_entry, _dump_subagent_timeout_diagnostic, _fabricated_entry,
    _apply_truncation_auto_continue, _get_continue_on_truncation_enabled,  # KENSEI CUSTOM: auto-continue
    _lease_child_credential, _merge_late_steer, _register_child, _start_heartbeat, _validate_child_output_schema,
)
from tools.delegate_tool_config import (  # noqa: F401
    _DEFAULT_MAX_CONCURRENT_CHILDREN, _get_child_timeout, _get_max_async_children, _get_max_concurrent_children,
    _get_max_spawn_depth, _get_oneshot_max_children, _get_orchestrator_enabled, _get_subagent_approval_callback, _get_worktree_isolation,
    _inherit_parent_capabilities, _load_config, _merge_request_overrides, _profile_model_cfg,  # KENSEI CUSTOM
    _registry_surface_chain,  # KENSEI CUSTOM: delegation fallback chain
    _resolve_child_credential_pool,
    _resolve_child_runtime, _resolve_delegation_credentials,
    _subagent_auto_approve, _subagent_auto_deny,
)
from tools.delegate_tool_dispatch import (  # noqa: F401
    _Batch, _announce_batch, _capture_origin, _run_batch,
    _split_child_budget,  # KENSEI CUSTOM: batch budget split (tests + facade callers)
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
    _MAX_TASK_IMAGES, _NESTED_DEFAULT_SCHEMA, _coerce_task_images, _coerce_task_schemas,  # KENSEI CUSTOM
    _normalize_task_images, _normalize_task_list,
)
from tools.delegate_tool_toolsets import (  # noqa: F401
    DELEGATE_BLOCKED_TOOLS, _expand_parent_toolsets, _resolve_child_toolsets, _strip_blocked_tools,
)
from tools.delegate_tool_results import (  # noqa: F401
    _apply_summary_budget, _build_child_preserving_parent_tools, _has_receipts,  # KENSEI CUSTOM: _has_receipts
    _run_child_lifecycle, _summarize_tool_arguments,
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


# ── KENSEI CUSTOM — profile home/scope helpers (ported) ──
def _profile_home_for_content(profile: Optional[str], profile_content: Optional[Dict[str, Any]]) -> Optional[Path]:
    """Resolve a target profile home without falling back to the parent's home silently."""
    if not profile:
        return None
    content = profile_content if isinstance(profile_content, dict) else {}
    explicit_home = content.get("hermes_home") or content.get("home")
    if explicit_home:
        return Path(explicit_home)
    try:
        from hermes_constants import get_hermes_home
        candidate = get_hermes_home() / "profiles" / str(profile)
        return candidate if candidate.is_dir() else None
    except Exception:
        return None


def _profile_scope_or_null(profile: Optional[str], profile_content: Optional[Dict[str, Any]]):
    """Return the shared profile scope for a real target profile, else a no-op for pre-resolved test fixtures."""
    from contextlib import nullcontext

    home = _profile_home_for_content(profile, profile_content)
    if not home:
        return nullcontext()
    from agent.profile_runtime_scope import profile_runtime_scope
    return profile_runtime_scope(home)


def _child_profile_scope(child):
    """Re-enter a child's target scope for its complete run, including cleanup.

    Test doubles (MagicMock/SimpleNamespace without a real profile home) fall
    through to a no-op: a MagicMock attribute is truthy but ``Path()`` of it
    would point at a ``MagicMock/...`` relative path and ``ensure_hermes_home``
    would scaffold profile dirs under the repo. Only real directory homes
    enter the scope.
    """
    from contextlib import nullcontext
    from pathlib import Path

    home = getattr(child, "_delegate_profile_home", None)
    if not home:
        return nullcontext()
    try:
        if not Path(home).is_dir():
            return nullcontext()
    except Exception:
        return nullcontext()
    from agent.profile_runtime_scope import profile_runtime_scope
    return profile_runtime_scope(home)
# ── END KENSEI CUSTOM ──


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

    # ACP transport overrides from trusted delegation config.
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    # Configuration block that owns the selected provider/model route. Internal
    # callers such as /review pass auxiliary.review here so fallback policy is
    # not accidentally read from the general delegation block.
    routing_cfg: Optional[Dict[str, Any]] = None,
    # Legacy; accepted for wire compat but ignored (capability is depth-derived).
    role: str = "leaf",
    # ── KENSEI CUSTOM — profile scoping (ported) ──
    profile: Optional[str] = None,
    profile_content: Optional[Dict[str, Any]] = None,
    isolate_profile_credentials: bool = False,
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
    effective_role = "orchestrator" if _get_orchestrator_enabled() and child_depth < max_spawn else "leaf"

    # One subagent_id shared by the progress callback, spawn_requested event and
    # the live registry; parent_id is set when THIS parent is itself a subagent.
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)

    # General delegation behavior (reasoning, compression, capabilities) stays
    # global. Only fallback policy follows the owner of a per-call route such
    # as auxiliary.review.
    delegation_cfg = _load_config()
    child_toolsets, child_disabled_toolsets = _resolve_child_toolsets(parent_agent, toolsets, effective_role)
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
        routing_cfg=routing_cfg,
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
    # ── KENSEI CUSTOM — stamp target profile for scope re-entry (ported) ──
    if profile:
        setattr(child, "_delegate_profile_name", profile)
        setattr(child, "_delegate_profile_home", _profile_home_for_content(profile, profile_content))
    # ── END KENSEI CUSTOM ──
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
    child_pool = _resolve_child_credential_pool(
        rt["provider"], parent_agent, rt["base_url"], effective_requested_provider=rt.get("requested_provider"),
    )
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
    """Run a pre-built child agent (called from a worker thread) inside its
    target profile runtime scope (KENSEI CUSTOM) and return its result entry.

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
    with _child_profile_scope(child):
        return _run_single_child_impl(
            task_index=task_index, goal=goal, child=child, parent_agent=parent_agent,
            owner_session_id=owner_session_id, owner_transport=owner_transport,
            owner_session_record=owner_session_record, **_kwargs,
        )


def _run_single_child_impl(
    task_index: int, goal: str, child=None, parent_agent=None, *, owner_session_id: Optional[str] = None,
    owner_transport: Any = None, owner_session_record: Any = None, **_kwargs,
) -> Dict[str, Any]:
    """Impl of ``_run_single_child`` — runs the full child lifecycle."""
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

        # KENSEI CUSTOM: one bounded continuation turn for a budget-cut child, before
        # schema validation so a continuation that completes the shape can satisfy it.
        _continued, _continuation_text = _apply_truncation_auto_continue(
            child, result, task_index, run.child_task_id, run.relay_text,
            _get_continue_on_truncation_enabled(),
        )
        schema = _validate_child_output_schema(child, result, task_index, run.child_task_id, run.relay_text)
        _merge_late_steer(result, _subagent_id, child)
        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, "_flush"):
            with _quiet("Progress callback flush failed: %s"):
                child_progress_cb._flush()

        duration = run.elapsed()
        entry = _build_result_entry(child, result, task_index, duration, schema,
                                    continued=_continued, continuation=_continuation_text)
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


def _build_children(
    task_list: List[Dict[str, Any]], task_schemas: List[Optional[Dict[str, Any]]], creds: Dict[str, Any], *,
    top_role: str, max_iterations: int, parent_agent, routing_cfg: Optional[Dict[str, Any]] = None,
    live_deleg_id: Optional[str], live_writers: list,
    task_images: Optional[List[Optional[List[str]]]] = None,
    # ── KENSEI CUSTOM — profile flow (ported) ──
    profile: Optional[str] = None, profile_content: Optional[Dict[str, Any]] = None,
    profile_explicit_pin: bool = False,
) -> tuple[List[tuple], Optional[str]]:
    """Build every child on the main thread (construction is not thread-safe);
    ``(children, None)`` or ``([], error)`` on an explicit-pin preflight failure."""
    from tools.delegation_live_log import wrap_progress_callback
    from tools.delegation_output_schema import append_output_contract

    # ── KENSEI CUSTOM — profile-based child config (ported) ──
    # Resolve the child profile ONCE per batch: config.yaml + SOUL.md from
    # ~/.hermes/profiles/<profile>/ (profile_content pre-resolved by the
    # caller wins — avoids re-parsing YAML per task).
    loaded_profile_cfg = profile_content
    if loaded_profile_cfg is None and profile:
        import yaml as _yaml
        import os as _os
        from hermes_constants import get_hermes_home
        _pdir = _os.fspath(get_hermes_home() / "profiles" / profile)
        _config_path = _os.path.join(_pdir, "config.yaml")
        _soul_path = _os.path.join(_pdir, "SOUL.md")
        if not _os.path.isdir(_pdir):
            return [], f"Profile '{profile}' not found at {_pdir}"
        loaded_profile_cfg = {"name": profile, "config": {}, "soul_md": None, "skills": [], "home": _pdir}
        try:
            with open(_config_path, encoding="utf-8") as _f:
                loaded_profile_cfg["config"] = _yaml.safe_load(_f) or {}
        except FileNotFoundError:
            logger.warning("Profile '%s' has no config.yaml", profile)
        if _os.path.isfile(_soul_path):
            with open(_soul_path, encoding="utf-8") as _f:
                loaded_profile_cfg["soul_md"] = _f.read()

    # Tier gate: Tier-3 profiles are dormant/specialized — they cannot be
    # delegated to without explicit Sahil approval and runtime proof.
    if loaded_profile_cfg:
        _pcfg = loaded_profile_cfg.get("config") or {}
        _tier = _pcfg.get("tier")
        if _tier is not None:
            try:
                _tier_int = int(_tier)
            except (ValueError, TypeError):
                pass  # Non-integer tier — let it pass
            else:
                if _tier_int == 3:
                    _pname = loaded_profile_cfg.get("name") or profile or "unknown"
                    return [], (
                        f"Profile '{_pname}' is tier 3 (dormant/specialized). "
                        "Tier 3 profiles require explicit Sahil approval and "
                        "runtime proof before delegation."
                    )
    # ── END KENSEI CUSTOM ──

    _profile_cfg = (loaded_profile_cfg or {}).get("config") or {}
    _profile_model_values = _profile_model_cfg(_profile_cfg)
    _profile_credentials = _profile_cfg.get("credentials")
    if not isinstance(_profile_credentials, dict):
        _profile_credentials = {}
    _profile_provider = _profile_cfg.get("provider") or _profile_model_values.get("provider") or _profile_credentials.get("provider")
    _profile_base_url = _profile_cfg.get("base_url") or _profile_model_values.get("base_url") or _profile_credentials.get("base_url")
    _profile_api_key = _profile_cfg.get("api_key") or _profile_model_values.get("api_key") or _profile_credentials.get("api_key")
    _profile_api_mode = _profile_cfg.get("api_mode") or _profile_credentials.get("api_mode")
    _profile_request_overrides = _profile_cfg.get("request_overrides")
    _profile_fallback_providers = _profile_cfg.get("fallback_providers")
    _profile_toolsets = (
        _profile_cfg.get("toolsets")
        or (loaded_profile_cfg or {}).get("toolsets")
        or (loaded_profile_cfg or {}).get("skills")
    )
    overrides = {
        "override_provider": creds["provider"], "override_base_url": creds["base_url"],
        "override_api_key": creds["api_key"], "override_api_mode": creds["api_mode"],
        "override_request_overrides": creds.get("request_overrides"),
        "override_acp_command": creds.get("command"),
        "override_acp_args": creds.get("args"),
        "routing_cfg": routing_cfg,
    }
    if loaded_profile_cfg and not profile_explicit_pin:
        if _profile_provider:
            overrides["override_provider"] = _profile_provider
        if _profile_base_url is not None:
            overrides["override_base_url"] = _profile_base_url
        if _profile_api_key is not None:
            overrides["override_api_key"] = _profile_api_key
        if _profile_api_mode is not None:
            overrides["override_api_mode"] = _profile_api_mode
        if isinstance(_profile_request_overrides, dict):
            overrides["override_request_overrides"] = dict(_profile_request_overrides)
        if isinstance(_profile_fallback_providers, (list, dict)):
            overrides["override_fallback_providers"] = _profile_fallback_providers
    _profile_home = _profile_home_for_content(profile, loaded_profile_cfg)
    children = []
    for i, t in enumerate(task_list):
        _task_schema = task_schemas[i] if i < len(task_schemas) else None
        _child_context = t.get("context")
        if _task_schema is not None:
            _child_context = append_output_contract(_child_context, _task_schema)
        # ── KENSEI CUSTOM — per-child profile application (ported) ──
        _child_model = creds["model"]
        _child_prompt_extra = None
        if loaded_profile_cfg:
            _pcfg = loaded_profile_cfg.get("config", {}) or {}
            _pmodel = _profile_model_cfg(_pcfg)
            # Profile model/provider/base_url apply as DEFAULTS only — explicit wins.
            if not _child_model and _pmodel.get("default"):
                _child_model = _pmodel["default"]
            # Inject profile SOUL.md into the child system prompt via context
            _soul = (loaded_profile_cfg.get("soul_md") or "").strip()
            if _soul:
                _child_prompt_extra = (
                    f"# Profile: {loaded_profile_cfg['name']}\n\n"
                    f"--- BEGIN PROFILE IDENTITY ---\n"
                    f"{_soul}\n"
                    f"--- END PROFILE IDENTITY ---\n\n"
                    f"--- DELEGATED TASK ---\n"
                    f"{_child_context or ''}"
                )
        _effective_context = _child_prompt_extra if _child_prompt_extra is not None else _child_context
        # ── END KENSEI CUSTOM ──
        try:
            with _profile_scope_or_null(profile, loaded_profile_cfg):
                child = _build_child_preserving_parent_tools(
                    task_index=i, goal=t["goal"], context=_effective_context,
                    toolsets=_profile_toolsets if loaded_profile_cfg else None,  # KENSEI: profile-scoped toolsets
                    model=_child_model, max_iterations=max_iterations, task_count=len(task_list),
                    parent_agent=parent_agent, role=_normalize_role(t.get("role") or top_role),
                    profile=profile, profile_content=loaded_profile_cfg,
                    isolate_profile_credentials=bool(profile and loaded_profile_cfg and not profile_explicit_pin),
                    **overrides,
                )
        except ValueError as exc:
            return [], str(exc)
        # ── KENSEI CUSTOM — stamp profile name for cycle detection ──
        if profile:
            setattr(child, "_delegate_profile_name", (loaded_profile_cfg or {}).get("name") or profile)
            if _profile_home is not None:
                setattr(child, "_delegate_profile_home", _profile_home)
        # ── END KENSEI CUSTOM ──
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
        children.append((i, t, child))
    return children, None


def _oneshot_spawn_budget(parent_agent: Any, requested: int) -> Optional[str]:
    """Charge *requested* children against the finite one-shot session's total (delegation.oneshot_max_children);
    the error text tells the model to do the work inline. Interactive and gateway sessions are never charged."""
    from agent.oneshot_footprint import is_single_query_session
    if not is_single_query_session():
        return None
    cap = _get_oneshot_max_children()
    if cap <= 0:
        return None
    spent = getattr(parent_agent, "_oneshot_children_spawned", 0)
    if spent + requested > cap:
        return (
            f"Delegation budget for this one-shot run is exhausted ({spent}/{cap} subagents used; "
            f"delegation.oneshot_max_children). Do the remaining work yourself in this session — reviewing "
            f"your own diff and running the tests inline is expected here, not a delegated review."
        )
    parent_agent._oneshot_children_spawned = spent + requested
    return None


def delegate_task(
    goal: Optional[str] = None, context: Optional[str] = None, tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None, role: Optional[str] = None, background: Optional[bool] = None,
    output_schema: Optional[Dict[str, Any]] = None, images: Optional[List[str]] = None, action: Optional[str] = None,
    subagent_id: Optional[str] = None, message: Optional[str] = None, parent_agent=None,
    credentials_cfg: Optional[Dict[str, Any]] = None,
    # ── KENSEI CUSTOM — verify/synthesis primitives ──
    verify: bool = False, verify_rubric: Optional[str] = None,
    synthesize: bool = False, synthesis_prompt: Optional[str] = None,
    profile: Optional[str] = None, profile_content: Optional[Dict[str, Any]] = None,
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
        # KENSEI CUSTOM: nested spawns without a contract get the minimal
        # default schema so validate + one-retry always enforces a shape.
        _nested_default = _NESTED_DEFAULT_SCHEMA if depth >= 1 else None
        task_schemas, err = _coerce_task_schemas(task_list, output_schema, _nested_default)
    if not err:
        task_images, err = _coerce_task_images(task_list, images)
    if err:
        return tool_error(err)
    err = _oneshot_spawn_budget(parent_agent, len(task_list))
    if err:
        return tool_error(err)

    # ── KENSEI CUSTOM — child budget split (ported) ──
    # _split_child_budget divides effective_max_iter across the batch
    # (configurable via delegation.max_iterations, default 50). Single-child
    # batches keep the full cap; multi-child batches split it so total
    # iterations across parent + subagents stay bounded by design, with a
    # floor of 1 per child.
    from tools.delegate_tool_dispatch import _split_child_budget
    default_max_iter = _split_child_budget(default_max_iter, len(task_list))
    # ── END KENSEI CUSTOM ──

    # ── KENSEI CUSTOM — profile validation (ported) ──
    # Profile validation must precede transcript creation and ancestry validation.
    # Explicit profile_content is already pre-resolved and bypasses filesystem validation.
    if profile and profile_content is None:
        from hermes_constants import get_hermes_home
        _profile_dir = get_hermes_home() / "profiles" / profile
        if not _profile_dir.is_dir():
            return tool_error(f"Profile '{profile}' not found at {_profile_dir}")
        _profile_config = _profile_dir / "config.yaml"
        if not _profile_config.is_file():
            return tool_error(f"Profile '{profile}' has no config.yaml at {_profile_config}")

    overall_start = time.monotonic()
    # Live transcripts: cache/delegation/live/<id>/task-<n>.log per task, a side channel with zero effect on message
    # content or prompt caching. Best-effort: on failure live_paths is empty and delegation proceeds.
    from tools.delegation_live_log import create_live_transcripts
    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        task_list, context, model=creds.get("model"), provider=creds.get("provider")
    )
    _announce_batch(parent_agent, len(task_list), live_deleg_id)
    origin = _capture_origin()

    # ── KENSEI CUSTOM — delegation cycle guard (ported) ──
    # Reject a spawn that would recurse into its own ancestor profile.
    try:
        from tools.delegate_tool_dispatch import _check_delegation_cycle
        _check_delegation_cycle(parent_agent, profile)
    except ValueError as _cycle_err:
        return tool_error(str(_cycle_err))
    # ── END KENSEI CUSTOM ──

    children, err = _build_children(
        task_list, task_schemas, creds, top_role=top_role, max_iterations=default_max_iter, parent_agent=parent_agent,
        routing_cfg=routing_cfg, live_deleg_id=live_deleg_id, live_writers=live_writers, task_images=task_images,
    )
    if err:
        return tool_error(err)
    batch = _Batch(
        task_list, children, parent_agent, creds, context, top_role, max_children,
        live_deleg_id, live_writers, live_paths, *origin, overall_start,
        # ── KENSEI CUSTOM — verify/synthesis primitive inputs ──
        verify=bool(verify), verify_rubric=verify_rubric,
        synthesize=bool(synthesize), synthesis_prompt=synthesis_prompt,
        profile=profile, profile_content=profile_content,
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
            # KENSEI CUSTOM (restored): the explicit single-goal surface is the
            # lead-routing contract's primary form. Upstream's tasks-only batch
            # interface sits alongside it for fan-out work. A per-task `role`
            # stays unadvertised (legacy, ignored: capability is depth-derived).
            "goal": _p(
                "string",
                "What the subagent should accomplish. Be specific and self-contained — it knows nothing "
                "about your conversation history.",
            ),
            "context": _p(
                "string",
                "Background THIS child needs: file paths, error messages, constraints. Each child sees only "
                "its own context — repeat shared background in every task that needs it.",
            ),
            "output_schema": _p(
                "object",
                "Optional JSON Schema this child's final answer must validate against (told to the child up "
                "front; parent validates with one bounded correction retry; result gains schema_valid, plus "
                "schema_errors on failure — the child's raw text is still returned as summary, never discarded). "
                "Keep it forgiving — require only fields you will read.",
            ),
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
                    },
                    "required": ["goal"],
                },
                "description": "(rebuilt at get_definitions() time)",
            },
            # ── KENSEI CUSTOM — profile routing + verify/synthesis primitive params ──
            "profile": {
                "type": "string",
                "description": (
                    "Profile name to load config, SOUL.md, and always_skills "
                    "from for the subagent. When set, the subagent uses the "
                    "profile's model, provider, toolsets, and identity instead "
                    "of inheriting the parent's. The profile must exist under "
                    "~/.hermes/profiles/<name>."
                ),
            },
            "synthesize": {
                "type": "boolean",
                "description": (
                    "KENSEI CUSTOM. After all children complete in batch mode, "
                    "spawn a synthesis agent to merge results. Only active when "
                    "delegation.synthesis_enabled=true and n_tasks > 1."
                ),
            },
            "synthesis_prompt": {
                "type": "string",
                "description": (
                    "Custom prompt for the synthesis agent when synthesize=true."
                ),
            },
            "verify": {
                "type": "boolean",
                "description": (
                    "KENSEI CUSTOM. After single-task child produces a finding, "
                    "spawn skeptic in CLEAN context to refute. Only active when "
                    "delegation.verify_enabled=true and n_tasks == 1."
                ),
            },
            "verify_rubric": {
                "type": "string",
                "description": (
                    "Custom rubric for the skeptic when verify=true."
                ),
            },
            # ── END KENSEI CUSTOM ──
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
        # ── KENSEI CUSTOM — profile routing + verify/synthesis primitives ──
        synthesize=args.get("synthesize"),
        synthesis_prompt=args.get("synthesis_prompt"),
        verify=args.get("verify"),
        verify_rubric=args.get("verify_rubric"),
        profile=args.get("profile"),
        # ── END KENSEI CUSTOM ──
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
from agent.memory_manager import sanitize_context  # noqa: F401 — facade compatibility seam
from tools.delegate_tool_dispatch import _check_delegation_cycle  # noqa: F401 — facade compatibility seam
from tools.delegate_tool_results import _extract_finding  # noqa: F401 — facade compatibility seam
from tools.delegate_tool_config import _get_synthesis_enabled, _get_verify_enabled  # noqa: F401 — facade compatibility seam
