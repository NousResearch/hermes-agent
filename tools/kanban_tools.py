"""Kanban tools — structured tool-call surface for worker + orchestrator agents.

Registered only under the dispatcher (``HERMES_KANBAN_TASK`` set) or when the profile
enables the ``kanban`` toolset. Tools rather than ``hermes kanban`` shell-outs: they run
in the agent's process (reach ``kanban.db`` from a container/SSH terminal backend, no
shlex quoting of JSON metadata, structured-JSON failures). Humans use CLI/dashboard.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.goals import judge_goal
from tools.registry import no_cache_check_fn, registry, tool_error
from hermes_cli.config import cfg_get, load_config
from tools.kanban_tools_schemas import (
    KANBAN_ARCHIVE_SCHEMA,
    KANBAN_ATTACH_SCHEMA,
    KANBAN_ATTACH_URL_SCHEMA, KANBAN_ATTACHMENTS_SCHEMA, KANBAN_BLOCK_SCHEMA, KANBAN_COMMENT_SCHEMA,
    KANBAN_COMPLETE_SCHEMA, KANBAN_CREATE_SCHEMA, KANBAN_DECOMPOSE_SCHEMA, KANBAN_DISCOVER_SCHEMA,
    KANBAN_GRAPH_SCHEMA,
    KANBAN_HEARTBEAT_SCHEMA, KANBAN_LINK_SCHEMA,
    KANBAN_LIST_SCHEMA, KANBAN_PROMOTE_SCHEMA, KANBAN_REASSIGN_SCHEMA, KANBAN_REQUEST_CHANGES_SCHEMA,
    KANBAN_REQUEST_REVIEW_SCHEMA,
    KANBAN_SCHEDULE_SCHEMA, KANBAN_SHOW_SCHEMA, KANBAN_UNBLOCK_SCHEMA, KANBAN_UNLINK_SCHEMA)

logger = logging.getLogger(__name__)

KANBAN_LIST_DEFAULT_LIMIT = 50
KANBAN_LIST_MAX_LIMIT = 200


# --- Gating ---

def _profile_has_kanban_toolset() -> bool:
    from tools.kanban_toolset_context import kanban_toolset_requested

    requested = kanban_toolset_requested()
    if requested:
        return True
    try:
        config = load_config()
        # Preserve the legacy profile-wide opt-in for callers using bundles.
        if "kanban" in (config.get("toolsets") or []):
            return True
        if requested is not None:
            # Never borrow another platform's opt-in during schema assembly.
            return False
        # Offer-time skill discovery has no platform selection. A saved opt-in
        # makes the playbook relevant; actual schemas still use the scope above.
        from hermes_cli.tools_config import _get_platform_tools

        platforms = config.get("platform_toolsets") or {}
        return any(
            "kanban" in _get_platform_tools(config, platform, include_default_mcp_servers=False)
            for platform, names in platforms.items() if isinstance(names, list)
        )
    except Exception:
        return False


def _delegation_ctx(predicate: str, default: bool) -> bool:
    """``agent.delegation_context.<predicate>()``; ``default`` when it cannot be evaluated."""
    try:
        from agent import delegation_context
        return getattr(delegation_context, predicate)()
    except Exception:
        return default


def _is_delegated_child_context() -> bool:
    return _delegation_ctx("is_delegated_child_context", False)


def _is_dispatcher_owned_worker() -> bool:
    """False for delegate_task children AND for cron jobs fired in-process from
    a worker — i.e. whenever HERMES_KANBAN_* is present but not ours."""
    return _delegation_ctx("is_dispatcher_owned_worker_context", True)


def _visible(*, to_env_worker: bool) -> bool:
    """check_fn core: never for delegate children; dispatcher-spawned env workers
    (HERMES_KANBAN_TASK) per flag; else the profile toolset decides."""
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK") and _is_dispatcher_owned_worker():
        return to_env_worker
    return _profile_has_kanban_toolset()


@no_cache_check_fn
def _check_kanban_mode() -> bool:
    """Lifecycle tools: dispatcher workers + profiles with the ``kanban`` toolset."""
    return _visible(to_env_worker=True)


@no_cache_check_fn
def _check_kanban_complete_mode() -> bool:
    """``kanban_complete`` everywhere EXCEPT an implementation run on a
    review-gated card.

    The dispatcher stamps ``HERMES_KANBAN_REQUIRED_REVIEWER`` and
    ``HERMES_KANBAN_RUN_PHASE`` into the worker's startup context; only a
    review-lane claim earns ``review``, so a same-profile implementer never
    sees the tool either. Hiding is a usability control — the backend
    re-derives the same verdict from lifecycle/run state in
    ``kb.complete_task``, so a stale or absent env var cannot widen what is
    accepted. Ungated cards keep the tool exactly as before.
    """
    if not _check_kanban_mode():
        return False
    if (
        os.environ.get("HERMES_KANBAN_TASK")
        and os.environ.get("HERMES_KANBAN_REQUIRED_REVIEWER")
        and os.environ.get("HERMES_KANBAN_RUN_PHASE", "implementation") != "review"
    ):
        return False
    return True


@no_cache_check_fn
def _check_kanban_orchestrator_mode() -> bool:
    """Board-routing tools (kanban_list, kanban_unblock): hidden from task workers."""
    return _visible(to_env_worker=False)


# --- Shared helpers: validation failures raise _Reject; _kanban_handler renders it ---

# Worker tools that terminate or transition a run's ownership. An unbound worker
# (HERMES_KANBAN_RUN_ID unresolvable) must not run these: expected_run_id=None
# would silently skip the run-ownership CAS in kanban_db. Non-lifecycle tools
# (heartbeat / attach / attach_url) do not terminate a run and are not gated.
_RUN_LIFECYCLE_TOOLS = frozenset({
    "kanban_complete", "kanban_block", "kanban_schedule",
    "kanban_request_review", "kanban_request_changes",
})

class _Reject(Exception):
    """Carries a finished ``tool_error`` payload out of a validation helper."""

    def __init__(self, message: str):
        super().__init__(tool_error(message))


def _check(cond: Any, message: str) -> None:
    """Reject (as a tool error) unless ``cond`` is truthy."""
    if not cond:
        raise _Reject(message)


# Keys a handler reads that its LLM-facing schema deliberately does not declare:
# ``session_id`` is provenance stamped by internal callers (31fe2290393), ``project_id``
# the pre-``project`` alias still honoured by ``_handle_create`` (e7811345c17).
_UNDECLARED_ARGS: dict[str, frozenset[str]] = {
    "kanban_create": frozenset({"session_id", "project_id"}),
    # ``title`` is the pre-schema alias of ``filename`` that ``_handle_attach_url`` still honours.
    "kanban_attach_url": frozenset({"title"}),
}


def _persisted_identity() -> str:
    """Profile name persisted into board records (comment author, task creator).

    ``hermes_cli.profiles.current_profile_name`` resolves the profile this call runs FOR — the bound
    home override under a multiplexed tick or turn, else the dispatcher's ``HERMES_PROFILE`` pin,
    else the process home; the generic ``"worker"`` only when nothing names a profile. Never taken
    from tool args: board records are injected into future workers' prompts, so a caller-supplied
    identity could forge an authoritative-looking author (see #19713).
    """
    from hermes_cli.profiles import current_profile_name

    return current_profile_name("worker") or "worker"


def _kanban_handler(tool_name: str) -> Callable:
    """Wrap a handler so every failure is a structured tool error. ``ValueError``
    (invalid board slug, DB validation such as cycle/self-link, ``AttachmentTooLarge``)
    is reported without a traceback; anything else is logged with ``logger.exception``."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(args: dict, **kw) -> str:
            try:
                # Reject typos before a handoff can succeed without its artifacts.
                properties = registry.get_schema(tool_name)["parameters"]["properties"]
                allowed = set(properties) | _UNDECLARED_ARGS.get(tool_name, frozenset())
                unknown = sorted(set(args) - allowed)
                _check(not unknown,
                       f"{tool_name}: unknown parameter(s): {', '.join(unknown)}. "
                       f"Valid parameters: {', '.join(sorted(properties))}. Nothing changed.")
                return fn(args, **kw)
            except _Reject as e:
                return e.args[0]
            except Exception as e:
                if not isinstance(e, ValueError):
                    logger.exception(f"{tool_name} failed")
                return tool_error(f"{tool_name}: {e}")
        return wrapper
    return deco


def _reject_delegated_child_mutation(tool_name: str) -> None:
    """A delegate_task child shares the parent's process, so inherited HERMES_KANBAN_*
    env is not proof of ownership: it may report findings but must not mutate."""
    if _delegation_ctx("is_delegated_child_process_context", False):
        raise _Reject(
            f"{tool_name} refused: delegate_task child agents are not Kanban run owners. "
            "Return findings to the parent agent; the dispatcher worker or an explicitly "
            "configured Kanban orchestrator must perform board mutations.")


def _default_task_id(arg: Any) -> Optional[str]:
    """Resolve ``task_id`` arg or fall back to the env var the dispatcher set."""
    if arg is not None:
        val = str(arg).strip()
        if val:
            return val
    if _is_delegated_child_context():
        return None
    if not _is_dispatcher_owned_worker():
        # A cron job fired in-process from a worker must never inherit the
        # worker's task id as an implicit default.
        return None
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    if env_tid:
        val = env_tid.strip()
        if val:
            return val
    return None


def _require_task_id(args: dict) -> str:
    """Resolve the target task or reject with a message the caller can act on.

    The env default only exists for a dispatcher-spawned worker; every other
    caller must name a task, so a rejection points at ``kanban_list`` instead of
    an env var a chat caller cannot set (#91431).
    """
    tid = _default_task_id(args.get("task_id"))
    if tid:
        return tid
    if os.environ.get("HERMES_KANBAN_TASK"):
        # Env task present but not usable here (delegate child / cron run beside
        # a worker): it is not this session's to default to.
        raise _Reject(
            "task_id is required: this session does not own the inherited "
            "HERMES_KANBAN_TASK, so it is not a valid default. Pass an explicit "
            "task_id.")
    raise _Reject(
        "task_id is required: this session has no dispatcher-assigned task to "
        "default to. Pass an explicit task_id; discover task ids with kanban_list.")


def _own_task_env(task_id: str, var: str) -> Optional[str]:
    """``$var`` only when this worker is scoped to ``task_id``; else None."""
    return os.environ.get(var) if os.environ.get("HERMES_KANBAN_TASK") == task_id else None

def _worker_run_id(task_id: str) -> Optional[int]:
    """This worker's dispatcher run id when it is scoped to task_id."""
    raw = _own_task_env(task_id, "HERMES_KANBAN_RUN_ID")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _stamp_worker_session_metadata(task_id: str, metadata: Optional[dict]) -> Optional[dict]:
    """Add trusted worker session id metadata for this worker's own task."""
    session_id = _own_task_env(task_id, "HERMES_SESSION_ID")
    return {**(metadata or {}), "worker_session_id": session_id} if session_id else metadata


def _enforce_worker_task_ownership(tid: str) -> None:
    """A dispatcher-spawned worker may only mutate its own HERMES_KANBAN_TASK; a
    prompt-injected ``task_id`` must not corrupt sibling/cross-tenant runs.
    Orchestrators (toolset enabled, no env task) legitimately route child tasks.

    Tools like ``kanban_complete`` / ``kanban_block`` / ``kanban_heartbeat`` mutate run-lifecycle state, so
    a buggy or prompt-injected worker that passed an explicit ``task_id`` for some other task could corrupt
    sibling or cross-tenant runs (see #19534).
    """
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    if env_tid and tid != env_tid:
        raise _Reject(
            f"worker is scoped to task {env_tid}; refusing to mutate {tid}. Use kanban_comment "
            f"to hand off information to other tasks, or kanban_create to spawn follow-up work.")


def _worker_guard(tool_name: str, args: dict) -> str:
    """Worker mutation preamble, in order: delegate-child rejection, task id
    resolution, task-scope ownership, run-identity proof. Returns the task id.

    A dispatcher-spawned worker (``HERMES_KANBAN_TASK`` set) that cannot name
    its run id is refused on the run-lifecycle mutations: ``expected_run_id=None``
    would silently skip the run-ownership CAS in ``kanban_db`` (``complete_task`` /
    ``block_task`` / ``request_review`` / ``request_changes`` only append
    ``AND current_run_id = ?`` when the value is not ``None``), so an unbound
    stale worker could complete a card a live successor owns. This mirrors
    ``agent/kanban_stop.py``, which already treats an unbound run id as unknown
    and fails closed. CLI / human / orchestrator paths (no ``HERMES_KANBAN_TASK``)
    legitimately pass ``expected_run_id=None`` and are unaffected. Non-lifecycle
    worker tools (heartbeat / attach / attach_url) do not terminate a run and are
    not gated here.
    """
    _reject_delegated_child_mutation(tool_name)
    tid = _require_task_id(args)
    _enforce_worker_task_ownership(tid)
    if (
        tool_name in _RUN_LIFECYCLE_TOOLS
        and os.environ.get("HERMES_KANBAN_TASK")
        and _worker_run_id(tid) is None
    ):
        raise _Reject(
            f"{tool_name} refused: this worker cannot resolve its "
            "HERMES_KANBAN_RUN_ID, so it cannot prove ownership of the card's "
            "current run. A stale or unbound worker must not terminate a run a "
            "live successor owns. Re-run through the dispatcher so the run id is "
            "pinned, or use an orchestrator/CLI path that passes an explicit "
            "expected_run_id."
        )
    return tid


def _require_orchestrator_tool(tool_name: str) -> None:
    """The check_fn already hides orchestrator tools from workers; this catches
    a stale registration or test harness routing a worker here anyway."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        raise _Reject(
            f"{tool_name} is orchestrator-only; dispatcher-spawned workers must use "
            "kanban_complete, kanban_request_review, kanban_request_changes, kanban_block, "
            "kanban_heartbeat, or kanban_comment for their assigned task.")


@contextmanager
def _board(board: Optional[str], *, quiet_close: bool = False):
    """``with _board(slug) as (kb, conn)``; lazy import so the module loads in non-kanban
    contexts. ``board=None`` keeps the env/symlink resolution chain; an explicit slug
    overrides it per call. ``quiet_close`` swallows close() errors (best-effort bridges)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect(board=board)
    try:
        yield kb, conn
    finally:
        try:
            conn.close()
        except Exception:
            if not quiet_close:
                raise


def _existing_task(kb, conn, tid: str):
    task = kb.get_task(conn, tid)
    _check(task is not None, f"task {tid} not found")
    return task


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _ok_landed(kb, conn, tid: str, default_status: str, **extra: Any) -> str:
    """Success payload reporting where the task actually landed (routing may
    not leave it in the requested status)."""
    run = kb.latest_run(conn, tid)
    landed = kb.get_task(conn, tid)
    return _ok(task_id=tid, run_id=run.id if run else None,
               status=landed.status if landed else default_status, **extra)


def _redact(value: Any) -> str:
    return redact_sensitive_text(str(value), force=True)


def _redact_opt(value: Any) -> Any:
    return _redact(value) if value else value


def _redact_metadata(metadata: dict) -> Optional[dict]:
    """Redact via a JSON round-trip; None if the result can't be re-parsed."""
    try:
        return json.loads(redact_sensitive_text(json.dumps(metadata), force=True))
    except json.JSONDecodeError:
        return None


# --- Canonical profile validation / enumeration -------------------------------
#
# One kernel for every caller-supplied profile name on this surface
# (``kanban_create`` assignee/reviewer, ``kanban_reassign`` destination,
# ``kanban_request_review`` reviewer) plus the ``kanban_discover`` roster.
# The kernel itself lives in ``hermes_cli.kanban_validation`` so the DB layer
# (create/reassign/request_review) and these handlers can never drift: it
# resolves through ``hermes_cli.profiles``, which is what the CLI and the
# dispatcher's spawn gate already use — ``default`` resolves through the
# profile root (HERMES_HOME / env aware, never the current home), and a
# directory that is tombstoned, marker-less or not a valid profile id never
# passes.

def _require_installed_profile(what: str, value: Any, *, hint: str = "") -> str:
    """Reject ``value`` unless it names a profile this home can actually spawn.

    Returns the stripped name. Validated BEFORE the board is opened, so a
    typo'd assignee can never leave a task row, a dependency edge, an event
    or a workspace behind — the dispatcher would otherwise bucket it as
    ``skipped_nonspawnable`` forever.

    The wording is the shared kernel's (``hermes_cli.kanban_validation``): a
    profile is "not found", never "not installed", and the refusal always
    carries the roster, ``kanban_discover`` guidance and ``Nothing changed``.
    """
    from hermes_cli.kanban_validation import require_profile

    try:
        return require_profile(what, value, hint=hint)
    except ValueError as exc:
        raise _Reject(str(exc)) from None


def _require_reviewer_capability(profile: str) -> None:
    """A reviewer must already carry the skill the dispatcher force-loads for a
    review-lane worker; reject before the board is opened."""
    from hermes_cli.kanban_validation import REVIEW_SKILLS, require_skills

    try:
        require_skills(profile, REVIEW_SKILLS)
    except ValueError as exc:
        raise _Reject(str(exc)) from None


def _require_assignee_skills(profile: Optional[str], skills: Optional[list]) -> None:
    """Every explicitly forced skill must resolve in the assignee's effective
    skill library — checked before the board is opened so a mixed
    valid/missing list is rejected atomically with zero writes."""
    if not skills or not profile:
        return
    from hermes_cli.kanban_validation import require_skills

    try:
        require_skills(profile, skills)
    except ValueError as exc:
        raise _Reject(str(exc)) from None


def _coerce_str_list(value: Any, name: str, what: str, *, strip: bool = False):
    """Accept a single string (convenience) or a list/tuple; with ``strip`` the
    items are stringified, stripped, and empties dropped."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise _Reject(f"{name} must be a list of {what}, got {type(value).__name__}")
    if strip:
        value = [str(x).strip() for x in value if str(x).strip()]
    return value


def _require_dict_metadata(metadata: Any) -> None:
    _check(metadata is None or isinstance(metadata, dict),
           f"metadata must be an object/dict, got {type(metadata).__name__}")


def _merge_artifacts(metadata: Any, artifacts: list[str]) -> dict:
    """Fold ``artifacts`` into ``metadata["artifacts"]`` (merged with, never overwriting, a
    list the worker passed manually). Artifacts ride inside metadata so the completed-event
    payload needs no DB schema change; the gateway notifier uploads each as an attachment."""
    _require_dict_metadata(metadata)
    metadata = {} if metadata is None else metadata
    existing = metadata.get("artifacts")
    if isinstance(existing, (list, tuple)):
        merged = (str(item).strip() for item in [*existing, *artifacts])
        metadata["artifacts"] = list(dict.fromkeys(s for s in merged if s))
    else:
        metadata["artifacts"] = artifacts
    return metadata


def _require_text(args: dict, name: str, message: Optional[str] = None) -> Any:
    """``args[name]``; rejects when missing or blank."""
    value = args.get(name)
    _check(value and str(value).strip(), message or f"{name} is required")
    return value


_BOOL_WORDS = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def _parse_bool_arg(args: dict, name: str) -> bool:
    value = args.get(name)
    if value is None or isinstance(value, bool):
        return bool(value)
    parsed = _BOOL_WORDS.get(str(value).strip().lower())
    _check(parsed is not None, f"{name} must be a boolean or 'true'/'false'")
    return parsed


def _opt_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    return int(value) if value is not None else default


_TASK_FIELDS = tuple(
    "id title body assignee status tenant priority workspace_kind workspace_path created_by "
    "created_at started_at completed_at result current_run_id model_override "
    "provider_override completion_contract last_failure_error required_reviewer".split())
_TASK_SUMMARY_FIELDS = tuple(
    "id title assignee status priority tenant workspace_kind workspace_path project_id created_by "
    "created_at started_at completed_at current_run_id model_override provider_override "
    "required_reviewer".split())
_RUN_FIELDS = tuple("id profile status outcome summary error metadata started_at ended_at".split())
_COMMENT_FIELDS = ("author", "body", "created_at")
_EVENT_FIELDS = ("kind", "payload", "created_at", "run_id")
_ATTACHMENT_FIELDS = tuple(
    "id filename content_type size uploaded_by stored_path created_at".split())
_CREATED_FIELDS = ("status", "workspace_kind", "workspace_path", "project_id")


def _fields(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """``{name: getattr(obj, name)}``; every value None when ``obj`` is None."""
    return {n: getattr(obj, n) if obj is not None else None for n in names}


def _task_summary_dict(kb, conn, task) -> dict[str, Any]:
    parents = kb.parent_ids(conn, task.id)
    children = kb.child_ids(conn, task.id)
    return {
        **_fields(task, _TASK_SUMMARY_FIELDS), "parents": parents, "children": children,
        "parent_count": len(parents), "child_count": len(children)}


# --- Goal-mode judge gate ---

_GOAL_MODE_BLOCK_ALLOWED_KINDS = frozenset({"dependency", "needs_input"})


def _goal_judge_available() -> bool:
    """``judge_goal`` fails open (no auxiliary model -> ``"continue"``), which is
    indistinguishable from "not done yet" and would wedge every goal_mode
    worker; so the gate is enforced only when a judge is actually reachable."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


# Per-tool guidance for a judge rejection: verdict -> message. ``{reason}``/``{tid}`` are filled in.
_GOAL_GATE_MESSAGES = {
    "kanban_complete": {
        "blocked": (
            "Goal completion rejected: judge ruled the goal unachievable — {reason}. The task "
            "will NOT complete silently. Either re-scope the task with kanban_edit, or record "
            "the block with kanban_block and hand the decision to a human / reviewer."),
        "continue": (
            "Goal completion rejected by judge: {reason}. To proceed, either: (1) provide "
            "explicit acceptance evidence in your summary matching the task's criteria, or (2) "
            "create continuation tasks with parents=[{tid}] and keep this task alive.")},
    "kanban_request_review": {
        "blocked": (
            "Goal review handoff rejected: judge ruled the goal unachievable — {reason}. "
            "Record the block with kanban_block instead of requesting review."),
        "continue": (
            "Goal review handoff rejected by judge: {reason}. Provide acceptance evidence "
            "matching the card before requesting review.")}}


def _goal_gate(tool_name: str, task, tid: str, evidence: str) -> None:
    """Goal-mode pre-handoff judge gate: a worker must not complete / request
    review before acceptance criteria are met. ``blocked`` gets its own
    guidance; any other non-``done`` verdict gets the ``continue`` guidance.
    A broken judge fails open (logged) so it cannot permanently wedge work."""
    if not task or not task.goal_mode or not _goal_judge_available():
        return
    try:
        # Headless gate runs outside any agent turn: bind the per-task relay-affinity scope
        # (mirrors kanban_specify) so the relay does not reject the judge call (#113669).
        from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
        affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{tid}")
        try:
            verdict, reason, _, _, transport_failed = judge_goal(
                goal=f"{task.title}\n\n{task.body or ''}".strip(), last_response=evidence.strip())
        finally:
            if affinity_token is not None:
                reset_affinity_scope(affinity_token)
    except Exception as judge_exc:
        logger.warning(
            "goal judge check failed, allowing lifecycle handoff: %s", judge_exc, exc_info=True)
        return
    if transport_failed:
        # ``judge_goal`` fails open to ``continue`` on transport errors (relay 400, auth, timeout);
        # an unreachable judge is not a human "not done" and must not reject the handoff (#83610).
        logger.warning("goal judge unreachable (%s), allowing lifecycle handoff", reason)
        return
    if verdict == "done":
        return
    key = "blocked" if verdict == "blocked" else "continue"
    raise _Reject(_GOAL_GATE_MESSAGES[tool_name][key].format(reason=reason, tid=tid))


# --- Runtime-activity → board bridges (auto-heartbeat, live comment injection) ---
# The dispatcher watchdog reads ``tasks.last_heartbeat_at``, not the agent's in-process
# activity timestamp, so normal work is mirrored onto the board here (``kanban_heartbeat``
# stays for notes / pre-extending a claim). Best-effort: never raise into the agent loop;
# rate-limited per process (a race costs one harmless extra write); no-op outside a
# dispatcher-spawned worker.

# --------------------------------------------------------------------------- Runtime-activity →
# board-heartbeat bridge (#31752)
# --------------------------------------------------------------------------- When the agent ticks
# ``_touch_activity`` during normal work (between tool calls, mid-stream chunks, etc.), we want the kanban
# board's ``last_heartbeat_at`` columns to reflect that liveness so the dispatcher watchdog (which reads
# ``tasks.last_heartbeat_at``, not the agent's in-process timestamp) doesn't reclaim an actively-running
# worker as stale. The model is not required to call the explicit ``kanban_heartbeat`` tool for this to work
# — that tool stays available for workers that want to attach a note or pre-emptively extend a claim across
# a known-long op. Constraints: - Best-effort: never raise. The agent loop must not care if the bridge fails
# (board missing, DB locked, etc.). - Rate-limited to one DB write per 60s per-process; runtime activity can
# tick on every chunk/tool result and we don't need that resolution. - No-op outside dispatcher-spawned
# worker context (no ``HERMES_KANBAN_TASK``). - No durable note on these auto-heartbeats; that's reserved
# for the explicit tool which carries a model-supplied note.
_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_auto_heartbeat_last_attempt: float = 0.0
_auto_heartbeat_fence_warned = False


def register_current_worker_from_env() -> bool:
    """Record this worker's pid on its run when the dispatcher died before it could
    (``adopt_worker_pid``). False only when the board says the run was already reclaimed:
    the caller must exit. Anything unreadable (no run id, delegate child, board error)
    lets the worker run, as before."""
    tid = os.environ.get("HERMES_KANBAN_TASK")
    run_id = _worker_run_id(tid) if tid else None
    if run_id is None or _is_delegated_child_context():
        return True
    try:
        from hermes_cli import kanban_db_dispatch as kbd
        with _board(None, quiet_close=True) as (_kb, conn):
            return kbd.adopt_worker_pid(conn, tid, run_id, os.getpid())
    except Exception:
        logger.debug("kanban worker registration for %s failed", tid, exc_info=True)
        return True


def heartbeat_current_worker_from_env() -> bool:
    """Claim extension + board heartbeat for the current worker; True iff both writes
    succeed. ``HERMES_KANBAN_RUN_ID`` pins the run row so a reclaimed stale run is not
    heartbeated; ``HERMES_KANBAN_CLAIM_LOCK`` absent -> default claimer (local workers)."""
    global _auto_heartbeat_last_attempt, _auto_heartbeat_fence_warned
    tid = os.environ.get("HERMES_KANBAN_TASK")
    now = time.monotonic()
    if not tid or (now - _auto_heartbeat_last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
        return False
    if _is_delegated_child_context():
        # An in-process delegate child's activity is not the worker's liveness; checked before
        # stamping the window so a chatty child cannot starve the worker's own heartbeat.
        return False
    _auto_heartbeat_last_attempt = now
    try:
        from hermes_cli import kanban_db_dispatch as kbd
        with _board(None, quiet_close=True) as (kb, conn):
            ops = ((kb.heartbeat_claim, {"claimer": os.environ.get("HERMES_KANBAN_CLAIM_LOCK")}),
                   (kbd.heartbeat_worker, {"note": None, "expected_run_id": _worker_run_id(tid)}))
            succeeded = True
            for fn, kwargs in ops:
                op = fn.__name__
                try:
                    succeeded = bool(fn(conn, tid, **kwargs)) and succeeded
                except PermissionError as exc:
                    # The board fence rejected the worker's own liveness write: this process
                    # inherited HERMES_DELEGATED_CHILD_CONTEXT next to HERMES_KANBAN_TASK, so it is
                    # a delegate descendant, not the dispatcher's worker (kanban_complete refuses
                    # too). Loud once: at DEBUG the board just showed a worker that never beats.
                    succeeded = False
                    if not _auto_heartbeat_fence_warned:
                        _auto_heartbeat_fence_warned = True
                        logger.warning(
                            "kanban auto-heartbeat for task %s refused (%s): this process carries "
                            "HERMES_DELEGATED_CHILD_CONTEXT together with HERMES_KANBAN_TASK, so the board "
                            "treats it as a delegate_task descendant and its claim will not be extended by "
                            "activity. Only the dispatcher's own spawn grants worker scope; do not copy a "
                            "worker's environment into a hand-launched process.", tid, exc)
                except Exception:
                    logger.debug("auto-heartbeat: %s failed", op, exc_info=True)
                    succeeded = False
        return succeeded
    except Exception:
        logger.debug("auto-heartbeat: bridge failed", exc_info=True)
        return False


# Live operator-note injection: poll the task for new comments and steer them in
# OUT-OF-BAND, so a user can talk to a running task without block → comment → unblock.
# Watermarked per task (seeded on first poll: that history is already in the context).
_COMMENT_POLL_MIN_INTERVAL_SECONDS = 6.0
_comment_poll_last_attempt: float = 0.0
_comment_watermark: dict[str, int] = {}


def inject_new_comments_from_env(agent: Any) -> bool:
    """Steer new operator comments on the worker's task into ``agent``; True iff a
    steer was injected; never raises. Own comments (``_persisted_identity``) are skipped."""
    global _comment_poll_last_attempt
    # Operator notes address the dispatcher-owned worker; a delegate_task child sharing
    # this process must neither receive them nor advance the shared watermark (#112817).
    tid = os.environ.get("HERMES_KANBAN_TASK") if _is_dispatcher_owned_worker() else None
    now = time.monotonic()
    if (not tid or agent is None or not hasattr(agent, "steer")
            or (now - _comment_poll_last_attempt) < _COMMENT_POLL_MIN_INTERVAL_SECONDS):
        return False
    _comment_poll_last_attempt = now
    seen = _comment_watermark.get(tid)
    try:
        with _board(None, quiet_close=True) as (kb, conn):
            rows = kb.list_comments_after(conn, tid, after_id=seen or 0)
    except Exception:
        logger.debug("comment-inject: bridge failed", exc_info=True)
        return False
    if seen is None:
        _comment_watermark[tid] = max((c.id for c in rows), default=0)
    if seen is None or not rows:
        return False
    # Advance past everything read (including our own notes) so nothing is re-injected.
    _comment_watermark[tid] = max(c.id for c in rows)
    # Same resolution the write side used, so a worker skips its OWN comments even
    # when the dispatcher did not pin HERMES_PROFILE (echoed notes would otherwise
    # re-enter the live turn as fake operator steering).
    own = _persisted_identity()
    fresh = [c for c in rows if (c.author or "").strip() != own and (c.body or "").strip()]
    if not fresh:
        return False
    lines = [f"- {c.author or 'operator'}: {c.body.strip()}" for c in fresh]
    note = ("New note" + ("s" if len(fresh) > 1 else "")
            + " on your kanban task from the operator (delivered mid-run). "
            + "Take it into account for the work you're doing right now:\n" + "\n".join(lines))
    try:
        return bool(agent.steer(note))
    except Exception:
        logger.debug("comment-inject: steer failed", exc_info=True)
        return False


# --- Handlers ---

@_kanban_handler("kanban_show")
def _handle_show(args: dict, **kw) -> str:
    """Full task state: row, parents, children, comments, runs, last 50 events."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        # No dispatcher task in scope and no explicit id: the caller asked "what
        # should I be looking at". A chat profile cannot set HERMES_KANBAN_TASK,
        # so an error naming the env var is dead end (#91431) — answer instead.
        return json.dumps({
            "current_task": None,
            "hint": (
                "No dispatcher-assigned task in this session, so there is no "
                "task to show by default. Call kanban_list to see task ids on "
                "the board, then kanban_show(task_id=<id>) for full state."),
        })
    with _board(args.get("board")) as (kb, conn):
        task = _existing_task(kb, conn, tid)
        return json.dumps({
            "task": _fields(task, _TASK_FIELDS),
            "parents": kb.parent_ids(conn, tid),
            # Non-terminal parents; on a running card this means the dependency
            # gate is not holding it and kanban_complete will refuse.
            "unsatisfied_parents": [
                {"id": pid, "status": status} for pid, status in kb.unsatisfied_parents(conn, tid)],
            "children": kb.child_ids(conn, tid),
            "comments": [_fields(c, _COMMENT_FIELDS) for c in kb.list_comments(conn, tid)],
            # Capped; full log via CLI.
            "events": [_fields(e, _EVENT_FIELDS) for e in kb.list_events(conn, tid)[-50:]],
            "runs": [_fields(r, _RUN_FIELDS) for r in kb.list_runs(conn, tid)],
            # Same string build_worker_context hands the dispatcher at spawn time.
            "worker_context": kb.build_worker_context(conn, tid)})


@_kanban_handler("kanban_list")
def _handle_list(args: dict, **kw) -> str:
    """Task summaries with the same core filters as the CLI."""
    _require_orchestrator_tool("kanban_list")
    include_archived = _parse_bool_arg(args, "include_archived")
    limit = args.get("limit")
    try:
        limit = KANBAN_LIST_DEFAULT_LIMIT if limit is None else int(limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")
    _check(limit >= 1, "limit must be >= 1")
    _check(limit <= KANBAN_LIST_MAX_LIMIT, f"limit must be <= {KANBAN_LIST_MAX_LIMIT}")
    with _board(args.get("board")) as (kb, conn):
        # Match CLI list: dependencies cleared since the last dispatcher tick
        # should be visible to orchestrators immediately.
        promoted = kb.recompute_ready(conn)
        # One extra row lets the output report truncation without dumping the board.
        rows = kb.list_tasks(
            conn, assignee=args.get("assignee"), status=args.get("status"),
            tenant=args.get("tenant"), include_archived=include_archived, limit=limit + 1)
        truncated = len(rows) > limit
        tasks = rows[:limit]
        return json.dumps({
            "tasks": [_task_summary_dict(kb, conn, t) for t in tasks],
            "count": len(tasks), "limit": limit, "truncated": truncated,
            "next_limit": (min(limit * 2, KANBAN_LIST_MAX_LIMIT)
                           if truncated and limit < KANBAN_LIST_MAX_LIMIT else None),
            "promoted": promoted})


@_kanban_handler("kanban_complete")
def _handle_complete(args: dict, **kw) -> str:
    """Mark the current task done with a structured handoff."""
    tid = _worker_guard("kanban_complete", args)
    summary = _redact_opt(args.get("summary"))
    result = _redact_opt(args.get("result"))
    metadata = args.get("metadata")
    if isinstance(metadata, dict):
        # Keep the unredacted dict if the redacted JSON cannot be re-parsed.
        metadata = _redact_metadata(metadata) or metadata
    created_cards = _coerce_str_list(
        args.get("created_cards"), "created_cards", "task ids", strip=True)
    artifacts = _coerce_str_list(args.get("artifacts"), "artifacts", "file paths", strip=True)
    if artifacts:
        metadata = _merge_artifacts(metadata, artifacts)
    _check(summary or result, "provide at least one of: summary (preferred), result")
    _require_dict_metadata(metadata)
    metadata = _stamp_worker_session_metadata(tid, metadata)
    with _board(args.get("board")) as (kb, conn):
        # Goal-mode pre-completion judge gate (Issue #38367). Prevent workers from bypassing the auxiliary
        # judge by calling kanban_complete before acceptance criteria are met. Only enforce when a judge is
        # actually reachable — see _goal_judge_available for why an unavailable judge fails open.
        task = kb.get_task(conn, tid)
        _goal_gate("kanban_complete", task, tid, (summary or result or "").strip())
        try:
            ok = kb.complete_task(
                conn, tid, result=result, summary=summary, metadata=metadata,
                created_cards=created_cards, expected_run_id=_worker_run_id(tid))
        except kb.ArtifactPreservationError as artifact_err:
            # Structured rejection — surface the phantom ids so the worker can retry with a corrected list
            # or drop the field. Audit event already landed in the DB. The task itself was NOT mutated (the
            # gate runs before the write txn), so the worker can simply call kanban_complete again. Spell
            # that out — without it the model often interprets a tool_error as a terminal failure and either
            # blocks or crashes the run instead of retrying. See #22923.
            return tool_error(
                f"kanban_complete could not preserve the declared artifacts: {artifact_err}. "
                f"Your task is still in-flight and its scratch workspace was kept. Fix the "
                f"artifact path or storage error, then retry kanban_complete with the same "
                f"handoff.")
        except kb.LiveClaimError as claim_err:
            # Env-less caller (orchestrator, another session) on a card a dispatcher
            # worker is executing: refusing here is what keeps that worker's run open.
            return tool_error(
                f"kanban_complete refused: {claim_err}. Nothing changed. Wait for the worker "
                f"to finish, or an operator can run `hermes kanban complete --force {tid}`.")
        except kb.ReviewerGateError as gate_err:
            # Backend enforcement of the review gate (the tool is hidden from an
            # implementation run anyway — this is the alternate-route defence).
            return tool_error(f"kanban_complete refused: {gate_err}")
        except kb.HallucinatedCardsError as hall_err:
            # The gate runs before the write txn, so the task was NOT mutated;
            # say so explicitly or the model treats the error as terminal and
            # blocks/crashes instead of retrying. Audit event already landed.
            return tool_error(
                f"kanban_complete blocked: the following created_cards do not exist or were not "
                f"created by this worker: {', '.join(hall_err.phantom)}. Your task is still "
                f"in-flight (no state change). Retry kanban_complete with the same "
                f"summary/metadata and either drop these ids from created_cards, or pass "
                f"created_cards=[] to skip the card-claim check entirely.")
        except kb.EmptyCompletionError as empty_err:
            # Same shape as the card gate: nothing was mutated, the audit event
            # already landed; the worker retries with evidence instead of stalling.
            return tool_error(
                f"kanban_complete blocked: {empty_err}. Your task is still in-flight (no state "
                f"change). Retry kanban_complete with a non-empty summary or result describing "
                f"what was done.")
        task = kb.get_task(conn, tid)
        if not ok:
            # complete_task reports every refusal as bare False; a reopened or
            # never-finished parent is the actionable one. Name the blockers so
            # the worker/operator completes the parents instead of re-running.
            blockers = kb.unsatisfied_parents(conn, tid)
            if blockers:
                detail = ", ".join(f"{pid} ({status})" for pid, status in blockers)
                raise _Reject(
                    f"could not complete {tid}: unsatisfied parent dependencies: "
                    f"{detail}; complete the parents first (done or archived)")
            _check(False, (task.last_failure_error if task else None) or
                   f"could not complete {tid} (unknown id, stale run, or already terminal)")
        run = kb.latest_run(conn, tid)
        # Artifact staging is atomic with the completion write, so a worker that
        # read `kanban_attachments` before completing saw an empty list and has
        # no way to observe what its completion just registered (#117360).
        # Report the card's durable attachment set in the result.
        return _ok(task_id=tid, run_id=run.id if run else None,
                   attachments=[
                       _fields(a, _ATTACHMENT_FIELDS)
                       for a in kb.list_attachments(conn, tid)])


@_kanban_handler("kanban_block")
def _handle_block(args: dict, **kw) -> str:
    """Transition the task to blocked with a reason a human will read."""
    tid = _worker_guard("kanban_block", args)
    reason = _redact(
        _require_text(args, "reason", "reason is required — explain what input you need"))
    kind = args.get("kind")
    with _board(args.get("board")) as (kb, conn):
        _check(kind is None or kind in kb.VALID_BLOCK_KINDS,
               f"kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} (or omit it)")
        # The goal loop treats ANY blocked status as terminal, so kanban_block
        # would be an escape hatch around the completion judge: goal_mode tasks
        # may only block on genuine external blockers.
        # Goal-mode block gate (Issue #38696, sibling of the kanban_complete judge gate in #38367).
        # kanban_block is a second exit path out of the goal loop — run_kanban_goal_loop() treats ANY
        # `blocked` status as terminal, identically to `done`, regardless of kind. Without this, a worker
        # that learns kanban_complete is gated can just call kanban_block(reason="anything") to escape the
        # loop instead. Restrict goal_mode tasks to the kinds that represent a genuine external blocker the
        # worker cannot resolve itself; `capability` and `transient` (or an unset kind) route back through
        # kanban_complete, which the judge now gates.
        task = kb.get_task(conn, tid)
        _check(not (task and task.goal_mode and kind not in _GOAL_MODE_BLOCK_ALLOWED_KINDS),
               f"goal_mode tasks can only block with kind in "
               f"{sorted(_GOAL_MODE_BLOCK_ALLOWED_KINDS)} (got {kind!r}). If the task is actually "
               f"finished or cannot proceed for another reason, call kanban_complete instead — "
               f"the completion judge will evaluate it.")
        review_before = kb.review_association(conn, tid)
        ok, why = kb.block_task(
            conn, tid, reason=reason, kind=kind, expected_run_id=_worker_run_id(tid),
            with_reason=True,
        )
        _check(ok, why or f"could not block {tid} (unknown id or not in running/ready/review)")
        landed_kind = kb.get_task(conn, tid).block_kind
        extra: dict = {"block_kind": landed_kind}
        if why:
            # The card IS blocked, but its worker could not be proven stopped —
            # or a spawn for it is still in flight and unpublished. Quote the
            # exact blocker so the later archive refusal is never a surprise.
            extra["stop_blocker"] = why
        if kind == "dependency" and landed_kind != kind:
            # block_task re-kinds a dependency wait that no open parent can satisfy.
            extra["requested_kind"] = kind
            extra["note"] = (
                "kind='dependency' only waits on an incomplete parent; no parent is open, "
                "so this was recorded as needs_input (sticky until a human unblocks) "
                "instead of parking in todo where the dispatcher would respawn it."
            )
        extra["review"] = _review_receipt(review_before, kb.review_association(conn, tid))
        return _ok_landed(kb, conn, tid, "blocked", **extra)


@_kanban_handler("kanban_schedule")
def _handle_schedule(args: dict, **kw) -> str:
    """Park the current task until an orchestrator re-gates it."""
    tid = _worker_guard("kanban_schedule", args)
    raw_reason = args.get("reason")
    _check(raw_reason is None or isinstance(raw_reason, str), "reason must be a string")
    reason = _redact(raw_reason.strip()) if raw_reason and raw_reason.strip() else None
    with _board(args.get("board")) as (kb, conn):
        # The goal loop treats ``scheduled`` as terminal like ``blocked``, so
        # parking would bypass the completion judge (see kanban_block, #38696).
        task = kb.get_task(conn, tid)
        _check(not (task and task.goal_mode),
               "goal_mode tasks cannot be scheduled: use kanban_block with kind "
               f"in {sorted(_GOAL_MODE_BLOCK_ALLOWED_KINDS)} for a genuine external "
               "blocker, or kanban_complete so the completion judge can evaluate it.")
        ok = kb.schedule_task(
            conn, tid, reason=reason, expected_run_id=_worker_run_id(tid))
        _check(ok, f"could not schedule {tid} (unknown id or not in todo/ready/running/blocked)")
        return _ok_landed(kb, conn, tid, "scheduled", reason=reason)


@_kanban_handler("kanban_request_review")
def _handle_request_review(args: dict, **kw) -> str:
    """Move implementation into the first-class review phase."""
    tid = _worker_guard("kanban_request_review", args)
    summary = _redact(_require_text(
        args, "summary", "summary is required — describe what was implemented and how it "
        "was verified so the reviewer has context"))
    metadata = args.get("metadata")
    _require_dict_metadata(metadata)
    if metadata is not None:
        metadata = _redact_metadata(metadata)
        _check(metadata is not None, "metadata could not be safely serialized")
    artifacts = _coerce_str_list(args.get("artifacts"), "artifacts", "file paths", strip=True)
    if artifacts:
        metadata = _merge_artifacts(metadata, artifacts)
    metadata = _stamp_worker_session_metadata(tid, metadata)
    # Reviewer is model-supplied free text stored durably on the event payload.
    reviewer = _redact_opt(args.get("reviewer") or None)
    if reviewer:
        # A non-profile reviewer would park the card in `review` on an assignee
        # the dispatcher can never spawn (#106163). Same kernel as the
        # kanban_create / kanban_reassign assignee guards.
        reviewer = _require_installed_profile("reviewer", reviewer)
        # ...and it must carry the review-phase skill the dispatcher injects:
        # routing review to a profile without it spawns a reviewer that cannot
        # do the phase. (A card with a saved reviewer refuses any other name
        # inside kb.request_review, so the gate cannot be re-pointed here.)
        _require_reviewer_capability(reviewer)
    with _board(args.get("board")) as (kb, conn):
        _goal_gate("kanban_request_review", kb.get_task(conn, tid), tid, summary)
        try:
            ok, fail_reason = kb.request_review(
                conn, tid, summary=summary, metadata=metadata, reviewer=reviewer,
                expected_run_id=_worker_run_id(tid), with_reason=True)
        except kb.ArtifactPreservationError as artifact_err:
            # Same contract as kanban_complete (#22923): the transition rolled
            # back, the task is untouched and retryable — say so explicitly or
            # the model treats the tool_error as terminal.
            return tool_error(
                f"kanban_request_review could not preserve the declared artifacts: {artifact_err}. "
                f"Your task is still in-flight (no state change) and its scratch workspace was "
                f"kept. Fix the artifact path or storage error, then retry "
                f"kanban_request_review with the same handoff.")
        _check(ok, f"could not request review for {tid}: "
                   f"{fail_reason or 'unknown id or not in running/ready'}")
        return _ok_landed(kb, conn, tid, "review")


@_kanban_handler("kanban_request_changes")
def _handle_request_changes(args: dict, **kw) -> str:
    """Return a reviewer-owned running task to its implementer."""
    tid = _worker_guard("kanban_request_changes", args)
    reason = _redact(
        _require_text(args, "reason", "reason is required — describe the changes needed"))
    with _board(args.get("board")) as (kb, conn):
        ok, detail = kb.request_changes(
            conn, tid, reason=reason, expected_run_id=_worker_run_id(tid))
        _check(ok, f"could not request changes for {tid}: {detail or 'invalid review state'}")
        return _ok_landed(kb, conn, tid, "ready", implementer=detail)


@_kanban_handler("kanban_heartbeat")
def _handle_heartbeat(args: dict, **kw) -> str:
    """Signal liveness: extend the claim TTL AND record a heartbeat event.
    Without the claim half, a worker blocked in one long tool call would still
    be reclaimed by ``release_stale_claims``."""
    tid = _worker_guard("kanban_heartbeat", args)
    from hermes_cli import kanban_db_dispatch as kbd
    with _board(args.get("board")) as (kb, conn):
        # The dispatcher pins HERMES_KANBAN_CLAIM_LOCK at spawn; the default
        # claimer covers locally-driven workers that bypassed the dispatcher.
        kb.heartbeat_claim(conn, tid, claimer=os.environ.get("HERMES_KANBAN_CLAIM_LOCK"))
        ok = kbd.heartbeat_worker(
            conn, tid, note=args.get("note"), expected_run_id=_worker_run_id(tid))
        _check(ok, f"could not heartbeat {tid} (unknown id or not running)")
        return _ok(task_id=tid)


@_kanban_handler("kanban_comment")
def _handle_comment(args: dict, **kw) -> str:
    """Append a comment to a task's thread."""
    _reject_delegated_child_mutation("kanban_comment")
    tid = args.get("task_id")
    if not tid:
        # A comment writes to the board, so a bare call must stay an error — but
        # point at kanban_list, not an env var a chat caller cannot set (#91431).
        raise _Reject(
            "task_id is required: comments are per-task and this session has no "
            "dispatcher-assigned task to default to. Discover task ids with "
            "kanban_list, or pass the task id you mean.")
    body = _redact(_require_text(args, "body"))
    # Author comes from the worker's runtime identity (``_persisted_identity``), never
    # caller args: comments are injected into future workers' system prompts, so an
    # args["author"] override could forge a directive from ``hermes-system``.
    # Cross-task commenting stays unrestricted — it is the handoff channel between tasks.
    # Comments are injected into the next worker's system prompt by ``build_worker_context`` as
    # ``**{author}** (timestamp): {body}`` — accepting an ``args["author"]`` override let a worker forge a
    # comment from an authoritative-looking name like ``hermes-system`` and poison the future-worker context
    # with what reads as a system directive. See #19713.
    author = _persisted_identity()
    with _board(args.get("board")) as (kb, conn):
        cid = kb.add_comment(conn, tid, author=author, body=str(body))
        return _ok(task_id=tid, comment_id=cid)


def _store_attachment(board, tid, filename, data, content_type) -> str:
    """Store via ``kanban_db.store_attachment_bytes`` (shared size cap, per-task
    dir, metadata row) so agent, dashboard, and CLI surfaces stay in lockstep."""
    with _board(board) as (kb, conn):
        att_id = kb.store_attachment_bytes(
            conn, tid, str(filename), data,
            content_type=content_type, uploaded_by="agent", board=board)
        return _ok(task_id=tid, attachment_id=att_id, size=len(data))


@_kanban_handler("kanban_attach")
def _handle_attach(args: dict, **kw) -> str:
    """Attach an inline (base64) file to a task."""
    tid = _worker_guard("kanban_attach", args)
    filename = _require_text(args, "filename")
    content_b64 = _require_text(args, "content_base64")
    import base64
    import binascii
    try:
        data = base64.b64decode(str(content_b64), validate=True)
    except (binascii.Error, ValueError) as e:
        raise _Reject(f"content_base64 is not valid base64: {e}")
    return _store_attachment(args.get("board"), tid, filename, data, args.get("content_type"))


_MAX_ATTACH_URL_REDIRECTS = 5


def _download_url_with_cap(url: str, max_bytes: int) -> tuple[bytes, Optional[str]]:
    """Fetch ``url`` over http(s) capped at ``max_bytes`` -> ``(data, content_type)``.
    Every hop is SSRF-checked (redirects followed manually) so a model-controlled URL, or a
    public host 302ing, cannot reach loopback/private/cloud-metadata ranges. ``ValueError``
    for bad scheme, blocked target, too many redirects, or a body over the cap (checked
    while streaming, so nothing oversize is buffered)."""
    from urllib.parse import urljoin, urlparse
    import httpx
    from tools.url_safety import is_safe_url
    current_url = url
    for _ in range(_MAX_ATTACH_URL_REDIRECTS + 1):
        scheme = (urlparse(current_url).scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme {scheme!r}; only http/https are allowed")
        if not is_safe_url(current_url):
            raise ValueError(
                f"URL blocked by SSRF protection (private/internal address): {current_url}")
        chunks: list[bytes] = []
        total = 0
        with httpx.stream("GET", current_url, headers={"User-Agent": "hermes-kanban/attach"},
                          timeout=30, follow_redirects=False) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise ValueError(f"redirect without Location header from {current_url}")
                current_url = urljoin(current_url, location)
                continue
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
            for chunk in resp.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit")
                chunks.append(chunk)
        return b"".join(chunks), content_type
    raise ValueError(f"too many redirects fetching {url}")


@_kanban_handler("kanban_attach_url")
def _handle_attach_url(args: dict, **kw) -> str:
    """Attach a file fetched server-side from an http(s) URL (shared size cap)."""
    from hermes_cli import kanban_db as kb
    tid = _worker_guard("kanban_attach_url", args)
    url = str(_require_text(args, "url")).strip()
    filename = args.get("filename") or args.get("title")
    if not filename or not str(filename).strip():
        # Derive a name from the URL path's leaf component.
        from urllib.parse import unquote, urlparse
        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip() or "download"
    try:
        data, fetched_ct = _download_url_with_cap(url, kb.KANBAN_ATTACHMENT_MAX_BYTES)
    except ValueError as e:
        return tool_error(f"kanban_attach_url: {e}")
    except Exception as e:
        logger.exception("kanban_attach_url download failed")
        return tool_error(f"kanban_attach_url: failed to fetch {url}: {e}")
    return _store_attachment(
        args.get("board"), tid, filename, data, args.get("content_type") or fetched_ct)


@_kanban_handler("kanban_attachments")
def _handle_attachments(args: dict, **kw) -> str:
    """List a task's attachments (read-only; no ownership restriction)."""
    tid = _require_task_id(args)
    with _board(args.get("board")) as (kb, conn):
        _existing_task(kb, conn, tid)
        return json.dumps({
            "ok": True, "task_id": tid,
            "attachments": [
                _fields(a, _ATTACHMENT_FIELDS) for a in kb.list_attachments(conn, tid)]})


def _persisted_session_id(session_id: Optional[str]) -> Optional[str]:
    """Return a session id only when it is present in this profile's state.db."""
    if not session_id:
        return None
    try:
        from hermes_state import SessionDB
        from hermes_constants import get_hermes_home

        state = SessionDB(db_path=get_hermes_home() / "state.db", read_only=True)
    except Exception:  # state.db may not exist for a CLI/dashboard invocation
        logger.debug("Could not open state.db to verify Kanban provenance", exc_info=True)
        return None
    try:
        return session_id if state.get_session(session_id) else None
    finally:
        state.close()


@_kanban_handler("kanban_create")
def _handle_create(args: dict, **kw) -> str:
    """Create a (child) task; orchestrator workers use this to fan out."""
    _reject_delegated_child_mutation("kanban_create")
    title = _require_text(args, "title")
    assignee = args.get("assignee")
    _check(assignee, "assignee is required — name the profile that should execute this "
                     "task (the dispatcher will only spawn tasks with an assignee)")
    # Validated before `_board` is even opened: a profile that does not exist
    # must not leave a task row, a dependency edge, an event or a workspace
    # behind — the dispatcher would only ever bucket it as skipped_nonspawnable.
    assignee = _require_installed_profile("assignee", assignee)
    # Review gate, same zero-side-effect contract: the reviewer must exist AND
    # already carry the skill the dispatcher injects for review-phase startup.
    reviewer = args.get("reviewer") or None
    if reviewer is not None:
        reviewer = _require_installed_profile("reviewer", reviewer)
        _require_reviewer_capability(reviewer)
    # Workspace sharing is always explicit: omitted fields mean a fresh scratch workspace
    # even for a dispatcher-spawned creator (reusing the parent's path would let a child
    # mutate review evidence or race its checkout). Project identity is the one safe thing
    # to inherit implicitly (the DB turns it into a fresh per-task worktree).
    workspace_kind, workspace_path = args.get("workspace_kind"), args.get("workspace_path")
    # See #67567. ``project=""`` is an explicit "no project" (no ``or`` collapse, #106342).
    project_id = args["project"] if "project" in args else args.get("project_id")
    project_source_task_id = None
    triage, skills, goal_mode = (
        _parse_bool_arg(args, "triage"), _coerce_str_list(args.get("skills"), "skills", "skill names"),
        _parse_bool_arg(args, "goal_mode"))
    # Forced skills are validated against the ASSIGNEE's effective library here,
    # before the board is opened; kb.create_task re-checks so the backend/CLI/API
    # paths carry the same rule.
    _require_assignee_skills(assignee, skills)
    model_override, provider_override = args.get("model"), args.get("provider")
    _check(model_override or not provider_override, "'provider' requires 'model' to be set as well")
    parents = _coerce_str_list(args.get("parents") or [], "parents", "task ids")
    with _board(args.get("board")) as (kb, conn):
        from gateway.session_context import get_session_env
        from tools.async_delegation import _current_origin_session_id
        self_tid = (os.environ.get("HERMES_KANBAN_TASK")
                    if _is_dispatcher_owned_worker() else None)
        self_task = kb.get_task(conn, self_tid) if self_tid else None
        # The worker/API runtime may be transient; the owning task's origin is durable.
        # The ambient id is the request-scoped ContextVar binding, not the process-global
        # os.environ: in a multi-session gateway the env holds the LAST agent built, and an
        # id that never reached its ``sessions`` row is provenance pointing at nothing.
        session_id = (_persisted_session_id(args.get("session_id"))
                      or (self_task.session_id if self_task else None)
                      or _persisted_session_id(_current_origin_session_id())
                      or _persisted_session_id(get_session_env("HERMES_SESSION_ID", "")))
        if project_id is None and workspace_kind is None and workspace_path is None:
            if self_task is not None and self_task.project_id:
                project_id, project_source_task_id = self_task.project_id, self_task.id
        new_tid = kb.create_task(
            conn, title=str(title).strip(), body=args.get("body"), assignee=str(assignee),
            parents=tuple(parents), tenant=args.get("tenant") or os.environ.get("HERMES_TENANT"),
            priority=_opt_int(args.get("priority"), 0),
            workspace_kind=workspace_kind, workspace_path=workspace_path, project_id=project_id,
            # Board-project inheritance must read the board this call opened, not the
            # session's current board.
            board=args.get("board"),
            project_source_task_id=project_source_task_id, triage=triage,
            creator_task_id=self_tid,
            idempotency_key=args.get("idempotency_key"),
            max_runtime_seconds=_opt_int(args.get("max_runtime_seconds")), skills=skills,
            model_override=model_override, provider_override=provider_override,
            goal_mode=goal_mode, goal_max_turns=_opt_int(args.get("goal_max_turns")),
            completion_contract=args.get("completion_contract"),
            initial_status=str(args.get("initial_status") or "running"),
            reviewer=reviewer,
            created_by=_persisted_identity(), session_id=session_id)
        landed = _fields(kb.get_task(conn, new_tid), _CREATED_FIELDS)
        wait = [e for e in kb.list_events(conn, new_tid) if e.kind == "dependency_wait"]
        gate = {"gated": True, "gated_by": wait[-1].payload["parent"]} if wait else {"gated": False}
        return _ok(task_id=new_tid, **landed, **gate,
                   subscribed=_maybe_auto_subscribe(conn, new_tid))


def _live_tui_session_key(session_key: str, profile: Optional[str]) -> str:
    """Re-resolve a TUI session key at subscribe time: the inherited ``HERMES_SESSION_KEY``
    can name a session already superseded by a compaction fork, and a subscription bound to
    the dead key silently drops every later completion notification (#110068). Maps the key
    to its continuation tip via the session store's lineage walk. Best-effort, fail-open:
    any error (or an unresolvable key) returns *session_key* unchanged."""
    db_path = None
    if profile and profile != "default":
        try:
            from pathlib import Path
            from hermes_cli.profiles import get_profile_dir, profile_exists
            if profile_exists(profile):
                db_path = Path(get_profile_dir(profile)) / "state.db"
        except Exception:
            db_path = None
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire(db_path)
        try:
            return db.resolve_resume_session_id(session_key) or session_key
        finally:
            release_or_close(db)
    except Exception:
        return session_key


def _resolve_notify_target() -> Optional[dict[str, Any]]:
    """``kanban_db.add_notify_sub`` kwargs for the calling session, or None (CLI/cron/tests).
    Gateway sessions: ``HERMES_SESSION_PLATFORM``/``CHAT_ID`` ContextVars. TUI/desktop:
    those are cleared but the subprocess inherits ``HERMES_SESSION_KEY`` -> ``platform="tui"``
    for the TUI poller. ``HERMES_SESSION_ID`` is deliberately NOT a fallback: it is set for
    every CLI/ACP invocation and would auto-subscribe every CLI run."""
    from gateway.session_context import get_session_env as env
    platform, chat_id = env("HERMES_SESSION_PLATFORM", ""), env("HERMES_SESSION_CHAT_ID", "")
    if not platform or not chat_id:
        session_key = env("HERMES_SESSION_KEY", "") or os.environ.get("HERMES_SESSION_KEY", "")
        if not session_key:
            return None
        platform, chat_id = "tui", session_key
    chat_type = env("HERMES_SESSION_CHAT_TYPE", "") or None
    thread_id = env("HERMES_SESSION_THREAD_ID", "") or None
    message_id = env("HERMES_SESSION_MESSAGE_ID", "") or ""
    notifier_profile = env("HERMES_SESSION_PROFILE", "")
    if not notifier_profile:
        from hermes_cli.profiles import current_profile_name
        notifier_profile = current_profile_name("default")
    if platform == "tui":
        # The inherited key can be stale after a compaction fork (#110068): bind the
        # subscription to the live continuation tip, not the key the process started with.
        chat_id = _live_tui_session_key(chat_id, notifier_profile)
    delivery_metadata: dict[str, Any] = {
        k: v for k, v in (
            ("thread_id", thread_id), ("chat_type", chat_type),
            ("scope_id", env("HERMES_SESSION_SCOPE_ID", "")),
            ("parent_chat_id", env("HERMES_SESSION_PARENT_CHAT_ID", "")),
        ) if v}
    if (platform.lower() == "telegram" and thread_id
            and (chat_type or "").lower() in {"dm", "direct", "private"}):
        delivery_metadata["telegram_dm_topic_reply_fallback"] = True
        if str(thread_id) not in {"", "1"}:
            delivery_metadata["direct_messages_topic_id"] = str(thread_id)
        if message_id:
            delivery_metadata["telegram_reply_to_message_id"] = str(message_id)
    return dict(
        platform=platform, chat_id=chat_id, chat_type=chat_type, thread_id=thread_id,
        user_id=env("HERMES_SESSION_USER_ID", "") or None,
        user_id_alt=env("HERMES_SESSION_USER_ID_ALT", "") or None,
        notifier_profile=notifier_profile,
        delivery_mode="notify+wake" if platform != "tui" else None,
        delivery_metadata=delivery_metadata or None)


def _maybe_auto_subscribe(conn: Any, task_id: str) -> bool:
    """Subscribe the calling session to completion/block events; True iff a row was
    written (surfaced as ``subscribed`` so an orchestrator can fall back to explicit
    ``kanban_notify-subscribe``). Gated by ``kanban.auto_subscribe_on_create`` (default
    True). Failures are logged and swallowed: bookkeeping must never fail kanban_create."""
    try:
        if not cfg_get(load_config(), "kanban", "auto_subscribe_on_create", default=True):
            return False
    except Exception:
        pass  # unreadable config keeps the user-friendly default (True)
    target = None
    try:
        target = _resolve_notify_target()
        if target is None:
            return False  # CLI / cron / test — no persistent channel
        from hermes_cli import kanban_db_notify as _kbn
        # Inheritance and explicit subscriptions already encode the delivery policy.
        # Auto-subscribe must not turn a passive destination into an agent wake.
        if any(sub["platform"] == target["platform"] and sub["chat_id"] == target["chat_id"]
               and (sub["thread_id"] or "") == (target["thread_id"] or "")
               for sub in _kbn.list_notify_subs(conn, task_id)):
            return True
        _kbn.add_notify_sub(conn, task_id=task_id, **target)
        return True
    except Exception as _exc:
        logger.warning(
            "_maybe_auto_subscribe failed: %r (platform=%r key_set=%r)",
            _exc, target["platform"] if target else "", bool(target and target["chat_id"]))
        return False


@_kanban_handler("kanban_unblock")
def _handle_unblock(args: dict, **kw) -> str:
    """Transition a blocked task to ready, or todo while parents remain open."""
    _reject_delegated_child_mutation("kanban_unblock")
    _require_orchestrator_tool("kanban_unblock")
    tid = args.get("task_id")
    _check(tid, "task_id is required")
    tid = str(tid)
    _enforce_worker_task_ownership(tid)
    with _board(args.get("board")) as (kb, conn):
        _check(kb.unblock_task(conn, tid), f"could not unblock {tid} (not blocked or unknown)")
        return _ok(task_id=tid, **_fields(kb.get_task(conn, tid), ("status",)))


@_kanban_handler("kanban_link")
def _handle_link(args: dict, **kw) -> str:
    """Add a parent→child dependency edge after the fact (cycles/self-links/running
    children → ValueError). A worker linking its OWN running card proves ownership
    with its run id so the dependency-block handoff still works.

    The receipt reports what ACTUALLY happened: ``gated`` is True only when this
    link really demoted a ``ready`` child back to ``todo``, and the child's
    resulting status plus any parents still gating it are read back from the
    board rather than assumed.

    A link mutates the CHILD (edge row + possible demotion + event), so a
    task-scoped worker may only link edges whose child is its own card —
    exactly like ``kanban_unlink``. Orchestrators (no ``HERMES_KANBAN_TASK``)
    route any edge, and the run-id handoff below still proves ownership of a
    RUNNING own card.
    """
    _reject_delegated_child_mutation("kanban_link")
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    _check(parent_id and child_id, "both parent_id and child_id are required")
    parent_id, child_id = str(parent_id), str(child_id)
    # Prompt-injected foreign ids must not let a worker gate (or demote, or
    # append events to) a sibling's card — _reject_delegated_child_mutation
    # only covers delegate children, and _worker_run_id(foreign_child) is None
    # for a non-running foreign task, so the CAS below is NOT the guard.
    _enforce_worker_task_ownership(child_id)
    with _board(args.get("board")) as (kb, conn):
        _existing_task(kb, conn, parent_id)
        previous_status = _existing_task(kb, conn, child_id).status
        gated = kb.link_tasks(
            conn, parent_id=parent_id, child_id=child_id,
            expected_child_run_id=_worker_run_id(child_id))
        child = kb.get_task(conn, child_id)
        unsatisfied = _dependency_gates(kb, conn, child_id)
        return _ok(
            parent_id=parent_id, child_id=child_id,
            # True only when a demotion genuinely occurred, never as a claim.
            gated=gated,
            gated_by=parent_id if gated else None,
            previous_status=previous_status,
            status=child.status if child else previous_status,
            unsatisfied_parents=unsatisfied,
            remaining_gates=bool(unsatisfied),
        )


# --- Controlled planning / lifecycle tools ---------------------------------

def _dependency_gates(kb, conn, task_id: str) -> list[dict]:
    """``[{id, status}]`` for every direct parent still gating ``task_id``."""
    return [{"id": pid, "status": status} for pid, status in kb.unsatisfied_parents(conn, task_id)]


@_kanban_handler("kanban_graph")
def _handle_graph(args: dict, **kw) -> str:
    """Read-only direct parent/child graph for one task.

    Deliberately SELECT-only: no ``recompute_ready``, no readiness promotion and
    no event append ever runs here, so inspecting a graph cannot move a card.
    """
    tid = _require_task_id(args)
    with _board(args.get("board")) as (kb, conn):
        task = _existing_task(kb, conn, tid)
        graph = kb.task_graph_context(conn, tid)
        return _ok(
            task_id=tid,
            task={"id": task.id, "title": task.title, "status": task.status},
            parents=graph["parents"],
            children=graph["children"],
            read_only=True,
        )


@_kanban_handler("kanban_unlink")
def _handle_unlink(args: dict, **kw) -> str:
    """Drop a parent→child edge, mirroring ``kanban_link``'s validation.

    Unlinking can promote the child (releasing its last open parent), which is a
    lifecycle mutation of that card — so a task-scoped worker may only do it to
    its own card, never to a sibling's.
    """
    _reject_delegated_child_mutation("kanban_unlink")
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    _check(parent_id and child_id, "both parent_id and child_id are required")
    parent_id, child_id = str(parent_id), str(child_id)
    _check(parent_id != child_id, "a task cannot depend on itself")
    _enforce_worker_task_ownership(child_id)
    with _board(args.get("board")) as (kb, conn):
        _existing_task(kb, conn, parent_id)
        previous_status = _existing_task(kb, conn, child_id).status
        # A task-scoped worker may only move its OWN card. unlink_tasks'
        # board-wide recompute_ready would promote an unrelated
        # stale-but-eligible sibling (and append its `promoted` event) as a
        # side effect of this worker's unlink, so scope it to the affected
        # child. Orchestrators / CLI / dashboard (no HERMES_KANBAN_TASK) keep
        # the historical board-wide behaviour.
        removed = kb.unlink_tasks(
            conn, parent_id, child_id,
            board_wide_recompute=not _own_task_env(child_id, "HERMES_KANBAN_TASK"))
        child = kb.get_task(conn, child_id)
        status = child.status if child else previous_status
        unsatisfied = _dependency_gates(kb, conn, child_id)
        return _ok(
            parent_id=parent_id, child_id=child_id,
            removed=removed,
            previous_status=previous_status,
            status=status,
            # True only if the removal actually landed the child on 'ready'.
            promoted=bool(removed and status == "ready" and previous_status != "ready"),
            gated=bool(unsatisfied),
            unsatisfied_parents=unsatisfied,
        )


@_kanban_handler("kanban_promote")
def _handle_promote(args: dict, **kw) -> str:
    """Promote a triage task into the normal flow, reporting where it landed."""
    _reject_delegated_child_mutation("kanban_promote")
    _require_orchestrator_tool("kanban_promote")
    tid = args.get("task_id")
    _check(tid, "task_id is required")
    tid = str(tid)
    _enforce_worker_task_ownership(tid)
    reason = _redact_opt(args.get("reason") or None)
    with _board(args.get("board")) as (kb, conn):
        _existing_task(kb, conn, tid)
        ok, err, _landed = kb.promote_triage_task(
            conn, tid, actor=_persisted_identity(), reason=reason)
        _check(ok, err)
        task = kb.get_task(conn, tid)
        unmet = _dependency_gates(kb, conn, tid)
        out: dict = {
            "task_id": tid,
            "status": task.status if task else None,
            "ready": bool(task and task.status == "ready"),
            "unmet_parents": unmet,
        }
        if unmet:
            out["gate_reason"] = "unmet parent gate(s): " + ", ".join(
                f"{m['id']} ({m['status']})" for m in unmet)
        return _ok(**out)


def _archive_impact_receipt(kb, conn, dependents: list[str], before: dict) -> dict:
    """Classify each dependent AFTER the archive + readiness recompute.

    Three disjoint questions, answered from the board rather than assumed:
    ``changed`` — status actually moved (with before/after), ``waiting`` — still
    held back and why (remaining parent gates, a block hold, or BOTH at once —
    a blocked dependent is never reduced to one cause), ``ready_followup``
    — now runnable and needing assignment or dispatch.
    """
    changed, waiting, ready_followup = [], [], []
    for dep_id in dependents:
        task = kb.get_task(conn, dep_id)
        if task is None:
            continue
        prev = before.get(dep_id)
        prev_status = prev.status if prev else None
        if prev_status != task.status:
            changed.append({
                "id": task.id, "title": task.title,
                "before": prev_status, "after": task.status,
            })
        gates = _dependency_gates(kb, conn, dep_id)
        held = task.status == "blocked"
        if gates or held:
            # A blocked dependent can be GATED and HELD at the same time (its
            # block hold plus another parent still open). Expose both causes:
            # `hold` is reported whenever the card is blocked, and `reason`
            # names every remaining cause instead of picking the first one.
            hold_kind = getattr(task, "block_kind", None)
            causes: list[str] = []
            if gates:
                causes.append("waiting on unsatisfied parent dependencies")
            if held:
                causes.append(f"held in blocked ({hold_kind or 'unclassified'})")
            entry = {
                "id": task.id, "title": task.title, "status": task.status,
                "unsatisfied_parents": gates,
                "reason": "; ".join(causes),
            }
            if held:
                entry["hold"] = {"kind": hold_kind}
            waiting.append(entry)
        if task.status == "ready":
            ready_followup.append({
                "id": task.id, "title": task.title, "assignee": task.assignee,
                "needs_assignment": not task.assignee,
                "changed": prev_status != "ready",
            })
    return {"changed": changed, "waiting": waiting, "ready_followup": ready_followup}


def _review_receipt(before: dict, after: dict) -> dict:
    """Same-card review evidence for an archive/block receipt.

    The review flow is same-card (``request_review`` + ``claim_review_task``
    claim THIS card into a reviewer run), so the receipt names the review run
    the move closed or preserved — nothing is inferred into a second card, and
    nothing outside this card is stopped on its behalf. What the move affects
    elsewhere is carried by the ordinary dependency graph
    (``linked_before``/``linked_after``) plus the archive impact receipt, which
    report dependents without mutating them.
    """
    run = after.get("active_review_run") or before.get("active_review_run")
    if run:
        was = ("active before this move and closed by it"
               if before.get("active_review_run") and not after.get("active_review_run")
               else "active")
        note = (f"same-card review run {run['id']} (profile {run['profile']}, "
                f"outcome {run['outcome']}) identified from its claimed event "
                f"source_status=review — {was}")
    elif after.get("review_runs"):
        ids = ", ".join(str(r["id"]) for r in after["review_runs"])
        note = f"no active review run; same-card review run(s) {ids} preserved in history"
    else:
        note = "no same-card review run"
    return {
        "lane": after.get("lane"), "note": note,
        "reviewer": after.get("reviewer"), "active_review_run": run,
        "review_runs": after.get("review_runs"),
        "linked_before": before.get("linked"), "linked_after": after.get("linked"),
    }


def _archive_reason(args: dict) -> str:
    """Required archive rationale, checked BEFORE the board is opened.

    Missing, non-string, empty and whitespace-only input is rejected here, ahead
    of ``_board``/``_existing_task``/``archive_task``, so a bad reason can never
    leave a status flip, an event or a run behind.

    The wording is the agent's own (no normalization, no generic stand-in), but
    it crosses the SAME ``_redact`` boundary every other agent-authored free-text
    field on this board crosses before it is stored — ``kanban_block`` reason,
    ``kanban_complete`` summary/result, ``kanban_comment`` body, and this PR's
    own ``kanban_promote`` reason. The reason is durable (``task_events.payload``)
    and is echoed back by ``kanban_show``, so an unredacted secret here would be
    both stored and re-served to the model. Plain prose passes through verbatim;
    only credential-shaped text is masked.
    """
    reason = args.get("reason")
    if reason is None:
        raise _Reject(
            "reason is required — say why this task is being archived; it is "
            "recorded on the archive event. Nothing changed.")
    if not isinstance(reason, str):
        raise _Reject(
            f"reason must be a string, got {type(reason).__name__}. Nothing changed.")
    if not reason.strip():
        raise _Reject(
            "reason must not be empty or whitespace-only — say why this task is "
            "being archived. Nothing changed.")
    return _redact(reason)


# The agent-facing archive runs the ONE shared archive policy (see
# ``hermes_cli.kanban_db.ARCHIVE_PROTECTED_STATUSES``): ``ready``/``running``/
# ``review`` are refused with a block/stop-first instruction, everything else —
# every other valid status AND legacy raw statuses such as ``completed`` —
# archives directly, and ``archived`` is the already-archived no-op. This tool
# NEVER blocks a task on the caller's behalf, so a refusal never parks a card.
# It is also the ONLY surface that supplies a ``reason``, so it is the only one
# whose ``archived`` event carries a payload.
@_kanban_handler("kanban_archive")
def _handle_archive(args: dict, **kw) -> str:
    """Archive a task under the shared policy, with an impact + review receipt."""
    _reject_delegated_child_mutation("kanban_archive")
    _require_orchestrator_tool("kanban_archive")
    tid = args.get("task_id")
    _check(tid, "task_id is required")
    tid = str(tid)
    # Validated first: an invalid reason must fail with no status/event/run
    # side effects, so nothing below it may have run yet.
    reason = _archive_reason(args)
    _enforce_worker_task_ownership(tid)
    with _board(args.get("board")) as (kb, conn):
        task = _existing_task(kb, conn, tid)
        before_status = task.status
        if before_status == "archived":
            raise _Reject(f"{tid} is already archived; nothing changed")
        # Review association and dependents are snapshotted BEFORE the move so
        # the receipt reports real before/after instead of assuming it.
        review_before = kb.review_association(conn, tid)
        dependents = kb.child_ids(conn, tid)
        before = {d: kb.get_task(conn, d) for d in dependents}
        # One shared policy with the CLI/dashboard/DB lifecycle: the refusal
        # copy (block/stop first, live worker, in-flight spawn) comes from the
        # same writer that enforces it, and nothing is archived on refusal.
        # Only this tool hands the writer a payload, so only agent archives
        # carry provenance; refusals and the already-archived no-op above write
        # no event at all. ``actor`` is the persisted runtime identity, never a
        # caller argument (board records feed future workers' prompts).
        ok, why = kb.archive_task(
            conn, tid, with_reason=True,
            event_payload={
                "source": "kanban_archive",
                "actor": _persisted_identity(),
                "reason": reason,
            },
        )
        if not ok:
            raise _Reject(why or f"kanban_archive refused: {tid} was not archived")
        receipt = _archive_impact_receipt(kb, conn, dependents, before)
        return _ok(
            task_id=tid, status="archived", previous_status=before_status,
            reason=reason,
            dependents=receipt,
            review=_review_receipt(review_before, kb.review_association(conn, tid)),
        )


@_kanban_handler("kanban_decompose")
def _handle_decompose(args: dict, **kw) -> str:
    """Apply an agent-authored child graph with no auxiliary/LLM call."""
    _reject_delegated_child_mutation("kanban_decompose")
    _require_orchestrator_tool("kanban_decompose")
    tid = args.get("task_id")
    _check(tid, "task_id is required")
    tid = str(tid)
    _enforce_worker_task_ownership(tid)
    children = args.get("children")
    if not isinstance(children, list) or not children:
        return tool_error("kanban_decompose: children must be a non-empty array of child specs")
    with _board(args.get("board")) as (kb, conn):
        _existing_task(kb, conn, tid)
        from hermes_cli.kanban_decompose import apply_explicit_decomposition
        # Validation happens before any write and the fan-out is one atomic
        # transaction, so a bad graph leaves the board byte-for-byte unchanged.
        child_ids = apply_explicit_decomposition(
            conn, tid, children=list(children), author=_persisted_identity())
        root = kb.get_task(conn, tid)
        kids = [kb.get_task(conn, cid) for cid in child_ids]
        return _ok(
            task_id=tid,
            child_ids=child_ids,
            status=root.status if root else None,
            root=_fields(root, ("id", "status", "assignee")) | {
                # The root waits on every child; these are its parent edges.
                "parents": kb.parent_ids(conn, tid),
            },
            children=[
                {
                    "id": k.id, "title": k.title, "status": k.status,
                    "assignee": k.assignee, "parents": kb.parent_ids(conn, k.id),
                }
                for k in kids if k is not None
            ],
        )


# --- Profile routing: reassign + discovery ------------------------------------

# ``profile.yaml`` is the ONLY file discovery reads: it is deliberately tiny
# metadata ABOUT the profile (see hermes_cli/profiles.py), never config.yaml,
# .env, auth.json, SOUL.md or any prompt/skill content. The roster consumers
# (the decomposer prompt, ``hermes profile list``) are bounded, so the strings
# handed to a model here are bounded too — the describer's own contract caps a
# description at 280 characters.
_DESCRIPTOR_DESCRIPTION_MAX = 400
_DESCRIPTOR_DISPLAY_NAME_MAX = 64  # the writer refuses anything longer


def _bounded_text(value: Any, limit: int) -> str:
    """Single-line, stripped, hard-capped. Never raises."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _profile_descriptor(profile_dir) -> dict:
    """``profile.yaml`` status + the fields the roster contract exposes.

    Status is explicit rather than implied: ``missing`` (no descriptor — a
    perfectly normal profile), ``unreadable`` (present but the filesystem said
    no), ``invalid`` (present but not a safely-parsed mapping, or a mapping
    whose values the canonical loader refuses to normalise), ``ok``. A
    malformed descriptor must be *reported*, never silently downgraded to
    "absent" and never raised: one broken profile must not take discovery (or
    ``hermes profile list``) down with it. Parsing goes through the safe loader
    only, and only the exception class name is surfaced — a YAML scanner error
    is allowed to quote the document it choked on.
    """
    path = profile_dir / "profile.yaml"
    try:
        path.stat()
    except FileNotFoundError:
        return {"status": "missing"}
    except OSError:
        # Present for ``stat`` purposes but not readable — reported, never raised.
        return {"status": "unreadable"}
    if not path.is_file():
        # e.g. a directory named profile.yaml: it exists and is not a descriptor.
        return {"status": "invalid", "detail": "not-a-file"}
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return {"status": "invalid", "detail": "not-utf8"}
    except OSError:
        return {"status": "unreadable"}
    try:
        import hermes_yaml as yaml

        data = yaml.safe_load(raw)
    except Exception as exc:  # noqa: BLE001 - every parser failure is "invalid"
        return {"status": "invalid", "detail": type(exc).__name__}
    if not isinstance(data, dict):
        return {"status": "invalid", "detail": "not-a-mapping"}

    # Canonical loader for the OK path: the same normalisation
    # ``list_profiles()`` / the decomposer roster / ``hermes profile list`` use.
    from hermes_cli.profiles import read_profile_meta

    try:
        meta = read_profile_meta(profile_dir)
    except Exception as exc:  # noqa: BLE001 - a malformed descriptor must not abort the roster
        # The loader documents itself as never raising, but a syntactically
        # valid mapping can still carry a value it cannot normalise — a list
        # where ``role`` must be a scalar raises inside its ``PROFILE_ROLES``
        # membership test. That is a malformed descriptor, so it is reported as
        # ``invalid`` with the exception class only: the message may quote the
        # document it failed on, and one broken profile must not take
        # discovery down with it.
        return {"status": "invalid", "detail": type(exc).__name__}
    # Descriptor text is an operator-authored free-text field heading straight
    # to a model, so it crosses the same ``_redact`` boundary as every other
    # agent-visible free text here. (Nothing is written back — this is only the
    # view discovery returns.)
    return {
        "status": "ok",
        "description": _bounded_text(_redact(meta.get("description") or ""),
                                     _DESCRIPTOR_DESCRIPTION_MAX),
        "display_name": _bounded_text(_redact(meta.get("display_name") or ""),
                                      _DESCRIPTOR_DISPLAY_NAME_MAX),
        "role": meta.get("role"),
    }


@_kanban_handler("kanban_discover")
def _handle_discover(args: dict, **kw) -> str:
    """Read-only roster of every profile this home can spawn work to.

    Home-scoped, not board-scoped: no board is opened (opening one can
    initialize the database, which would be a write), no file is written and
    no describer is run. The enumeration is ``hermes_cli.profiles``' own, so
    the roster and the ``kanban_create`` / ``kanban_reassign`` guards can
    never disagree about what a valid assignee is.
    """
    from hermes_cli import profiles as profiles_mod

    names = profiles_mod.list_profile_names()
    entries = [
        {
            "name": name,
            "is_default": name == "default",
            "descriptor": _profile_descriptor(profiles_mod.get_profile_dir(name)),
        }
        for name in names
    ]
    return _ok(count=len(entries), profiles=entries)


@_kanban_handler("kanban_reassign")
def _handle_reassign(args: dict, **kw) -> str:
    """Hand a card to another profile through the CLI's own assign kernel.

    Deliberately not a second implementation: ``kanban_db.reassign_task`` /
    ``assign_task`` are what ``hermes kanban reassign`` runs, including its
    active-worker guard (a card with a live claim is refused inside the write
    transaction, before any UPDATE) and its ``assigned`` audit event. The
    destination profile is validated before the board is opened, so a typo
    changes nothing.
    """
    _reject_delegated_child_mutation("kanban_reassign")
    _require_orchestrator_tool("kanban_reassign")
    tid = args.get("task_id")
    _check(tid, "task_id is required")
    tid = str(tid)
    assignee = args.get("assignee")
    _check(assignee and str(assignee).strip(),
           "assignee is required — name the destination profile. Nothing changed.")
    assignee = _require_installed_profile("assignee", assignee)
    _enforce_worker_task_ownership(tid)
    reclaim = _parse_bool_arg(args, "reclaim")
    reason = _redact_opt(args.get("reason") or None)
    with _board(args.get("board")) as (kb, conn):
        task = _existing_task(kb, conn, tid)
        before = _fields(task, ("id", "status", "assignee"))
        events_before = len(kb.list_events(conn, tid))
        # Shared machinery: reclaim (optional) + assign, same guards as the CLI.
        ok = kb.reassign_task(conn, tid, assignee, reclaim_first=reclaim, reason=reason)
        if not ok:
            landed = kb.get_task(conn, tid)
            _check(landed is not None, f"task {tid} not found — nothing changed")
            _check(False,
                   f"cannot reassign {tid}: it is still running under a live claim "
                   f"(status {landed.status}). Set reclaim=true to release the claim "
                   f"first — the CLI equivalent is "
                   f"`hermes kanban reassign {tid} {assignee} --reclaim`. Nothing changed.")
        landed = _existing_task(kb, conn, tid)
        events = kb.list_events(conn, tid)
        audit = events[-1] if len(events) > events_before else None
        _check(audit is not None and audit.kind == "assigned",
               f"reassigned {tid} but could not read back its audit event")
        return _ok(
            task_id=tid,
            status=landed.status,
            assignee=landed.assignee,
            previous_assignee=before["assignee"],
            claim_reclaimed=reclaim,
            # Readback, not a restatement of the request: the stored event.
            audit=_fields(audit, _EVENT_FIELDS),
        )


# --- Registration (order preserved: it is the order tools appear in the schema) ---

# kanban_list / kanban_unblock route the board and are hidden from task workers.
# kanban_archive / kanban_promote / kanban_decompose are likewise board-level
# lifecycle moves: a task-scoped worker must not archive, promote or fan out a
# card (its own escape hatch around the complete/review gates, or a sibling's).
# kanban_reassign hands a card to a different profile — the same class of move.
# kanban_discover stays worker-visible: it is read-only (no board, no writes)
# and a worker that is about to fan out needs the same roster kanban_create
# validates against.
_ORCHESTRATOR_TOOLS = frozenset({
    "kanban_list", "kanban_unblock",
    "kanban_archive", "kanban_promote", "kanban_decompose", "kanban_reassign",
})
# Per-tool visibility overrides layered on the two gates above. A tool not
# named here keeps the default (orchestrator tools are hidden from workers,
# everything else is worker-visible).
_TOOL_GATES = {
    "kanban_complete": _check_kanban_complete_mode,
}
_TOOLS = (
    ("kanban_show", KANBAN_SHOW_SCHEMA, _handle_show, "📋"),
    ("kanban_list", KANBAN_LIST_SCHEMA, _handle_list, "📋"),
    ("kanban_discover", KANBAN_DISCOVER_SCHEMA, _handle_discover, "🔎"),
    ("kanban_graph", KANBAN_GRAPH_SCHEMA, _handle_graph, "🕸"),
    ("kanban_complete", KANBAN_COMPLETE_SCHEMA, _handle_complete, "✔"),
    ("kanban_block", KANBAN_BLOCK_SCHEMA, _handle_block, "⏸"),
    ("kanban_schedule", KANBAN_SCHEDULE_SCHEMA, _handle_schedule, "⏰"),
    ("kanban_request_review", KANBAN_REQUEST_REVIEW_SCHEMA, _handle_request_review, "👀"),
    ("kanban_request_changes", KANBAN_REQUEST_CHANGES_SCHEMA, _handle_request_changes, "↩"),
    ("kanban_heartbeat", KANBAN_HEARTBEAT_SCHEMA, _handle_heartbeat, "💓"),
    ("kanban_comment", KANBAN_COMMENT_SCHEMA, _handle_comment, "💬"),
    ("kanban_attach", KANBAN_ATTACH_SCHEMA, _handle_attach, "📎"),
    ("kanban_attach_url", KANBAN_ATTACH_URL_SCHEMA, _handle_attach_url, "📎"),
    ("kanban_attachments", KANBAN_ATTACHMENTS_SCHEMA, _handle_attachments, "📎"),
    ("kanban_create", KANBAN_CREATE_SCHEMA, _handle_create, "➕"),
    ("kanban_reassign", KANBAN_REASSIGN_SCHEMA, _handle_reassign, "🔀"),
    ("kanban_decompose", KANBAN_DECOMPOSE_SCHEMA, _handle_decompose, "🗂"),
    ("kanban_promote", KANBAN_PROMOTE_SCHEMA, _handle_promote, "⏫"),
    ("kanban_unblock", KANBAN_UNBLOCK_SCHEMA, _handle_unblock, "▶"),
    ("kanban_link", KANBAN_LINK_SCHEMA, _handle_link, "🔗"),
    ("kanban_unlink", KANBAN_UNLINK_SCHEMA, _handle_unlink, "⛓"),
    ("kanban_archive", KANBAN_ARCHIVE_SCHEMA, _handle_archive, "🗄"))

for _name, _sch, _handler, _emoji in _TOOLS:
    _gate = _TOOL_GATES.get(_name) or (
        _check_kanban_orchestrator_mode if _name in _ORCHESTRATOR_TOOLS else _check_kanban_mode)
    registry.register(name=_name, toolset="kanban", schema=_sch, handler=_handler, emoji=_emoji,
                      check_fn=_gate)
