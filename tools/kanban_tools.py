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
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.goals import judge_goal
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get, load_config
from tools.kanban_tools_schemas import (
    KANBAN_ATTACH_SCHEMA,
    KANBAN_ATTACH_URL_SCHEMA, KANBAN_ATTACHMENTS_SCHEMA, KANBAN_BLOCK_SCHEMA, KANBAN_COMMENT_SCHEMA,
    KANBAN_COMPLETE_SCHEMA, KANBAN_CREATE_SCHEMA, KANBAN_HEARTBEAT_SCHEMA, KANBAN_LINK_SCHEMA,
    KANBAN_LIST_SCHEMA, KANBAN_REQUEST_CHANGES_SCHEMA, KANBAN_REQUEST_REVIEW_SCHEMA,
    KANBAN_SHOW_SCHEMA, KANBAN_UNBLOCK_SCHEMA)

logger = logging.getLogger(__name__)

KANBAN_LIST_DEFAULT_LIMIT = 50
KANBAN_LIST_MAX_LIMIT = 200


# --- Gating ---

def _profile_has_kanban_toolset() -> bool:
    # load_config() is mtime-cached and check_fn results are TTL-cached (~30s).
    try:
        return "kanban" in load_config().get("toolsets", [])
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


def _check_kanban_mode() -> bool:
    """Lifecycle tools: dispatcher workers + profiles with the ``kanban`` toolset."""
    return _visible(to_env_worker=True)


def _check_kanban_orchestrator_mode() -> bool:
    """Board-routing tools (kanban_list, kanban_unblock): hidden from task workers."""
    return _visible(to_env_worker=False)


# --- Shared helpers: validation failures raise _Reject; _kanban_handler renders it ---

class _Reject(Exception):
    """Carries a finished ``tool_error`` payload out of a validation helper."""

    def __init__(self, message: str):
        super().__init__(tool_error(message))


def _check(cond: Any, message: str) -> None:
    """Reject (as a tool error) unless ``cond`` is truthy."""
    if not cond:
        raise _Reject(message)


def _kanban_handler(tool_name: str) -> Callable:
    """Wrap a handler so every failure is a structured tool error. ``ValueError``
    (invalid board slug, DB validation such as cycle/self-link, ``AttachmentTooLarge``)
    is reported without a traceback; anything else is logged with ``logger.exception``."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(args: dict, **kw) -> str:
            try:
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


def _default_task_id(arg: Optional[str]) -> Optional[str]:
    """``task_id`` arg or the dispatcher's env var. A delegate child or an
    in-process cron job must never inherit the worker's task id implicitly."""
    if arg:
        return arg
    if _is_delegated_child_context() or not _is_dispatcher_owned_worker():
        return None
    return os.environ.get("HERMES_KANBAN_TASK") or None


def _require_task_id(args: dict) -> str:
    tid = _default_task_id(args.get("task_id"))
    _check(tid, "task_id is required (or set HERMES_KANBAN_TASK in the env)")
    return tid


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


def _stamp_worker_session_metadata(
    task_id: str, metadata: Optional[dict], *, finalize_conclusive: bool = False,
) -> Optional[dict]:
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
    resolution, task-scope ownership. Returns the task id."""
    _reject_delegated_child_mutation(tool_name)
    tid = _require_task_id(args)
    _enforce_worker_task_ownership(tid)
    return tid


def _require_orchestrator_tool(tool_name: str) -> None:
    """The check_fn already hides orchestrator tools from workers; this catches
    a stale registration or test harness routing a worker here anyway."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        raise _Reject(
            f"{tool_name} is orchestrator-only; dispatcher-spawned workers must use "
            "kanban_complete, kanban_block, kanban_heartbeat, or kanban_comment for their "
            "assigned task.")


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
    "provider_override completion_contract last_failure_error".split())
_TASK_SUMMARY_FIELDS = tuple(
    "id title assignee status priority tenant workspace_kind workspace_path project_id created_by "
    "created_at started_at completed_at current_run_id model_override provider_override".split())
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
        verdict, reason, _, _, _ = judge_goal(
            goal=f"{task.title}\n\n{task.body or ''}".strip(), last_response=evidence.strip())
    except Exception as judge_exc:
        logger.warning(
            "goal judge check failed, allowing lifecycle handoff: %s", judge_exc, exc_info=True)
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


def heartbeat_current_worker_from_env() -> bool:
    """Claim extension + board heartbeat for the current worker; True iff a write was
    attempted. ``HERMES_KANBAN_RUN_ID`` pins the run row so a reclaimed stale run is not
    heartbeated; ``HERMES_KANBAN_CLAIM_LOCK`` absent -> default claimer (local workers)."""
    global _auto_heartbeat_last_attempt
    tid = os.environ.get("HERMES_KANBAN_TASK")
    now = time.monotonic()
    if not tid or (now - _auto_heartbeat_last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
        return False
    _auto_heartbeat_last_attempt = now
    try:
        from hermes_cli import kanban_db_dispatch as kbd
        with _board(None, quiet_close=True) as (kb, conn):
            ops = ((kb.heartbeat_claim, {"claimer": os.environ.get("HERMES_KANBAN_CLAIM_LOCK")}),
                   (kbd.heartbeat_worker, {"note": None, "expected_run_id": _worker_run_id(tid)}))
            for fn, kwargs in ops:
                op = fn.__name__
                try:
                    fn(conn, tid, **kwargs)
                except Exception:
                    logger.debug("auto-heartbeat: %s failed", op, exc_info=True)
        return True
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
    steer was injected; never raises. Own comments (``HERMES_PROFILE``) are skipped."""
    global _comment_poll_last_attempt
    tid = os.environ.get("HERMES_KANBAN_TASK")
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
    own = (os.environ.get("HERMES_PROFILE") or "").strip()
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
    tid = _require_task_id(args)
    with _board(args.get("board")) as (kb, conn):
        task = _existing_task(kb, conn, tid)
        return json.dumps({
            "task": _fields(task, _TASK_FIELDS),
            "parents": kb.parent_ids(conn, tid),
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
    metadata = _stamp_worker_session_metadata(tid, metadata, finalize_conclusive=True)
    # FLEET tool-evidence gate: refuse a completion from a run that made ZERO
    # non-kanban tool calls — there is no evidence the work was done. Fails
    # OPEN for orchestrator/CLI paths and unreadable transcripts (t_8fc16a73).
    evidence_rejection = _complete_tool_evidence_rejection(tid)
    if evidence_rejection is not None:
        return evidence_rejection
    # FLEET: the uncommitted-work gate was retired from core on 2026-09-09
    # (manifest gate-core-retire-20260909). Charter rule 3 ("done requires a
    # commit") is enforced by the kanban-completion-gate PLUGIN on upstream's
    # pre_tool_call hook, which runs before this body is reached.
    #
    # FLEET CI gate: for a tenant that declares ``ci_gate``, done requires a
    # GREEN CHECK RUN on the PR's head SHA — not a local test run in the card's
    # own worktree. Fails CLOSED, unlike the evidence gate above: an unreadable
    # check is not a green one. Out-of-scope tenants pass straight through.
    ci_rejection = _complete_ci_gate_rejection(tid)
    if ci_rejection is not None:
        return ci_rejection
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
        task = kb.get_task(conn, tid)
        _check(ok, (task.last_failure_error if task else None) or
               f"could not complete {tid} (unknown id, stale run, or already terminal)")
        run = kb.latest_run(conn, tid)
        return _ok(task_id=tid, run_id=run.id if run else None)


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
        ok = kb.block_task(conn, tid, reason=reason, kind=kind, expected_run_id=_worker_run_id(tid))
        _check(ok, f"could not block {tid} (unknown id or not in running/ready)")
        return _ok_landed(kb, conn, tid, "blocked", block_kind=kind)


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
    metadata = _stamp_worker_session_metadata(tid, metadata)
    # Reviewer is model-supplied free text stored durably on the event payload.
    reviewer = _redact_opt(args.get("reviewer") or None)
    if reviewer:
        from hermes_cli.profiles import list_profile_names, profile_exists

        # A non-profile reviewer would park the card in `review` on an assignee
        # the dispatcher can never spawn (#106163).
        _check(profile_exists(reviewer),
               f"reviewer profile {reviewer!r} is not installed. "
               f"Installed profiles: {', '.join(list_profile_names())}")
    with _board(args.get("board")) as (kb, conn):
        task = kb.get_task(conn, tid)
        _goal_gate("kanban_request_review", task, tid, summary)
        # FLEET pre-review build gate: refuse the transition when a worktree
        # card fails any rung of the review ladder (lint, typecheck,
        # import/build, focused tests). Zero LLM tokens. On failure the card
        # stays in its current builder lane, an auto-comment names the failing
        # rung and carries only that rung's output, and NO failure is counted
        # against the card.
        gate_bounce = _run_pre_review_gate(task)
        if gate_bounce is not None:
            tail = _run_gate_output_tail(gate_bounce.output)
            kb.add_comment(
                conn, tid, author="pre-review-gate",
                body=(
                    f"Pre-review gate FAILED on the '{gate_bounce.rung}' rung — "
                    "review was not started.\n\nThe worktree must pass the review "
                    "ladder (lint, typecheck, import/build, focused tests) before "
                    "entering review. Fix the failure and request review again.\n\n"
                    "```\n" + tail + "\n```"
                ),
            )
            # Record WHICH rung bounced so the ladder's own value stays
            # measurable — a rung that never catches anything is removable.
            with kb.write_txn(conn):
                kb._append_event(
                    conn, tid, "gate_bounced", {"rung": gate_bounce.rung},
                    run_id=_worker_run_id(tid),
                )
            return tool_error(
                f"Pre-review gate failed for {tid} on the '{gate_bounce.rung}' rung; "
                f"review not started. The task stays in its current lane (no failure "
                f"counted) and a comment carries the last {PRE_REVIEW_TAIL_LINES} lines "
                f"of {gate_bounce.rung} output. Fix the {gate_bounce.rung} and call "
                f"kanban_request_review again.\n\n" + tail
            )
        ok, fail_reason = kb.request_review(
            conn, tid, summary=summary, metadata=metadata, reviewer=reviewer,
            expected_run_id=_worker_run_id(tid), with_reason=True)
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
    _check(tid, "task_id is required (use the current task id if that's what "
                "you mean — pulls from env but kept explicit here)")
    body = _redact(_require_text(args, "body"))
    # Author comes from the worker's runtime identity, never caller args: comments are
    # injected into future workers' system prompts, so an args["author"] override could
    # forge a directive from ``hermes-system``. Cross-task commenting stays unrestricted —
    # it is the handoff channel between tasks.
    # Comments are injected into the next worker's system prompt by ``build_worker_context`` as
    # ``**{author}** (timestamp): {body}`` — accepting an ``args["author"]`` override let a worker forge a
    # comment from an authoritative-looking name like ``hermes-system`` and poison the future-worker context
    # with what reads as a system directive. See #19713.
    author = os.environ.get("HERMES_PROFILE") or "worker"
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


@_kanban_handler("kanban_create")
def _handle_create(args: dict, **kw) -> str:
    """Create a (child) task; orchestrator workers use this to fan out."""
    _reject_delegated_child_mutation("kanban_create")
    title = _require_text(args, "title")
    assignee = args.get("assignee")
    _check(assignee, "assignee is required — name the profile that should execute this "
                     "task (the dispatcher will only spawn tasks with an assignee)")
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
    model_override, provider_override = args.get("model"), args.get("provider")
    _check(model_override or not provider_override, "'provider' requires 'model' to be set as well")
    parents = _coerce_str_list(args.get("parents") or [], "parents", "task ids")
    # FLEET: a HELD card is blocked/operator_hold from birth — only a human
    # `kanban unblock` releases it (charter §5: every job parent and every
    # deploy card is created this way).
    hold = _parse_bool_arg(args, "hold")
    initial_status = "blocked" if hold else str(args.get("initial_status") or "running")
    _parked: dict = {}
    with _board(args.get("board")) as (kb, conn):
        from tools.async_delegation import _current_origin_session_id
        self_tid = (os.environ.get("HERMES_KANBAN_TASK")
                    if _is_dispatcher_owned_worker() else None)
        self_task = kb.get_task(conn, self_tid) if self_tid else None
        # The worker/API runtime may be transient; the owning task's origin is durable.
        session_id = (args.get("session_id") or (self_task.session_id if self_task else None)
                      or _current_origin_session_id() or os.environ.get("HERMES_SESSION_ID"))
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
            initial_status=initial_status,
            block_kind=("operator_hold" if hold else None),
            # FLEET: explicit max_cost wins; otherwise inherit
            # kanban.default_max_cost exactly as the CLI does, through the same
            # shared resolver. An absent config key leaves the card uncapped;
            # garbage or a negative value is rejected there, never silently
            # uncapped.
            max_cost=_resolved_create_max_cost(kb, args),
            # FLEET / charter §4: an explicit estimate writes that number; omit
            # it and create_task writes the conservative auto-points
            # placeholder comment, so no card leaves the mint path unestimated.
            points=_opt_int(args.get("points")),
            _assignee_parked=_parked,
            created_by=os.environ.get("HERMES_PROFILE") or "worker", session_id=session_id)
        landed = _fields(kb.get_task(conn, new_tid), _CREATED_FIELDS)
        payload = dict(task_id=new_tid, **landed,
                       subscribed=_maybe_auto_subscribe(conn, new_tid))
        if _parked:
            # FLEET: an unknown assignee parks the card in triage rather than
            # stranding it — say so, or the creator never learns why nothing ran.
            payload["assignee_parked"] = _parked.get("assignee")
            payload["notice"] = (
                f"assignee {_parked.get('assignee')!r} is not a real profile; task "
                "parked in triage for the PM to accept/reject. Transfer to a valid "
                "assignee to dispatch."
            )
        return _ok(**payload)


def _resolved_create_max_cost(kb, args: dict):
    """FLEET: explicit ``max_cost`` else ``kanban.default_max_cost``; None = uncapped."""
    max_cost = args.get("max_cost")
    if max_cost is not None:
        return max_cost
    try:
        return kb.resolve_default_max_cost()
    except ValueError as exc:
        raise _Reject(f"kanban_create: {exc}") from exc


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
    notifier_profile = env("HERMES_SESSION_PROFILE", "") or os.environ.get("HERMES_PROFILE")
    if not notifier_profile:
        try:
            from hermes_cli.profiles import get_active_profile_name
            notifier_profile = get_active_profile_name() or "default"
        except Exception:
            notifier_profile = "default"
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
        _check(kb.unblock_task(conn, tid), f"could not unblock/accept {tid} (not blocked/scheduled/triage or unknown)")
        return _ok(task_id=tid, **_fields(kb.get_task(conn, tid), ("status",)))


@_kanban_handler("kanban_link")
def _handle_link(args: dict, **kw) -> str:
    """Add a parent→child dependency edge after the fact (cycles/self-links → ValueError)."""
    _reject_delegated_child_mutation("kanban_link")
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    _check(parent_id and child_id, "both parent_id and child_id are required")
    with _board(args.get("board")) as (kb, conn):
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)
        return _ok(parent_id=parent_id, child_id=child_id)


# --- Registration (order preserved: it is the order tools appear in the schema) ---

# kanban_list / kanban_unblock route the board and are hidden from task workers.
_ORCHESTRATOR_TOOLS = frozenset({"kanban_list", "kanban_unblock"})
_TOOLS = (
    ("kanban_show", KANBAN_SHOW_SCHEMA, _handle_show, "📋"),
    ("kanban_list", KANBAN_LIST_SCHEMA, _handle_list, "📋"),
    ("kanban_complete", KANBAN_COMPLETE_SCHEMA, _handle_complete, "✔"),
    ("kanban_block", KANBAN_BLOCK_SCHEMA, _handle_block, "⏸"),
    ("kanban_request_review", KANBAN_REQUEST_REVIEW_SCHEMA, _handle_request_review, "👀"),
    ("kanban_request_changes", KANBAN_REQUEST_CHANGES_SCHEMA, _handle_request_changes, "↩"),
    ("kanban_heartbeat", KANBAN_HEARTBEAT_SCHEMA, _handle_heartbeat, "💓"),
    ("kanban_comment", KANBAN_COMMENT_SCHEMA, _handle_comment, "💬"),
    ("kanban_attach", KANBAN_ATTACH_SCHEMA, _handle_attach, "📎"),
    ("kanban_attach_url", KANBAN_ATTACH_URL_SCHEMA, _handle_attach_url, "📎"),
    ("kanban_attachments", KANBAN_ATTACHMENTS_SCHEMA, _handle_attachments, "📎"),
    ("kanban_create", KANBAN_CREATE_SCHEMA, _handle_create, "➕"),
    ("kanban_unblock", KANBAN_UNBLOCK_SCHEMA, _handle_unblock, "▶"),
    ("kanban_link", KANBAN_LINK_SCHEMA, _handle_link, "🔗"))

for _name, _sch, _handler, _emoji in _TOOLS:
    _gate = _check_kanban_orchestrator_mode if _name in _ORCHESTRATOR_TOOLS else _check_kanban_mode
    registry.register(name=_name, toolset="kanban", schema=_sch, handler=_handler, emoji=_emoji,
                      check_fn=_gate)


# ===========================================================================
# FLEET ADDITIONS (WeRoll) — the pre-review build ladder, the CI gate and the
# tool-evidence gate. No upstream equivalent; kept in this module so their
# internal references resolve unchanged.
# ===========================================================================

def _connect(board: Optional[str] = None):
    """FLEET: ``(kb, conn)`` for the gate helpers, which manage their own
    connection lifetime rather than nesting inside a ``_board`` block."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    return kb, kbc.connect(board=board)


_PYTHON_CACHE: dict[str, Optional[str]] = {}

_BASE_ARCHIVE_CACHE: dict[str, Optional[str]] = {}

# (rung_key, project_python) -> resolved tool name or None (=> rung skipped)
_TOOL_CACHE: dict[tuple[str, str], Optional[str]] = {}

# project_python -> pytest importable under that interpreter (bool)
_PYTEST_CACHE: dict[str, bool] = {}


def _count_non_kanban_tool_calls(db, session_id: str) -> int:
    """Count tool invocations in a session transcript that are NOT ``kanban_*``.

    A run whose only tool calls are kanban lifecycle calls has made no real
    work — the fabricated-completion signature the tool-evidence gate refuses.
    Assistant ``tool_calls`` and ``tool`` result rows both count; the kanban
    toolset is excluded so a bare ``kanban_complete`` / ``kanban_heartbeat``
    run never counts as evidence.
    """
    try:
        rows = db.get_messages(session_id)
    except Exception:
        return 0
    count = 0
    seen: set = set()
    for m in rows or []:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls")
            if isinstance(tcs, str):
                try:
                    tcs = json.loads(tcs)
                except (ValueError, TypeError):
                    tcs = None
            for tc in tcs or []:
                fn = ((tc or {}).get("function") or {}).get("name") or ""
                key = ("tc", fn)
                if fn and not fn.startswith("kanban_") and key not in seen:
                    seen.add(key)
                    count += 1
        elif role == "tool":
            name = (m.get("tool_name") or "").strip()
            key = ("tool", name)
            if name and not name.startswith("kanban_") and key not in seen:
                seen.add(key)
                count += 1
    return count


def _run_produced_kanban_children(db, session_id: str) -> bool:
    """True when the run called ``kanban_create`` / ``kanban_link``.

    An orchestrator worker decomposes a goal by fanning out kanban_create /
    kanban_link children and then completes its OWN card — with no non-kanban
    tool call in the run. That decomposition IS the work, so the tool-evidence
    gate must exempt it (else a legitimate orchestrator completion is bounced as
    "zero evidence"). Rodge round-1 (t_8fc16a73): an orchestrator worker has
    HERMES_KANBAN_TASK == its own id, so the ``!= task_id`` open path does not
    cover it.
    """
    try:
        rows = db.get_messages(session_id)
    except Exception:
        return False
    for m in rows or []:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if isinstance(tcs, str):
            try:
                tcs = json.loads(tcs)
            except (ValueError, TypeError):
                tcs = None
        for tc in tcs or []:
            fn = ((tc or {}).get("function") or {}).get("name") or ""
            if fn in ("kanban_create", "kanban_link"):
                return True
    return False


def _open_session_db(session_id: str):
    """Locate the session DB for ``session_id``, returning ``(db, profile)``
    or ``(None, None)`` when it cannot be resolved (fail open). Shares the
    lookup between the evidence count and the orchestrator exemption."""
    if not session_id:
        return None, None
    try:
        from tools.session_search_tool import _locate_session_db
        return _locate_session_db(session_id)
    except Exception:
        return None, None


def _consecutive_no_evidence_blocks(conn, task_id: str) -> int:
    """Count trailing ``completion_blocked_no_evidence`` events for ``task_id``.

    A fresh run (last non-this event is a ``completed`` or different kind)
    starts the count at 0; a worker that keeps calling ``kanban_complete``
    without doing work bumps it. Drives correction-first vs
    failed-complete-on-repeat.
    """
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? "
        "ORDER BY id DESC LIMIT 25",
        (task_id,),
    ).fetchall()
    count = 0
    for r in rows:
        if (r["kind"] or "") == "completion_blocked_no_evidence":
            count += 1
        else:
            break
    return count


def _dir_workspace_uncommitted(path: str) -> Optional[list[str]]:
    """Return TRACKED-but-uncommitted paths in ``path``'s git tree, or None.

    ``None`` means "cannot tell, allow" — not a git tree, no git binary, a
    timeout, anything. This gate must never be the reason a card cannot close.

    Untracked files (``??``) are deliberately EXCLUDED. A worker legitimately
    leaves scratch output lying around; what destroyed B1/B2/B3 on 2026-09-04
    was *tracked source edits* sitting in a shared working tree with no commit
    and no branch, which the next card clobbered.
    """
    if not path or not os.path.isdir(path):
        return None
    try:
        inside = subprocess.run(
            ["git", "-C", path, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        proc = subprocess.run(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return None
    except Exception:
        return None
    return [ln[3:].strip() for ln in proc.stdout.splitlines() if ln.strip()] or None


def _complete_ci_gate_rejection(task_id: str) -> Optional[str]:
    """Refuse `kanban_complete` without a green CI check. See tools/kanban_ci_gate.py."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None  # orchestrator / CLI path, as with the uncommitted gate.
    try:
        from tools.kanban_ci_gate import evaluate
    except Exception:  # noqa: BLE001 — module absent: gate not installed, not a crash
        return None
    kb, conn = _connect()
    try:
        task = kb.get_task(conn, task_id)
        if task is None:
            return None
        workspace = task.workspace_path or ""
        tenant = getattr(task, "tenant", None)
    finally:
        conn.close()
    try:
        result = evaluate(workspace, tenant)
    except Exception as exc:  # noqa: BLE001
        # The gate itself failing is uncertainty, and uncertainty blocks — but it
        # says so plainly rather than pretending to be a CI verdict.
        return tool_error(
            f"kanban_complete rejected: the CI gate could not run ({exc}). This is a fault "
            f"in the gate, not in your work. Call kanban_block so a human sees it.")
    if not result.blocked:
        return None
    return tool_error(
        "kanban_complete rejected: this card's tenant requires a passing CI check before "
        f"done.\n\n  {result.reason}\n\n"
        "Done means a green check on a clean checkout, not a local run in your own worktree. "
        "On 2026-09-04 four cards were built, reviewed and marked done on local evidence and "
        "the work was lost. If the check cannot go green for a reason outside this card, call "
        "kanban_block and say which. Your task is still in-flight; nothing was changed."
    )


def _complete_tool_evidence_rejection(task_id: str) -> Optional[str]:
    """Tool-evidence gate for ``kanban_complete``.

    Refuses a completion from a worker run that made ZERO non-kanban tool
    calls: no evidence the model did any work, so accepting it would fabricate
    a pass (a stalled card is visible; a fabricated completion is not). The
    first refusal returns a correction naming what is missing (the task stays
    in-flight); a repeat is counted as a failed completion so the failure
    budget sees it instead of the board silently rubber-stamping empty runs.

    Returns ``None`` to allow the completion. Orchestrator / CLI completions
    (no worker task scope), runs whose transcript cannot be read, and runs that
    produced kanban_create / kanban_link children (an orchestrator decomposition
    IS the work) all fail open.
    """
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None  # orchestrator / CLI path — not a worker completion.
    session_id = os.environ.get("HERMES_SESSION_ID") or ""
    db, _prof = _open_session_db(session_id)
    if db is None:
        return None  # cannot locate transcript — fail open.
    try:
        # An orchestrator worker decomposing a goal calls kanban_create /
        # kanban_link and then completes its own card with NO non-kanban tool
        # call. That fan-out is real work, so it is exempt from the evidence
        # gate (Rodge round-1, t_8fc16a73).
        if _run_produced_kanban_children(db, session_id):
            return None
        evidence = _count_non_kanban_tool_calls(db, session_id)
    finally:
        try:
            db.close()
        except Exception:
            pass
    if evidence is None or evidence > 0:
        return None  # fail open, or the run did real work — allow.

    kb, conn = _connect()
    try:
        run_id = _worker_run_id(task_id)
        with kb.write_txn(conn):
            kb._append_event(
                conn, task_id, "completion_blocked_no_evidence",
                {"violation_class": "no_evidence_complete"},
                run_id=run_id,
            )
        consecutive = _consecutive_no_evidence_blocks(conn, task_id)
        if consecutive == 1:
            return tool_error(
                "kanban_complete rejected: this run made ZERO non-kanban tool "
                "calls, so there is no evidence the task's work was actually "
                "done (a stalled card is visible, a fabricated completion is "
                "not). Do the reported work with real tool calls (file "
                "reads/edits, tests, searches, terminal) and then call "
                "kanban_complete again with the same handoff. Your task is "
                "still in-flight; nothing was changed."
            )
        # Repeat: count it as a failed completion so the failure budget sees
        # it. The run is closed as a crash and the card returns to its source
        # phase for re-dispatch. Failure accounting moved to the dispatch
        # module in the Sep-2026 decomposition; local import matches this
        # module's prevailing style and avoids an import cycle.
        from hermes_cli import kanban_db_dispatch as kbd

        kbd._record_task_failure(
            conn, task_id,
            error=("fabricated completion: kanban_complete called again with "
                   "zero non-kanban tool calls in the run (no evidence of "
                   "work); counted as a failed complete on repeat"),
            outcome="crashed",
            release_claim=True,
            end_run=True,
            event_payload_extra={"violation_class": "no_evidence_complete"},
        )
        return tool_error(
            "kanban_complete rejected for a second time with no non-kanban "
            "tool evidence in this run. This repeat is counted as a failed "
            "completion: the run has been closed as a crash and the card "
            "returned to its source phase for re-dispatch. Redo the work with "
            "real tool calls before calling kanban_complete again."
        )
    finally:
        conn.close()


PRE_REVIEW_TAIL_LINES = 30


_GATE_RUN_TIMEOUT = 600  # generous: a focused suite can take minutes


def _resolve_primary_repo(worktree_root: str) -> Path:
    """Return the primary repo a worktree workspace is checked out under.

    A project-linked worktree lives at ``<repo>/.worktrees/<task-id>``; the
    primary repo (where the project venv lives) is two levels up.  For an
    unanchored worktree the root is the repo.
    """
    p = Path(worktree_root).resolve()
    if p.parent.name == ".worktrees":
        return p.parent.parent
    return p


def _project_python(worktree_root: str) -> Optional[str]:
    """Resolve the project's python interpreter for gate execution.

    Checks ``venv``/``.venv`` under the worktree root first, then under the
    primary repo (the venv is often only in the primary checkout).  Cached per
    root so the happy path resolves once.
    """
    root = str(Path(worktree_root).resolve())
    if root in _PYTHON_CACHE:
        return _PYTHON_CACHE[root] or None
    cands = []
    for base in (Path(root), _resolve_primary_repo(root)):
        for name in ("venv", ".venv"):
            cands.append(base / name / "bin" / "python")
    found = next(
        (str(c) for c in cands if c.is_file() and os.access(c, os.X_OK)), None
    )
    # Cache positive hits only.  "Not found" is re-probed on the next call so
    # a venv created after a first (empty) probe is picked up.
    if found:
        _PYTHON_CACHE[root] = found
    return found


def _run_capture(args: list[str], cwd: str) -> tuple[int, str]:
    """Run a command locally, capturing combined output.  Returns (rc, output)."""
    try:
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=_GATE_RUN_TIMEOUT
        )
        combined = proc.stdout or ""
        if proc.stderr:
            combined += "\n" + proc.stderr
        return proc.returncode, combined.strip()
    except subprocess.TimeoutExpired:
        return 124, f"gate command timed out after {_GATE_RUN_TIMEOUT}s"
    except FileNotFoundError:
        return 127, f"gate command not found: {args[0]}"


_BASE_REF_CANDIDATES = ("origin/main", "main", "master", "HEAD~1")


def _resolve_base_ref(worktree_root: str) -> Optional[str]:
    """Return the base ref for a worktree, or None when none resolves.

    Walks the candidate refs and picks the one whose merge-base against
    HEAD is the DEEPEST — ie. the common ancestor closest to HEAD, i.e. the
    candidate with the fewest commits reachable from HEAD but not from the
    merge-base.  This is the diff base that reports only the card's own
    changes rather than sweeping in unrelated backlog.

    The default-candidate hardcoded order still applies as a tiebreaker for
    equally-deep candidates (prefer origin/main, then main, then master,
    then the parent).  But a STALE remote-tracking ref must never win just
    because it sorts first: origin/main can lag behind local main by many
    commits, and ``diff <stale-origin>...HEAD`` would report every file
    pushed since as changed, mapping to red baseline tests that are not this
    card's responsibility (t_13af5268: a frontend-only card bounced on
    backend search/ws AC15 failures it has nothing to do with.  A
        stale ancestor (merge-base != the ref itself) is a weaker base than an
        up-to-date one.
    """
    resolved: list[tuple[str, int]] = []
    for candidate in _BASE_REF_CANDIDATES:
        rc, _ = _run_capture(
            ["git", "-C", worktree_root, "rev-parse", "--verify", "-q", candidate],
            cwd=worktree_root,
        )
        if rc != 0:
            continue
        mrc, mb = _run_capture(
            ["git", "-C", worktree_root, "merge-base", candidate, "HEAD"],
            cwd=worktree_root,
        )
        if mrc != 0 or not mb.strip():
            continue
        # Depth: commits reachable from HEAD but not the merge-base.  The
        # deeper the merge-base (closer to HEAD), the smaller this count and
        # the more the diff is scoped to the card's own work.
        cc, count_out = _run_capture(
            ["git", "-C", worktree_root, "rev-list", "--count", f"{mb.strip()}..HEAD"],
            cwd=worktree_root,
        )
        count = int(count_out.strip()) if cc == 0 and count_out.strip().isdigit() else 10**9
        resolved.append((candidate, count))
    if not resolved:
        return None
    resolved.sort(key=lambda x: (x[1], _BASE_REF_CANDIDATES.index(x[0])))
    return resolved[0][0]


def _changed_python_files(worktree_root: str) -> list[str]:
    """Return python files changed in the worktree vs its base branch.

    Uses ``git diff <base>...HEAD`` for committed changes plus ``git status
    --porcelain`` for uncommitted ones.  Falls back to looking only at the
    working tree when no base branch resolves.
    """
    base = _resolve_base_ref(worktree_root)
    changed: set[str] = set()
    if base:
        rc, out = _run_capture(
            ["git", "-C", worktree_root, "diff", "--name-only", f"{base}...HEAD"],
            cwd=worktree_root,
        )
        if rc == 0:
            changed.update(
                line.strip() for line in out.splitlines() if line.strip()
            )
    rc, out = _run_capture(
        ["git", "-C", worktree_root, "status", "--porcelain", "-uall"],
        cwd=worktree_root,
    )
    if rc == 0:
        for line in out.splitlines():
            if not line:
                continue
            # porcelain: "XY path" — path after two status chars + a space.
            # Keep the raw line; .strip() would strip the leading status char.
            path = line[2:].lstrip()
            # porcelain rename, e.g. "R  old.py -> new.py", produces a
            # pseudo-path "old.py -> new.py".  Feed only the destination, never
            # the pseudo-path, into the gate (it would be a false FileNotFound).
            if " -> " in path:
                path = path.split(" -> ")[-1].strip()
            if path.endswith(".py"):
                changed.add(path)
    return sorted(p for p in changed if p.endswith(".py"))


def _focused_test_paths(repo_root: str, changed_py: list[str]) -> list[str]:
    """Map changed python files to matching test paths that exist.

    If a changed file is itself a test (name contains ``test`` or lives under
    a ``tests`` dir) it is used directly.  Otherwise guess the conventional
    mirror under ``tests/`` and keep the first guess that exists.
    """
    root = Path(repo_root).resolve()
    paths: set[str] = set()
    for rel in changed_py:
        p = Path(rel)
        if "tests" in p.parts or "test" in p.name.lower():
            if (root / p).is_file():
                paths.add(str(p))
            continue
        name = p.name
        if not name.endswith(".py"):
            continue
        stem = name[: -len(".py")]
        guesses = []
        if str(p.parent) != ".":
            guesses.append(str(Path("tests") / p.parent / f"test_{stem}.py"))
            guesses.append(str(Path("tests") / p.parent / "tests" / f"test_{stem}.py"))
        guesses.append(str(Path("tests") / f"test_{stem}.py"))
        for guess in guesses:
            if (root / guess).is_file():
                paths.add(guess)
                break
    return sorted(paths)


def _gate_command(project_python: str, tests: list[str]) -> list[str]:
    """Build the gate command for the focused tests.

    Default: ``<python> -m pytest <tests> -q``.  An override lives in the
    worker config key ``kanban.review_gate.command`` (a global knob, not
    project-scoped) — a string or list of argv fragments with ``{python}``
    and ``{tests}`` placeholders; every ``{tests}`` expands to one argv
    element per focused test path.
    """
    from hermes_cli.config import cfg_get, load_config

    overrides: Any = None
    try:
        overrides = cfg_get(load_config(), "kanban", "review_gate", "command", default=None)
    except Exception:
        overrides = None
    if overrides:
        try:
            if isinstance(overrides, str):
                overrides = shlex.split(overrides)
            if isinstance(overrides, (list, tuple)):
                argv: list[str] = []
                for a in overrides:
                    token = str(a).replace("{python}", project_python)
                    if "{tests}" not in token:
                        if token:
                            argv.append(token)
                        continue
                    before, _, after = token.partition("{tests}")
                    for t in tests:
                        seg = before + t + after
                        if seg:
                            argv.append(seg)
                return argv
        except Exception:
            pass  # fall through to the sane default on any malformed override
    return [project_python, "-m", "pytest", *tests, "-q"]


_PYTEST_FAILURE_LINE_RE = re.compile(r"^FAILED\s+([^\s]+)")


def _parse_focused_test_failures(output: str) -> frozenset[str]:
    """Extract the set of failed test IDs from a pytest ``-q`` run.

    Pytest ``-q`` prints one ``FAILED tests/...::Test::test_x - reason``
    line per failure in its short summary.  We key on the node id (the
    leading path::class::method token) so a failure can be matched against
    the baseline run and pre-existing failures excluded.  A run whose
    output we cannot parse (non-pytest rc, crash, etc.) yields the sentinel
    empty set — callers must not treat that as 'a failing test' unless the
    run was otherwise green; see ``_focused_tests_new_failures``.
    """
    fails: set[str] = set()
    for line in output.splitlines():
        m = _PYTEST_FAILURE_LINE_RE.match(line)
        if m:
            fails.add(m.group(1).strip())
    return frozenset(fails)


def _focused_test_failures(
    project_python: str, cwd: str, tests: list[str]
) -> tuple[int, str, frozenset[str]]:
    """Run the focused tests and return (rc, output, failing-test-ids).

    The failing-test-id set drives the baseline comparison; it is only
    meaningful when ``rc != 0`` AND the output parses as a pytest run (ie.
    the short summary carries ``FAILED ...`` lines for the expected node
    shape).  A crash/import error that never reaches the summary is
    reported by ``rc != 0`` with an empty/idempotent set, so a
    genuinely-broken runner is not masked by a 'no failures' baseline.
    """
    cmd = _gate_command(project_python, tests)
    rc, out = _run_capture(cmd, cwd=cwd)
    return rc, out, _parse_focused_test_failures(out)


def _focused_tests_new_failures(
    project_python: str,
    cwd: str,
    tests: list[str],
    base_dir: Optional[str],
    *,
    worktree_rc: int,
    worktree_out: str,
    worktree_fails: frozenset[str],
) -> tuple[Optional[frozenset[str]], Optional[str]]:
    """Compare focused tests against a merge-base baseline.

    The worktree selection has already been run by the caller (``worktree_rc``
    / ``worktree_out`` / ``worktree_fails``); this only establishes the
    baseline failure set and diffs. Returns ``(new_failures, error)`` where
    ``new_failures`` are the failures present in the worktree run but NOT in
    the merge-base baseline — the failures a card is responsible for.

    ``error`` is set (and ``new_failures`` is ``None``) when no reliable
    comparison is possible — callers fall back to the strict behaviour (any
    focused failure blocks review): a worktree run that crashed before pytest
    could emit a failure summary (no fault baseline can exonerate), or a
    baseline run that crashed rather than exercising the tests. ``error`` is
    ``None`` when both runs parsed cleanly.

    Zero LLM tokens: every step is a pure subprocess (git archive + pytest).
    """
    if worktree_rc != 0 and not worktree_fails:
        # Worktree run crashed before pytest could emit a failure summary
        # (eg. import error at collection, no exec).  Can't attribute this to
        # a pre-existing failure — bounce the card.
        return None, worktree_out or f"focused tests rc={worktree_rc} (no parseable failures)"
    if not base_dir:
        # No baseline source: fall back to strict (any failure blocks).
        return worktree_fails or frozenset(), None
    base_python, base_tests, base_err = _baseline_archive_selection(
        project_python, base_dir, tests
    )
    if base_err:
        return None, base_err
    if base_python is None or not base_tests:
        # Baseline archive has no usable python / no matching tests: fall
        # back to strict.
        return worktree_fails or frozenset(), None
    brc, bout, base_fails = _focused_test_failures(
        base_python, str(base_dir), base_tests
    )
    if brc != 0 and not base_fails:
        # Baseline run crashed rather than exercised tests — cannot compare.
        # This is a soft failure: fall back to strict rather than false-pass.
        return None, "baseline run crashed (no parseable failures)"
    return worktree_fails - base_fails, None


def _baseline_archive_selection(
    project_python: str, base_dir: str, tests: list[str]
) -> tuple[Optional[str], list[str], Optional[str]]:
    """Resolve (python, focused-test-paths) against a base-commit archive.

    The archive lives at ``base_dir`` (an extracted ``git archive`` of the
    merge-base).  Its venv is absent, so the interpreter falls back to the
    project's (``project_python``); pytest resolves there only when the
    project venv itself carries pytest.  Tests mirror the same heuristic as
    ``_focused_test_paths`` — the changed-route guess must land on an
    existing path inside the archive, else baseline-vs-worktree compare
    silently no-ops.
    """
    if not os.path.isdir(base_dir):
        return None, [], "baseline archive missing"
    # No venv in an archive; reuse the project interpreter.  The archive code
    # importing pytest is enough — that is what _focused_test_failures probes.
    base_python = project_python
    # Map the focused tests into paths that exist under the base archive.
    base_tests: list[str] = []
    root = Path(base_dir).resolve()
    for rel in tests:
        p = Path(_test_file_part(rel))
        if (root / p).is_file() or (root / p).is_dir():
            base_tests.append(rel)
    return base_python, base_tests, None


def _make_base_commit_archive(
    worktree_root: str, base_ref: str
) -> Optional[str]:
    """Materialize the merge-base tree into a fresh temp dir and return its path.

    Returns None when the archive cannot be produced (no git, empty tree, or
    subprocess failure).  The returned directory is the caller's to clean up,
    else it accumulates under the hosting temp dir.
    """
    rc, out = _run_capture(
        ["git", "-C", worktree_root, "merge-base", base_ref, "HEAD"],
        cwd=worktree_root,
    )
    if rc != 0:
        return None
    tokens = out.split()
    mb = tokens[0] if tokens else None
    if not mb:
        return None
    tmp = tempfile.mkdtemp(prefix="kanban_gate_base_")
    # Emit the tar to a temp file (binary), then extract — _run_capture reads
    # text and would corrupt the tar bytes.  Drop capture_output: it would
    # raise ValueError alongside an explicit stdout= stream (subprocess.run
    # forbids combining the two).
    tar_path = os.path.join(tmp, "base.tar")
    try:
        with open(tar_path, "wb") as fh:
            subprocess.run(
                ["git", "-C", worktree_root, "archive", "--format=tar", mb],
                cwd=worktree_root,
                stdout=fh,
                stderr=subprocess.PIPE,
                timeout=_GATE_RUN_TIMEOUT,
                check=True,
            )
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=tmp)
        os.unlink(tar_path)
        return tmp
    except Exception:
        return None


def _base_archive_for(worktree_root: str) -> Optional[str]:
    """Return a cached merge-base archive dir for a worktree, or None.

    The archive is keyed by the worktree root and cached for the life of the
    gate process so the per-card baseline run happens at most once.  A failed
    archive is cached as ``None`` (so we don't re-attempt on every rung), and
    callers fall back to strict gating when no baseline can be established.
    """
    key = str(Path(worktree_root).resolve())
    if key in _BASE_ARCHIVE_CACHE:
        return _BASE_ARCHIVE_CACHE[key]
    base_ref = _resolve_base_ref(worktree_root)
    archive = _make_base_commit_archive(worktree_root, base_ref) if base_ref else None
    _BASE_ARCHIVE_CACHE[key] = archive
    return archive


def _build_sanity_command(
    project_python: str, worktree_root: str, changed_py: list[str]
) -> list[str]:
    """Build the import/build sanity command for the changed python files.

    Real import resolution — not ``py_compile``, which checks syntax only and
    never resolves imports.  ``py_compile`` on a module that ``import``s an
    untracked/missing module exits 0, so the unbuildable-import class
    (2026-08-31 ce14358: a module importing from untracked files) would slip
    through a compile-only gate.  This child script imports each changed
    module by its dotted name (derived from the root-relative path, e.g.
    ``gateway/delivery.py`` -> ``gateway.delivery``) with the worktree root on
    ``sys.path``, so a missing/untracked import, a syntax error, or an
    import-time error yields rc != 0.

    Driving the real import machinery (rather than ``exec_module`` on a raw
    file) matters for two reasons: relative imports (``from .config import
    ...``) need the module's package context, and circular imports that the
    standard import system resolves (``gateway/__init__.py`` pulling
    ``.delivery``) must not be falsely bounced.  One subprocess for all
    changed files (happy-path cheap).
    """
    check = (
        "import importlib, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(root))\n"
        "failures = 0\n"
        "for rel in sys.argv[2:]:\n"
        "    path = (root / rel).resolve()\n"
        "    if not path.is_file():\n"
        "        print(f'import sanity: missing {rel}')\n"
        "        failures += 1\n"
        "        continue\n"
        "    dotted = '.'.join(pathlib.Path(rel).with_suffix('').parts)\n"
        "    if dotted == '__main__':\n"
        "        # Cannot import __main__ by name; it is the running script.\n"
        "        print(f'import sanity: skip {rel}')\n"
        "        continue\n"
        "    try:\n"
        "        importlib.import_module(dotted)\n"
        "    except Exception as exc:\n"
        "        print(f'import sanity FAIL {rel}: {type(exc).__name__}: {exc}')\n"
        "        failures += 1\n"
        "    else:\n"
        "        print(f'import sanity ok {rel}')\n"
        "sys.exit(1 if failures else 0)\n"
    )
    return [project_python, "-c", check, worktree_root, *changed_py]


def _run_gate_output_tail(output: str) -> str:
    """Keep the last ~TAIL lines of gate output for the auto-comment."""
    lines = [l for l in (output or "").splitlines() if l.strip()]
    return "\n".join(lines[-PRE_REVIEW_TAIL_LINES:])


_LINT_TOOLS = ("ruff", "flake8", "pylint")


_TYPECHECK_TOOLS = ("mypy", "pyright", "basedpyright")


_FOCUSED_TEST_CMD_RE = re.compile(
    # 2026-09-06 (G1): the first version accepted ONLY ``pytest <path> -q|-x``.
    # Every card on the afternoon of 6 Sep wrote
    # ``.venv/bin/python -m pytest tests/test_ui_v2.py::test_name -v``, which
    # did not match, so the rung fell back to the diff and swept the repo's
    # red baseline (7 more bounces after the "fix"). Now: any ``pytest`` token
    # (bare or ``-m pytest``), optional flags, then every following token that
    # looks like a root-relative test path (contains ``/``, may carry
    # ``::node``), stopping at the first option or shell token. Flags are
    # ignored — the gate builds its own command.
    r"(?:^|[\s`\"'=])pytest\s+(?:-[\w=-]+\s+)*"
    r"(?P<paths>"
    r"[^\s`\"';()|&<>-][^\s`\"';()|&<>]*/[^\s`\"';()|&<>]*"
    r"(?:\s+[^\s`\"';()|&<>-][^\s`\"';()|&<>]*/[^\s`\"';()|&<>]*)*"
    r")"
)


def _scoped_test_paths_from_body(body: Optional[str], ws_root: Path) -> list[str]:
    """Parse a per-card scoped pytest command from the card body.

    The pre-review gate's focused-tests rung must honor the card's own scope
    rather than derive the run set from the worktree diff.  When sibling cards
    commit onto the shared main while this card works a per-area worktree, the
    worktree diff is polluted with out-of-scope files (their modules mirror to
    red baseline tests), and the gate bounces a card whose scoped AC is green
    on failures it is explicitly forbidden to fix (t_4eea8efe: 9 red
    ``tests/test_search_backend.py`` cases on a card whose body says "Do NOT
    modify backend/app/search.py or its scoped tests").

    So: the body is the source of truth for what this card is accountable for.
    We extract the scoped pytest path from the acceptance criteria — any
    ``pytest <path> -q`` or ``pytest <path> -x`` invocation in the body (the
    exact command an AC line names).  Collect them in order of appearance and
    keep only paths that resolve under the worktree, so a stale/typo'd pointer
    silently degrades to the diff-derived fallback rather than bouncing.

    Returns an empty list when no per-card command is parseable (the caller
    falls back to the diff-derived selection, NOT the whole ``tests/`` dir).
    """
    root = Path(ws_root).resolve()
    if not body:
        return []
    found: list[str] = []
    for line in body.splitlines():
        m = _FOCUSED_TEST_CMD_RE.search(line)
        if m:
            # A single body AC may invoke pytest on MULTIPLE space-separated
            # test paths (e.g. t_eafe2bd3's
            # ``-m pytest tests/test_worker_pool.py tests/test_worker_pool_regressions.py -q``).
            # Collect them all, in order of appearance, so the scoped rung runs
            # exactly the card's declared files rather than degrading to the
            # diff-derived fallback (which sweeps in out-of-scope red baseline
            # tests).  Each token is a root-relative path that must carry a
            # ``/`` and contain no spaces/backticks/quotes.
            found.extend(m.group("paths").split())
    # De-dupe while preserving order, then keep only paths that exist under
    # the worktree.
    seen: set[str] = set()
    result: list[str] = []
    for p in found:
        p = p.strip()
        # A ``<worktree>/`` or ``./`` prefix in prose is stripped; a
        # ``::node`` selector is kept on the token (pytest accepts it) but
        # existence is checked on the file part (a directory is fine too).
        for pre in ("<worktree>/", "./"):
            if p.startswith(pre):
                p = p[len(pre):]
        if p in seen:
            continue
        seen.add(p)
        fp = root / _test_file_part(p)
        if fp.is_file() or fp.is_dir():
            result.append(p)
    return result


def _test_file_part(token: str) -> str:
    """``tests/x.py::test_a`` -> ``tests/x.py`` (G1)."""
    return token.split("::", 1)[0]


def _is_test_py(rel: str) -> bool:
    """True when a root-relative python path is (or lives under) a test file."""
    p = Path(rel)
    return "tests" in p.parts or "test" in p.name.lower()


def _is_browser_ui_test(rel: str) -> bool:
    """True when a root-relative test path needs a built frontend bundle.

    A worktree ships gitignored node_modules/dist absent, so a browser or
    UI test in the focused set would ERROR (cannot build the bundle it
    drives) rather than exercise anything — an environment gap, not the
    card's defect.  These tests are conventionally named/directed at the
    served UI: a ``test_ui*``/``*_ui*`` module, a ``regression`` directory
    (browser pass), or any path that mentions browser/playwright.  This
    lets the gate skip (log, never fail) the browser rung on build-less
    worktree-only runs while the card's own static tests still gate.
    """
    low = rel.lower()
    p = Path(low)
    if "tests/regression" in low or "test_regression" in low:
        return True
    if "ui" in p.name:
        return True
    if "browser" in low or "playwright" in low or "real_click" in low:
        return True
    return False


def _pytest_importable(project_python: str, cwd: str) -> bool:
    """True when ``pytest`` imports under the project interpreter.

    Cached per interpreter so the happy path probes once.  The focused-tests
    rung and the import rung's handling of test files both depend on this: a
    worktree venv without pytest (it is not part of the stdlib venv) would
    otherwise false-bounce a good card whose changed test file does ``import
    pytest`` (2026-09-01 self-test defect).  A gate-toolchain gap is SKIPPED
    (logged), never a failure.
    """
    if project_python in _PYTEST_CACHE:
        return _PYTEST_CACHE[project_python]
    rc, _ = _run_capture([project_python, "-c", "import pytest"], cwd=cwd)
    _PYTEST_CACHE[project_python] = rc == 0
    return _PYTEST_CACHE[project_python]


def _resolve_tool(python: str, key: str, candidates: tuple[str, ...]) -> Optional[str]:
    """Pick the first available lint/typecheck tool for a project python.

    Cached per (key, python) so the happy path probes the venv once.  A tool
    is 'available in the project' when its executable lives in the project's
    venv bin (the sibling of ``<python>/bin/python``).  Deliberately scoped to
    the project venv — NOT the ambient ``PATH`` — so the ladder is
    deterministic per project and a project that does not vendor a linter
    skips the rung rather than silently using some globally-installed tool
    that isn't part of its environment.  Projects that rely on system tooling
    can pin ``kanban.review_gate.lint_command`` / ``typecheck_command``.
    """
    ck = (key, python)
    if ck in _TOOL_CACHE:
        return _TOOL_CACHE[ck]
    chosen: Optional[str] = None
    bin_dir = Path(python).parent  # venv/bin when python is venv/bin/python
    try:
        for tool in candidates:
            if (bin_dir / tool).is_file():
                chosen = tool
                break
    except Exception:
        chosen = None
    _TOOL_CACHE[ck] = chosen
    return chosen


def _config_rung_override(key: str) -> Any:
    """Return the ``kanban.review_gate.<key>`` override (str/list) or None."""
    from hermes_cli.config import cfg_get, load_config

    try:
        return cfg_get(load_config(), "kanban", "review_gate", key, default=None)
    except Exception:
        return None


def _expand_rung_argv(
    tokens: Any, project_python: str, files: list[str]
) -> list[str]:
    """Expand a lint/typecheck override (like the tests ``command`` override).

    ``{python}`` becomes the project interpreter; every ``{files}`` expands to
    one argv element per changed python file (a shared placeholder cannot be a
    single space-joined element — ruff/mypy would read one bogus path).
    """
    argv: list[str] = []
    for a in tokens:
        token = str(a).replace("{python}", project_python)
        if "{files}" not in token:
            if token:
                argv.append(token)
            continue
        before, _, after = token.partition("{files}")
        for f in files:
            seg = before + f + after
            if seg:
                argv.append(seg)
    return argv


def _rung_command(
    project_python: str,
    key: str,
    candidates: tuple[str, ...],
    files: list[str],
    *,
    extra_prefix: tuple[str, ...] = (),
) -> Optional[list[str]]:
    """Build the argv for a lint/typecheck rung, or None when the tool is absent.

    Precedence: config override (``kanban.review_gate.<key>_command``) →
    first available default tool.  ``None`` means 'skip this rung', never a
    failure.

    The default tool is invoked as the exact executable ``_resolve_tool``
    detected (a console script in the project venv ``bin``), never ``python
    -m <tool>``.  A standalone tool (ruff, pyright) installs a binary but not
    a ``<tool>`` module on every interpreter — ``python -m pyright`` dies with
    "No module named pyright", inverting the missing-tool → skip contract
    into a false bounce (2026-09-01 self-test defect).  Ruff needs its
    ``check`` subcommand.
    """
    override = _config_rung_override(f"{key}_command")
    if override:
        try:
            if isinstance(override, str):
                override = shlex.split(override)
            if isinstance(override, (list, tuple)):
                return _expand_rung_argv(override, project_python, files)
        except Exception:
            pass  # malformed override -> fall through to the tool default
    tool = _resolve_tool(project_python, key, candidates)
    if tool is None:
        return None
    bin_dir = Path(project_python).parent  # same bin _resolve_tool probed
    argv = [str(bin_dir / tool), *extra_prefix]
    if tool == "ruff":
        argv.append("check")
    argv.extend(files)
    return argv


class _GateBounce(NamedTuple):
    """Which ladder rung bounced and that rung's output (this rung only)."""

    rung: str
    output: str


def _run_pre_review_gate(task: Any) -> Optional[_GateBounce]:
    """Run the zero-token review-gate ladder for a worktree-backed task.

    Returns ``None`` when the gate passes (or does not apply), otherwise a
    ``_GateBounce`` naming the rung that failed and carrying only that rung's
    output.  Pure subprocess — no LLM tokens.  The ladder short-circuits on
    the first failing rung; a rung whose tool is absent from the project is
    skipped (logged), not a failure.
    """
    kind = getattr(task, "workspace_kind", None)
    if kind != "worktree":
        return None
    # Respect the config kill-switch (kanban.review_gate.enabled) — a project
    # that opts out must not have reviews gated.
    from hermes_cli.config import cfg_get, load_config

    try:
        if not cfg_get(
            load_config(), "kanban", "review_gate", "enabled", default=True
        ):
            return None
    except Exception:
        pass  # fail open on config read failure — never silently block reviews
    ws = getattr(task, "workspace_path", None)
    if not ws or not os.path.isdir(ws):
        # Can't gate a missing/unresolvable worktree — don't block the flow.
        return None
    pypath = _project_python(str(ws))
    if not pypath:
        return None
    changed_py = _changed_python_files(str(ws))

    def _fail(rung: str, cmd: list[str], rc: int, out: str) -> _GateBounce:
        return _GateBounce(rung, f"[{rung}: {' '.join(cmd)} rc={rc}]\n{out}")

    # Rung 1: lint (cheapest).  Skip when no linter tool is available.
    if changed_py:
        lint = _rung_command(pypath, "lint", _LINT_TOOLS, changed_py)
        if lint is None:
            logger.info(
                "review gate: lint rung skipped (no linter tool in project venv for %s)",
                pypath,
            )
        else:
            rc, out = _run_capture(lint, cwd=ws)
            if rc != 0:
                return _fail("lint", lint, rc, out)

    # Rung 2: typecheck.  Skip when no typechecker tool is available.
    if changed_py:
        tc = _rung_command(pypath, "typecheck", _TYPECHECK_TOOLS, changed_py)
        if tc is None:
            logger.info(
                "review gate: typecheck rung skipped (no typechecker tool in project venv for %s)",
                pypath,
            )
        else:
            rc, out = _run_capture(tc, cwd=ws)
            if rc != 0:
                return _fail("typecheck", tc, rc, out)

    # Rung 3: import/build sanity (today's check).
    if changed_py:
        # A changed test file may ``import pytest`` at module load; if pytest
        # isn't importable under the project interpreter that would be a
        # gate-toolchain false bounce, not a card defect.  When pytest is
        # absent, import only the non-test changed modules and log the skip.
        sanity_py = changed_py
        pytest_ok = _pytest_importable(pypath, str(ws))
        if not pytest_ok and any(_is_test_py(f) for f in changed_py):
            sanity_py = [f for f in changed_py if not _is_test_py(f)]
            logger.info(
                "review gate: import sanity skipping %d test file(s); "
                "pytest not importable under %s",
                len(changed_py) - len(sanity_py),
                pypath,
            )
        if sanity_py:
            build_cmd = _build_sanity_command(pypath, str(ws), sanity_py)
            rc, out = _run_capture(build_cmd, cwd=str(ws))
            if rc != 0:
                return _fail("import/build sanity", build_cmd, rc, out)

    # Rung 4: focused tests (least cheap — constructor + execution).
    # The run set honors the card's own scope first: parse a per-card
    # scoped-test command from the body (its AC's pytest invocation, a
    # ``tests/...`` line).  Only when the body names nothing parseable do we
    # fall back to the diff-derived selection — never the whole tests/ dir.
    task_body = getattr(task, "body", None) or ""
    tests = _scoped_test_paths_from_body(task_body, Path(str(ws)))
    if not tests:
        # 2026-09-06 (G1): NEVER derive the run set from the diff. In a shared
        # dir the diff is every sibling's files; in a worktree it is whatever
        # the base ref guessed; both swept the repo's red baseline onto cards
        # forbidden to fix it (28 bounces across two jobs). The card body is
        # the only source of truth; a body naming no test command runs
        # nothing (E1 makes the command mandatory at mint).
        logger.info(
            "review gate: focused-tests rung skipped — card body names no test command"
        )
    # A worktree ships gitignored node_modules/dist absent, so a browser/UI
    # test in the focused set would ERROR (no built bundle) rather than
    # exercise anything — an env gap, not a card defect.  Skip (log, never
    # fail) any focused test that needs a built frontend when dist is missing.
    # The card's own static/lock-in tests still run and remain the real gate.
    if tests:
        dist_root = Path(str(ws)) / "frontend" / "dist"
        built = dist_root.is_dir() and any(dist_root.glob("index*.html"))
        if not built:
            kept, dropped = [], []
            for t in tests:
                if _is_browser_ui_test(t):
                    dropped.append(t)
                else:
                    kept.append(t)
            if dropped:
                logger.info(
                    "review gate: skipping %d browser/UI focused test(s) — "
                    "frontend/dist absent in worktree (%s)",
                    len(dropped), ", ".join(dropped),
                )
            tests = kept
    if tests:
        if not _pytest_importable(pypath, str(ws)):
            logger.info(
                "review gate: focused-tests rung skipped (pytest not importable under %s)",
                pypath,
            )
        else:
            test_cmd = _gate_command(pypath, tests)
            rc, out = _run_capture(test_cmd, cwd=str(ws))
            if rc == 0:
                # All green — nothing else to compare.
                pass
            else:
                worktree_fails = _parse_focused_test_failures(out)
                base_dir = _base_archive_for(ws)
                new_fails, cmp_err = _focused_tests_new_failures(
                    pypath,
                    str(ws),
                    tests,
                    base_dir,
                    worktree_rc=rc,
                    worktree_out=out,
                    worktree_fails=worktree_fails,
                )
                if cmp_err is not None:
                    # No reliable baseline — fall back to strict (a failing
                    # focused run blocks review, exactly as before).
                    return _fail("focused tests", test_cmd, rc, out)
                if new_fails:
                    # A failure this card introduced — bounce with output.
                    return _fail("focused tests", test_cmd, rc, out)
                # Only pre-existing failures present in the merge-base
                # baseline: the card is not responsible — let it through.
                logger.info(
                    "review gate: %d focused failure(s) present on merge-base baseline; card not responsible",
                    len(worktree_fails),
                )

    # Every rung green (or skipped) — gate passes.
    return None
