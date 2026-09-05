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
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get, load_config
from tools.kanban_tools_schemas import (
    KANBAN_ATTACH_SCHEMA,
    KANBAN_ATTACH_URL_SCHEMA, KANBAN_ATTACHMENTS_SCHEMA, KANBAN_BLOCK_SCHEMA, KANBAN_COMMENT_SCHEMA,
    KANBAN_COMPLETE_SCHEMA, KANBAN_CREATE_SCHEMA, KANBAN_HEARTBEAT_SCHEMA, KANBAN_LINK_SCHEMA,
    KANBAN_UNLINK_SCHEMA,
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
    if _is_delegated_child_context():
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
    "provider_override".split())
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
        _check(ok, f"could not complete {tid} (unknown id or already terminal)")
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
    with _board(args.get("board")) as (kb, conn):
        _goal_gate("kanban_request_review", kb.get_task(conn, tid), tid, summary)
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
    # Prefer the request-scoped api_server origin binding over HERMES_SESSION_ID: the env
    # var is clobbered with a subagent's internal id whenever a child agent is constructed
    # in-process, which would stamp — and later wake — the wrong session.
    from tools.async_delegation import _current_origin_session_id
    session_id = (args.get("session_id") or _current_origin_session_id()
                  or os.environ.get("HERMES_SESSION_ID"))
    # Workspace sharing is always explicit: omitted fields mean a fresh scratch workspace
    # even for a dispatcher-spawned creator (reusing the parent's path would let a child
    # mutate review evidence or race its checkout). Project identity is the one safe thing
    # to inherit implicitly (the DB turns it into a fresh per-task worktree).
    workspace_kind, workspace_path = args.get("workspace_kind"), args.get("workspace_path")
    # See #67567.
    project_id = args.get("project") or args.get("project_id")
    project_source_task_id = None
    triage, skills, goal_mode = (
        _parse_bool_arg(args, "triage"), _coerce_str_list(args.get("skills"), "skills", "skill names"),
        _parse_bool_arg(args, "goal_mode"))
    model_override, provider_override = args.get("model"), args.get("provider")
    _check(model_override or not provider_override, "'provider' requires 'model' to be set as well")
    parents = _coerce_str_list(args.get("parents") or [], "parents", "task ids")
    with _board(args.get("board")) as (kb, conn):
        if project_id is None and workspace_kind is None and workspace_path is None:
            self_tid = os.environ.get("HERMES_KANBAN_TASK")
            self_task = kb.get_task(conn, self_tid) if self_tid else None
            if self_task is not None and self_task.project_id:
                project_id, project_source_task_id = self_task.project_id, self_task.id
        new_tid = kb.create_task(
            conn, title=str(title).strip(), body=args.get("body"), assignee=str(assignee),
            parents=tuple(parents), tenant=args.get("tenant") or os.environ.get("HERMES_TENANT"),
            priority=_opt_int(args.get("priority"), 0),
            workspace_kind=str(workspace_kind if workspace_kind is not None else "scratch"),
            workspace_path=workspace_path, project_id=project_id,
            project_source_task_id=project_source_task_id, triage=triage,
            idempotency_key=args.get("idempotency_key"),
            max_runtime_seconds=_opt_int(args.get("max_runtime_seconds")), skills=skills,
            model_override=model_override, provider_override=provider_override,
            goal_mode=goal_mode, goal_max_turns=_opt_int(args.get("goal_max_turns")),
            initial_status=str(args.get("initial_status") or "running"),
            created_by=os.environ.get("HERMES_PROFILE") or "worker", session_id=session_id)
        landed = _fields(kb.get_task(conn, new_tid), _CREATED_FIELDS)
        return _ok(task_id=new_tid, **landed, subscribed=_maybe_auto_subscribe(conn, new_tid))


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
        k: v for k, v in (("thread_id", thread_id), ("chat_type", chat_type)) if v}
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
        from hermes_cli import kanban_db as _kb
        from hermes_cli import kanban_db_notify as _kbn
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
    """Add a parent→child dependency edge after the fact (cycles/self-links → ValueError)."""
    _reject_delegated_child_mutation("kanban_link")
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    _check(parent_id and child_id, "both parent_id and child_id are required")
    with _board(args.get("board")) as (kb, conn):
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)
        return _ok(parent_id=parent_id, child_id=child_id)


@_kanban_handler("kanban_unlink")
def _handle_unlink(args: dict, **kw) -> str:
    """Remove one edge, retaining worker support only for an owned endpoint.

    Dispatcher-owned workers need this to break dependency deadlocks, but a
    delegated child shares its parent's environment and is never a Kanban
    owner.  Orchestrators may route any board; workers may not cross their
    dispatcher-pinned board.
    """
    _reject_delegated_child_mutation("kanban_unlink")
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    _check(parent_id and child_id, "both parent_id and child_id are required")
    parent_id, child_id = str(parent_id), str(child_id)
    _check(parent_id != child_id, "kanban_unlink: parent_id and child_id must differ")
    env_task = os.environ.get("HERMES_KANBAN_TASK")
    if env_task:
        _check(env_task in (parent_id, child_id),
               f"worker is scoped to task {env_task}; refusing to mutate this edge")
        pinned_board = os.environ.get("HERMES_KANBAN_BOARD")
        requested_board = args.get("board")
        _check(not requested_board or not pinned_board or requested_board == pinned_board,
               f"worker is scoped to board {pinned_board}; refusing cross-board mutation")
    board = args.get("board")
    from hermes_cli import kanban_db as kb
    if board is not None:
        _check(kb.board_exists(board), f"kanban_unlink: board {board!r} does not exist")
    with _board(args.get("board")) as (kb, conn):
        _check(kb.get_task(conn, parent_id) is not None,
               f"kanban_unlink: unknown task {parent_id}")
        _check(kb.get_task(conn, child_id) is not None,
               f"kanban_unlink: unknown task {child_id}")
        removed = kb.unlink_tasks(conn, parent_id=parent_id, child_id=child_id)
        return _ok(parent_id=parent_id, child_id=child_id, removed=removed)


# --- Registration (order preserved: it is the order tools appear in the schema) ---

# kanban_list / kanban_unblock route the board and are hidden from task workers.
# kanban_unlink remains available to dispatcher-owned workers, but its handler
# limits them to an owned endpoint and their pinned board.
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
    ("kanban_link", KANBAN_LINK_SCHEMA, _handle_link, "🔗"),
    ("kanban_unlink", KANBAN_UNLINK_SCHEMA, _handle_unlink, "🔗"))

for _name, _sch, _handler, _emoji in _TOOLS:
    _gate = _check_kanban_orchestrator_mode if _name in _ORCHESTRATOR_TOOLS else _check_kanban_mode
    registry.register(name=_name, toolset="kanban", schema=_sch, handler=_handler, emoji=_emoji,
                      check_fn=_gate)
