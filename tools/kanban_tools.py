"""Kanban tools — structured tool-call surface for worker + orchestrator agents.

These tools are registered into the model's schema when the agent is
running under the dispatcher (env var ``HERMES_KANBAN_TASK`` set) or when
the active profile explicitly enables the ``kanban`` toolset for
orchestrator work. A normal ``hermes chat`` session still sees **zero**
kanban tools in its schema unless configured.

Why tools instead of just shelling out to ``hermes kanban``?

1. **Backend portability.** A worker whose terminal tool points at Docker
   / Modal / Singularity / SSH would run ``hermes kanban complete …``
   inside the container, where ``hermes`` isn't installed and the DB
   isn't mounted. Tools run in the agent's Python process, so they
   always reach ``~/.hermes/kanban.db`` regardless of terminal backend.

2. **No shell-quoting footguns.** Passing ``--metadata '{"x": [...]}'``
   through shlex+argparse is fragile. Structured tool args skip it.

3. **Better errors.** Tool-call failures return structured JSON the
   model can reason about, not stderr strings it has to parse.

Humans continue to use the CLI (``hermes kanban …``), the dashboard
(``hermes dashboard``), and the slash command (``/kanban …``) — all
three bypass the agent entirely. The tools are for dispatcher-spawned
worker handoffs and for configured orchestrator profiles that route work
through the board.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.goals import judge_goal
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

KANBAN_LIST_DEFAULT_LIMIT = 50
KANBAN_LIST_MAX_LIMIT = 200


def _profile_has_kanban_toolset() -> bool:
    # Uses load_config() which has mtime-based caching, so this adds
    # negligible overhead. The check_fn results are further TTL-cached
    # (~30s) by the tool registry.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        toolsets = cfg.get("toolsets", [])
        return "kanban" in toolsets
    except Exception:
        return False


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _is_dispatcher_owned_worker() -> bool:
    """False for delegate_task children AND for cron jobs fired in-process from
    a worker — i.e. whenever HERMES_KANBAN_* is present but not ours."""
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context

        return is_dispatcher_owned_worker_context()
    except Exception:
        return True


def _reject_delegated_child_mutation(tool_name: str) -> Optional[str]:
    """Deny Kanban mutations from delegate_task children.

    A delegate_task child runs in the same process as its parent, so stale or
    inherited HERMES_KANBAN_* env vars are not proof of dispatcher ownership.
    The child may summarize findings to its parent, but it must not complete,
    block, heartbeat, comment, create, link, or unblock board tasks directly.
    """
    if not _is_delegated_child_context():
        return None
    return tool_error(
        f"{tool_name} refused: delegate_task child agents are not Kanban "
        "run owners. Return findings to the parent agent; the dispatcher "
        "worker or an explicitly configured Kanban orchestrator must perform "
        "board mutations."
    )


def _check_kanban_mode() -> bool:
    """Task-lifecycle tools are available when:

    1. ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), OR
    2. The current profile has ``kanban`` in its toolsets config
       (orchestrator profiles like techlead that route work via Kanban).

    Humans running ``hermes chat`` without the kanban toolset see zero
    kanban tools. Workers spawned by the kanban dispatcher (gateway-
    embedded by default) and orchestrator profiles with the kanban
    toolset enabled see the Kanban lifecycle tool surface.
    """
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK") and _is_dispatcher_owned_worker():
        return True
    return _profile_has_kanban_toolset()


def _check_kanban_orchestrator_mode() -> bool:
    """Board-routing tools (kanban_list, kanban_unblock) are intentionally
    hidden from task workers.

    Dispatcher-spawned workers should close their own task via the
    lifecycle tools (complete/block/heartbeat), not enumerate or unblock
    board state. Profiles that explicitly opt into the kanban toolset
    and are NOT scoped to a single task are the orchestrator surface.
    """
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK") and _is_dispatcher_owned_worker():
        return False
    return _profile_has_kanban_toolset()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_task_id(arg: Optional[str]) -> Optional[str]:
    """Resolve ``task_id`` arg or fall back to the env var the dispatcher set."""
    if arg:
        return arg
    if _is_delegated_child_context():
        return None
    if not _is_dispatcher_owned_worker():
        # A cron job fired in-process from a worker must never inherit the
        # worker's task id as an implicit default.
        return None
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    return env_tid or None


def _worker_run_id(task_id: str) -> Optional[int]:
    """Return this worker's dispatcher run id when it is scoped to task_id."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _stamp_worker_session_metadata(
    task_id: str, metadata: Optional[dict]
) -> Optional[dict]:
    """Add trusted worker session id metadata for this worker's own task."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return metadata
    session_id = os.environ.get("HERMES_SESSION_ID")
    if not session_id:
        return metadata
    stamped = dict(metadata or {})
    stamped["worker_session_id"] = session_id
    return stamped


def _enforce_worker_task_ownership(tid: str) -> Optional[str]:
    """Reject worker-driven destructive calls on foreign task IDs.

    A process spawned by the dispatcher has ``HERMES_KANBAN_TASK`` set
    to its own task id. Tools like ``kanban_complete`` / ``kanban_block``
    / ``kanban_heartbeat`` mutate run-lifecycle state, so a buggy or
    prompt-injected worker that passed an explicit ``task_id`` for some
    other task could corrupt sibling or cross-tenant runs (see #19534).

    Orchestrator profiles (kanban toolset enabled but **no**
    ``HERMES_KANBAN_TASK`` in env) aren't subject to this check — their
    job is routing, and they sometimes legitimately close out child
    tasks or reopen blocked ones. Workers are narrowly scoped to their
    one task.

    Returns ``None`` when the call is allowed, or a tool-error string
    when it must be rejected. Callers should ``return`` the error
    verbatim.
    """
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    if not env_tid:
        # Orchestrator or CLI context — no task-scope restriction.
        return None
    if tid != env_tid:
        return tool_error(
            f"worker is scoped to task {env_tid}; refusing to mutate "
            f"{tid}. Use kanban_comment to hand off information to other "
            f"tasks, or kanban_create to spawn follow-up work."
        )
    return None


def _connect(board: Optional[str] = None):
    """Import + connect lazily so the module imports cleanly in non-kanban
    contexts (e.g. test rigs that import every tool module).

    When ``board`` is provided it's forwarded to :func:`kb.connect`, which
    routes the connection to that board's sqlite file. ``None`` (the
    default) preserves the legacy resolution chain
    (``HERMES_KANBAN_DB`` → ``HERMES_KANBAN_BOARD`` env → current symlink
    → ``default``). Per-tool ``board`` lets a Telegram-side agent override
    the env-pinned active board without restarting Hermes.
    """
    from hermes_cli import kanban_db as kb
    return kb, kb.connect(board=board)


_GOAL_MODE_BLOCK_ALLOWED_KINDS = frozenset({"dependency", "needs_input"})


def _goal_judge_available() -> bool:
    """True when an auxiliary client is configured for the goal judge.

    ``judge_goal`` is fail-open at the source: when no auxiliary model can
    be reached it returns a ``"continue"`` verdict that is indistinguishable
    from a real "not done yet" judgment. The completion gate must not treat
    that as a rejection, or an unconfigured/degraded auxiliary model would
    wedge every ``goal_mode`` worker (it could never close its own task).

    So we probe availability first and only enforce the gate when a judge is
    actually reachable. This mirrors the same client lookup ``judge_goal``
    performs internally.
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


def _goal_mode_handoff_rejection(task, evidence: str) -> Optional[str]:
    """Return a rejection reason when a goal-mode terminal handoff is premature."""
    if not task or not task.goal_mode or not _goal_judge_available():
        return None
    verdict = "done"
    reason = ""
    try:
        verdict, reason, _, _, _ = judge_goal(
            goal=f"{task.title}\n\n{task.body or ''}".strip(),
            last_response=evidence.strip(),
        )
    except Exception as judge_exc:
        # Keep the existing fail-open semantics: an unavailable/broken
        # auxiliary judge must not permanently wedge goal-mode work.
        logger.warning(
            "goal judge check failed, allowing lifecycle handoff: %s",
            judge_exc,
            exc_info=True,
        )
    return reason if verdict != "done" else None


# ---------------------------------------------------------------------------
# Runtime-activity → board-heartbeat bridge (#31752)
# ---------------------------------------------------------------------------
# When the agent ticks ``_touch_activity`` during normal work (between
# tool calls, mid-stream chunks, etc.), we want the kanban board's
# ``last_heartbeat_at`` columns to reflect that liveness so the dispatcher
# watchdog (which reads ``tasks.last_heartbeat_at``, not the agent's
# in-process timestamp) doesn't reclaim an actively-running worker as
# stale. The model is not required to call the explicit ``kanban_heartbeat``
# tool for this to work — that tool stays available for workers that want
# to attach a note or pre-emptively extend a claim across a known-long op.
#
# Constraints:
#   - Best-effort: never raise. The agent loop must not care if the bridge
#     fails (board missing, DB locked, etc.).
#   - Rate-limited to one DB write per 60s per-process; runtime activity
#     can tick on every chunk/tool result and we don't need that resolution.
#   - No-op outside dispatcher-spawned worker context (no ``HERMES_KANBAN_TASK``).
#   - No durable note on these auto-heartbeats; that's reserved for the
#     explicit tool which carries a model-supplied note.

_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_auto_heartbeat_last_attempt: float = 0.0


def heartbeat_current_worker_from_env() -> bool:
    """Best-effort: extend the kanban claim + bump board heartbeat for the
    current dispatcher-spawned worker, using identity from env vars.

    Returns True if a write was attempted (whether or not it succeeded);
    False if the call was skipped (not a kanban worker, rate-limited, or
    swallowed exception). The boolean is informational — callers should
    not branch on it.

    Identity comes from:
      * ``HERMES_KANBAN_TASK`` — task id (required; absence means no-op)
      * ``HERMES_KANBAN_RUN_ID`` — pins the run row so we don't heartbeat
        a stale run that may have already been reclaimed
      * ``HERMES_KANBAN_CLAIM_LOCK`` — claim lock for ``heartbeat_claim``;
        falls back to the default ``_claimer_id()`` for locally-driven
        workers that never went through the dispatcher path

    Rate-limited via the module-level ``_auto_heartbeat_last_attempt``
    timestamp (monotonic clock); not thread-safe in the strict sense, but
    the worst case is one extra DB write per race, which is harmless.
    """
    global _auto_heartbeat_last_attempt
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid:
        return False
    import time as _time
    now = _time.monotonic()
    if (now - _auto_heartbeat_last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
        return False
    _auto_heartbeat_last_attempt = now
    try:
        kb, conn = _connect()
        try:
            claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
            try:
                kb.heartbeat_claim(conn, tid, claimer=claim_lock)
            except Exception:
                logger.debug("auto-heartbeat: heartbeat_claim failed", exc_info=True)
            run_id_raw = os.environ.get("HERMES_KANBAN_RUN_ID")
            run_id: Optional[int]
            try:
                run_id = int(run_id_raw) if run_id_raw else None
            except (TypeError, ValueError):
                run_id = None
            try:
                kb.heartbeat_worker(conn, tid, note=None, expected_run_id=run_id)
            except Exception:
                logger.debug("auto-heartbeat: heartbeat_worker failed", exc_info=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True
    except Exception:
        logger.debug("auto-heartbeat: bridge failed", exc_info=True)
        return False


# Live operator-note injection: poll the worker's task for new comments and
# fold them into the running agent via the OUT-OF-BAND steer channel, so a user
# can "talk to" a running kanban task without the block → comment → unblock
# dance (or a restart). Rate-limited on its own (tighter than the 60s heartbeat
# so notes land within a few seconds), watermarked per task id.
_COMMENT_POLL_MIN_INTERVAL_SECONDS = 6.0
_comment_poll_last_attempt: float = 0.0
# task_id -> highest comment id already seen (seeded on first poll so history
# already present in build_worker_context isn't re-injected).
_comment_watermark: dict[str, int] = {}


def inject_new_comments_from_env(agent: Any) -> bool:
    """Fold new operator comments on the current worker's task into ``agent``.

    Best-effort and self-gating: no-op unless this process is a kanban worker
    (``HERMES_KANBAN_TASK`` set) and ``agent`` exposes ``steer``. Returns True
    if a steer was injected, else False. Never raises into the agent loop.

    The first poll only *seeds* the watermark to the newest existing comment —
    those are already in the worker's context — so only comments added after
    the run started are injected. The worker's own authored comments (matched
    by ``HERMES_PROFILE``) are skipped to avoid echoing itself.
    """
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid or agent is None or not hasattr(agent, "steer"):
        return False
    global _comment_poll_last_attempt
    import time as _time
    now = _time.monotonic()
    if (now - _comment_poll_last_attempt) < _COMMENT_POLL_MIN_INTERVAL_SECONDS:
        return False
    _comment_poll_last_attempt = now

    seen = _comment_watermark.get(tid)
    try:
        kb, conn = _connect()
        try:
            rows = kb.list_comments_after(conn, tid, after_id=seen or 0)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        logger.debug("comment-inject: bridge failed", exc_info=True)
        return False

    if seen is None:
        # First poll for this task: seed past the existing thread, inject nothing.
        _comment_watermark[tid] = max((c.id for c in rows), default=0)
        return False
    if not rows:
        return False

    # Advance the watermark past everything we just read (including our own
    # notes) so nothing is re-injected next poll.
    _comment_watermark[tid] = max(c.id for c in rows)

    own = (os.environ.get("HERMES_PROFILE") or "").strip()
    fresh = [c for c in rows if (c.author or "").strip() != own and (c.body or "").strip()]
    if not fresh:
        return False

    lines = [f"- {c.author or 'operator'}: {c.body.strip()}" for c in fresh]
    note = (
        "New note"
        + ("s" if len(fresh) > 1 else "")
        + " on your kanban task from the operator (delivered mid-run). "
        + "Take it into account for the work you're doing right now:\n"
        + "\n".join(lines)
    )
    try:
        return bool(agent.steer(note))
    except Exception:
        logger.debug("comment-inject: steer failed", exc_info=True)
        return False


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _normalize_profile(value: Any) -> Optional[str]:
    """Normalize CLI-compatible assignee sentinels for the tool surface."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "-", "null"}:
        return None
    return text


def _parse_bool_arg(args: dict, name: str, *, default: bool = False):
    value = args.get(name)
    if value is None:
        return default, None
    if isinstance(value, bool):
        return value, None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True, None
    if text in {"false", "0", "no"}:
        return False, None
    return default, f"{name} must be a boolean or 'true'/'false'"


def _require_orchestrator_tool(tool_name: str) -> Optional[str]:
    """Belt-and-suspenders runtime guard for orchestrator-only handlers.

    The check_fn (`_check_kanban_orchestrator_mode`) keeps these tools
    out of the worker schema entirely, but in case a stale registration
    or test harness routes a worker to one of them anyway, return a
    structured tool_error so the model gets a clear refusal instead of
    silently mutating board state from a worker context.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return tool_error(
            f"{tool_name} is orchestrator-only; dispatcher-spawned workers "
            "must use kanban_complete, kanban_block, kanban_heartbeat, or "
            "kanban_comment for their assigned task."
        )
    return None


def _task_summary_dict(kb, conn, task) -> dict[str, Any]:
    """Compact task shape for board-listing tools."""
    parents = kb.parent_ids(conn, task.id)
    children = kb.child_ids(conn, task.id)
    return {
        "id": task.id,
        "title": task.title,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "project_id": task.project_id,
        "created_by": task.created_by,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "current_run_id": task.current_run_id,
        "model_override": task.model_override,
        "theme": task.theme,
        "provider_override": task.provider_override,
        "parents": parents,
        "children": children,
        "parent_count": len(parents),
        "child_count": len(children),
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_show(args: dict, **kw) -> str:
    """Read a task's full state: task row, parents, children, comments,
    runs (attempt history), and the last N events."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            task = kb.get_task(conn, tid)
            if task is None:
                return tool_error(f"task {tid} not found")
            comments = kb.list_comments(conn, tid)
            events = kb.list_events(conn, tid)
            runs = kb.list_runs(conn, tid)
            parents = kb.parent_ids(conn, tid)
            children = kb.child_ids(conn, tid)

            def _task_dict(t):
                return {
                    "id": t.id, "title": t.title, "body": t.body,
                    "assignee": t.assignee, "status": t.status,
                    "tenant": t.tenant, "priority": t.priority,
                    "workspace_kind": t.workspace_kind,
                    "workspace_path": t.workspace_path,
                    "created_by": t.created_by, "created_at": t.created_at,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                    "result": t.result,
                    "current_run_id": t.current_run_id,
                    "model_override": t.model_override,
                    "theme": t.theme,
                    "provider_override": t.provider_override,
                }

            def _run_dict(r):
                return {
                    "id": r.id, "profile": r.profile,
                    "status": r.status, "outcome": r.outcome,
                    "summary": r.summary, "error": r.error,
                    "metadata": r.metadata,
                    "started_at": r.started_at, "ended_at": r.ended_at,
                }

            return json.dumps({
                "task": _task_dict(task),
                "parents": parents,
                "children": children,
                "comments": [
                    {"author": c.author, "body": c.body,
                     "created_at": c.created_at}
                    for c in comments
                ],
                "events": [
                    {"kind": e.kind, "payload": e.payload,
                     "created_at": e.created_at, "run_id": e.run_id}
                    for e in events[-50:]   # cap; full log via CLI
                ],
                "runs": [_run_dict(r) for r in runs],
                # Also surface the worker's own context block so the
                # agent can include it directly if it wants. This is
                # the same string build_worker_context returns to the
                # dispatcher at spawn time.
                "worker_context": kb.build_worker_context(conn, tid),
            })
        finally:
            conn.close()
    except ValueError as e:
        # Invalid board slug surfaces as ValueError from _normalize_board_slug.
        return tool_error(f"kanban_show: {e}")
    except Exception as e:
        logger.exception("kanban_show failed")
        return tool_error(f"kanban_show: {e}")


def _handle_list(args: dict, **kw) -> str:
    """List task summaries with the same core filters as the CLI."""
    guard = _require_orchestrator_tool("kanban_list")
    if guard:
        return guard
    assignee = args.get("assignee")
    status = args.get("status")
    tenant = args.get("tenant")
    include_archived, bool_error = _parse_bool_arg(args, "include_archived")
    if bool_error:
        return tool_error(bool_error)
    limit = args.get("limit")
    if limit is None:
        limit = KANBAN_LIST_DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")
    if limit < 1:
        return tool_error("limit must be >= 1")
    if limit > KANBAN_LIST_MAX_LIMIT:
        return tool_error(f"limit must be <= {KANBAN_LIST_MAX_LIMIT}")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # Match CLI list: dependencies that cleared since the last
            # dispatcher tick should be visible to orchestrators immediately.
            promoted = kb.recompute_ready(conn)
            # Fetch one extra row so model-facing output can report that
            # a bounded listing was truncated without dumping the board.
            rows = kb.list_tasks(
                conn,
                assignee=assignee,
                status=status,
                tenant=tenant,
                session_id=args.get("session_id"),
                theme=args.get("theme"),
                include_archived=include_archived,
                limit=limit + 1,
            )
            truncated = len(rows) > limit
            tasks = rows[:limit]
            return json.dumps({
                "tasks": [_task_summary_dict(kb, conn, t) for t in tasks],
                "count": len(tasks),
                "limit": limit,
                "truncated": truncated,
                "next_limit": (
                    min(limit * 2, KANBAN_LIST_MAX_LIMIT)
                    if truncated and limit < KANBAN_LIST_MAX_LIMIT else None
                ),
                "promoted": promoted,
            })
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_list: {e}")
    except Exception as e:
        logger.exception("kanban_list failed")
        return tool_error(f"kanban_list: {e}")


def _handle_complete(args: dict, **kw) -> str:
    """Mark the current task done with a structured handoff."""
    delegated_err = _reject_delegated_child_mutation("kanban_complete")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    summary = args.get("summary")
    metadata = args.get("metadata")
    result = args.get("result")
    if summary:
        summary = redact_sensitive_text(str(summary), force=True)
    if result:
        result = redact_sensitive_text(str(result), force=True)
    if metadata is not None and isinstance(metadata, dict):
        meta_json = json.dumps(metadata)
        meta_json = redact_sensitive_text(meta_json, force=True)
        try:
            metadata = json.loads(meta_json)
        except json.JSONDecodeError:
            pass
    created_cards = args.get("created_cards")
    artifacts = args.get("artifacts")
    if created_cards is not None:
        if isinstance(created_cards, str):
            # Accept a single id as a string for convenience.
            created_cards = [created_cards]
        if not isinstance(created_cards, (list, tuple)):
            return tool_error(
                f"created_cards must be a list of task ids, got "
                f"{type(created_cards).__name__}"
            )
        # Normalise: strings only, stripped, non-empty.
        created_cards = [
            str(c).strip() for c in created_cards if str(c).strip()
        ]
    if artifacts is not None:
        if isinstance(artifacts, str):
            # Accept a single path as a string for convenience.
            artifacts = [artifacts]
        if not isinstance(artifacts, (list, tuple)):
            return tool_error(
                f"artifacts must be a list of file paths, got "
                f"{type(artifacts).__name__}"
            )
        artifacts = [
            str(p).strip() for p in artifacts if str(p).strip()
        ]
        # Carry the artifact list inside metadata so it rides the
        # existing completed-event payload without a schema change at
        # the DB layer.  The gateway notifier reads payload['artifacts']
        # off the completion event and uploads each path as a native
        # attachment.
        if artifacts:
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                return tool_error(
                    f"metadata must be an object/dict, got "
                    f"{type(metadata).__name__}"
                )
            # Don't overwrite an existing metadata.artifacts the worker
            # passed manually — merge instead.
            existing = metadata.get("artifacts")
            if isinstance(existing, (list, tuple)):
                merged: list[str] = []
                seen: set[str] = set()
                for item in list(existing) + artifacts:
                    s = str(item).strip()
                    if s and s not in seen:
                        seen.add(s)
                        merged.append(s)
                metadata["artifacts"] = merged
            else:
                metadata["artifacts"] = artifacts
    if not (summary or result):
        return tool_error(
            "provide at least one of: summary (preferred), result"
        )
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(
            f"metadata must be an object/dict, got {type(metadata).__name__}"
        )
    metadata = _stamp_worker_session_metadata(tid, metadata)
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # Goal-mode pre-completion judge gate (Issue #38367).
            # Prevent workers from bypassing the auxiliary judge by
            # calling kanban_complete before acceptance criteria are met.
            # Only enforce when a judge is actually reachable — see
            # _goal_judge_available for why an unavailable judge fails open.
            task = kb.get_task(conn, tid)
            rejection = _goal_mode_handoff_rejection(
                task,
                (summary or result or "").strip(),
            )
            if rejection is not None:
                return tool_error(
                    f"Goal completion rejected by judge: {rejection}. "
                    f"To proceed, either: (1) provide explicit acceptance "
                    f"evidence in your summary matching the task's criteria, "
                    f"or (2) create continuation tasks with parents=[{tid}] "
                    f"and keep this task alive."
                )

            try:
                ok = kb.complete_task(
                    conn, tid,
                    result=result, summary=summary, metadata=metadata,
                    created_cards=created_cards,
                    expected_run_id=_worker_run_id(tid),
                )
            except kb.ArtifactPreservationError as artifact_err:
                return tool_error(
                    f"kanban_complete could not preserve the declared artifacts: "
                    f"{artifact_err}. Your task is still in-flight and its "
                    f"scratch workspace was kept. Fix the artifact path or "
                    f"storage error, then retry kanban_complete with the same handoff."
                )
            except kb.HallucinatedCardsError as hall_err:
                # Structured rejection — surface the phantom ids so the
                # worker can retry with a corrected list or drop the
                # field. Audit event already landed in the DB.
                #
                # The task itself was NOT mutated (the gate runs before
                # the write txn), so the worker can simply call
                # kanban_complete again. Spell that out — without it the
                # model often interprets a tool_error as a terminal
                # failure and either blocks or crashes the run instead
                # of retrying. See #22923.
                return tool_error(
                    f"kanban_complete blocked: the following created_cards "
                    f"do not exist or were not created by this worker: "
                    f"{', '.join(hall_err.phantom)}. "
                    f"Your task is still in-flight (no state change). "
                    f"Retry kanban_complete with the same summary/metadata "
                    f"and either drop these ids from created_cards, or pass "
                    f"created_cards=[] to skip the card-claim check entirely."
                )
            if not ok:
                return tool_error(
                    f"could not complete {tid} (unknown id or already terminal)"
                )
            run = kb.latest_run(conn, tid)
            return _ok(task_id=tid, run_id=run.id if run else None)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_complete: {e}")
    except Exception as e:
        logger.exception("kanban_complete failed")
        return tool_error(f"kanban_complete: {e}")



def _handle_complete_pipeline(args: dict, **kw) -> str:
    """Return a pipeline worker to its originating stage after writing the artifact."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    summary = args.get("summary")
    result = args.get("result")
    if not (summary or result):
        return tool_error(
            "provide at least one of: summary (preferred), result"
        )
    board = args.get("board")
    try:
        kb_mod, conn = _connect(board=board)
        try:
            ok = kb_mod.complete_pipeline_task(
                conn, tid,
                result=result, summary=summary,
            )
            if not ok:
                return tool_error(
                    f"could not complete pipeline task {tid} (not running or not a pipeline task)"
                )
            run = kb_mod.latest_run(conn, tid)
            return _ok(task_id=tid, run_id=run.id if run else None)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_complete_pipeline: {e}")
    except Exception as e:
        logger.exception("kanban_complete_pipeline failed")
        return tool_error(f"kanban_complete_pipeline: {e}")

def _handle_block(args: dict, **kw) -> str:
    """Transition the task to blocked with a reason a human will read."""
    delegated_err = _reject_delegated_child_mutation("kanban_block")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required — explain what input you need")
    reason = redact_sensitive_text(str(reason), force=True)
    kind = args.get("kind")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        if kind is not None and kind not in kb.VALID_BLOCK_KINDS:
            conn.close()
            return tool_error(
                f"kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} (or omit it)"
            )
        # Goal-mode block gate (Issue #38696, sibling of the kanban_complete
        # judge gate in #38367). kanban_block is a second exit path out of
        # the goal loop — run_kanban_goal_loop() treats ANY `blocked` status
        # as terminal, identically to `done`, regardless of kind. Without
        # this, a worker that learns kanban_complete is gated can just call
        # kanban_block(reason="anything") to escape the loop instead.
        # Restrict goal_mode tasks to the kinds that represent a genuine
        # external blocker the worker cannot resolve itself; `capability`
        # and `transient` (or an unset kind) route back through
        # kanban_complete, which the judge now gates.
        task = kb.get_task(conn, tid)
        if (
            task
            and task.goal_mode
            and kind not in _GOAL_MODE_BLOCK_ALLOWED_KINDS
        ):
            conn.close()
            return tool_error(
                f"goal_mode tasks can only block with kind in "
                f"{sorted(_GOAL_MODE_BLOCK_ALLOWED_KINDS)} (got {kind!r}). "
                f"If the task is actually finished or cannot proceed for "
                f"another reason, call kanban_complete instead — the "
                f"completion judge will evaluate it."
            )
        try:
            ok = kb.block_task(
                conn, tid,
                reason=reason,
                kind=kind,
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(
                    f"could not block {tid} (unknown id or not in "
                    f"running/ready)"
                )
            run = kb.latest_run(conn, tid)
            # Tell the worker where the task actually landed so it doesn't
            # assume it's sitting in 'blocked' when routing sent it elsewhere.
            landed = kb.get_task(conn, tid)
            return _ok(
                task_id=tid,
                run_id=run.id if run else None,
                status=landed.status if landed else "blocked",
                block_kind=kind,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_block: {e}")
    except Exception as e:
        logger.exception("kanban_block failed")
        return tool_error(f"kanban_block: {e}")


def _handle_request_review(args: dict, **kw) -> str:
    """Move implementation into the first-class review phase."""
    delegated_err = _reject_delegated_child_mutation("kanban_request_review")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    summary = args.get("summary")
    if not summary or not str(summary).strip():
        return tool_error(
            "summary is required — describe what was implemented and how it "
            "was verified so the reviewer has context"
        )
    summary = redact_sensitive_text(str(summary), force=True)
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(
            f"metadata must be an object/dict, got {type(metadata).__name__}"
        )
    if metadata is not None:
        metadata_json = redact_sensitive_text(json.dumps(metadata), force=True)
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return tool_error("metadata could not be safely serialized")
    metadata = _stamp_worker_session_metadata(tid, metadata)
    reviewer = args.get("reviewer") or None
    if reviewer:
        # Model-supplied free text stored durably on the event payload —
        # redact like summary / kanban_block's reason.
        reviewer = redact_sensitive_text(str(reviewer), force=True)
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            task = kb.get_task(conn, tid)
            rejection = _goal_mode_handoff_rejection(task, summary)
            if rejection is not None:
                return tool_error(
                    f"Goal review handoff rejected by judge: {rejection}. "
                    "Provide acceptance evidence matching the card before "
                    "requesting review."
                )
            ok, fail_reason = kb.request_review(
                conn, tid,
                summary=summary,
                metadata=metadata,
                reviewer=reviewer,
                expected_run_id=_worker_run_id(tid),
                with_reason=True,
            )
            if not ok:
                detail = fail_reason or "unknown id or not in running/ready"
                return tool_error(
                    f"could not request review for {tid}: {detail}"
                )
            run = kb.latest_run(conn, tid)
            landed = kb.get_task(conn, tid)
            return _ok(
                task_id=tid,
                run_id=run.id if run else None,
                status=landed.status if landed else "review",
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_request_review: {e}")
    except Exception as e:
        logger.exception("kanban_request_review failed")
        return tool_error(f"kanban_request_review: {e}")


def _handle_request_changes(args: dict, **kw) -> str:
    """Return a reviewer-owned running task to its implementer."""
    delegated_err = _reject_delegated_child_mutation("kanban_request_changes")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required — describe the changes needed")
    reason = redact_sensitive_text(str(reason), force=True)
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            ok, detail = kb.request_changes(
                conn,
                tid,
                reason=reason,
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(
                    f"could not request changes for {tid}: {detail or 'invalid review state'}"
                )
            landed = kb.get_task(conn, tid)
            run = kb.latest_run(conn, tid)
            return _ok(
                task_id=tid,
                run_id=run.id if run else None,
                status=landed.status if landed else "ready",
                implementer=detail,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_request_changes: {e}")
    except Exception as e:
        logger.exception("kanban_request_changes failed")
        return tool_error(f"kanban_request_changes: {e}")


def _handle_heartbeat(args: dict, **kw) -> str:
    """Signal that the worker is still alive during a long operation.

    Extends the claim TTL via ``heartbeat_claim`` AND records a heartbeat
    event via ``heartbeat_worker``. Without the ``heartbeat_claim`` half,
    a diligent worker that loops this tool while a single tool call
    blocks the agent for >DEFAULT_CLAIM_TTL_SECONDS still gets reclaimed
    by ``release_stale_claims`` — which is exactly the trap that
    ``heartbeat_claim``'s docstring warns against.
    """
    delegated_err = _reject_delegated_child_mutation("kanban_heartbeat")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    note = args.get("note")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # Extend the claim TTL first. The dispatcher pins
            # HERMES_KANBAN_CLAIM_LOCK in the worker env at spawn time
            # (see _default_spawn in kanban_db.py); falling back to the
            # default _claimer_id() covers locally-driven workers that
            # never went through the dispatcher path.
            claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
            kb.heartbeat_claim(conn, tid, claimer=claim_lock)

            ok = kb.heartbeat_worker(
                conn,
                tid,
                note=note,
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(
                    f"could not heartbeat {tid} (unknown id or not running)"
                )
            return _ok(task_id=tid)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_heartbeat: {e}")
    except Exception as e:
        logger.exception("kanban_heartbeat failed")
        return tool_error(f"kanban_heartbeat: {e}")


def _handle_comment(args: dict, **kw) -> str:
    """Append a comment to a task's thread."""
    delegated_err = _reject_delegated_child_mutation("kanban_comment")
    if delegated_err:
        return delegated_err
    tid = args.get("task_id")
    if not tid:
        return tool_error(
            "task_id is required (use the current task id if that's what "
            "you mean — pulls from env but kept explicit here)"
        )
    body = args.get("body")
    if not body or not str(body).strip():
        return tool_error("body is required")
    body = redact_sensitive_text(str(body), force=True)
    # Author is intentionally derived from the worker's own runtime
    # identity, NOT from caller-supplied args. Comments are injected
    # into the next worker's system prompt by ``build_worker_context``
    # as ``**{author}** (timestamp): {body}`` — accepting an
    # ``args["author"]`` override let a worker forge a comment from
    # an authoritative-looking name like ``hermes-system`` and poison
    # the future-worker context with what reads as a system directive.
    # Cross-task commenting itself remains unrestricted (see #19713) —
    # comments are the deliberate handoff channel between tasks.
    author = os.environ.get("HERMES_PROFILE") or "worker"
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            cid = kb.add_comment(conn, tid, author=author, body=str(body))
            return _ok(task_id=tid, comment_id=cid)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_comment: {e}")
    except Exception as e:
        logger.exception("kanban_comment failed")
        return tool_error(f"kanban_comment: {e}")


def _handle_attach(args: dict, **kw) -> str:
    """Attach an inline (base64) file to a task.

    Mirrors the dashboard's upload endpoint for the agent surface: decode
    the payload, enforce the shared size cap, write it under the per-task
    attachments dir, and record the metadata row — all via
    ``kanban_db.store_attachment_bytes`` so the three surfaces stay in lockstep.
    """
    from hermes_cli import kanban_db as kb

    delegated_err = _reject_delegated_child_mutation("kanban_attach")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    filename = args.get("filename")
    if not filename or not str(filename).strip():
        return tool_error("filename is required")
    content_b64 = args.get("content_base64")
    if not content_b64 or not str(content_b64).strip():
        return tool_error("content_base64 is required")
    import base64
    import binascii
    try:
        data = base64.b64decode(str(content_b64), validate=True)
    except (binascii.Error, ValueError) as e:
        return tool_error(f"content_base64 is not valid base64: {e}")
    content_type = args.get("content_type")
    board = args.get("board")
    try:
        _, conn = _connect(board=board)
        try:
            att_id = kb.store_attachment_bytes(
                conn,
                tid,
                str(filename),
                data,
                content_type=content_type,
                uploaded_by="agent",
                board=board,
            )
            return _ok(task_id=tid, attachment_id=att_id, size=len(data))
        finally:
            conn.close()
    except kb.AttachmentTooLarge as e:
        return tool_error(f"kanban_attach: {e}")
    except ValueError as e:
        return tool_error(f"kanban_attach: {e}")
    except Exception as e:
        logger.exception("kanban_attach failed")
        return tool_error(f"kanban_attach: {e}")


_MAX_ATTACH_URL_REDIRECTS = 5


def _download_url_with_cap(url: str, max_bytes: int) -> tuple[bytes, Optional[str]]:
    """Fetch ``url`` over http(s) with SSRF guarding, capped at ``max_bytes``.

    Every hop — the initial URL and each redirect target — is validated with
    ``tools.url_safety.is_safe_url`` before it is fetched, so a
    model-controlled URL (or a public host 302ing to one) cannot reach
    loopback, private/CGNAT ranges, or cloud metadata endpoints. Redirects
    are followed manually (``follow_redirects=False``) so each Location is
    re-checked, mirroring ``tools.skills_hub._guarded_http_get``.

    Returns ``(data, content_type)``. Raises ``ValueError`` for a non-http(s)
    scheme, an SSRF-blocked target, too many redirects, or a body that
    overruns the cap (the caller maps it to a clean tool error). Reads in
    chunks so an oversize response is rejected without buffering the whole
    thing.
    """
    from urllib.parse import urljoin, urlparse

    import httpx

    from tools.url_safety import is_safe_url

    current_url = url
    for _ in range(_MAX_ATTACH_URL_REDIRECTS + 1):
        scheme = (urlparse(current_url).scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported URL scheme {scheme!r}; only http/https are allowed"
            )
        if not is_safe_url(current_url):
            raise ValueError(
                f"URL blocked by SSRF protection (private/internal address): {current_url}"
            )
        chunks: list[bytes] = []
        total = 0
        with httpx.stream(
            "GET",
            current_url,
            headers={"User-Agent": "hermes-kanban/attach"},
            timeout=30,
            follow_redirects=False,
        ) as resp:
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
                    raise ValueError(
                        f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)
        return b"".join(chunks), content_type
    raise ValueError(f"too many redirects fetching {url}")


def _handle_attach_url(args: dict, **kw) -> str:
    """Attach a file fetched server-side from a URL.

    The agent passes a URL; Hermes downloads it (with the shared size cap)
    and stores it as a real attachment. Useful when the agent has a link
    rather than the bytes. Only http/https URLs are accepted.
    """
    from hermes_cli import kanban_db as kb

    delegated_err = _reject_delegated_child_mutation("kanban_attach_url")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    url = args.get("url")
    if not url or not str(url).strip():
        return tool_error("url is required")
    url = str(url).strip()
    filename = args.get("filename") or args.get("title")
    if not filename or not str(filename).strip():
        # Derive a name from the URL path's leaf component.
        from urllib.parse import unquote, urlparse
        leaf = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip()
        filename = leaf or "download"
    content_type = args.get("content_type")
    board = args.get("board")
    try:
        data, fetched_ct = _download_url_with_cap(url, kb.KANBAN_ATTACHMENT_MAX_BYTES)
    except ValueError as e:
        return tool_error(f"kanban_attach_url: {e}")
    except Exception as e:
        logger.exception("kanban_attach_url download failed")
        return tool_error(f"kanban_attach_url: failed to fetch {url}: {e}")
    try:
        _, conn = _connect(board=board)
        try:
            att_id = kb.store_attachment_bytes(
                conn,
                tid,
                str(filename),
                data,
                content_type=content_type or fetched_ct,
                uploaded_by="agent",
                board=board,
            )
            return _ok(task_id=tid, attachment_id=att_id, size=len(data))
        finally:
            conn.close()
    except kb.AttachmentTooLarge as e:
        return tool_error(f"kanban_attach_url: {e}")
    except ValueError as e:
        return tool_error(f"kanban_attach_url: {e}")
    except Exception as e:
        logger.exception("kanban_attach_url failed")
        return tool_error(f"kanban_attach_url: {e}")


def _handle_attachments(args: dict, **kw) -> str:
    """List a task's attachments (read-only; no ownership restriction)."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            if kb.get_task(conn, tid) is None:
                return tool_error(f"task {tid} not found")
            atts = kb.list_attachments(conn, tid)
            return json.dumps({
                "ok": True,
                "task_id": tid,
                "attachments": [
                    {
                        "id": a.id,
                        "filename": a.filename,
                        "content_type": a.content_type,
                        "size": a.size,
                        "uploaded_by": a.uploaded_by,
                        "stored_path": a.stored_path,
                        "created_at": a.created_at,
                    }
                    for a in atts
                ],
            })
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_attachments: {e}")
    except Exception as e:
        logger.exception("kanban_attachments failed")
        return tool_error(f"kanban_attachments: {e}")


def _handle_create(args: dict, **kw) -> str:
    """Create a child task. Orchestrator workers use this to fan out.

    ``parents`` can be a list of task ids; dependency-gated promotion
    works as usual.
    """
    delegated_err = _reject_delegated_child_mutation("kanban_create")
    if delegated_err:
        return delegated_err
    title = args.get("title")
    if not title or not str(title).strip():
        return tool_error("title is required")
    assignee = args.get("assignee")
    if not assignee:
        return tool_error(
            "assignee is required — name the profile that should execute this "
            "task (the dispatcher will only spawn tasks with an assignee)"
        )
    body = args.get("body")
    parents = args.get("parents") or []
    tenant = args.get("tenant") or os.environ.get("HERMES_TENANT")
    # Stamp the originating session id when the agent loop runs under
    # ACP (which sets HERMES_SESSION_ID before invoking tools). NULL on
    # CLI / dashboard paths and on legacy hosts that don't set the env.
    # Prefer the request-scoped api_server origin binding: HERMES_SESSION_ID
    # is clobbered with a subagent's internal id whenever a child agent is
    # constructed in-process (agent_init calls set_current_session_id), which
    # would stamp — and later wake — the wrong session.
    from tools.async_delegation import _current_origin_session_id

    session_id = (
        args.get("session_id")
        or _current_origin_session_id()
        or os.environ.get("HERMES_SESSION_ID")
    )
    priority = args.get("priority")
    # Resolve workspace. Workspace sharing is always explicit: omitted fields
    # mean a fresh scratch workspace, even when a dispatcher-spawned worker
    # creates the task. Reusing a parent's literal path would let a child
    # mutate review evidence or race the parent's checkout (#67567).
    #
    # Project identity is the one safe context to inherit implicitly. The DB
    # resolves a project-linked scratch request into a fresh per-task worktree,
    # preserving the repository/branch convention without sharing a checkout.
    workspace_kind = args.get("workspace_kind")
    workspace_path = args.get("workspace_path")
    project_id = args.get("project") or args.get("project_id")
    project_source_task_id = None
    _inherit_project = workspace_kind is None and workspace_path is None
    if workspace_kind is None:
        workspace_kind = "scratch"
    triage, bool_error = _parse_bool_arg(args, "triage")
    if bool_error:
        return tool_error(bool_error)
    idempotency_key = args.get("idempotency_key")
    max_runtime_seconds = args.get("max_runtime_seconds")
    initial_status = args.get("initial_status") or "running"
    skills = args.get("skills")
    if isinstance(skills, str):
        # Accept a single skill name as a string for convenience.
        skills = [skills]
    if skills is not None and not isinstance(skills, (list, tuple)):
        return tool_error(
            f"skills must be a list of skill names, got {type(skills).__name__}"
        )
    goal_mode, goal_bool_error = _parse_bool_arg(args, "goal_mode")
    if goal_bool_error:
        return tool_error(goal_bool_error)
    goal_max_turns = args.get("goal_max_turns")
    model_override = args.get("model")
    provider_override = args.get("provider")
    if provider_override and not model_override:
        return tool_error("'provider' requires 'model' to be set as well")
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, (list, tuple)):
        return tool_error(
            f"parents must be a list of task ids, got {type(parents).__name__}"
        )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # A project link is safe to inherit because ``create_task`` turns
            # it into a fresh per-task worktree. Never inherit the parent's
            # literal workspace kind/path; directory sharing must be explicit.
            if _inherit_project and project_id is None:
                _self_tid = os.environ.get("HERMES_KANBAN_TASK")
                if _self_tid:
                    _self_task = kb.get_task(conn, _self_tid)
                    if _self_task is not None and _self_task.project_id:
                        project_id = _self_task.project_id
                        project_source_task_id = _self_task.id
            new_tid = kb.create_task(
                conn,
                title=str(title).strip(),
                body=body,
                assignee=str(assignee),
                parents=tuple(parents),
                tenant=tenant,
                priority=int(priority) if priority is not None else 0,
                workspace_kind=str(workspace_kind),
                workspace_path=workspace_path,
                project_id=project_id,
                project_source_task_id=project_source_task_id,
                triage=triage,
                idempotency_key=idempotency_key,
                max_runtime_seconds=(
                    int(max_runtime_seconds)
                    if max_runtime_seconds is not None else None
                ),
                skills=skills,
                model_override=model_override,
                provider_override=provider_override,
                goal_mode=goal_mode,
                goal_max_turns=(
                    int(goal_max_turns) if goal_max_turns is not None else None
                ),
                initial_status=str(initial_status),
                theme=args.get("theme"),
                created_by=os.environ.get("HERMES_PROFILE") or "worker",
                session_id=session_id,
                )
            new_task = kb.get_task(conn, new_tid)
            subscribed = _maybe_auto_subscribe(conn, new_tid)
            return _ok(
                task_id=new_tid,
                status=new_task.status if new_task else None,
                workspace_kind=new_task.workspace_kind if new_task else None,
                workspace_path=new_task.workspace_path if new_task else None,
                project_id=new_task.project_id if new_task else None,
                subscribed=subscribed,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_create: {e}")
    except Exception as e:
        logger.exception("kanban_create failed")
        return tool_error(f"kanban_create: {e}")


def _maybe_auto_subscribe(conn: Any, task_id: str) -> bool:
    """Auto-subscribe the calling session to task completion / block events.

    Returns True if a subscription row was written, False otherwise (no
    session context, config gate disabled, or best-effort failure). The
    caller surfaces this in the ``subscribed`` field of the kanban_create
    response so an orchestrator can decide whether to fall back to an
    explicit ``kanban_notify-subscribe`` or to polling.

    Gated by ``kanban.auto_subscribe_on_create`` in config.yaml (default
    True). Disable to mirror pre-feature behaviour, e.g. when the
    originating user/chat opted out via the per-platform notification
    toggle (see ``hermes dashboard``).

    Subscription paths:

    - **Gateway** (telegram/discord/slack/etc): ``HERMES_SESSION_PLATFORM``,
      ``HERMES_SESSION_CHAT_ID``, and ``HERMES_SESSION_CHAT_TYPE`` are set in
      ContextVars by the messaging gateway before agent dispatch. The
      notification poller already keys off these, so we just register a row.

    - **TUI** (herm desktop / herm TUI): the platform/chat_id ContextVars
      are intentionally cleared (TUI is a single-channel local UI, not
      a multi-tenant chat surface), but the agent subprocess inherits
      ``HERMES_SESSION_KEY`` from the parent session. We subscribe with
      ``platform="tui"`` and ``chat_id=<key>``; the TUI notification
      poller (``tui_gateway/server.py``) reads ``kanban_notify_subs``
      for these rows and posts the completion message into the running
      session.

    - **CLI / cron / test / unattached**: no persistent delivery channel,
      no-op.

    Failure mode: any exception inside the function is logged at WARNING
    with the offending exception + diagnostic env vars and swallowed.
    We never want a notification bookkeeping failure to fail the
    kanban_create that the agent is mid-conversation about.
    """
    try:
        cfg = load_config()
        if not cfg_get(cfg, "kanban", "auto_subscribe_on_create", default=True):
            return False
    except Exception:
        # If config can't load we still default to True — this is the
        # user-friendly behaviour that mirrors the pre-gate implementation.
        pass

    platform = ""
    chat_id = ""
    try:
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        if not platform or not chat_id:
            # TUI / desktop fallback: platform/chat_id ContextVars are
            # cleared for TUI sessions, but the parent process exports
            # HERMES_SESSION_KEY into the subprocess env. Treat that
            # as a "tui" subscription so the TUI notification poller
            # (tui_gateway/server.py) can pick it up.
            #
            # HERMES_SESSION_ID is intentionally NOT a fallback here:
            # it is set by ACP / the agent subprocess for telemetry
            # regardless of whether the parent is a TUI or a CLI, so
            # treating it as a notification target would auto-subscribe
            # every CLI invocation, which is exactly the over-eager
            # behaviour that got #19718 reverted upstream. The TUI
            # poller keys on HERMES_SESSION_KEY.
            session_key = (
                get_session_env("HERMES_SESSION_KEY", "")
                or os.environ.get("HERMES_SESSION_KEY", "")
            )
            if not session_key:
                return False  # CLI / cron / test — no persistent channel
            platform = "tui"
            chat_id = session_key
        is_gateway_session = platform != "tui"
        chat_type = get_session_env("HERMES_SESSION_CHAT_TYPE", "") or None
        delivery_mode = "notify+wake" if is_gateway_session else None
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "") or None
        user_id = get_session_env("HERMES_SESSION_USER_ID", "") or None
        user_id_alt = get_session_env("HERMES_SESSION_USER_ID_ALT", "") or None
        message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "") or ""
        notifier_profile = (
            get_session_env("HERMES_SESSION_PROFILE", "")
            or os.environ.get("HERMES_PROFILE")
        )
        if not notifier_profile:
            try:
                from hermes_cli.profiles import get_active_profile_name
                notifier_profile = get_active_profile_name() or "default"
            except Exception:
                notifier_profile = "default"
        delivery_metadata: dict[str, Any] = {}
        if thread_id:
            delivery_metadata["thread_id"] = thread_id
        if chat_type:
            delivery_metadata["chat_type"] = chat_type
        if (
            platform.lower() == "telegram"
            and thread_id
            and (chat_type or "").lower() in {"dm", "direct", "private"}
        ):
            delivery_metadata["telegram_dm_topic_reply_fallback"] = True
            if str(thread_id) not in {"", "1"}:
                delivery_metadata["direct_messages_topic_id"] = str(thread_id)
            if message_id:
                delivery_metadata["telegram_reply_to_message_id"] = str(message_id)

        # Lazy-import to keep the module-level dependency light
        from hermes_cli import kanban_db as _kb
        _kb.add_notify_sub(
            conn, task_id=task_id,
            platform=platform, chat_id=chat_id,
            thread_id=thread_id, user_id=user_id, user_id_alt=user_id_alt,
            chat_type=chat_type,
            notifier_profile=notifier_profile,
            delivery_mode=delivery_mode,
            delivery_metadata=delivery_metadata or None,
        )
        return True
    except Exception as _exc:
        logger.warning(
            "_maybe_auto_subscribe failed: %r (platform=%r key_set=%r)",
            _exc, platform, bool(chat_id),
        )
        return False


def _handle_unblock(args: dict, **kw) -> str:
    """Transition a blocked task back to ready.

    Returns a specific error when the task is in ``decision-needed`` or
    otherwise decision-gated (see :func:`hermes_cli.kanban_db.unblock_task`
    for the refusal predicate) so the operator knows it needs a human
    decision rather than a cron-style retry.
    """
    delegated_err = _reject_delegated_child_mutation("kanban_unblock")
    if delegated_err:
        return delegated_err
    guard = _require_orchestrator_tool("kanban_unblock")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    ownership_err = _enforce_worker_task_ownership(str(tid))
    if ownership_err:
        return ownership_err
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            ok = kb.unblock_task(conn, str(tid))
            if not ok:
                # Distinguish decision-gated refusals from generic
                # "not blocked" so operators get an actionable message.
                gated_row = conn.execute(
                    "SELECT status, block_kind, escalation_target "
                    "FROM tasks WHERE id = ?",
                    (str(tid),),
                ).fetchone()
                if gated_row is None:
                    return tool_error(
                        f"could not unblock {tid} (not found)"
                    )
                status = gated_row["status"]
                block_kind = gated_row["block_kind"]
                escalation = gated_row["escalation_target"]
                if status == "decision-needed":
                    return tool_error(
                        f"task {tid} is in 'decision-needed' state "
                        f"(escalation_target={escalation!r}); a human "
                        f"decision is required — use 'hermes kanban "
                        f"set-status' to advance it directly"
                    )
                if (
                    status == "blocked"
                    and block_kind in kb.DECISION_BLOCK_KINDS
                    and escalation
                ):
                    return tool_error(
                        f"task {tid} is decision-gated "
                        f"(block_kind={block_kind!r}, "
                        f"escalation_target={escalation!r}); a human "
                        f"decision is required"
                    )
                return tool_error(
                    f"could not unblock {tid} (not blocked or unknown)"
                )
            task = kb.get_task(conn, str(tid))
            return _ok(task_id=str(tid), status=task.status if task else None)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_unblock: {e}")
    except Exception as e:
        logger.exception("kanban_unblock failed")
        return tool_error(f"kanban_unblock: {e}")


def _handle_request_review(args: dict, **kw) -> str:
    """Worker tool: flip the current task to the review column."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error("task_id is required (or set HERMES_KANBAN_TASK)")
    ownership_err = _enforce_worker_task_ownership(str(tid))
    if ownership_err:
        return ownership_err
    summary = args.get("summary")
    if not summary or not str(summary).strip():
        return tool_error("summary is required")
    artefacts = args.get("artefacts") or args.get("artifacts") or []
    if isinstance(artefacts, str):
        artefacts = [artefacts]
    if not isinstance(artefacts, (list, tuple)):
        return tool_error("artefacts must be a list of strings")
    next_steps = args.get("next_steps")
    board = args.get("board")
    expected_run_id = _worker_run_id(str(tid))
    try:
        kb, conn = _connect(board=board)
        try:
            ok = kb.request_review(
                conn, str(tid),
                summary=str(summary),
                artefacts=artefacts,
                next_steps=next_steps,
                expected_run_id=expected_run_id,
            )
            if not ok:
                return tool_error(
                    f"could not request review for {tid} (not running or "
                    "run id mismatch)"
                )
            return _ok(task_id=str(tid), status="review")
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_request_review: {e}")
    except Exception as e:
        logger.exception("kanban_request_review failed")
        return tool_error(f"kanban_request_review: {e}")


def _approver_profile_from_env() -> Optional[str]:
    raw = os.environ.get("HERMES_PROFILE")
    if not raw:
        return None
    text = str(raw).strip()
    return text or None


def _handle_approve(args: dict, **kw) -> str:
    """Reviewer tool: three-outcome approval."""
    guard = _require_orchestrator_tool("kanban_approve")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    outcome = args.get("outcome")
    if outcome not in {"terminal", "chained", "conditional"}:
        return tool_error(
            "outcome must be one of terminal | chained | conditional"
        )
    approver = _approver_profile_from_env()
    if not approver:
        return tool_error(
            "kanban_approve requires HERMES_PROFILE in env to record the "
            "approver"
        )
    follow_up_spec = args.get("follow_up_spec")
    if follow_up_spec is not None and not isinstance(follow_up_spec, dict):
        return tool_error("follow_up_spec must be an object")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            res = kb.approve_review_task(
                conn, str(tid),
                outcome=outcome,
                approver_profile=approver,
                summary=args.get("summary"),
                follow_up_spec=follow_up_spec,
                next_reviewer=args.get("next_reviewer"),
                comment=args.get("comment"),
            )
            return json.dumps({"ok": True, "task_id": str(tid), **{k: v for k, v in res.items() if k != "ok"}})
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_approve: {e}")
    except Exception as e:
        logger.exception("kanban_approve failed")
        return tool_error(f"kanban_approve: {e}")


def _handle_reject(args: dict, **kw) -> str:
    """Reviewer tool: send task to blocked with structured findings."""
    guard = _require_orchestrator_tool("kanban_reject")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    findings = args.get("findings")
    if not findings or not isinstance(findings, dict):
        return tool_error("findings is required and must be an object")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            ok = kb.reject_review_task(
                conn, str(tid),
                findings=findings,
                reviewer_profile=_approver_profile_from_env() or "orchestrator",
            )
            if not ok:
                return tool_error(
                    f"could not reject review for {tid} (not in review or "
                    "reviewer profile mismatch)"
                )
            return _ok(task_id=str(tid), status="blocked")
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_reject: {e}")
    except Exception as e:
        logger.exception("kanban_reject failed")
        return tool_error(f"kanban_reject: {e}")


def _handle_reassign(args: dict, **kw) -> str:
    """Orchestrator tool: change assignee with handoff note."""
    guard = _require_orchestrator_tool("kanban_reassign")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    new_assignee = args.get("new_assignee")
    if not new_assignee or not str(new_assignee).strip():
        return tool_error("new_assignee is required")
    handoff_note = args.get("handoff_note")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            current = kb.reassign_count(conn, str(tid))
            ok = kb.reassign_task_with_note(
                conn, str(tid), str(new_assignee).strip(),
                handoff_note=handoff_note,
                author=_approver_profile_from_env() or "orchestrator",
            )
            if not ok:
                return tool_error(
                    f"could not reassign {tid} (unknown or refused)"
                )
            return _ok(
                task_id=str(tid),
                new_assignee=str(new_assignee).strip(),
                reassign_count=current + 1,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_reassign: {e}")
    except Exception as e:
        logger.exception("kanban_reassign failed")
        return tool_error(f"kanban_reassign: {e}")


def _handle_edit(args: dict, **kw) -> str:
    """Orchestrator tool: replace the skills list. Scope-locked."""
    guard = _require_orchestrator_tool("kanban_edit")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    skills = args.get("skills")
    if skills is None or not isinstance(skills, (list, tuple)):
        return tool_error("skills must be a list of skill names (use [] to clear)")
    # Refuse any other mutation field even though the schema doesn't list
    # them; defends against a caller that hand-rolls extra args.
    forbidden = {"assignee", "priority", "max_runtime_seconds", "tenant", "theme"}
    leaked = forbidden & set(args.keys())
    if leaked:
        return tool_error(
            f"kanban_edit is scope-locked to `skills`; refused fields: "
            f"{sorted(leaked)}. Use CLI for these mutations."
        )
    author = _approver_profile_from_env() or "orchestrator"
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            res = kb.edit_task_skills(
                conn, str(tid), skills=skills, author=author,
            )
            return json.dumps({"ok": True, "task_id": str(tid), **{k: v for k, v in res.items() if k != "ok"}})
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_edit: {e}")
    except Exception as e:
        logger.exception("kanban_edit failed")
        return tool_error(f"kanban_edit: {e}")


def _handle_link(args: dict, **kw) -> str:
    """Add a parent→child dependency edge after the fact."""
    delegated_err = _reject_delegated_child_mutation("kanban_link")
    if delegated_err:
        return delegated_err
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    if not parent_id or not child_id:
        return tool_error("both parent_id and child_id are required")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)
            return _ok(parent_id=parent_id, child_id=child_id)
        finally:
            conn.close()
    except ValueError as e:
        # Covers cycle + self-parent rejections
        return tool_error(f"kanban_link: {e}")
    except Exception as e:
        logger.exception("kanban_link failed")
        return tool_error(f"kanban_link: {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_DESC_TASK_ID_DEFAULT = (
    "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the env "
    "(the task the dispatcher spawned you to work on)."
)

_DESC_BOARD = (
    "Kanban board slug to target. When omitted, the call resolves the "
    "active board the usual way: HERMES_KANBAN_DB env → "
    "HERMES_KANBAN_BOARD env → the 'current' symlink under the kanban "
    "home → 'default'. Pass an explicit slug only when the caller (e.g. "
    "a Telegram routing layer) needs to override the env-pinned active "
    "board for this one call."
)


def _board_schema_prop() -> dict[str, str]:
    """Schema fragment for the optional ``board`` parameter.

    Centralised so a future tweak to the description / validation hint
    only has to land in one place.
    """
    return {"type": "string", "description": _DESC_BOARD}

KANBAN_SHOW_SCHEMA = {
    "name": "kanban_show",
    "description": (
        "Read a task's full state — title, body, assignee, parent task "
        "handoffs, your prior attempts on this task if any, comments, "
        "and recent events. Use this to (re)orient yourself before "
        "starting work, especially on retries. The response includes a "
        "pre-formatted ``worker_context`` string suitable for inclusion "
        "verbatim in your reasoning."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_LIST_SCHEMA = {
    "name": "kanban_list",
    "description": (
        "List Kanban task summaries so an orchestrator profile can discover "
        "work to route. Supports the same core filters as the CLI: assignee, "
        "status, tenant, include_archived, and limit. Returns compact rows "
        "with ids, title, status, assignee, priority, parent/child ids, and "
        "counts. Bounded to 50 rows by default, 200 max, with truncation "
        "metadata. Also recomputes ready tasks before listing, matching the "
        "CLI. Orchestrator-only — dispatcher-spawned task workers never see "
        "this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "assignee": {
                "type": "string",
                "description": "Optional assignee/profile filter.",
            },
            "status": {
                "type": "string",
                "enum": [
                    "triage", "todo", "ready", "running",
                    "blocked", "review", "done", "archived", "backlog",
                ],
                "description": "Optional task status filter.",
            },
            "tenant": {
                "type": "string",
                "description": "Optional tenant/project namespace filter.",
            },
            "session_id": {
                "type": "string",
                "description": "Filter by originating chat/agent session id.",
            },
            "theme": {
                "type": "string",
                "description": (
                    "Optional theme tag filter (exact match). Use to "
                    "list all tasks under a project codename or "
                    "milestone shorthand."
                ),
            },
            "include_archived": {
                "type": "boolean",
                "description": "Include archived tasks. Defaults to false.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional maximum rows to return (default 50, max 200).",
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_COMPLETE_SCHEMA = {
    "name": "kanban_complete",
    "description": (
        "Mark your current task done with a structured handoff for "
        "downstream workers and humans. Prefer ``summary`` for a "
        "human-readable 1-3 sentence description of what you did; put "
        "machine-readable facts in ``metadata`` (changed_files, "
        "tests_run, decisions, findings, etc). At least one of "
        "``summary`` or ``result`` is required. If you created new "
        "tasks via ``kanban_create`` during this run, list their ids "
        "in ``created_cards`` — the kernel verifies them so phantom "
        "references are caught before they leak into downstream "
        "automation. If you produced deliverable files (charts, PDFs, "
        "spreadsheets, generated images), list their absolute paths "
        "in ``artifacts`` — the gateway notifier will upload them as "
        "native attachments to the human who subscribed to the task, "
        "so the deliverable lands in their chat alongside the summary "
        "instead of being a path they have to fetch by hand."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "summary": {
                "type": "string",
                "description": (
                    "Human-readable handoff, 1-3 sentences. Appears in "
                    "Run History on the dashboard and in downstream "
                    "workers' context."
                ),
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Free-form dict of structured facts about this "
                    "attempt — {\"changed_files\": [...], \"tests_run\": 12, "
                    "\"findings\": [...]}. Surfaced to downstream "
                    "workers alongside ``summary``."
                ),
            },
            "result": {
                "type": "string",
                "description": (
                    "Short result log line (legacy field, maps to "
                    "task.result). Use ``summary`` instead when "
                    "possible; this exists for compatibility with "
                    "callers that still set --result on the CLI."
                ),
            },
            "created_cards": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional structured manifest of task ids you "
                    "created via ``kanban_create`` during this run. "
                    "The kernel verifies each id exists and was "
                    "created by this worker's profile; any phantom "
                    "id blocks the completion with an error listing "
                    "what went wrong (auditable in the task's events). "
                    "Only list ids you got back from a successful "
                    "``kanban_create`` call — do not invent or "
                    "remember ids from prose. Omit the field if you "
                    "did not create any cards."
                ),
            },
            "artifacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of absolute paths to deliverable "
                    "files you produced during this run — generated "
                    "charts, PDFs, spreadsheets, images, archives. "
                    "Examples: [\"/tmp/q3-revenue.png\", "
                    "\"/tmp/report.pdf\"]. The gateway notifier "
                    "uploads each path as a native attachment to the "
                    "subscribed chat (images embed inline, everything "
                    "else uploads as a file) so the deliverable "
                    "lands with the completion notification. Skip "
                    "intermediate scratch files and references that "
                    "are not the deliverable. The path must exist "
                    "on disk at completion. Files inside a managed scratch "
                    "workspace are copied to durable task attachments before "
                    "cleanup; a missing declared scratch artifact keeps the "
                    "task in-flight so you can fix the path and retry."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_BLOCK_SCHEMA = {
    "name": "kanban_block",
    "description": (
        "Stop work on this task and route it according to WHY you're stuck. "
        "Set ``kind`` to say which: 'dependency' (waiting on another task — "
        "goes to todo and auto-resumes when that task finishes, no human "
        "needed), 'needs_input' (you need a human decision/answer), "
        "'capability' (a hard wall: no access, missing credentials, an action "
        "no agent can do), or 'transient' (a flaky failure that may clear). "
        "``reason`` is shown to the human on the board. If a task keeps "
        "getting unblocked and re-blocked for the same reason, it is "
        "auto-escalated to triage. Use for genuine blockers only — don't "
        "block on things you can resolve yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "reason": {
                "type": "string",
                "description": (
                    "What you need answered or what stopped you, in one or "
                    "two sentences. Don't paste the whole conversation; the "
                    "human has the board and can ask follow-ups via comments."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["dependency", "needs_input", "capability", "transient"],
                "description": (
                    "Why you're blocked. 'dependency' waits in todo and "
                    "resumes automatically; the others surface to a human. "
                    "Omit only if none apply."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["reason"],
    },
}

KANBAN_REQUEST_REVIEW_SCHEMA = {
    "name": "kanban_request_review",
    "description": (
        "Hand the task off for review: implementation, self-review, and "
        "verification are complete and you want a human (or reviewer) to "
        "look before it is marked done. Moves the task to the 'review' "
        "column and notifies the subscriber. Unlike ``kanban_block`` this is "
        "NOT a blocker — it never counts toward unblock-loop detection, so a "
        "task can cycle through review across follow-ups without ever being "
        "falsely escalated to triage. Use this instead of blocking with a "
        "free-form 'review-required:' reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "summary": {
                "type": "string",
                "description": (
                    "What was implemented and how it was verified, in one or "
                    "two sentences — shown to the reviewer. Don't paste "
                    "the whole diff; the reviewer has the board and the PR."
                ),
            },
            "reviewer": {
                "type": "string",
                "description": (
                    "Optional reviewer profile. When provided, the task is "
                    "reassigned to that profile before review dispatch."
                ),
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Optional structured handoff facts for the reviewer, such "
                    "as changed_files, tests_run, commit, or decisions."
                ),
                "additionalProperties": True,
            },
            "board": _board_schema_prop(),
        },
        "required": ["summary"],
    },
}

KANBAN_REQUEST_CHANGES_SCHEMA = {
    "name": "kanban_request_changes",
    "description": (
        "Reviewer verdict: return the current review run to the original "
        "implementer with concrete required changes. This closes the review "
        "run, reapplies parent dependency gating, and requeues the task without "
        "using block-loop accounting. Only use from a task claimed from the "
        "review column; use kanban_block only for a genuine external blocker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "reason": {
                "type": "string",
                "description": (
                    "Specific, actionable changes the implementer must make "
                    "before requesting another review."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["reason"],
    },
}

KANBAN_HEARTBEAT_SCHEMA = {
    "name": "kanban_heartbeat",
    "description": (
        "Signal that you're still alive during a long operation "
        "(training, encoding, large crawls). Call every few minutes so "
        "humans see liveness separately from PID checks. Pure side "
        "effect — no work changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "note": {
                "type": "string",
                "description": (
                    "Optional short note describing current progress. "
                    "Shown in the event log."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_COMMENT_SCHEMA = {
    "name": "kanban_comment",
    "description": (
        "Append a comment to a task's thread. Use for durable notes "
        "that should outlive this run (questions for the next worker, "
        "partial findings, rationale). Ephemeral reasoning doesn't "
        "belong here — use your normal response instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "Task id. Required (may be your own task or "
                    "another's — comment threads are per-task)."
                ),
            },
            "body": {
                "type": "string",
                "description": "Markdown-supported comment body.",
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id", "body"],
    },
}

KANBAN_ATTACH_SCHEMA = {
    "name": "kanban_attach",
    "description": (
        "Attach a file to a task by passing its bytes inline (base64). "
        "Use for genuine file artifacts the next worker or a human should "
        "be able to download — generated reports, images, exports. The "
        "file is stored as a real attachment (not a comment link) under "
        "the task's attachments dir, capped at 25 MB. Prefer "
        "kanban_attach_url when you only have a URL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "filename": {
                "type": "string",
                "description": (
                    "File name to store it under (e.g. 'report.pdf'). "
                    "Directory components are stripped; only the leaf is kept."
                ),
            },
            "content_base64": {
                "type": "string",
                "description": "The file contents, base64-encoded. Max 25 MB decoded.",
            },
            "content_type": {
                "type": "string",
                "description": "Optional MIME type (e.g. 'application/pdf').",
            },
            "board": _board_schema_prop(),
        },
        "required": ["filename", "content_base64"],
    },
}

KANBAN_ATTACH_URL_SCHEMA = {
    "name": "kanban_attach_url",
    "description": (
        "Attach a file to a task by URL — Hermes downloads it server-side "
        "and stores it as a real attachment (capped at 25 MB). Use when "
        "you have a link rather than the bytes. Only http/https URLs are "
        "accepted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "url": {
                "type": "string",
                "description": "http(s) URL to fetch and store.",
            },
            "filename": {
                "type": "string",
                "description": (
                    "Optional name to store it under. Defaults to the URL "
                    "path's leaf component."
                ),
            },
            "content_type": {
                "type": "string",
                "description": (
                    "Optional MIME type override. Defaults to the "
                    "Content-Type the server returns."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["url"],
    },
}

KANBAN_ATTACHMENTS_SCHEMA = {
    "name": "kanban_attachments",
    "description": (
        "List the files attached to a task: id, filename, content_type, "
        "size, who uploaded it, and the absolute on-disk path you can read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_CREATE_SCHEMA = {
    "name": "kanban_create",
    "description": (
        "Create a new kanban task, optionally as a child of the current "
        "one (pass the current task id in ``parents``). Used by "
        "orchestrator workers to fan out — decompose work into child "
        "tasks with specific assignees, link them into a pipeline, "
        "then complete your own task. The dispatcher picks up the new "
        "tasks on its next tick and spawns the assigned profiles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short task title (required).",
            },
            "assignee": {
                "type": "string",
                "description": (
                    "Profile name that should execute this task "
                    "(e.g. 'researcher-a', 'reviewer', 'writer'). "
                    "Required — tasks without an assignee are never "
                    "dispatched."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Opening post: full spec, acceptance criteria, "
                    "links. The assigned worker reads this as part of "
                    "its context."
                ),
            },
            "parents": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Parent task ids. The new task stays in 'todo' "
                    "until every parent reaches 'done'; then it "
                    "auto-promotes to 'ready'. Typical fan-in: list "
                    "all the researcher task ids when creating a "
                    "synthesizer task."
                ),
            },
            "tenant": {
                "type": "string",
                "description": (
                    "Optional namespace for multi-project isolation. "
                    "Defaults to HERMES_TENANT env if set."
                ),
            },
            "priority": {
                "type": "integer",
                "description": (
                    "Dispatcher tiebreaker. Higher = picked sooner "
                    "when multiple ready tasks share an assignee."
                ),
            },
            "workspace_kind": {
                "type": "string",
                "enum": ["scratch", "dir", "worktree"],
                "description": (
                    "Workspace flavor: 'scratch' (fresh tmp dir, "
                    "default), 'dir' (shared directory, requires "
                    "absolute workspace_path), 'worktree' (git worktree)."
                ),
            },
            "workspace_path": {
                "type": "string",
                "description": (
                    "Absolute path for 'dir' or 'worktree' workspace. "
                    "Relative paths are rejected at dispatch."
                ),
            },
            "project": {
                "type": "string",
                "description": (
                    "Optional project id or slug to link the task to. When "
                    "set, the task becomes a git worktree under the project's "
                    "primary repo with a deterministic branch (project slug + "
                    "task id), instead of a random branch."
                ),
            },
            "triage": {
                "type": "boolean",
                "description": (
                    "If true, task lands in 'triage' instead of 'todo' "
                    "— a specifier profile is expected to flesh out "
                    "the body before work starts."
                ),
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "If a non-archived task with this key already "
                    "exists, return that task's id instead of creating "
                    "a duplicate. Useful for retry-safe automation."
                ),
            },
            "max_runtime_seconds": {
                "type": "integer",
                "description": (
                    "Per-task runtime cap. When exceeded, the "
                    "dispatcher SIGTERMs the worker and re-queues the "
                    "task with outcome='timed_out'."
                ),
            },
            "initial_status": {
                "type": "string",
                "enum": ["running", "blocked", "backlog"],
                "description": (
                    "Initial card status. Use 'blocked' for tasks that "
                    "require immediate human ops (R3 gate) to skip the "
                    "brief running-to-blocked transition. Use 'backlog' "
                    "to park an approved follow-up spec until a human "
                    "or specifier calls `hermes kanban promote-backlog`. "
                    "Defaults to 'running', which preserves the usual "
                    "dispatch path."
                ),
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Skill names to force-load into the dispatched "
                    "worker. The kanban lifecycle is already injected "
                    "automatically; use this to pin a task to a specialist "
                    "context — e.g. ['translation'] for a translation "
                    "task, ['github-code-review'] for a reviewer task. "
                    "The names must match skills installed on the "
                    "assignee's profile."
                ),
            },
            "goal_mode": {
                "type": "boolean",
                "description": (
                    "Run the dispatched worker in a goal loop. When true, "
                    "after each turn an auxiliary judge checks the worker's "
                    "response against this card's title/body; if the work "
                    "isn't done and budget remains, the worker keeps going "
                    "in the same session until the judge agrees it's "
                    "complete (or the goal-turn budget is exhausted, which "
                    "blocks the task for human review). Use this for "
                    "open-ended cards where one shot rarely finishes the "
                    "work. Defaults to false (classic single-shot worker)."
                ),
            },
            "goal_max_turns": {
                "type": "integer",
                "description": (
                    "Turn budget for goal_mode workers. Caps how many "
                    "continuation turns the worker may take before the task "
                    "is blocked for review. Ignored unless goal_mode is "
                    "true. Defaults to the goal-engine default (20)."
                ),
            },
            "theme": {
                "type": "string",
                "description": (
                    "Optional flat tag for grouping related work "
                    "(e.g. 'atm10', 'compliance-q2'). Set at create "
                    "time; kanban_edit does not mutate this in v1."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Pin the dispatched worker to this model instead of "
                    "the assignee profile's configured model. Use the "
                    "exact model name the target provider expects. Omit "
                    "to use the profile default."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Provider the 'model' belongs to (e.g. 'openrouter', "
                    "'anthropic', 'nous'). Set this whenever the model "
                    "is not from the assignee profile's configured "
                    "provider — a model name alone is resolved against "
                    "the profile's provider and will fail if it belongs "
                    "to a different one. Requires 'model'."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["title", "assignee"],
    },
}

KANBAN_UNBLOCK_SCHEMA = {
    "name": "kanban_unblock",
    "description": (
        "Unblock a Kanban task. It moves to ready when all parents are done, "
        "or todo while any parent remains open. Orchestrator-only — only "
        "profiles with the kanban toolset can unblock routed work; "
        "dispatcher-spawned task workers never see this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Blocked task id to move to ready or parent-gated todo.",
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id"],
    },
}

KANBAN_REQUEST_REVIEW_SCHEMA = {
    "name": "kanban_request_review",
    "description": (
        "Worker tool: hand off your current running task to a reviewer "
        "by flipping its column to 'review'. The dispatcher claims the "
        "next tick under a reviewer profile (sdlc-review or, if the task "
        "has a prior rejection, the same reviewer who logged it). The "
        "reviewer reads `summary`, `artefacts`, and `next_steps` via "
        "their first kanban_show; no separate JSON-in-comment is needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "summary": {
                "type": "string",
                "description": (
                    "Human-readable handoff, 1-3 sentences. The reviewer "
                    "sees this on their first kanban_show."
                ),
            },
            "artefacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of file paths or PR URLs the reviewer "
                    "should examine. Same shape as kanban_complete's "
                    "`artifacts`."
                ),
            },
            "next_steps": {
                "type": "string",
                "description": (
                    "Optional follow-up guidance for the reviewer (e.g. "
                    "'confirm the fallback choice; if approved, chain to "
                    "security review')."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["summary"],
    },
}

KANBAN_APPROVE_SCHEMA = {
    "name": "kanban_approve",
    "description": (
        "Reviewer tool: terminate a review with one of three outcomes. "
        "`terminal` (-> done; optional follow_up_spec lands a child in "
        "backlog). `chained` (stays in review, reassigned to "
        "next_reviewer). `conditional` (-> todo, assignee pinned to the "
        "original worker for a small fix). Approver guard rejects "
        "self-approval: first-pass approver cannot equal the most "
        "recent worker; chained approver cannot equal the previous "
        "reviewer. Chain capped at kanban.max_review_passes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task id in the review column.",
            },
            "outcome": {
                "type": "string",
                "enum": ["terminal", "chained", "conditional"],
                "description": (
                    "Approval mode. `terminal` completes the task. "
                    "`chained` hands off to next_reviewer. "
                    "`conditional` returns to todo with the original "
                    "worker pinned."
                ),
            },
            "summary": {
                "type": "string",
                "description": "Signoff note recorded on the run + event.",
            },
            "follow_up_spec": {
                "type": "object",
                "description": (
                    "Only valid with outcome='terminal'. Creates a new "
                    "task in the `backlog` column linked under the "
                    "approved task. Required fields: title, assignee. "
                    "Optional: body, theme, sub_goals (list)."
                ),
                "properties": {
                    "title": {"type": "string"},
                    "assignee": {"type": "string"},
                    "body": {"type": "string"},
                    "theme": {"type": "string"},
                    "sub_goals": {"type": "array", "items": {"type": "string"}},
                },
            },
            "next_reviewer": {
                "type": "string",
                "description": (
                    "Required for outcome='chained'. Profile name of "
                    "the next reviewer; must not equal the current "
                    "approver."
                ),
            },
            "comment": {
                "type": "string",
                "description": (
                    "Optional for outcome='conditional'. Appended to "
                    "the task as a comment so the pinned worker reads "
                    "the requested change on their next claim."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id", "outcome"],
    },
}

KANBAN_REJECT_SCHEMA = {
    "name": "kanban_reject",
    "description": (
        "Reviewer tool: send the current review task to `blocked` with "
        "structured `findings`. The rejecting reviewer's profile is "
        "recorded so the sticky-reviewer logic routes the next review "
        "pass back to the same reviewer who has context on the "
        "original findings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task id in the review column (claimed).",
            },
            "findings": {
                "type": "object",
                "description": (
                    "Structured rejection payload. Recommended keys: "
                    "`reasons` (list[str]) and `requested_changes` "
                    "(list[str])."
                ),
                "properties": {
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "requested_changes": {
                        "type": "array", "items": {"type": "string"}
                    },
                },
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id", "findings"],
    },
}

KANBAN_REASSIGN_SCHEMA = {
    "name": "kanban_reassign",
    "description": (
        "Orchestrator tool: change a task's assignee with an optional "
        "handoff note. Capped at kanban.max_reassigns per task; the "
        "Nth+1 attempt errors so chains stop thrashing. Useful for "
        "access escalation (worker hands off to a lead, lead grants "
        "skills via kanban_edit, then reassigns back)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id."},
            "new_assignee": {
                "type": "string",
                "description": "New profile name.",
            },
            "handoff_note": {
                "type": "string",
                "description": (
                    "Optional rationale; appended as a comment so the "
                    "new assignee reads it via build_worker_context."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id", "new_assignee"],
    },
}

KANBAN_EDIT_SCHEMA = {
    "name": "kanban_edit",
    "description": (
        "Orchestrator tool: replace a task's `skills` list (full list, "
        "not a delta). Scope-locked to `skills` only; other fields "
        "(assignee, priority, max_runtime_seconds) stay CLI-only. "
        "Rejected on terminal status. On a running task the row "
        "updates but the live worker keeps its already-loaded skill "
        "set; the response carries applies_on_next_spawn=true so the "
        "caller knows whether to reassign or wait for natural respawn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id."},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Full desired skills list. Pass the complete list, "
                    "not a delta; an empty list explicitly clears skills."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id", "skills"],
    },
}

KANBAN_LINK_SCHEMA = {
    "name": "kanban_link",
    "description": (
        "Add a parent→child dependency edge after both tasks already "
        "exist. The child won't promote to 'ready' until all parents "
        "are 'done'. Cycles and self-links are rejected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parent_id": {"type": "string", "description": "Parent task id."},
            "child_id":  {"type": "string", "description": "Child task id."},
            "board": _board_schema_prop(),
        },
        "required": ["parent_id", "child_id"],
    },
}


# ---------------------------------------------------------------------------
# WS-5 + WS-8 handlers
# ---------------------------------------------------------------------------

def _handle_collate_children(args: dict, **kw) -> str:
    """Return done child task results for a parent/integrator task."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error("task_id is required")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        children = kb.collate_children(conn, tid)
        return json.dumps({"children": children, "count": len(children)}, indent=2, default=str)
    except Exception as exc:
        return tool_error(f"collate_children failed: {exc}")


def _handle_request_human_approval(args: dict, **kw) -> str:
    """Block a full-tier task awaiting explicit human go/no-go."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error("task_id is required")
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    detail = args.get("detail")
    if not detail or not str(detail).strip():
        return tool_error("detail is required")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            ok = kb.request_human_approval(
                conn, tid, detail=str(detail).strip(),
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(f"could not request approval for {tid}")
            run = kb.latest_run(conn, tid)
            return _ok(task_id=tid, run_id=run.id if run else None)
        finally:
            conn.close()
    except Exception as exc:
        return tool_error(f"request_human_approval failed: {exc}")


def _handle_request_subprofile(args: dict, **kw) -> str:
    """Request creation of a new sub-profile via the PROFILE-GATE.

    Blocks the task and records a pending profile_lifecycle_approval with
    op="create"; Sahil approves or rejects on Discord. The lead has no
    direct path to create profiles, only this request channel.
    """
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error("task_id is required")
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err

    profile = args.get("profile")
    if not profile or not str(profile).strip():
        return tool_error("profile is required")
    profile = str(profile).strip()

    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required")
    reason = str(reason).strip()

    clone_from = args.get("clone_from")
    if clone_from is not None:
        clone_from = str(clone_from).strip() or None

    try:
        from hermes_cli.profiles import profile_exists
    except Exception as exc:
        return tool_error(f"could not load hermes_cli.profiles: {exc}")

    if profile_exists(profile):
        return tool_error(f"profile '{profile}' already exists; refusing to request a duplicate")
    if clone_from and not profile_exists(clone_from):
        return tool_error(f"clone_from profile '{clone_from}' does not exist")

    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            from hermes_cli.profile_lifecycle_gate import submit_lifecycle_request
            from tools.skills_tool import _current_profile

            lifecycle_args = {"clone_from": clone_from} if clone_from else None
            approval_id = submit_lifecycle_request(
                conn, tid,
                op="create", profile=profile, args=lifecycle_args,
                requested_by=_current_profile(),
                expected_run_id=_worker_run_id(tid),
            )
            return _ok(
                task_id=tid,
                approval_id=approval_id,
                status="pending_human_approval",
                message=(
                    "Sub-profile creation requested; awaiting Sahil's "
                    "approval via Discord. The task is blocked until then."
                ),
            )
        finally:
            conn.close()
    except Exception as exc:
        return tool_error(f"request_subprofile failed: {exc}")


# WS-7: profile_editor handlers — call the profile_editor.py script
import subprocess as _sp
import json as _json

def _handle_profile_edit(args: dict, **kw) -> str:
    """Edit a profile file with git-backed version control.

    Gated by blast-radius (P2-3) for autonomous edits: edit caps,
    fleet-health tripwire, and canary validation.
    """
    profile = args.get("profile")
    file = args.get("file")
    key_path = args.get("key_path") or ""
    new_value = args.get("new_value")
    reason = args.get("reason")

    if not all([profile, file, new_value, reason]):
        return tool_error("profile, file, new_value, and reason are required")

    # ── Blast-radius gate (P2-3) ──
    guard = None
    canary_edit_id = None
    try:
        from hermes_cli.blast_radius import EditGuard, eval_domain_for
        # Fail CLOSED: refuse an autonomous edit to a profile with no
        # governance.eval_domains mapping. Without a mapping the canary
        # observe loop can never evaluate or revert the edit, so it would
        # persist unchecked. Require evaluability up front.
        if not eval_domain_for(profile):
            return tool_error(
                f"Profile '{profile}' has no governance.eval_domains mapping; "
                f"refusing autonomous edit (fail-closed). Map it to a domain "
                f"that has a golden task set before editing."
            )
        guard = EditGuard()
        result = guard.try_edit(
            profile=profile,
            patch=f"{file}:{key_path}={new_value}",
            patch_summary=reason,
        )
        if not result.allowed:
            return tool_error(
                f"Blast-radius guard blocked profile edit: {result.reason}. "
                f"{'Deferred until ' + result.defer_until if result.defer_until else ''}"
            )
        # Apply to canary — records the edit for observation
        if result.canary:
            guard.apply_canary(result.canary)
            canary_edit_id = result.canary.edit_id
    except ImportError:
        # Fail CLOSED: profile mutation is the most dangerous path in the
        # fork. If the blast-radius guard cannot load we refuse the edit
        # rather than proceeding unprotected. Matches the fail-closed
        # posture used everywhere else (e.g. skill_grants).
        return tool_error(
            "Blast-radius guard unavailable; refusing profile edit "
            "(fail-closed). Restore hermes_cli.blast_radius before editing "
            "profiles."
        )
    except Exception as exc:
        return tool_error(f"Blast-radius guard error: {exc}")

    script = str(Path(os.environ.get(
        "HERMES_HOME", "/home/kensei/.hermes"
    )) / "scripts" / "profile_editor.py")

    cmd = [
        sys.executable, script,
        str(profile), str(file), str(key_path), str(new_value), str(reason),
    ]
    try:
        r = _sp.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return tool_error(f"profile_edit failed: {_try_json_error(r)}")
        result = _json.loads(r.stdout)
        # Record the commit on the canary so the observe loop can revert
        # this exact edit if it later regresses (P2-3 close-the-loop).
        if guard is not None and canary_edit_id and result.get("commit"):
            try:
                guard.attach_commit(canary_edit_id, str(result["commit"]))
            except Exception:
                logger.warning("could not attach commit to canary %s", canary_edit_id)
        return _json.dumps(result, indent=2)
    except Exception as exc:
        return tool_error(f"profile_edit error: {exc}")


def _handle_profile_rollback(args: dict, **kw) -> str:
    """Revert a profile edit by commit hash.

    Gated by the same blast-radius guard as profile_edit: a rollback is a
    profile mutation and an agent could otherwise silently revert a
    security-hardening edit. A reason is required for the audit trail.
    Sahil retains the manual git/script path if the guard is down.
    """
    commit_hash = args.get("commit_hash")
    if not commit_hash:
        return tool_error("commit_hash is required")
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required (rollbacks are audited)")

    # ── Blast-radius gate (mirror of profile_edit, fail-closed) ──
    try:
        from hermes_cli.blast_radius import EditGuard
        guard = EditGuard()
        result = guard.try_edit(
            profile=str(args.get("profile") or "fleet"),
            patch=f"rollback:{commit_hash}",
            patch_summary=str(reason).strip(),
        )
        if not result.allowed:
            return tool_error(
                f"Blast-radius guard blocked profile rollback: {result.reason}. "
                f"{'Deferred until ' + result.defer_until if result.defer_until else ''}"
            )
    except ImportError:
        return tool_error(
            "Blast-radius guard unavailable; refusing profile rollback "
            "(fail-closed). Restore hermes_cli.blast_radius, or roll back "
            "manually via git."
        )
    except Exception as exc:
        return tool_error(f"Blast-radius guard error: {exc}")

    script = str(Path(os.environ.get(
        "HERMES_HOME", "/home/kensei/.hermes"
    )) / "scripts" / "profile_editor.py")

    cmd = [sys.executable, script, "--rollback", str(commit_hash)]
    try:
        r = _sp.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return tool_error(f"profile_rollback failed: {_try_json_error(r)}")
        result = _json.loads(r.stdout)
        return _json.dumps(result, indent=2)
    except Exception as exc:
        return tool_error(f"profile_rollback error: {exc}")


def _try_json_error(r) -> str:
    try:
        err = _json.loads(r.stdout)
        return err.get("error", r.stderr or "unknown error")
    except Exception:
        return r.stderr or r.stdout or "unknown error"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="kanban_show",
    toolset="kanban",
    schema=KANBAN_SHOW_SCHEMA,
    handler=_handle_show,
    check_fn=_check_kanban_mode,
    emoji="📋",
)

registry.register(
    name="kanban_list",
    toolset="kanban",
    schema=KANBAN_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="📋",
)

registry.register(
    name="kanban_complete",
    toolset="kanban",
    schema=KANBAN_COMPLETE_SCHEMA,
    handler=_handle_complete,
    check_fn=_check_kanban_mode,
    emoji="✔",
)

registry.register(
    name="kanban_block",
    toolset="kanban",
    schema=KANBAN_BLOCK_SCHEMA,
    handler=_handle_block,
    check_fn=_check_kanban_mode,
    emoji="⏸",
)

registry.register(
    name="kanban_request_review",
    toolset="kanban",
    schema=KANBAN_REQUEST_REVIEW_SCHEMA,
    handler=_handle_request_review,
    check_fn=_check_kanban_mode,
    emoji="👀",
)

registry.register(
    name="kanban_request_changes",
    toolset="kanban",
    schema=KANBAN_REQUEST_CHANGES_SCHEMA,
    handler=_handle_request_changes,
    check_fn=_check_kanban_mode,
    emoji="↩",
)

registry.register(
    name="kanban_heartbeat",
    toolset="kanban",
    schema=KANBAN_HEARTBEAT_SCHEMA,
    handler=_handle_heartbeat,
    check_fn=_check_kanban_mode,
    emoji="💓",
)

registry.register(
    name="kanban_comment",
    toolset="kanban",
    schema=KANBAN_COMMENT_SCHEMA,
    handler=_handle_comment,
    check_fn=_check_kanban_mode,
    emoji="💬",
)

registry.register(
    name="kanban_attach",
    toolset="kanban",
    schema=KANBAN_ATTACH_SCHEMA,
    handler=_handle_attach,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_attach_url",
    toolset="kanban",
    schema=KANBAN_ATTACH_URL_SCHEMA,
    handler=_handle_attach_url,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_attachments",
    toolset="kanban",
    schema=KANBAN_ATTACHMENTS_SCHEMA,
    handler=_handle_attachments,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_create",
    toolset="kanban",
    schema=KANBAN_CREATE_SCHEMA,
    handler=_handle_create,
    check_fn=_check_kanban_mode,
    emoji="➕",
)

registry.register(
    name="kanban_unblock",
    toolset="kanban",
    schema=KANBAN_UNBLOCK_SCHEMA,
    handler=_handle_unblock,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="▶",
)

registry.register(
    name="kanban_request_review",
    toolset="kanban",
    schema=KANBAN_REQUEST_REVIEW_SCHEMA,
    handler=_handle_request_review,
    check_fn=_check_kanban_mode,
    emoji="🔍",
)

registry.register(
    name="kanban_approve",
    toolset="kanban",
    schema=KANBAN_APPROVE_SCHEMA,
    handler=_handle_approve,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="✅",
)

registry.register(
    name="kanban_reject",
    toolset="kanban",
    schema=KANBAN_REJECT_SCHEMA,
    handler=_handle_reject,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="🚫",
)

registry.register(
    name="kanban_reassign",
    toolset="kanban",
    schema=KANBAN_REASSIGN_SCHEMA,
    handler=_handle_reassign,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="↔️",
)

registry.register(
    name="kanban_edit",
    toolset="kanban",
    schema=KANBAN_EDIT_SCHEMA,
    handler=_handle_edit,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="✏️",
)

registry.register(
    name="kanban_link",
    toolset="kanban",
    schema=KANBAN_LINK_SCHEMA,
    handler=_handle_link,
    check_fn=_check_kanban_mode,
    emoji="🔗",
)

# WS-5: collate_children — registers the function as a kanban tool so
# integrator/parent tasks can read done child results and produce a
# rolled-up deliverable before calling kanban_complete.
KANBAN_COLLATE_CHILDREN_SCHEMA = {
    "name": "kanban_collate_children",
    "description": (
        "Read the results of all done child tasks linked to this task. "
        "Returns a list of {id, title, result, assignee} dicts, ordered "
        "by completion time. Use this as an integrator/parent task to "
        "produce a rolled-up summary from completed child work before "
        "calling kanban_complete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "board": {
                "type": "string",
                "description": "Board slug — uses default when omitted",
            },
        },
        "required": [],
    },
}

# WS-8: request_human_approval — blocks a full-tier task awaiting
# explicit human go/no-go via the Discord approval handler.
KANBAN_REQUEST_HUMAN_APPROVAL_SCHEMA = {
    "name": "kanban_request_human_approval",
    "description": (
        "Block this task and request explicit human approval before it can "
        "proceed. Use for full-tier go/no-go decisions — the task will be "
        "blocked with reason 'needs_human_approval: <detail>' and must be "
        "explicitly unblocked by a human. The Discord approval handler cron "
        "surfaces these in #governance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "detail": {
                "type": "string",
                "description": "What the human needs to decide, in a sentence",
            },
            "board": {
                "type": "string",
                "description": "Board slug — uses default when omitted",
            },
        },
        "required": ["detail"],
    },
}

registry.register(
    name="kanban_collate_children",
    toolset="kanban",
    schema=KANBAN_COLLATE_CHILDREN_SCHEMA,
    handler=_handle_collate_children,
    check_fn=_check_kanban_mode,
    emoji="📊",
)

registry.register(
    name="kanban_request_human_approval",
    toolset="kanban",
    schema=KANBAN_REQUEST_HUMAN_APPROVAL_SCHEMA,
    handler=_handle_request_human_approval,
    check_fn=_check_kanban_mode,
    emoji="🛑",
)

# WS-7: profile_editor — governed profile mutation with git-backed rollback.
# Restricted to governance profiles (Denji, KENSEI) only.
KANBAN_PROFILE_EDIT_SCHEMA = {
    "name": "kanban_profile_edit",
    "description": (
        "Edit a profile's config.yaml, SOUL.md, or USER.md with git-backed "
        "version control. Every edit is committed to the profiles git repo "
        "and logged to the profile-change-ledger.md. Supports rollback via "
        "kanban_profile_rollback. RESTRICTED: governance profiles only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": "Profile name (e.g. 'octacon', 'wesker')",
            },
            "file": {
                "type": "string",
                "enum": ["config.yaml", "SOUL.md", "USER.md"],
                "description": "Which file to edit",
            },
            "key_path": {
                "type": "string",
                "description": "Dot-separated YAML key path for config.yaml "
                               "(e.g. 'agent.reasoning_effort'). Ignored for .md files.",
            },
            "new_value": {
                "type": "string",
                "description": "New value to set",
            },
            "reason": {
                "type": "string",
                "description": "Why this change is needed (logged to ledger)",
            },
        },
        "required": ["profile", "file", "new_value", "reason"],
    },
}

KANBAN_PROFILE_ROLLBACK_SCHEMA = {
    "name": "kanban_profile_rollback",
    "description": (
        "Revert a previous profile edit by commit hash. Only the most recent "
        "commits can be rolled back. RESTRICTED: governance profiles only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "commit_hash": {
                "type": "string",
                "description": "The git commit hash to revert (from kanban_profile_edit output)",
            },
            "reason": {
                "type": "string",
                "description": "Why the rollback is needed (required — rollbacks are audited and blast-radius gated)",
            },
            "profile": {
                "type": "string",
                "description": "Optional profile the rollback targets, for blast-radius scoping (defaults to fleet-wide)",
            },
        },
        "required": ["commit_hash", "reason"],
    },
}

registry.register(
    name="kanban_profile_edit",
    toolset="kanban",
    schema=KANBAN_PROFILE_EDIT_SCHEMA,
    handler=_handle_profile_edit,
    check_fn=_check_kanban_mode,
    emoji="✏️",
)

registry.register(
    name="kanban_profile_rollback",
    toolset="kanban",
    schema=KANBAN_PROFILE_ROLLBACK_SCHEMA,
    handler=_handle_profile_rollback,
    check_fn=_check_kanban_mode,
    emoji="↩️",
)

# PROFILE-GATE front door: lead requests a new sub-profile, gated by
# explicit human (Sahil) approval via Discord.
KANBAN_REQUEST_SUBPROFILE_SCHEMA = {
    "name": "kanban_request_subprofile",
    "description": (
        "Request creation of a new sub-profile. This does NOT create the "
        "profile directly: it blocks the task and records a pending "
        "PROFILE-GATE approval, which Sahil must explicitly approve or "
        "reject via Discord. The task stays blocked until that decision "
        "is made. Use this when a lead determines it needs a new "
        "specialist sub-profile (optionally cloned from an existing one)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "profile": {
                "type": "string",
                "description": "Name of the new sub-profile to create (e.g. 'remii-deep')",
            },
            "clone_from": {
                "type": "string",
                "description": "Optional existing profile to clone config/skills from",
            },
            "reason": {
                "type": "string",
                "description": "Why this sub-profile is needed, for the Discord approval prompt",
            },
            "board": {
                "type": "string",
                "description": "Board slug; uses default when omitted",
            },
        },
        "required": ["profile", "reason"],
    },
}

registry.register(
    name="kanban_request_subprofile",
    toolset="kanban",
    schema=KANBAN_REQUEST_SUBPROFILE_SCHEMA,
    handler=_handle_request_subprofile,
    check_fn=_check_kanban_mode,
    emoji="🐣",
)

# Pipeline worker completion — returns a pipeline worker to its originating
# stage (e.g. research, prd, spec) instead of marking the task done.
KANBAN_COMPLETE_PIPELINE_SCHEMA = {
    "name": "kanban_complete_pipeline",
    "description": (
        "Complete a pipeline-stage task (research/prd/spec audit) and "
        "return it to its originating stage so the gate can re-check on "
        "the next dispatcher tick.  Use this when you have written the "
        "stage artifact (research-brief.md, prd.md, spec.md) and want "
        "the pipeline to advance.  Do NOT use kanban_complete — that "
        "would mark the task done and break the pipeline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "The task id.  Defaults to HERMES_KANBAN_TASK."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "Human-readable summary of what was done (preferred "
                    "over result for handoff context)."
                ),
            },
            "result": {
                "type": "string",
                "description": (
                    "The raw result / output produced by this stage."
                ),
            },
            "board": {
                "type": "string",
                "description": "Board slug (e.g. apps, research). Auto-detected if omitted.",
            },
        },
    },
}

registry.register(
    name="kanban_complete_pipeline",
    toolset="kanban",
    schema=KANBAN_COMPLETE_PIPELINE_SCHEMA,
    handler=_handle_complete_pipeline,
    check_fn=_check_kanban_mode,
    emoji="🔁",
)

