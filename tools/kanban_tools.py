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
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Optional

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
    task_id: str, metadata: Optional[dict], *, finalize_conclusive: bool = False
) -> Optional[dict]:
    """Add trusted worker session id metadata for this worker's own task."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return metadata
    # Only build a new dict when there is actually something to stamp — a plain
    # worker run (no session id, no finalize turn) must return the input
    # unchanged so callers that treat a `None` metadata as absent keep working.
    to_stamp: dict = {}
    session_id = os.environ.get("HERMES_SESSION_ID")
    if session_id:
        to_stamp["worker_session_id"] = session_id
    # Finalize-in-process instrumentation: if a forced finalize turn fired this
    # run, stamp whether it ultimately produced a terminal tool. Reading the
    # module flags here (a terminal-tool handler) is correct: reaching a
    # terminal tool WITH a prior finalize turn proves the finalize closed the
    # run instead of falling through to a protocol violation. If the worker
    # instead exited cleanly without ever calling a terminal tool, the run is
    # closed by the dispatcher as a protocol violation and the metrics snapshot
    # was already recorded at the moment the finalize turn fired (see the loop
    # hook) — this stamp just ties the successful path to the run row.
    #
    # `finalize_conclusive` distinguishes the two terminal classes: `finalize
    # _turn_succeeded` is only marked True by a CONCLUSIVE close (kanban_complete;
    # kanban_block writes no metadata so carries no stamp). A review/return
    # handoff (kanban_request_review / changes) is also a valid terminal that
    # closes the run, but it does NOT mark success — doing so would skew the
    # measured "finalize concluded the work" rate with handoffs that hand off
    # rather than conclude. For those, only the fired flag is stamped.
    # 2026-09-07: the finalize-turn metrics stamp was retired with the turn
    # itself. `finalize_conclusive` stays in the signature so callers keep
    # working; it no longer drives a stamp. The stop-nudge is measured by the
    # dispatcher's protocol_violation accounting, which needs no per-process
    # global to be correct.
    if not to_stamp:
        return metadata
    stamped = dict(metadata or {})
    stamped.update(to_stamp)
    return stamped


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


_UNCOMMITTED_MARKER = "completion_blocked_uncommitted_work"


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


def _complete_uncommitted_work_rejection(task_id: str) -> Optional[str]:
    """Refuse ``kanban_complete`` on a ``dir`` card with uncommitted source.

    **2026-09-05, from the 2026-09-04 work-loss incident.** Cards `t_d354546b`,
    `t_f6325bc3`, `t_fe5cb6f5` and `t_1fbada04` were built, verified live by
    Rodge, marked ``done`` — and the commit exists in no branch and no reflog.
    All four were ``workspace_kind='dir'`` pointing at the SHARED BackupBrain
    working tree, so the edits were never committed and the next card clobbered
    them. 146 of the board's 147 ``dir`` cards share that exposure.

    The charter's rule 3 ("done requires a commit") was written as prose. This
    is the enforcement: a ``dir`` card cannot report done while tracked edits
    sit uncommitted in its workspace.

    Deliberately NARROW and fail-open:
      * worker completions only (orchestrator / CLI pass straight through);
      * ``dir`` workspaces only — a ``worktree`` card has its own branch, and a
        ``scratch`` card has no repo to lose;
      * tracked changes only, never untracked scratch;
      * any uncertainty (no git, not a repo, timeout) allows the completion.

    Never crashes the run. The work is real and uncommitted — closing the run
    is exactly how it gets lost — so a repeat refusal points the worker at
    ``kanban_block`` instead, which is a terminal call the escalator now routes.
    """
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None  # orchestrator / CLI path.
    kb, conn = _connect()
    try:
        task = kb.get_task(conn, task_id)
        if task is None or (task.workspace_kind or "") != "dir":
            return None
        dirty = _dir_workspace_uncommitted(task.workspace_path or "")
        if not dirty:
            return None
        run_id = _worker_run_id(task_id)
        with kb.write_txn(conn):
            kb._append_event(
                conn, task_id, _UNCOMMITTED_MARKER,
                {"violation_class": "uncommitted_dir_workspace",
                 "files": dirty[:20], "count": len(dirty)},
                run_id=run_id,
            )
        repeat = _consecutive_uncommitted_blocks(conn, task_id) > 1
    finally:
        conn.close()

    shown = "\n".join(f"  - {p}" for p in dirty[:15])
    more = f"\n  ...and {len(dirty) - 15} more" if len(dirty) > 15 else ""
    if not repeat:
        return tool_error(
            "kanban_complete rejected: this card uses a `dir` workspace and has "
            f"{len(dirty)} TRACKED file(s) modified but NOT COMMITTED:\n"
            f"{shown}{more}\n\n"
            "A `dir` workspace is a SHARED working tree with no branch of its "
            "own. On 2026-09-04 four cards were built, reviewed and marked done "
            "exactly like this, and the work was clobbered by the next card — "
            "it exists in no branch and no reflog. Commit your work before "
            "reporting done:\n"
            "  git -C <workspace> add -A && git -C <workspace> commit -m '<what you did>'\n"
            "then call kanban_complete again with the commit SHA in your handoff. "
            "Your task is still in-flight; nothing was changed."
        )
    return tool_error(
        "kanban_complete rejected again: tracked edits are still uncommitted in "
        f"this `dir` workspace ({len(dirty)} file(s)). Do NOT keep retrying the "
        "completion — the run closing with this work uncommitted is precisely "
        "how it gets lost. If you cannot commit (no branch, wrong base, "
        "conflicting tree, missing identity), call kanban_block with the reason "
        "and the file list so it routes to someone who can. Blocking is safe; "
        "completing is not."
    )


def _consecutive_uncommitted_blocks(conn, task_id: str) -> int:
    """How many uncommitted-work refusals in a row on this card."""
    count = 0
    try:
        rows = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 40",
            (task_id,),
        ).fetchall()
    except Exception:
        return 1
    for row in rows:
        if (row[0] if not isinstance(row, dict) else row["kind"]) == _UNCOMMITTED_MARKER:
            count += 1
        elif count:
            break
    return count


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
        # phase for re-dispatch.
        kb._record_task_failure(
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


# ---------------------------------------------------------------------------
# Pre-review build gate (zero-token)
# ---------------------------------------------------------------------------
# A worker calling ``kanban_request_review`` on a card backed by a git
# worktree gets its work gated *before* the transition is accepted: the
# project's gate command (default ``python -m pytest <focused tests> -q``)
# plus an import/build sanity check of the changed python files.  The gate
# is pure subprocess — no LLM tokens.  On failure the review is refused, a
# comment carrying the last ~TAIL lines of gate output is posted, and the
# card stays in its current (builder) lane with no failure counted.  A card
# with a non-worktree workspace, or a worktree we cannot resolve a python
# for, skips the gate (never blocks the happy path).
#
# Real-world motivation (2026-08-31): an unbuildable commit (ce14358,
# imports from untracked files) reached review and cost ~8 cards and 90
# minutes of churn.  The cheapest gates run first: build/tests as an
# automated block BEFORE any LLM review.

PRE_REVIEW_TAIL_LINES = 30
_GATE_RUN_TIMEOUT = 600  # generous: a focused suite can take minutes
_PYTHON_CACHE: dict[str, Optional[str]] = {}


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


_BASE_ARCHIVE_CACHE: dict[str, Optional[str]] = {}


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


# ---------------- Review-gate ladder (cheapest rung first) ----------------
# The pre-review gate is an ordered ladder: lint -> typecheck -> import/build
# (today's check) -> focused tests.  Every rung that bounces a card saves a
# full LLM review run (review is the fleet's most expensive step), and the
# cheapest rungs run first.  Order is fixed: lint and typecheck are nearly
# free vs import/tests, so they go ahead of the constructive checks.
#
# A rung whose tool is not available in the project is skipped (logged, not a
# failure) so a project with no linter never blocks.  Per-project commands use
# the same override mechanism as the focused-tests rung (``kanban.review_gate
# .lint_command`` / ``.typecheck_command``), defaulting to a sane tool; the
# tool resolution is cached per (rung, python) so the happy path does not
# re-probe the venv.

_LINT_TOOLS = ("ruff", "flake8", "pylint")
_TYPECHECK_TOOLS = ("mypy", "pyright", "basedpyright")
# (rung_key, project_python) -> resolved tool name or None (=> rung skipped)
_TOOL_CACHE: dict[tuple[str, str], Optional[str]] = {}
# project_python -> pytest importable under that interpreter (bool)
_PYTEST_CACHE: dict[str, bool] = {}


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


def _goal_mode_handoff_rejection(task, evidence: str):
    """Return ``(verdict, reason_or_None)`` for a goal-mode terminal handoff.

    ``{"done", None}`` means the judge allows the handoff; anything else is
    a rejection whose verdict disambiguates the guidance the caller gives
    the worker (``continue`` = not done yet, ``blocked`` = judged
    unachievable — see #100954).
    """
    if not task or not task.goal_mode or not _goal_judge_available():
        return ("done", None)
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
    return (verdict, None if verdict == "done" else reason)


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
    metadata = _stamp_worker_session_metadata(tid, metadata, finalize_conclusive=True)
    # Tool-evidence gate: refuse a completion from a run that made ZERO
    # non-kanban tool calls (no evidence the work was done). Fails open for
    # orchestrator/CLI paths and unreadable transcripts (t_8fc16a73).
    evidence_rejection = _complete_tool_evidence_rejection(tid)
    if evidence_rejection is not None:
        return evidence_rejection
    # Uncommitted-work gate: a `dir` card cannot report done while TRACKED
    # source edits sit uncommitted in its shared working tree. This is the
    # enforcement of charter rule 3 ("done requires a commit") after the
    # 2026-09-04 loss of B1/B2/B3. Fails open on anything uncertain.
    uncommitted_rejection = _complete_uncommitted_work_rejection(tid)
    if uncommitted_rejection is not None:
        return uncommitted_rejection
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
            gate_verdict, rejection = _goal_mode_handoff_rejection(
                task,
                (summary or result or "").strip(),
            )
            if gate_verdict == "blocked":
                return tool_error(
                    f"Goal completion rejected: judge ruled the goal "
                    f"unachievable — {rejection}. The task will NOT complete "
                    f"silently. Either re-scope the task with kanban_edit, "
                    f"or record the block with kanban_block and hand the "
                    f"decision to a human / reviewer."
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
    # A review handoff is a valid terminal close but NOT a conclusive complete:
    # stamp worker session + the fired flag, never finalize_turn_succeeded.
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
            gate_verdict, rejection = _goal_mode_handoff_rejection(task, summary)
            if gate_verdict == "blocked":
                return tool_error(
                    f"Goal review handoff rejected: judge ruled the goal "
                    f"unachievable — {rejection}. Record the block with "
                    f"kanban_block instead of requesting review."
                )
            if rejection is not None:
                return tool_error(
                    f"Goal review handoff rejected by judge: {rejection}. "
                    "Provide acceptance evidence matching the card before "
                    "requesting review."
                )
            # Pre-review build gate: refuse the transition when a worktree
            # card fails any rung of the review ladder (lint, typecheck,
            # import/build, focused tests).  Zero LLM tokens; on failure the
            # card stays in its current (builder) lane, an auto-comment names
            # the failing rung and carries only that rung's output, and no
            # failure is counted against the card.
            gate_bounce = _run_pre_review_gate(task)
            if gate_bounce is not None:
                tail = _run_gate_output_tail(gate_bounce.output)
                kb.add_comment(
                    conn,
                    tid,
                    author="pre-review-gate",
                    body=(
                        f"Pre-review gate FAILED on the '{gate_bounce.rung}' "
                        "rung — review was not started.\n\nThe worktree must "
                        "pass the review ladder (lint, typecheck, "
                        "import/build, focused tests) before entering review. "
                        "Fix the failure and request review again.\n\n"
                        "```\n"
                        + tail
                        + "\n```"
                    ),
                )
                # Record WHICH rung bounced so the ladder's own value is
                # measurable (e.g. if lint never catches anything in a month,
                # that rung is removable).
                with kb.write_txn(conn):
                    kb._append_event(
                        conn,
                        tid,
                        "gate_bounced",
                        {"rung": gate_bounce.rung},
                        run_id=_worker_run_id(tid),
                    )
                return tool_error(
                    f"Pre-review gate failed for {tid} on the "
                    f"'{gate_bounce.rung}' rung; review not started. The task "
                    "stays in its current lane (no failure counted) and a "
                    f"comment carries the last {PRE_REVIEW_TAIL_LINES} lines "
                    f"of {gate_bounce.rung} output. Fix the "
                    f"{gate_bounce.rung} and call kanban_request_review "
                    "again.\n\n"
                    + tail
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
    hold, bool_error = _parse_bool_arg(args, "hold")
    if bool_error:
        return tool_error(bool_error)
    if hold:
        # Charter §5 (2026-09-03): a held card is blocked/operator_hold from
        # birth; only a human `kanban unblock` releases it.
        initial_status = "blocked"
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
            # Explicit max_cost wins; otherwise inherit
            # kanban.default_max_cost exactly as the CLI does (single
            # shared helper). Absent config key = None = uncapped
            # (backward compat). Garbage/negative config is rejected, not
            # silently uncapped.
            max_cost = args.get("max_cost")
            if max_cost is None:
                try:
                    max_cost = kb.resolve_default_max_cost()
                except ValueError as exc:
                    return tool_error(f"kanban_create: {exc}")
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
            _parked: dict = {}
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
                max_cost=max_cost,
                skills=skills,
                model_override=model_override,
                provider_override=provider_override,
                goal_mode=goal_mode,
                goal_max_turns=(
                    int(goal_max_turns) if goal_max_turns is not None else None
                ),
                initial_status=str(initial_status),
                block_kind=("operator_hold" if hold else None),
                created_by=os.environ.get("HERMES_PROFILE") or "worker",
                session_id=session_id,
                _assignee_parked=_parked,
            )
            new_task = kb.get_task(conn, new_tid)
            subscribed = _maybe_auto_subscribe(conn, new_tid)
            payload = dict(
                task_id=new_tid,
                status=new_task.status if new_task else None,
                workspace_kind=new_task.workspace_kind if new_task else None,
                workspace_path=new_task.workspace_path if new_task else None,
                project_id=new_task.project_id if new_task else None,
                subscribed=subscribed,
            )
            if _parked:
                payload["assignee_parked"] = _parked.get("assignee")
                payload["notice"] = (
                    f"assignee {_parked.get('assignee')!r} is not a real profile; "
                    "task parked in triage for the PM to accept/reject. "
                    "Transfer to a valid assignee to dispatch."
                )
            return _ok(**payload)
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
    """Transition a blocked task to ready, or todo while parents remain open."""
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
                return tool_error(
                    f"could not unblock/accept {tid} (not blocked/scheduled/triage or unknown)"
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
                    "blocked", "done", "archived",
                ],
                "description": "Optional task status filter.",
            },
            "tenant": {
                "type": "string",
                "description": "Optional tenant/project namespace filter.",
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
            "max_cost": {
                "type": "number",
                "description": (
                    "Optional per-task cumulative spend cap in USD. When "
                    "omitted, inherits kanban.default_max_cost from config "
                    "(0.60 default) exactly like the CLI; an absent config "
                    "key leaves the card uncapped."
                ),
            },
            "hold": {
                "type": "boolean",
                "description": (
                    "Create the card HELD (blocked / operator_hold). Nothing runs "
                    "until Richie unblocks it. Charter §5: every job parent and "
                    "every deploy card is created held. Defaults to false."
                ),
            },
            "initial_status": {
                "type": "string",
                "enum": ["running", "blocked"],
                "description": (
                    "Initial card status. Use 'blocked' for tasks that "
                    "require immediate human ops (R3 gate) to skip the "
                    "brief running-to-blocked transition. Defaults to "
                    "'running', which preserves the usual dispatch path."
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
        "Unblock or accept a Kanban task. A blocked/scheduled task moves to "
        "ready when all parents are done, or todo while any parent remains "
        "open. A triage task (a parked decision) is accepted: triage -> todo, "
        "then lifted to ready. Orchestrator-only — only profiles with the "
        "kanban toolset can unblock/accept routed work; dispatcher-spawned "
        "task workers never see this tool."
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
    name="kanban_link",
    toolset="kanban",
    schema=KANBAN_LINK_SCHEMA,
    handler=_handle_link,
    check_fn=_check_kanban_mode,
    emoji="🔗",
)
