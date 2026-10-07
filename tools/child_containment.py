"""Fail-closed containment registry for worktree-isolated delegated children.

Goal A of the delegation hardening: when a delegated child has an ACTIVE
isolated worktree, a mutating operation that resolves OUTSIDE that child's
approved worktree must fail closed BEFORE mutation — for the structured
mutation paths Hermes actually mediates:

* ``terminal`` with an explicit ``workdir`` (the per-command cwd override)
* ``write_file`` / ``patch`` (and their V4A multi-file headers)
* the post-command session-cwd recorder (an escaped anchor must not be
  persisted as the child's new base directory)

Enforcement is keyed on the PARENT-SIDE registry by the child's ``task_id``
— the same trust anchor as the verified cwd-key fix — not on the
delegated-child ContextVar. That choice closes a real bypass: a child's
``execute_code`` kernel calls tools back through RPC handler threads where
the ContextVar is absent, so a contextvar-gated guard would let a
kernel-mediated ``write_file`` through unguarded. The registry is populated
by the parent at dispatch (in this process) and is visible from every
thread. A task id that was never registered belongs to a non-child caller
(or a child whose parent did not opt into worktree isolation): those keep
their historical behavior exactly — the mechanism is opt-in per dispatch,
like the isolation itself.

What this module is NOT (stated plainly, per the hardening contract):
* It is NOT OS-level sandboxing. A child's arbitrary shell text (``cd``,
  absolute paths, redirection, spawning processes) and raw ``execute_code``
  kernel code that touches the filesystem directly (``open()``, ``os.*``)
  run with the full privileges of the Hermes process and CANNOT be contained
  by any in-process guard. Those mechanisms are MATERIAL residual risks and
  are reported as such; true containment there requires OS-level sandboxing
  (containers, seatbelt, AppContainer) which Hermes does not provide on the
  local backend.
* It is not adversarial-proof against a hostile child that shells out.
  It is fail-closed for every STRUCTURED mutation path Hermes mediates.

Registry lifecycle mirrors ``_ChildRun``: the parent registers the worktree
in ``_create_isolated_worktree`` (before the child conversation starts) and
unregisters it in ``_ChildRun.cleanup`` (finally-path teardown).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

logger = logging.getLogger("tools.delegate_tool")

# child task_id -> {"path": approved worktree root, "repo_root": parent repo root}
_active_child_worktrees: Dict[str, Dict[str, str]] = {}
_registry_lock = threading.Lock()

# Reason text shared by every denying guard so denials are recognizable.
DENIED_PREFIX = "Blocked: delegated-child containment"


def register_child_worktree(child_task_id: str, worktree_info: Dict[str, str]) -> None:
    """Record the approved worktree for *child_task_id* (parent side, at dispatch).

    Idempotent per child; overwrites a stale entry for the same id (a re-dispatch
    under the same id must not keep the previous run's worktree approved).
    """
    path = str((worktree_info or {}).get("path") or "")
    if not child_task_id or not path:
        return
    with _registry_lock:
        _active_child_worktrees[str(child_task_id)] = {
            "path": os.path.realpath(path),
            "repo_root": os.path.realpath(str((worktree_info or {}).get("repo_root") or path)),
        }
    logger.info("child containment: approved worktree for %s -> %s", child_task_id, path)


def unregister_child_worktree(child_task_id: str) -> None:
    """Drop the approval when the child run finishes (finally-path teardown)."""
    if not child_task_id:
        return
    with _registry_lock:
        removed = _active_child_worktrees.pop(str(child_task_id), None)
    if removed is not None:
        logger.info("child containment: released worktree approval for %s", child_task_id)


def approved_worktree_for(child_task_id: Optional[str]) -> Optional[str]:
    """The approved worktree root for *child_task_id*, or None when not isolated."""
    if not child_task_id:
        return None
    with _registry_lock:
        entry = _active_child_worktrees.get(str(child_task_id))
    return entry["path"] if entry else None


def _norm(path: Optional[str]) -> str:
    """Best-effort normalized absolute form; empty string when unresolvable.

    On Windows a Git Bash/MSYS-form path (``/c/Users/...``) is first translated
    to native form so a workdir/target spelled either way compares equal —
    without this the guard would FALSELY DENY an in-worktree MSYS-form path.
    """
    try:
        text = str(path or "")
        if os.name == "nt":
            from tools.environments.local import _msys_to_windows_path

            text = _msys_to_windows_path(text)
        return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(text))))
    except Exception:
        return ""


def is_within(child_path: str, root: str) -> bool:
    """True when *child_path* equals or lies under *root* (case/symlink-normalized)."""
    c, r = _norm(child_path), _norm(root)
    if not c or not r:
        return False
    return c == r or c.startswith(r + os.sep)


def _denial(operation: str, target: str, worktree: str, hint: str = "") -> str:
    base = (
        f"{DENIED_PREFIX}: {operation} resolving outside this delegated child's "
        f"approved worktree was refused before mutation. Approved worktree: "
        f"{worktree}. Target: {target}. The target was NOT modified."
    )
    if hint:
        base += f" {hint}"
    return base


def check_terminal_workdir(*, workdir: Optional[str], task_id: Optional[str]) -> Optional[str]:
    """Guard a child's explicit terminal ``workdir``.

    Returns the denial error string when a worktree-isolated child asked to run
    a command in a directory outside its approved worktree, else None. Runs
    BEFORE the command executes, so nothing has mutated yet.

    A child recorded as downgraded (isolation requested but failed — Goal B)
    is denied any explicit workdir: it is not repository-isolated, so
    directory-targeted commands are refused. Workdir-less commands keep
    resolving to the child's inherited cwd. Task ids with no registry entry
    (isolation never engaged for them) keep historical behavior.
    """
    if not workdir:
        return None
    worktree = approved_worktree_for(task_id)
    if worktree is not None:
        if is_within(workdir, worktree):
            return None
        return _denial("a terminal command's workdir", str(workdir), worktree)
    if is_isolation_downgraded(task_id):
        record = downgrade_record(task_id) or {}
        return (
            f"{DENIED_PREFIX}: a terminal command's explicit workdir was refused for a "
            f"delegated child whose worktree isolation FAILED (reason: "
            f"{record.get('reason', 'unknown')}). This child is NOT repository-isolated; "
            f"commands may run only without a workdir, from its inherited cwd. "
            f"Target: {workdir}. Nothing was executed."
        )
    return None


def check_mutation_target(
    *, target: Optional[str], task_id: Optional[str], operation: str = "a file write",
) -> Optional[str]:
    """Guard one resolved write target (write_file / patch / V4A headers).

    Returns the denial error string when a worktree-isolated child's resolved
    target lies outside its approved worktree, else None. The caller must run
    this BEFORE any disk mutation. Task ids with no registry entry no-op
    (historical behavior for non-isolated callers).
    """
    worktree = approved_worktree_for(task_id)
    if worktree is None or target is None:
        return None
    if is_within(target, worktree):
        return None
    return _denial(operation, str(target), worktree)


def check_session_cwd_record(*, proposed_cwd: Optional[str], task_id: Optional[str]) -> bool:
    """May the post-command recorder persist *proposed_cwd* as this child's cwd?

    False when a worktree-isolated child's observed cwd escaped its approved
    worktree (an in-shell ``cd`` escape): the escaped directory must not become
    the child's new base directory for later commands and file-tool anchoring.
    The command itself already ran (arbitrary shell is outside containment by
    design — MATERIAL residual); this guard stops the escape from PERSISTING.
    Task ids with no registry entry are untouched (historical behavior).
    """
    worktree = approved_worktree_for(task_id)
    if worktree is None or not proposed_cwd:
        return True
    return is_within(proposed_cwd, worktree)


# ── Goal B: explicit isolation-downgrade records ─────────────────────────
# A repo-isolated delegation that could not establish a worktree must never be
# silently treated as isolated. The dispatch records the downgrade here; the
# child result entry carries it; and repo-mutating tool calls for a downgraded
# child are denied (fail-closed for repo work).

_downgraded_children: Dict[str, Dict[str, str]] = {}


def register_isolation_downgrade(child_task_id: str, reason: str, parent_cwd: str = "") -> None:
    """Record that *child_task_id* requested worktree isolation but did not get it."""
    if not child_task_id:
        return
    with _registry_lock:
        _downgraded_children[str(child_task_id)] = {
            "reason": str(reason or "unknown"), "parent_cwd": str(parent_cwd or "")}
    logger.warning(
        "child containment: worktree isolation FAILED for %s (%s) — child is NOT "
        "repository-isolated; repo-mutating tool calls will be denied", child_task_id, reason)


def is_isolation_downgraded(child_task_id: Optional[str]) -> bool:
    """True when *child_task_id* was recorded as a failed-isolation downgrade."""
    if not child_task_id:
        return False
    with _registry_lock:
        return str(child_task_id) in _downgraded_children


def downgrade_record(child_task_id: Optional[str]) -> Optional[Dict[str, str]]:
    """The recorded downgrade entry for *child_task_id*, or None."""
    if not child_task_id:
        return None
    with _registry_lock:
        entry = _downgraded_children.get(str(child_task_id))
        return dict(entry) if entry else None


def unregister_isolation_downgrade(child_task_id: str) -> None:
    """Drop the downgrade record when the child run finishes."""
    if not child_task_id:
        return
    with _registry_lock:
        _downgraded_children.pop(str(child_task_id), None)


def check_downgraded_repo_mutation(
    *, task_id: Optional[str], operation: str = "a file write",
) -> Optional[str]:
    """Goal B guard: a downgraded child may not write files via the covered tools.

    A downgraded child shares the PARENT's workspace (no worktree exists), so a
    write from it can hit the parent checkout. Fail closed: deny file writes via
    write_file/patch for downgraded children rather than let them run in a
    workspace that was never approved as theirs. Task ids with no registry
    entry no-op (historical behavior for everyone else).
    """
    if not is_isolation_downgraded(task_id):
        return None
    record = downgrade_record(task_id) or {}
    reason = record.get("reason", "unknown")
    return (
        f"{DENIED_PREFIX}: {operation} refused for a delegated child whose worktree "
        f"isolation FAILED and was NOT silently downgraded (reason: {reason}). This "
        f"child is NOT repository-isolated; it shares the parent's workspace, so a "
        f"write could hit the parent checkout. The target was NOT modified."
    )


def _reset_for_tests() -> None:
    """Test helper: clear both registries."""
    with _registry_lock:
        _active_child_worktrees.clear()
        _downgraded_children.clear()
